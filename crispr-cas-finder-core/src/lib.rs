#![feature(portable_simd)]

// Bundled implementation shared with the unpublished compatibility crates.
// Public only for the workspace's bindings, benchmarks, and regression tests.
#[doc(hidden)]
pub mod hmmer_core;
#[doc(hidden)]
pub mod hmmer_io;

use anyhow::Result;
use log::info;
use rayon::prelude::*;
use std::borrow::Cow;
use std::path::Path;

pub mod annotation;
pub mod association;
pub mod cas_pipeline;
pub mod cas_types;
pub mod casfinder;
pub mod casparser;
pub mod conservation;
pub mod detect;
mod dna;
pub mod io;
pub mod macsy;
pub mod repeat_db;
pub mod report;
pub mod types;
pub mod vmatch;

pub use cas_pipeline::{GeneRecord, ModelDefinition};
pub use cas_types::RepliconTopology;
pub use casparser::{CasCluster, CasGene, GeneCoordinates};
pub use conservation::ArrayStatistics;
pub use repeat_db::{RepeatLookup, lookup_repeat};
pub use report::{
    Analysis, AnalysisOptions, AnalysisReport, SequenceRecord, analyze, build_report, parse_fasta,
};
pub use types::{CasFinderConfig, CrisprArray, DetectionParams, Orientation, Repeat, Spacer};

use detect::{
    RepeatHit, SpacerSpan, allowed_copy_mismatches, candidate_score, check_short_crispr,
    cluster_region, cluster_repeats, define_spacers, extend_with_truncated_repeats,
    find_direct_repeats, infer_orientation, repeat_candidates, repeat_occurrences, slide_search,
};

/// Detect CRISPR arrays in a set of sequences.
///
/// This is the main programmatic entry point. It operates entirely in memory —
/// no file I/O, no subprocess calls. The caller provides sequences directly
/// and gets back structured results.
///
/// The case of a sequence does not change the arrays found. Repeats and
/// spacers are returned in the case the sequence has them in, the consensus
/// repeat in uppercase.
///
/// # Arguments
/// * `sequences` — slice of `(seq_id, sequence_bytes)` pairs
/// * `params` — algorithm parameters controlling detection
///
/// # Returns
/// A `Vec<CrisprArray>` of detected arrays, each with repeats, spacers, and
/// an evidence level.
pub fn detect_crisprs(sequences: &[(&str, &[u8])], params: &DetectionParams) -> Vec<CrisprArray> {
    let params = &*params.effective();
    let min_repeat_count = params.min_spacer_count.max(1) + 1;
    // Small arrays must survive detection to be reclassified, so the
    // evidence filter is applied afterwards in that case.
    let search_params = if params.classify_small_arrays && params.min_evidence_level > 1 {
        std::borrow::Cow::Owned(DetectionParams {
            min_evidence_level: 1,
            ..params.clone()
        })
    } else {
        std::borrow::Cow::Borrowed(params)
    };
    let search_params = &*search_params;

    let mut arrays: Vec<CrisprArray> = if sequences.len() > 1 {
        let arrays_by_sequence: Vec<Vec<CrisprArray>> = sequences
            .par_iter()
            .map(|&(seq_id, seq)| {
                detect_crisprs_for_sequence(seq_id, seq, search_params, min_repeat_count)
            })
            .collect();
        arrays_by_sequence.into_iter().flatten().collect()
    } else {
        sequences
            .iter()
            .flat_map(|&(seq_id, seq)| {
                detect_crisprs_for_sequence(seq_id, seq, search_params, min_repeat_count)
            })
            .collect()
    };

    if params.classify_small_arrays {
        reclassify_small_arrays(&mut arrays);
        arrays.retain(|array| array.evidence_level >= params.min_evidence_level);
    }

    let mut arrays_per_sequence: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for array in &mut arrays {
        let count = arrays_per_sequence.entry(array.seq_id.clone()).or_default();
        *count += 1;
        array.id = format!("{}_CRISPR_{count}", array.seq_id);
    }
    arrays
}

/// Raise level-1 arrays to level 4 when their consensus repeat, or its
/// reverse complement, is the consensus of a level-4 array in the input.
pub fn reclassify_small_arrays(arrays: &mut [CrisprArray]) {
    let mut trusted_repeats = std::collections::HashSet::new();
    for array in arrays.iter().filter(|array| array.evidence_level == 4) {
        let repeat = array.consensus_repeat.to_ascii_uppercase();
        trusted_repeats.insert(
            String::from_utf8_lossy(&dna::reverse_complement(repeat.as_bytes())).into_owned(),
        );
        trusted_repeats.insert(repeat);
    }
    for array in arrays.iter_mut().filter(|array| array.evidence_level == 1) {
        if trusted_repeats.contains(&array.consensus_repeat.to_ascii_uppercase()) {
            array.reclassified_from = Some(1);
            array.evidence_level = 4;
        }
    }
}

fn detect_crisprs_for_sequence(
    seq_id: &str,
    seq: &[u8],
    params: &DetectionParams,
    min_repeat_count: usize,
) -> Vec<CrisprArray> {
    const GAP_THRESHOLD: usize = 1500;

    if seq.len() < params.min_sequence_length {
        return Vec::new();
    }

    // Vmatch and fuzznuc ignore case, and a masked stretch may end anywhere
    // in an array. Arrays are searched in an uppercase copy, made only if
    // the sequence holds lowercase letters, so that every comparison below
    // ignores case and the consensus repeat is in uppercase.
    let input = seq;
    let uppercase = dna::uppercase(input);
    let seq = &*uppercase;

    let hits = find_direct_repeats(seq, params);
    info!("  {} — {} DR candidates", seq_id, hits.len());
    let occurrences = repeat_occurrences(&hits);

    let mut arrays = Vec::new();
    for cluster in cluster_repeats(&hits, GAP_THRESHOLD) {
        // Hits are sorted by position.
        let first = hits.partition_point(|hit| hit.pos1 < cluster.0);
        let after = hits.partition_point(|hit| hit.pos1 <= cluster.1);
        arrays.extend(process_cluster(
            seq_id,
            seq,
            &hits[first..after],
            cluster,
            &occurrences,
            params,
            min_repeat_count,
        ));
    }

    if matches!(uppercase, Cow::Owned(_)) {
        for array in &mut arrays {
            restore_input_case(array, input);
        }
    }
    arrays
}

/// Give the repeats and spacers of an array found in an uppercase copy of
/// `input` the case they have in `input`: they are reported as the input has
/// them, as CRISPRCasFinder does.
fn restore_input_case(array: &mut CrisprArray, input: &[u8]) {
    // `start` and `end` are 1-based and inclusive.
    let written =
        |start: usize, end: usize| String::from_utf8_lossy(&input[start - 1..end]).to_string();
    for repeat in &mut array.repeats {
        repeat.sequence = written(repeat.start, repeat.end);
    }
    for spacer in &mut array.spacers {
        spacer.sequence = written(spacer.start, spacer.end);
    }
}

/// The consensus repeat chosen for a cluster.
struct ChosenRepeat<'a> {
    sequence: &'a str,
    /// Its copies in the region of the cluster, as `(start, mismatches)`.
    copies: Vec<(usize, usize)>,
    /// Share of mismatching bases among the copies, see [`candidate_score`].
    mismatch_share: f64,
}

/// Turn a cluster of repeat hits into the arrays it holds, in the order
/// CRISPRCasFinder works:
///
/// 1. One of the hit sequences is chosen as the consensus repeat (see
///    [`repeat_candidates`] and [`candidate_score`]). It is reported as it
///    is; its boundaries are not adjusted afterwards.
/// 2. Its copies around the cluster give the spacers ([`define_spacers`]).
/// 3. The locus is dropped if its spacers are too alike, as in a tandem
///    repeat, or if its only spacer is too much like the repeat.
/// 4. It is divided into several arrays at spacers that are too long.
///
/// CRISPRCasFinder divides a locus only once the sequence has given an
/// array of three spacers or more, and loses arrays when one side of a
/// division has no spacer. Neither is followed: a locus is always divided,
/// and each part holding enough spacers is reported.
fn process_cluster(
    seq_id: &str,
    seq: &[u8],
    cluster_hits: &[RepeatHit],
    cluster: (usize, usize),
    occurrences: &std::collections::HashMap<&str, usize>,
    params: &DetectionParams,
    min_repeat_count: usize,
) -> Vec<CrisprArray> {
    let region_range = cluster_region(cluster, seq.len());
    let region_start = region_range.start;
    let region = &seq[region_range];

    let candidates = repeat_candidates(cluster_hits, occurrences);
    let mut best: Option<ChosenRepeat> = None;
    let mut best_score = 100_000.0f64;
    for &candidate in &candidates {
        let copies = slide_search(
            region,
            candidate.as_bytes(),
            allowed_copy_mismatches(candidate.len(), params),
        );
        let (score, mismatch_share) = candidate_score(&copies, region, candidate.len(), params);
        // On equal penalties the later candidate wins.
        if candidates.len() == 1 || score <= best_score {
            best_score = score;
            best = Some(ChosenRepeat {
                sequence: candidate,
                copies,
                mismatch_share,
            });
        }
    }
    let Some(ChosenRepeat {
        sequence: consensus,
        copies,
        mismatch_share,
    }) = best
    else {
        return Vec::new();
    };
    let repeat_length = consensus.len();

    let spacers = define_spacers(&copies, repeat_length, params);
    if spacers.len() + 1 < min_repeat_count {
        return Vec::new();
    }
    // Spacer starts are 1-based within the region.
    let spacer_sequence = |spacer: &SpacerSpan| {
        let start = region_start + spacer.start - 1;
        std::str::from_utf8(&seq[start..start + spacer.length]).unwrap_or_default()
    };
    let spacer_sequences: Vec<&str> = spacers
        .iter()
        .map(|(spacer, _)| spacer_sequence(spacer))
        .collect();
    let spacers_rule_it_out = match spacer_sequences.as_slice() {
        [spacer] => !check_short_crispr(consensus, spacer),
        _ => {
            let alignment = conservation::align_shifted_copies(&spacer_sequences);
            conservation::aligned_residue_identity(&alignment) >= params.spacer_similarity_threshold
        }
    };
    if spacers_rule_it_out || mismatch_share > params.repeat_mismatch_percent / 100.0 {
        return Vec::new();
    }

    let repeat_at = |start: usize| RepeatHit {
        repeat_length,
        pos1: start,
        repeat_sequence: String::from_utf8_lossy(&seq[start - 1..start - 1 + repeat_length])
            .to_string(),
    };
    spacers
        .split(|(_, divides)| *divides)
        .filter_map(|part| {
            let (last, _) = part.last()?;
            let mut repeats: Vec<RepeatHit> = part
                .iter()
                .map(|(spacer, _)| repeat_at(region_start + spacer.start - repeat_length))
                .collect();
            repeats.push(repeat_at(region_start + last.start + last.length));
            (repeats.len() >= min_repeat_count).then_some(repeats)
        })
        .filter_map(|repeats| build_array(seq_id, seq, consensus.to_string(), repeats, params))
        .collect()
}

/// Evaluate one chain of repeats as a CRISPR array.
///
/// The repeats are in order and do not touch: the stretch between two
/// neighbours is a spacer of the array.
fn build_array(
    seq_id: &str,
    seq: &[u8],
    best_consensus: String,
    mut best_refined: Vec<RepeatHit>,
    params: &DetectionParams,
) -> Option<CrisprArray> {
    if params.better_detect_truncated {
        extend_with_truncated_repeats(&best_consensus, &mut best_refined, seq, params);
    }
    // One spacer for each pair of neighbouring repeats, its sequence read at
    // the positions reported for it.
    let spacers: Vec<Spacer> = best_refined
        .windows(2)
        .map(|repeat_window| {
            let start = repeat_window[0].pos1 + repeat_window[0].repeat_length;
            let end = repeat_window[1].pos1 - 1;
            Spacer {
                start,
                end,
                sequence: String::from_utf8_lossy(&seq[start - 1..end]).to_string(),
            }
        })
        .collect();

    let repeat_sequences: Vec<&str> = best_refined
        .iter()
        .map(|hit| hit.repeat_sequence.as_str())
        .collect();
    let spacer_sequences: Vec<&str> = spacers
        .iter()
        .map(|spacer| spacer.sequence.as_str())
        .collect();
    let statistics =
        conservation::array_statistics(&best_consensus, &repeat_sequences, &spacer_sequences);

    let evidence_level = conservation::evidence_level(spacers.len(), &statistics);

    if evidence_level < params.min_evidence_level {
        return None;
    }

    let repeats: Vec<Repeat> = best_refined
        .iter()
        .map(|h| Repeat {
            start: h.pos1,
            end: h.pos1 + h.repeat_length - 1,
            sequence: h.repeat_sequence.clone(),
        })
        .collect();

    let array_start = best_refined.first().map(|h| h.pos1).unwrap_or(0);
    let array_end = best_refined
        .last()
        .map(|h| h.pos1 + h.repeat_length - 1)
        .unwrap_or(0);

    let orientation = infer_orientation(seq, array_start, array_end, params.flank);

    let db_entry = repeat_db::lookup_repeat(&best_consensus);

    Some(CrisprArray {
        id: String::new(),
        seq_id: seq_id.to_string(),
        start: array_start,
        end: array_end,
        consensus_repeat: best_consensus,
        repeats,
        spacers,
        evidence_level,
        statistics,
        orientation,
        repeat_id: db_entry.repeat_id,
        crispr_direction: db_entry.crispr_direction,
        crisprdb_repeat_count: db_entry.crisprdb_count,
        known_in_crisprdb: None,
        reclassified_from: None,
    })
}

/// Convenience: detect CRISPRs from a FASTA-formatted string.
///
/// Parses the FASTA content in memory and delegates to [`detect_crisprs`].
pub fn detect_crisprs_in_fasta_str(
    fasta_content: &str,
    params: &DetectionParams,
) -> Result<Vec<CrisprArray>> {
    params.validate().map_err(anyhow::Error::msg)?;
    let reader = bio::io::fasta::Reader::new(std::io::Cursor::new(fasta_content.as_bytes()));
    let mut sequences: Vec<(String, Vec<u8>)> = Vec::new();
    for result in reader.records() {
        let record = result?;
        sequences.push((record.id().to_string(), record.seq().to_vec()));
    }
    let seq_refs: Vec<(&str, &[u8])> = sequences
        .iter()
        .map(|(id, seq)| (id.as_str(), seq.as_slice()))
        .collect();
    Ok(detect_crisprs(&seq_refs, params))
}

/// Convenience: detect CRISPRs from a single raw sequence string.
pub fn detect_crisprs_in_sequence_str(
    seq_id: &str,
    sequence: &str,
    params: &DetectionParams,
) -> Vec<CrisprArray> {
    detect_crisprs(&[(seq_id, sequence.as_bytes())], params)
}

/// Convenience: detect CRISPRs directly from a FASTA file path.
pub fn detect_crisprs_in_fasta_path(
    fasta_path: &Path,
    params: &DetectionParams,
) -> Result<Vec<CrisprArray>> {
    params.validate().map_err(anyhow::Error::msg)?;
    // Open separately so callers can inspect the underlying I/O error kind.
    let reader = bio::io::fasta::Reader::new(std::fs::File::open(fasta_path)?);
    let mut sequences: Vec<(String, Vec<u8>)> = Vec::new();
    for result in reader.records() {
        let record = result?;
        sequences.push((record.id().to_string(), record.seq().to_vec()));
    }
    let seq_refs: Vec<(&str, &[u8])> = sequences
        .iter()
        .map(|(id, seq)| (id.as_str(), seq.as_slice()))
        .collect();
    Ok(detect_crisprs(&seq_refs, params))
}

#[cfg(test)]
pub(crate) mod test_support {
    /// Deterministic sequence without repeats, to keep parts of a test locus
    /// apart.
    pub(crate) fn filler(length: usize, seed: u64) -> String {
        let mut state = seed;
        (0..length)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                b"ACGT"[(state >> 62) as usize] as char
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::filler;
    use super::*;

    const DR: &str = "GTTCACTGCCGTATAGGCAGCTAAGAAA";
    const SPACERS: [&str; 5] = [
        "CAGTTTCAGCGATACGATCGATCGATCGTTTT",
        "AAAACCCCGGGGTTTTACGATCGATCGATCGA",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "TTGACCATGGCATTACGGATCCAATGCCGTAA",
        "CCATGGTACGTTAGCAAGTCCGATTAGGCATC",
    ];

    fn array_sequence(spacers: &[&str]) -> String {
        let flank = "TGCA".repeat(60);
        let mut sequence = flank.clone();
        for spacer in spacers {
            sequence.push_str(DR);
            sequence.push_str(spacer);
        }
        sequence.push_str(DR);
        sequence.push_str(&flank);
        sequence
    }

    #[test]
    fn arrays_are_numbered_per_sequence() {
        let long = array_sequence(&SPACERS);
        // The first two spacers share half their bases, which rules a locus
        // of two spacers out.
        let short = array_sequence(&SPACERS[3..]);
        let arrays = detect_crisprs(
            &[("a", long.as_bytes()), ("b", short.as_bytes())],
            &DetectionParams::default(),
        );
        let ids: Vec<&str> = arrays.iter().map(|array| array.id.as_str()).collect();
        assert_eq!(ids, ["a_CRISPR_1", "b_CRISPR_1"]);
        assert_eq!(arrays[0].evidence_level, 4);
        assert_eq!(arrays[1].evidence_level, 1);
    }

    #[test]
    fn small_arrays_sharing_a_trusted_repeat_are_reclassified_before_filtering() {
        let long = array_sequence(&SPACERS);
        // The first two spacers share half their bases, which rules a locus
        // of two spacers out.
        let short = array_sequence(&SPACERS[3..]);
        let sequences = [("a", long.as_bytes()), ("b", short.as_bytes())];
        let strict = DetectionParams {
            min_evidence_level: 4,
            ..DetectionParams::default()
        };
        assert_eq!(detect_crisprs(&sequences, &strict).len(), 1);

        let reclassifying = DetectionParams {
            classify_small_arrays: true,
            ..strict
        };
        let arrays = detect_crisprs(&sequences, &reclassifying);
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[1].evidence_level, 4);
        assert_eq!(arrays[1].reclassified_from, Some(1));
        assert_eq!(arrays[0].reclassified_from, None);
    }

    #[test]
    fn every_array_has_one_more_repeat_than_spacers() {
        // A query matching only the first repeats exactly still yields the
        // whole array once the consensus is rescanned with mismatches.
        let mut sequence = array_sequence(&SPACERS).into_bytes();
        let last_repeat = sequence.len() - 240 - DR.len();
        sequence[last_repeat + 3] = b'G';
        sequence[last_repeat + 9] = b'A';
        let params = DetectionParams {
            query_repeats: vec![DR.to_string()],
            ..DetectionParams::default()
        };
        let arrays = detect_crisprs(&[("a", sequence.as_slice())], &params);
        assert_eq!(arrays.len(), 1);
        assert_eq!(arrays[0].repeats.len(), 6);
        assert_eq!(arrays[0].spacers.len(), 5);
        assert_eq!(
            arrays[0].spacers.last().unwrap().end + 1,
            arrays[0].repeats.last().unwrap().start
        );
    }

    #[test]
    fn copies_side_by_side_leave_no_spacer_between_them() {
        // Two copies of the repeat follow the second spacer. With a ratio of
        // 0 only the lack of room for a spacer rules the second one out.
        let sequence = array_sequence(&[
            SPACERS[0], SPACERS[1], "", SPACERS[2], SPACERS[3], SPACERS[4],
        ]);
        let params = DetectionParams {
            min_spacer_to_repeat_ratio: 0.0,
            ..DetectionParams::default()
        };
        let arrays = detect_crisprs(&[("a", sequence.as_bytes())], &params);
        assert_eq!(arrays.len(), 1);
        let array = &arrays[0];

        // The second copy is part of the third spacer.
        let third = format!("{DR}{}", SPACERS[2]);
        let spacers: Vec<&str> = array
            .spacers
            .iter()
            .map(|spacer| spacer.sequence.as_str())
            .collect();
        assert_eq!(
            spacers,
            [SPACERS[0], SPACERS[1], &third, SPACERS[3], SPACERS[4]]
        );
        assert_eq!(array.repeats.len(), array.spacers.len() + 1);
        for (spacer, repeats) in array.spacers.iter().zip(array.repeats.windows(2)) {
            assert_eq!(spacer.start, repeats[0].end + 1);
            assert_eq!(spacer.end + 1, repeats[1].start);
            assert_eq!(spacer.sequence, sequence[spacer.start - 1..spacer.end]);
        }
        // The default ratio rules it out as well.
        assert_eq!(
            arrays,
            detect_crisprs(&[("a", sequence.as_bytes())], &DetectionParams::default())
        );
        assert_same_arrays_in_the_case_of_the_input(
            &soft_masked(sequence.as_bytes(), |index| index % 2 == 0),
            &params,
            &arrays,
            "every other base",
        );
    }

    #[test]
    fn neighbouring_arrays_are_reported_separately() {
        // 1,080 bp between the arrays: one cluster of seed hits holds both.
        let sequence = format!(
            "{}{}{}",
            array_sequence(&SPACERS[..3]),
            filler(600, 1),
            array_sequence(&SPACERS[3..])
        );
        let arrays = detect_crisprs(&[("a", sequence.as_bytes())], &DetectionParams::default());
        let spacer_counts: Vec<usize> = arrays.iter().map(|array| array.spacers.len()).collect();
        assert_eq!(spacer_counts, [3, 2]);
        assert_eq!(arrays[0].consensus_repeat, DR);
        assert_eq!(arrays[1].consensus_repeat, DR);
        assert_eq!(
            arrays[1].start,
            sequence.rfind(SPACERS[3]).unwrap() + 1 - DR.len()
        );
    }

    #[test]
    fn degenerate_copy_before_an_array_does_not_hide_it() {
        // A copy with six mismatches lies 400 bp before the array: inside the
        // window rescanned for repeats, too far to be the first repeat.
        let mut weak = DR.as_bytes().to_vec();
        for index in [1, 5, 9, 13, 17, 21] {
            weak[index] = if weak[index] == b'A' { b'C' } else { b'A' };
        }
        let sequence = format!(
            "{}{}{}{}",
            filler(300, 2),
            String::from_utf8(weak).unwrap(),
            filler(160, 3),
            array_sequence(&SPACERS)
        );
        let arrays = detect_crisprs(&[("a", sequence.as_bytes())], &DetectionParams::default());
        assert_eq!(arrays.len(), 1);
        assert_eq!(arrays[0].spacers.len(), 5);
        assert_eq!(
            arrays[0].start,
            sequence.find(SPACERS[0]).unwrap() + 1 - DR.len()
        );
    }

    /// The arrays CRISPRCasFinder 4.2.30 reports for E. coli K-12 MG1655, with
    /// the start of every repeat. Four of them are in CRISPRdb
    /// (`CRISPR_crisprdb.csv`, accession NC_000913) with the same coordinates.
    #[test]
    fn ecoli_arrays_are_those_of_crisprcasfinder() {
        let fasta = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crispr-cas-finder-cli/data/ecoli.fasta");
        let arrays = detect_crisprs_in_fasta_path(&fasta, &DetectionParams::default()).unwrap();
        let found: Vec<(usize, usize, &str, Vec<usize>)> = arrays
            .iter()
            .map(|array| {
                (
                    array.end,
                    array.evidence_level,
                    array.consensus_repeat.as_str(),
                    array.repeats.iter().map(|repeat| repeat.start).collect(),
                )
            })
            .collect();
        let expected: [(usize, usize, &str, Vec<usize>); 5] = [
            (
                411_220,
                1,
                "ATGCCTGATGCGACGCTTGCCGCGTCTTATCAGGCCTACAAAA",
                vec![411_076, 411_178],
            ),
            (
                551_462,
                1,
                "GCTTGACGCGTCTTATCAGGCCTACAA",
                vec![551_366, 551_436],
            ),
            (
                2_347_327,
                1,
                "TTTGTAGGCCTGATAAGACGCGCCAGCGTCGCATCAGGC",
                vec![2_347_201, 2_347_289],
            ),
            (
                2_878_463,
                4,
                "CGGTTTATCCCCGCTGGCGCGGGGAACTC",
                vec![
                    2_877_701, 2_877_762, 2_877_823, 2_877_884, 2_877_945, 2_878_006, 2_878_067,
                    2_878_128, 2_878_190, 2_878_252, 2_878_313, 2_878_374, 2_878_435,
                ],
            ),
            (
                2_904_407,
                4,
                "GGTTTATCCCCGCTGGCGCGGGGAACAC",
                vec![
                    2_904_014, 2_904_075, 2_904_136, 2_904_197, 2_904_258, 2_904_319, 2_904_380,
                ],
            ),
        ];
        assert_eq!(found, expected);
        for array in &arrays {
            assert_eq!(array.repeats.len(), array.spacers.len() + 1, "{}", array.id);
            assert_eq!(array.start, array.repeats[0].start, "{}", array.id);
        }
    }

    /// `sequence` with the bases at the indices `masked` picks in lowercase.
    fn soft_masked(sequence: &[u8], masked: impl Fn(usize) -> bool) -> Vec<u8> {
        sequence
            .iter()
            .enumerate()
            .map(|(index, base)| {
                if masked(index) {
                    base.to_ascii_lowercase()
                } else {
                    *base
                }
            })
            .collect()
    }

    /// Check that `sequence` gives the arrays in `expected`, those of the
    /// same sequence in uppercase, with its repeats and spacers as it has
    /// them.
    fn assert_same_arrays_in_the_case_of_the_input(
        sequence: &[u8],
        params: &DetectionParams,
        expected: &[CrisprArray],
        masking: &str,
    ) {
        let mut arrays = detect_crisprs(&[("a", sequence)], params);
        for array in &mut arrays {
            for (start, end, text) in array
                .repeats
                .iter_mut()
                .map(|repeat| (repeat.start, repeat.end, &mut repeat.sequence))
                .chain(
                    array
                        .spacers
                        .iter_mut()
                        .map(|spacer| (spacer.start, spacer.end, &mut spacer.sequence)),
                )
            {
                assert_eq!(
                    text.as_bytes(),
                    &sequence[start - 1..end],
                    "{masking}: {start}-{end}"
                );
                text.make_ascii_uppercase();
            }
        }
        assert_eq!(arrays, expected, "{masking}");
    }

    #[test]
    fn arrays_do_not_depend_on_the_case_of_the_sequence() {
        let sequence = array_sequence(&SPACERS).into_bytes();
        // The array starts after 240 bases; a repeat and its spacer take 60.
        let within_array = sequence.len() / 2;
        let maskings: [(&str, &dyn Fn(usize) -> bool); 6] = [
            ("all", &|_| true),
            ("every other line of 70", &|index| index / 70 % 2 == 0),
            ("every other base", &|index| index % 2 == 0),
            ("up to the middle of the array", &|index| {
                index < within_array
            }),
            ("the repeats", &|index| {
                index >= 240 && (index - 240) % 60 < DR.len()
            }),
            ("one repeat", &|index| {
                (300..300 + DR.len()).contains(&index)
            }),
        ];
        for params in [
            DetectionParams::default(),
            DetectionParams {
                query_repeats: vec![DR.to_lowercase()],
                ..DetectionParams::default()
            },
        ] {
            let expected = detect_crisprs(&[("a", sequence.as_slice())], &params);
            assert_eq!(expected.len(), 1);
            assert_eq!(expected[0].consensus_repeat, DR);
            assert_eq!(expected[0].spacers.len(), 5);
            assert_eq!(expected[0].statistics.exact_repeat_count, 6);
            for (masking, masked) in maskings {
                assert_same_arrays_in_the_case_of_the_input(
                    &soft_masked(&sequence, masked),
                    &params,
                    &expected,
                    masking,
                );
            }
        }
    }

    /// E. coli K-12 MG1655 in lowercase, and with every other line of its
    /// FASTA file in lowercase: no array has two copies of its repeat in one
    /// case throughout.
    #[test]
    fn ecoli_arrays_do_not_depend_on_the_case_of_the_sequence() {
        let fasta = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crispr-cas-finder-cli/data/ecoli.fasta");
        let records = parse_fasta(std::fs::File::open(fasta).unwrap()).unwrap();
        let sequence = records[0].sequence.as_slice();
        let params = DetectionParams::default();
        let expected = detect_crisprs(&[("a", sequence)], &params);
        assert_eq!(expected.len(), 5);

        let maskings: [(&str, &dyn Fn(usize) -> bool); 2] = [
            ("all", &|_| true),
            ("every other line of 70", &|index| index / 70 % 2 == 1),
        ];
        for (masking, masked) in maskings {
            assert_same_arrays_in_the_case_of_the_input(
                &soft_masked(sequence, masked),
                &params,
                &expected,
                masking,
            );
        }
    }

    /// Clostridioides difficile 630 (NC_009089.1) from base 373,041: a gene
    /// with a tandem repeat of 42 bases. Every other copy of the repeat
    /// passes for a repeat of an array, with a copy inside each "spacer".
    const TANDEM_REPEAT: &str = concat!(
        "ACTAGGCATAACTACAGGCATACCTTGCATTGTATTATGACATATAACTCCTGGTATCATCATAAGTACATTAGGAGGCA",
        "TATTTTGATTACATGAGAAGTTAGGAATACCCATAGTATTATTCATATTATACCCCATATTACTGTTCATGTTATTAGGC",
        "ATACCTGTATTACCATACATATTGTTATTCATATTGCCAGACATATTGTTAGGCATACCCATATTACTATTCATATTATT",
        "AGACATATTGTTAGGCATACCCATATTACTATTCATGTTATTAGGAACATTGTTAGGCATGCCCATGTTACTATTCATAT",
        "TATTAGGCATATTGTTAGGCATACCCATATTACTATTCATGTTATTAGACATATTGTTAGGCATACCCATATTACTATTC",
        "ATATTGTTAGACATATTATTAGGCATACCCGTATTATTATTCATGTTATTAGGCATATTATTTGTAGCAGATGTATTGTA",
        "ATTCATATTTCCGGATGTATTATTTGTGTTAGGCAAGTTG",
    );

    #[test]
    fn tandem_repeat_is_told_from_an_array_by_its_spacers() {
        let sequences = [("tandem", TANDEM_REPEAT.as_bytes())];
        assert!(detect_crisprs(&sequences, &DetectionParams::default()).is_empty());

        // Without the limit on spacer similarity it is reported, with the
        // repeat exactly as the seed search gave it.
        let permissive = DetectionParams {
            spacer_similarity_threshold: 101.0,
            ..DetectionParams::default()
        };
        let arrays = detect_crisprs(&sequences, &permissive);
        assert_eq!(arrays.len(), 1);
        assert_eq!(
            arrays[0].consensus_repeat,
            "CATATTGTTAGGCATACCCATATTACTATTCATGTTATTAG"
        );
        let starts: Vec<usize> = arrays[0]
            .repeats
            .iter()
            .map(|repeat| repeat.start)
            .collect();
        assert_eq!(starts, [94, 202, 286, 370]);
    }

    #[test]
    fn query_repeats_restrict_detection_to_the_given_repeat() {
        let sequence = array_sequence(&SPACERS);
        let matching = DetectionParams {
            query_repeats: vec![DR.to_string()],
            ..DetectionParams::default()
        };
        let arrays = detect_crisprs(&[("a", sequence.as_bytes())], &matching);
        assert_eq!(arrays.len(), 1);
        assert_eq!(arrays[0].spacers.len(), 5);
        assert_eq!(arrays[0].repeats.len(), 6);

        let other = DetectionParams {
            query_repeats: vec!["ACGTTGCAACGTTGCAACGTTGCAAC".to_string()],
            ..DetectionParams::default()
        };
        assert!(detect_crisprs(&[("a", sequence.as_bytes())], &other).is_empty());
    }
}
