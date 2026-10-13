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
    RepeatHit, check_short_crispr, cluster_repeats, conserved_consensus_range,
    extend_with_truncated_repeats, extract_spacers, find_direct_repeats, foster_score,
    get_repeat_candidates, infer_orientation, refine_cluster_with_consensus,
};

/// Detect CRISPR arrays in a set of sequences.
///
/// This is the main programmatic entry point. It operates entirely in memory —
/// no file I/O, no subprocess calls. The caller provides sequences directly
/// and gets back structured results.
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

    let hits = find_direct_repeats(seq, params);
    info!("  {} — {} DR candidates", seq_id, hits.len());

    let clusters = cluster_repeats(&hits, GAP_THRESHOLD);

    let filtered_clusters: Vec<(usize, usize)> = clusters
        .into_iter()
        .filter(|(start, end)| {
            hits.iter()
                .filter(|h| h.pos1 >= *start && h.pos1 <= *end)
                .count()
                >= min_repeat_count
        })
        .collect();

    let mut arrays = Vec::new();
    for cluster in filtered_clusters {
        arrays.extend(process_cluster(
            seq_id,
            seq,
            &hits,
            cluster,
            params,
            min_repeat_count,
        ));
    }

    arrays
}

/// First and last repeat start, the range `extract_spacers` works over.
fn repeat_span(repeats: &[RepeatHit]) -> (usize, usize) {
    (
        repeats.first().map_or(0, |hit| hit.pos1),
        repeats.last().map_or(0, |hit| hit.pos1),
    )
}

/// Percentage of positions at which a repeat differs from the consensus.
fn mismatch_percent(consensus: &str, repeat: &RepeatHit) -> f64 {
    let mismatches = repeat
        .repeat_sequence
        .bytes()
        .zip(consensus.bytes())
        .filter(|(a, b)| a != b)
        .count();
    mismatches as f64 * 100.0 / consensus.len() as f64
}

/// Whether the repeats differ from the consensus by more than
/// `repeat_mismatch_percent` on average.
fn exceeds_repeat_mismatches(
    consensus: &str,
    repeats: &[RepeatHit],
    params: &DetectionParams,
) -> bool {
    let total: f64 = repeats
        .iter()
        .map(|repeat| mismatch_percent(consensus, repeat))
        .sum();
    total / repeats.len() as f64 > params.repeat_mismatch_percent
}

/// A chain of repeats settled on the consensus it is reported with.
struct SettledChain {
    consensus: String,
    repeats: Vec<RepeatHit>,
    /// Spacers the chain explains inside the seed cluster, used to rank
    /// candidate consensus repeats.
    seed_spacers: usize,
}

/// Trim the poorly conserved edges of a consensus and rescan with the result.
///
/// Returns `None` when too few repeats are left.
fn settle_chain(
    consensus: &str,
    repeats: Vec<RepeatHit>,
    cluster: (usize, usize),
    seq: &[u8],
    params: &DetectionParams,
    min_repeat_count: usize,
) -> Option<SettledChain> {
    let seed_spacers = extract_spacers(&repeats, cluster, seq, params).len();
    let kept = conserved_consensus_range(consensus, &repeats, params.min_repeat_length);
    if kept.len() == consensus.len() {
        return Some(SettledChain {
            consensus: consensus.to_string(),
            repeats,
            seed_spacers,
        });
    }
    let trimmed = &consensus[kept];
    // The rescan reaches neighbouring arrays again; keep the chain that holds
    // the repeats of this one. Trimming moves their starts to the right.
    let span = repeat_span(&repeats);
    let starts = span.0..=span.1 + consensus.len();
    let repeats = refine_cluster_with_consensus(trimmed, span, seq, params)
        .into_iter()
        .map(|chain| {
            let shared = chain
                .iter()
                .filter(|hit| starts.contains(&hit.pos1))
                .count();
            (shared, chain)
        })
        .filter(|(shared, _)| *shared > 0)
        .max_by_key(|(shared, _)| *shared)
        .map(|(_, chain)| chain)?;
    (repeats.len() >= min_repeat_count).then(|| SettledChain {
        consensus: trimmed.to_string(),
        // A degenerate copy beside an array can match the untrimmed
        // candidate only. The spacer it adds is not in the reported array
        // and must not help the candidate win.
        seed_spacers: seed_spacers.min(repeats.len() - 1),
        repeats,
    })
}

/// Turn a cluster of DR hits into the arrays it holds.
///
/// One consensus repeat is chosen for the cluster. Each chain of its copies
/// is then evaluated as an array of its own, so two arrays closer than the
/// clustering distance are both reported.
fn process_cluster(
    seq_id: &str,
    seq: &[u8],
    hits: &[RepeatHit],
    cluster: (usize, usize),
    params: &DetectionParams,
    min_repeat_count: usize,
) -> Vec<CrisprArray> {
    let repeat_candidates = get_repeat_candidates(hits, cluster);

    let mut best_chains: Vec<SettledChain> = Vec::new();
    let mut best_length = 0usize;
    let mut best_spacer_count = 0usize;
    let mut best_foster_score = 0usize;

    for consensus in &repeat_candidates {
        let chains: Vec<SettledChain> =
            refine_cluster_with_consensus(consensus, cluster, seq, params)
                .into_iter()
                .filter(|chain| {
                    chain.len() >= min_repeat_count
                        && !exceeds_repeat_mismatches(consensus, chain, params)
                })
                .filter_map(|chain| {
                    settle_chain(consensus, chain, cluster, seq, params, min_repeat_count)
                })
                .collect();

        // Candidates are ranked by the spacers they explain inside the seed
        // cluster, which favours the consensus the seed hits agree on.
        let spacer_count: usize = chains.iter().map(|chain| chain.seed_spacers).sum();

        // Pick the candidate that produces the most spacers.
        // Tie-break: prefer longer DR (vmatch finds maximal repeats).
        // With `force_detection`, fostered repeat properties come first.
        let candidate_foster_score: usize = if params.force_detection {
            chains
                .iter()
                .map(|chain| foster_score(&chain.repeats, params))
                .sum()
        } else {
            0
        };
        let more_spacers = spacer_count > best_spacer_count
            || (spacer_count == best_spacer_count
                && best_spacer_count > 0
                && consensus.len() > best_length);
        if candidate_foster_score > best_foster_score
            || (candidate_foster_score == best_foster_score && more_spacers)
        {
            best_foster_score = candidate_foster_score;
            best_length = consensus.len();
            best_chains = chains;
            best_spacer_count = spacer_count;
        }
    }

    best_chains
        .into_iter()
        .filter_map(|chain| build_array(seq_id, seq, chain.consensus, chain.repeats, params))
        .collect()
}

/// Evaluate one chain of repeats as a CRISPR array.
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
    // Refinement rescans beyond the seed cluster, so the reported spacers
    // cover the span of the refined repeats, not of the seed hits.
    let best_spacers = extract_spacers(&best_refined, repeat_span(&best_refined), seq, params);

    let repeat_sequences: Vec<&str> = best_refined
        .iter()
        .map(|hit| hit.repeat_sequence.as_str())
        .collect();
    let spacer_sequences: Vec<&str> = best_spacers.iter().map(String::as_str).collect();
    let statistics =
        conservation::array_statistics(&best_consensus, &repeat_sequences, &spacer_sequences);

    // Multi-spacer: reject when the aligned spacers are on average at least
    // `spacer_similarity_threshold` percent identical (tandem repeats).
    if best_spacers.len() > 1
        && statistics.spacer_average_identity >= params.spacer_similarity_threshold
    {
        return None;
    }
    // Single-spacer: reject if DR aligns too well against the spacer
    if best_spacers.len() == 1 && !check_short_crispr(&best_consensus, &best_spacers[0]) {
        return None;
    }

    let evidence_level = conservation::evidence_level(best_spacers.len(), &statistics);

    if evidence_level < params.min_evidence_level {
        return None;
    }

    // Build the structured CrisprArray from the refined hits + extracted spacers
    let repeats: Vec<Repeat> = best_refined
        .iter()
        .map(|h| Repeat {
            start: h.pos1,
            end: h.pos1 + h.repeat_length - 1,
            sequence: h.repeat_sequence.clone(),
        })
        .collect();

    let spacers: Vec<Spacer> = best_refined
        .windows(2)
        .zip(best_spacers.iter())
        .map(|(repeat_window, spacer_sequence)| {
            let spacer_start = repeat_window[0].pos1 + repeat_window[0].repeat_length;
            let spacer_end = repeat_window[1].pos1 - 1;
            Spacer {
                start: spacer_start,
                end: spacer_end,
                sequence: spacer_sequence.clone(),
            }
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
        let short = array_sequence(&SPACERS[..2]);
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
        let short = array_sequence(&SPACERS[..2]);
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

    /// The four E. coli K-12 MG1655 arrays recorded in CRISPRdb
    /// (`CRISPR_crisprdb.csv`, accession NC_000913) with their spacer counts.
    #[test]
    fn ecoli_arrays_match_crisprdb_coordinates_exactly() {
        let fasta = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crispr-cas-finder-cli/data/ecoli.fasta");
        let arrays = detect_crisprs_in_fasta_path(&fasta, &DetectionParams::default()).unwrap();
        let found: Vec<(usize, usize, usize)> = arrays
            .iter()
            .map(|array| (array.start, array.end, array.spacers.len()))
            .collect();
        for expected in [
            (411_076, 411_220, 1),
            (2_347_201, 2_347_327, 1),
            (2_877_701, 2_878_463, 12),
            (2_904_014, 2_904_407, 6),
        ] {
            assert!(found.contains(&expected), "{expected:?} not in {found:?}");
        }
        for array in &arrays {
            assert_eq!(array.repeats.len(), array.spacers.len() + 1, "{}", array.id);
        }
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
