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
    RepeatHit, check_short_crispr, cluster_repeats, extend_with_truncated_repeats, extract_spacers,
    find_direct_repeats, foster_score, get_repeat_candidates, infer_orientation,
    refine_cluster_with_consensus, trim_consensus,
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
        if let Some(array) = process_cluster(seq_id, seq, &hits, cluster, params, min_repeat_count)
        {
            arrays.push(array);
        }
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

/// Process a single cluster of DR hits into a CrisprArray (if valid).
fn process_cluster(
    seq_id: &str,
    seq: &[u8],
    hits: &[RepeatHit],
    cluster: (usize, usize),
    params: &DetectionParams,
    min_repeat_count: usize,
) -> Option<CrisprArray> {
    let repeat_candidates = get_repeat_candidates(hits, cluster);
    if repeat_candidates.is_empty() {
        return None;
    }

    let mut best_consensus = String::new();
    let mut best_refined: Vec<RepeatHit> = Vec::new();
    let mut best_spacers: Vec<String> = Vec::new();
    let mut best_foster_score = 0usize;

    for consensus in &repeat_candidates {
        let refined = refine_cluster_with_consensus(consensus, cluster, seq, params);
        if refined.len() < min_repeat_count {
            continue;
        }

        // DR error rate filter
        let repeat_length = consensus.len();
        let consensus_bytes = consensus.as_bytes();
        let total_mism_frac: f64 = refined
            .iter()
            .map(|h| {
                let mismatches = h
                    .repeat_sequence
                    .as_bytes()
                    .iter()
                    .zip(consensus_bytes.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                mismatches as f64 / repeat_length as f64
            })
            .sum();
        let avg_mism_frac = total_mism_frac / refined.len() as f64;
        if avg_mism_frac * 100.0 > params.repeat_mismatch_percent {
            continue;
        }

        // Candidates are ranked by the spacers they explain inside the seed
        // cluster, which favours the consensus the seed hits agree on.
        let spacers = extract_spacers(&refined, cluster, seq, params);

        // Pick the candidate that produces the most spacers.
        // Tie-break: prefer longer DR (vmatch finds maximal repeats).
        // With `force_detection`, fostered repeat properties come first.
        let candidate_foster_score = if params.force_detection {
            foster_score(&refined, params)
        } else {
            0
        };
        let more_spacers = spacers.len() > best_spacers.len()
            || (spacers.len() == best_spacers.len()
                && !best_spacers.is_empty()
                && consensus.len() > best_consensus.len());
        if candidate_foster_score > best_foster_score
            || (candidate_foster_score == best_foster_score && more_spacers)
        {
            best_foster_score = candidate_foster_score;
            best_consensus = consensus.clone();
            best_refined = refined;
            best_spacers = spacers;
        }
    }

    if best_consensus.is_empty() {
        return None;
    }

    // Refinement rescans beyond the seed cluster, so the reported spacers
    // cover the span of the refined repeats, not of the seed hits.
    let mut cluster = repeat_span(&best_refined);
    best_spacers = extract_spacers(&best_refined, cluster, seq, params);

    // Trim poorly-conserved edges from the consensus, re-refine if trimmed
    let trimmed = trim_consensus(&best_consensus, &best_refined, params.min_repeat_length);
    if trimmed != best_consensus {
        best_consensus = trimmed;
        best_refined = refine_cluster_with_consensus(&best_consensus, cluster, seq, params);
        if best_refined.len() < min_repeat_count {
            return None;
        }
        // Update cluster bounds to cover all refined hits (positions may have shifted)
        if let (Some(first), Some(last)) = (best_refined.first(), best_refined.last()) {
            cluster = (first.pos1, last.pos1);
        }
        best_spacers = extract_spacers(&best_refined, cluster, seq, params);
    }

    if params.better_detect_truncated
        && extend_with_truncated_repeats(&best_consensus, &mut best_refined, seq, params) > 0
        && let (Some(first), Some(last)) = (best_refined.first(), best_refined.last())
    {
        cluster = (first.pos1, last.pos1);
        best_spacers = extract_spacers(&best_refined, cluster, seq, params);
    }

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
mod tests {
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
