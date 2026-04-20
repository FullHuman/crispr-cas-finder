use anyhow::Result;
use log::info;
use rayon::prelude::*;
use std::path::Path;

pub mod cas_pipeline;
pub mod cas_types;
pub mod casfinder;
pub mod casparser;
pub mod detect;
pub mod io;
pub mod types;

pub use cas_pipeline::{GeneRecord, ModelDefinition};
pub use casparser::{CasCluster, CasGene, GeneCoordinates};
pub use types::{
    CasFinderConfig, CrisprArray, DetectionParams, FullAnalysisResult, Orientation, Repeat, Spacer,
};

use detect::{
    RepeatHit, check_short_crispr, check_spacer_similarity, cluster_repeats, extract_spacers,
    find_direct_repeats, get_repeat_candidates, infer_orientation, refine_cluster_with_consensus,
    trim_consensus,
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
    let min_repeat_count = params.min_spacer_count.max(1) + 1;

    if sequences.len() > 1 {
        let arrays_by_sequence: Vec<Vec<CrisprArray>> = sequences
            .par_iter()
            .map(|&(seq_id, seq)| {
                detect_crisprs_for_sequence(seq_id, seq, params, min_repeat_count)
            })
            .collect();
        return arrays_by_sequence.into_iter().flatten().collect();
    }

    sequences
        .iter()
        .flat_map(|&(seq_id, seq)| {
            detect_crisprs_for_sequence(seq_id, seq, params, min_repeat_count)
        })
        .collect()
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

    let hits = find_direct_repeats(seq, seq_id, params);
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

/// Process a single cluster of DR hits into a CrisprArray (if valid).
fn process_cluster(
    seq_id: &str,
    seq: &[u8],
    hits: &[RepeatHit],
    cluster: (usize, usize),
    params: &DetectionParams,
    min_repeat_count: usize,
) -> Option<CrisprArray> {
    let dr_cands = get_repeat_candidates(hits, cluster);
    if dr_cands.is_empty() {
        return None;
    }

    let mut best_consensus = String::new();
    let mut best_refined: Vec<RepeatHit> = Vec::new();
    let mut best_spacers: Vec<String> = Vec::new();

    for consensus in &dr_cands {
        let refined = refine_cluster_with_consensus(consensus, cluster, seq_id, seq, params);
        if refined.len() < min_repeat_count {
            continue;
        }

        // DR error rate filter
        let dr_len = consensus.len();
        let cons_bytes = consensus.as_bytes();
        let total_mism_frac: f64 = refined
            .iter()
            .map(|h| {
                let mismatches = h
                    .repeat_sequence
                    .as_bytes()
                    .iter()
                    .zip(cons_bytes.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                mismatches as f64 / dr_len as f64
            })
            .sum();
        let avg_mism_frac = total_mism_frac / refined.len() as f64;
        if avg_mism_frac * 100.0 > params.repeat_mismatch_percent {
            continue;
        }

        let spacers = extract_spacers(&refined, cluster, seq, params);

        // Pick the candidate that produces the most spacers.
        // Tie-break: prefer longer DR (vmatch finds maximal repeats).
        if spacers.len() > best_spacers.len()
            || (spacers.len() == best_spacers.len()
                && !best_spacers.is_empty()
                && consensus.len() > best_consensus.len())
        {
            best_consensus = consensus.clone();
            best_refined = refined;
            best_spacers = spacers;
        }
    }

    if best_consensus.is_empty() {
        return None;
    }

    // Trim poorly-conserved edges from the consensus, re-refine if trimmed
    let trimmed = trim_consensus(&best_consensus, &best_refined, params.min_repeat_length);
    let mut cluster = cluster;
    if trimmed != best_consensus {
        best_consensus = trimmed;
        best_refined = refine_cluster_with_consensus(&best_consensus, cluster, seq_id, seq, params);
        if best_refined.len() < min_repeat_count {
            return None;
        }
        // Update cluster bounds to cover all refined hits (positions may have shifted)
        if let (Some(first), Some(last)) = (best_refined.first(), best_refined.last()) {
            cluster = (first.pos1, last.pos1);
        }
        best_spacers = extract_spacers(&best_refined, cluster, seq, params);
    }

    // Multi-spacer: reject if ANY pairwise spacer identity >= spacer_similarity_threshold
    if best_spacers.len() > 1
        && !check_spacer_similarity(&best_spacers, params.spacer_similarity_threshold)
    {
        return None;
    }
    // Single-spacer: reject if DR aligns too well against the spacer
    if best_spacers.len() == 1 && !check_short_crispr(&best_consensus, &best_spacers[0]) {
        return None;
    }

    let spacer_count = best_spacers.len();
    let spacers_pass_similarity =
        check_spacer_similarity(&best_spacers, params.spacer_similarity_threshold);
    let evidence_level = if spacer_count <= 3 {
        1
    } else if spacers_pass_similarity {
        4
    } else {
        2
    };

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
        .map(|(w, sp_seq)| {
            let sp_start = w[0].pos1 + w[0].repeat_length;
            let sp_end = w[1].pos1 - 1;
            Spacer {
                start: sp_start,
                end: sp_end,
                sequence: sp_seq.clone(),
            }
        })
        .collect();

    let array_start = best_refined.first().map(|h| h.pos1).unwrap_or(0);
    let array_end = best_refined
        .last()
        .map(|h| h.pos1 + h.repeat_length - 1)
        .unwrap_or(0);

    let orientation = infer_orientation(seq, array_start, array_end, params.flank);

    Some(CrisprArray {
        seq_id: seq_id.to_string(),
        start: array_start,
        end: array_end,
        consensus_repeat: best_consensus,
        repeats,
        spacers,
        evidence_level,
        orientation,
    })
}

/// Convenience: detect CRISPRs from a FASTA-formatted string.
///
/// Parses the FASTA content in memory and delegates to [`detect_crisprs`].
pub fn detect_crisprs_in_fasta_str(
    fasta_content: &str,
    params: &DetectionParams,
) -> Result<Vec<CrisprArray>> {
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
    let reader = bio::io::fasta::Reader::from_file(fasta_path)?;
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
