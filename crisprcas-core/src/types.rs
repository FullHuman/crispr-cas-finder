use crate::casparser::CasCluster;
use serde::{Deserialize, Serialize};

/// A single direct repeat within a CRISPR array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Repeat {
    /// 1-based start position on the sequence.
    pub start: usize,
    /// 1-based end position on the sequence (inclusive).
    pub end: usize,
    /// DNA sequence of the repeat.
    pub sequence: String,
}

/// A single spacer between two direct repeats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spacer {
    /// 1-based start position on the sequence.
    pub start: usize,
    /// 1-based end position on the sequence (inclusive).
    pub end: usize,
    /// DNA sequence of the spacer.
    pub sequence: String,
}

/// Potential orientation of a CRISPR array based on AT% of flanking sequences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Orientation {
    /// Forward strand (+); the leader is on the left flank.
    Forward,
    /// Reverse strand (-); the leader is on the right flank.
    Reverse,
    /// Orientation could not be determined.
    Unknown,
}

impl std::fmt::Display for Orientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Orientation::Forward => write!(f, "+"),
            Orientation::Reverse => write!(f, "-"),
            Orientation::Unknown => write!(f, "."),
        }
    }
}

/// A detected CRISPR array with its repeats, spacers, and evidence level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisprArray {
    /// Identifier of the source sequence (FASTA header).
    pub seq_id: String,
    /// 1-based start position of the first repeat.
    pub start: usize,
    /// 1-based end position of the last repeat (inclusive).
    pub end: usize,
    /// Consensus direct repeat sequence chosen for this array.
    pub consensus_repeat: String,
    /// Ordered list of direct repeats.
    pub repeats: Vec<Repeat>,
    /// Ordered list of spacers (one fewer than repeats).
    pub spacers: Vec<Spacer>,
    /// Evidence level (1–4). Higher is more confident.
    pub evidence_level: usize,
    /// Potential orientation inferred from flanking AT% content.
    pub orientation: Orientation,
}

/// Algorithm-only parameters for CRISPR detection.
///
/// This struct contains no file paths, output flags, or subprocess
/// configuration — only the knobs that control the detection algorithm.
#[derive(Debug, Clone)]
pub struct DetectionParams {
    pub min_repeat_length: usize,
    pub max_repeat_length: usize,
    pub min_spacer_length: usize,
    pub max_spacer_length: usize,
    pub repeat_mismatch_percent: f64,
    pub truncated_repeat_mismatch_percent: f64,
    pub no_mismatch: bool,
    pub min_spacer_to_repeat_ratio: f64,
    pub max_spacer_to_repeat_ratio: f64,
    pub spacer_similarity_threshold: f64,
    pub min_spacer_count: usize,
    pub min_evidence_level: usize,
    pub min_sequence_length: usize,
    pub force_detection: bool,
    pub flank: usize,
    pub foster_repeat_length: usize,
    pub foster_repeat_begin: String,
    pub foster_repeat_end: String,
    pub better_detect_truncated: bool,
    pub truncated_mismatch_percent: f64,
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self {
            min_repeat_length: 23,
            max_repeat_length: 55,
            min_spacer_length: 25,
            max_spacer_length: 60,
            repeat_mismatch_percent: 20.0,
            truncated_repeat_mismatch_percent: 33.3,
            no_mismatch: false,
            min_spacer_to_repeat_ratio: 0.6,
            max_spacer_to_repeat_ratio: 2.5,
            spacer_similarity_threshold: 60.0,
            min_spacer_count: 1,
            min_evidence_level: 1,
            min_sequence_length: 0,
            force_detection: false,
            flank: 100,
            foster_repeat_length: 30,
            foster_repeat_begin: "G".to_string(),
            foster_repeat_end: "AA.".to_string(),
            better_detect_truncated: false,
            truncated_mismatch_percent: 4.0,
        }
    }
}

/// Configuration for the CasFinder gene-prediction and Cas-system detection
/// pipeline.
#[derive(Debug, Clone)]
pub struct CasFinderConfig {
    pub genetic_code: usize,
    pub metagenome: bool,
    pub workers: usize,
    pub definition: String,
    pub vicinity: usize,
    pub clustering_threshold: usize,
    pub quiet: bool,
    pub fast: bool,
    /// Directory containing CAS model definitions (XML or TOML).
    pub cas_models_dir: Option<std::path::PathBuf>,
    /// Directory containing CAS HMM profile files.
    pub cas_profiles_dir: Option<std::path::PathBuf>,
}

impl Default for CasFinderConfig {
    fn default() -> Self {
        Self {
            genetic_code: 11,
            metagenome: false,
            workers: 0,
            definition: "SubTyping".to_string(),
            vicinity: 600,
            clustering_threshold: 0,
            quiet: false,
            fast: false,
            cas_models_dir: None,
            cas_profiles_dir: None,
        }
    }
}

/// Combined CRISPR + Cas analysis result for higher-level library callers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FullAnalysisResult {
    pub crisprs: Vec<CrisprArray>,
    pub cas_clusters: Vec<CasCluster>,
}
