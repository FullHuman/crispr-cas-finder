//! Shared constants for hmmer-core.
//!
//! Hot-path DP/SIMD indexing constants remain in their local modules for
//! readability and cache-local code organization.

/// Maximum size of canonical alphabet (4 for DNA, 20 for amino acids).
pub const MAX_CANONICAL_ALPHABET: usize = 20;

/// Maximum size of full/degenerate alphabet (18 for DNA/RNA, 29 for amino).
pub const MAX_FULL_ALPHABET: usize = 29;

// ---------------------------------------------------------------
// Version metadata
// ---------------------------------------------------------------

pub const HMMER_VERSION: &str = concat!(
    "crispr-cas-finder ",
    env!("CARGO_PKG_VERSION"),
    " Rust port"
);
pub const HMMER_DATE: &str = "Independent implementation; HMMER 3.4 reference";
pub const HMMER_COPYRIGHT: &str = "Upstream HMMER/Easel copyrights: see THIRD_PARTY_NOTICES.md";
pub const HMMER_LICENSE: &str =
    "BSD-3-Clause upstream portions; GPL-3.0-or-later project. See distributed notices.";
pub const HMMER_URL: &str = "http://hmmer.org";

// ---------------------------------------------------------------
// E-value/statistics array sizes and sentinels
// ---------------------------------------------------------------

pub const NUM_EV_PARAMS: usize = 6;
pub const NUM_CUTOFFS: usize = 6;
pub const NUM_OFFSETS: usize = 3;

pub const EV_PARAM_UNSET: f32 = -99999.0;
pub const CUTOFF_UNSET: f32 = -99999.0;
pub const COMPOSITION_UNSET: f32 = -1.0;

// ---------------------------------------------------------------
// HMM/Profile layout constants
// ---------------------------------------------------------------

pub const HMM_NUM_TRANSITIONS: usize = 7;

pub const HMM_NUM_MATCH_TRANSITIONS: usize = 3;
pub const HMM_NUM_DELETE_TRANSITIONS: usize = 2;
pub const HMM_NUM_INSERT_TRANSITIONS: usize = 2;

pub const NUM_TRACE_STATE_TYPES: usize = 12;

pub const PROFILE_NUM_TRANSITIONS: usize = 8;
pub const PROFILE_NUM_EMISSIONS: usize = 2;

// ---------------------------------------------------------------
// HMM flags (P7_HMM.flags bitset)
// ---------------------------------------------------------------

pub const HMM_FLAG_DESC: u32 = 1 << 1;
pub const HMM_FLAG_RF: u32 = 1 << 2;
pub const HMM_FLAG_CS: u32 = 1 << 3;

pub const HMM_FLAG_STATS: u32 = 1 << 7;
pub const HMM_FLAG_MAP: u32 = 1 << 8;
pub const HMM_FLAG_ACC: u32 = 1 << 9;
pub const HMM_FLAG_GA: u32 = 1 << 10;
pub const HMM_FLAG_TC: u32 = 1 << 11;
pub const HMM_FLAG_NC: u32 = 1 << 12;
pub const HMM_FLAG_CA: u32 = 1 << 13;
pub const HMM_FLAG_COMPO: u32 = 1 << 14;
pub const HMM_FLAG_CHKSUM: u32 = 1 << 15;
pub const HMM_FLAG_CONS: u32 = 1 << 16;
pub const HMM_FLAG_MMASK: u32 = 1 << 17;

pub mod background {
    /// Default null1 geometric length parameter: mean length 350.
    pub const DEFAULT_P1: f32 = 350.0 / 351.0;
    /// Default prior weight on null2/null3 model.
    pub const DEFAULT_OMEGA: f32 = 1.0 / 256.0;
}

pub mod dynamic_programming {
    pub mod logsum {
        pub const LOGSUM_SCALE: f32 = 1000.0;
    }
}

pub mod pipeline {
    pub mod search {
        pub const LOG2: f32 = std::f32::consts::LN_2;
        pub const DEFAULT_MSV_THRESHOLD: f64 = 0.02;
        pub const DEFAULT_VITERBI_THRESHOLD: f64 = 1e-3;
        pub const DEFAULT_FORWARD_THRESHOLD: f64 = 1e-5;

        pub const DEFAULT_MSV_BIAS_WINDOW: usize = 100;
        pub const DEFAULT_VITERBI_BIAS_WINDOW: usize = 240;
        pub const DEFAULT_FORWARD_BIAS_WINDOW: usize = 1000;

        pub const SCORE_LENGTH_PRIOR_OFFSET: f32 = 3.0;
        pub const MIN_PVALUE_CLAMP: f64 = 1e-300;
    }
}

pub mod results {
    pub mod alidisplay {
        // Bit-vector constants for serialization.
        #[allow(dead_code)]
        pub const RFLINE_PRESENT: u8 = 1 << 0;
        #[allow(dead_code)]
        pub const MMLINE_PRESENT: u8 = 1 << 1;
        #[allow(dead_code)]
        pub const CSLINE_PRESENT: u8 = 1 << 2;
        #[allow(dead_code)]
        pub const PPLINE_PRESENT: u8 = 1 << 3;
        #[allow(dead_code)]
        pub const ASEQ_PRESENT: u8 = 1 << 4;
        #[allow(dead_code)]
        pub const NTSEQ_PRESENT: u8 = 1 << 5;

        // BLOSUM62-based similarity groups.
        pub const SIMILARITY_GROUPS: &[&[u8]] = &[
            b"STA",  // small/hydroxyl
            b"NEQK", // amide/charged
            b"NHQK", // amide+
            b"NDEQ", // acid/amide
            b"QHRK", // charged
            b"MILV", // hydrophobic
            b"MILF", // hydrophobic+
            b"HY",   // aromatic-like
            b"FYW",  // aromatic
        ];
    }

    pub mod tophits {
        pub const DEFAULT_INITIAL_CAPACITY: usize = 256;
    }
}
