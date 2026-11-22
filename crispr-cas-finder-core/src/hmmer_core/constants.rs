//! Shared constants for hmmer-core.
//!
//! Hot-path DP/SIMD indexing constants remain in their local modules for
//! readability and cache-local code organization.

/// Maximum size of canonical alphabet (4 for DNA, 20 for amino acids).
pub const MAX_CANONICAL_ALPHABET: usize = 20;

/// Maximum size of full/degenerate alphabet (18 for DNA/RNA, 29 for amino).
pub const MAX_FULL_ALPHABET: usize = 29;

// ---------------------------------------------------------------
// E-value/statistics array sizes and sentinels
// ---------------------------------------------------------------

pub const NUM_EV_PARAMS: usize = 6;

pub const EV_PARAM_UNSET: f32 = -99999.0;
pub const COMPOSITION_UNSET: f32 = -1.0;

// ---------------------------------------------------------------
// HMM/Profile layout constants
// ---------------------------------------------------------------

pub const HMM_NUM_TRANSITIONS: usize = 7;

pub const PROFILE_NUM_TRANSITIONS: usize = 8;
pub const PROFILE_NUM_EMISSIONS: usize = 2;

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

        pub const SCORE_LENGTH_PRIOR_OFFSET: f32 = 3.0;
        pub const MIN_PVALUE_CLAMP: f64 = 1e-300;
    }
}
