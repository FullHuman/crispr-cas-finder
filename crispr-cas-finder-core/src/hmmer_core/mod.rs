//! Pure Rust profile-HMM algorithms, scoring, and search.
//! Requires the pinned nightly compiler for `std::simd`.

pub use rayon;

// Re-export error types
pub use errors::HmmerError;

// --- Foundation modules ---
pub mod alphabet;
pub mod config;
pub mod constants;
pub mod errors;
pub mod hmmer;
pub mod rng;
pub mod sequence;

// --- Core data structures ---
pub mod background;
pub mod hmm;
pub mod modelconfig;
pub mod profile;
pub mod trace;

// --- Module groups ---
pub mod dynamic_programming;
pub mod pipeline;
pub mod results;
pub mod stats;

// --- Other ---
pub mod test_helpers;

// --- Re-exports for convenience ---
pub use dynamic_programming::forward_backward;
pub use dynamic_programming::logsum;
pub use dynamic_programming::null2;
pub use dynamic_programming::optimal_accuracy;
pub use dynamic_programming::score_matrix;
pub use dynamic_programming::simd;
pub use pipeline::domain;
pub use pipeline::domaindef;
pub use pipeline::search as pipeline_search;
pub use results::alidisplay;
pub use results::hit;
pub use results::tophits;
pub use stats::evalues;
