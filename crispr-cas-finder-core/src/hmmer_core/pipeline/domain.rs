// p7_domain.rs - Domain data structure
//
// Port of src/p7_domain.c

use crate::hmmer_core::alidisplay::AliDisplay;
use crate::hmmer_core::trace::Trace;

/// A single domain hit within a sequence.
///
/// Coordinate fields (`ienv`, `jenv`, `iali`, `jali`, `hmm_from`, `hmm_to`)
/// use **1-based** numbering, matching HMMER's output conventions.
#[derive(Debug, Clone, Default)]
pub struct Domain {
    pub envelope_start: usize,       // envelope start in sequence (1-based)
    pub envelope_end: usize,         // envelope end in sequence (1-based)
    pub alignment_start: usize,      // alignment start in sequence (1-based)
    pub alignment_end: usize,        // alignment end in sequence (1-based)
    pub hmm_from: usize,             // first match state on model (1-based, 1..=M)
    pub hmm_to: usize,               // last match state on model (1-based, 1..=M)
    pub envelope_score: f32,         // Forward score in envelope (nats)
    pub domain_correction: f32,      // null2 correction for domain
    pub domain_bias: f32,            // null2 bias contribution (nats)
    pub optimal_accuracy_score: f32, // expected number of correctly decoded positions
    pub bitscore: f32,               // total score in bits, null corrected
    pub log_pvalue: f64,             // log(P-value) of the bitscore
    pub is_reported: bool,
    pub is_included: bool,
    pub scores_per_pos: Option<Vec<f32>>, // per-position scores (nhmmer only)
    pub alignment_display: Option<Box<AliDisplay>>,
    pub trace: Option<Trace>, // OA traceback (used to build AliDisplay)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let d = Domain::default();
        assert_eq!(d.envelope_start, 0);
        assert_eq!(d.hmm_from, 0);
        assert!(d.alignment_display.is_none());
    }
}
