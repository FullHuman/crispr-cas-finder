// p7_trace.rs - Traceback (alignment of seq to profile)
//
// Port of src/p7_trace.c
//
// A traceback path records a state sequence in a profile HMM alignment.

use crate::hmmer_core::config::*;

/// A single step in a traceback path.
#[derive(Debug, Clone, Copy)]
pub struct TraceStep {
    pub state: TraceStateType,
    pub node_index: i32,
    pub sequence_position: i32,
    pub posterior_prob: f32,
}

/// Domain coordinates identified within a trace (B..E pair).
#[derive(Debug, Clone)]
pub struct TraceDomain {
    pub sequence_start: i32, // first M-emitted residue on sequence
    pub sequence_end: i32,   // last M-emitted residue on sequence
    pub hmm_start: i32,      // first M state on model
    pub hmm_end: i32,        // last M state on model
}

/// A traceback path — alignment of a sequence to a profile, every step
/// carrying the posterior probability of the residue it emits.
///
/// Corresponds to P7_TRACE in the C code.
#[derive(Debug, Clone, Default)]
pub struct Trace {
    pub steps: Vec<TraceStep>,
}

impl Trace {
    /// Create a new growable traceback.
    pub fn new() -> Self {
        Trace {
            steps: Vec::with_capacity(256),
        }
    }

    /// Append a state with the posterior probability of its residue, zero
    /// for a state that emits none.
    pub fn append(&mut self, st: TraceStateType, k: i32, i: i32, pp_val: f32) {
        self.steps.push(TraceStep {
            state: st,
            node_index: k,
            sequence_position: i,
            posterior_prob: pp_val,
        });
    }

    /// Reverse the trace in place.
    pub fn reverse(&mut self) {
        self.steps.reverse();
    }

    /// Identify domains (B..E pairs) in the trace.
    pub fn compute_domains(&self) -> Vec<TraceDomain> {
        let mut domains = Vec::new();
        let n = self.steps.len();
        let mut trace_idx = 0;
        while trace_idx < n {
            if self.steps[trace_idx].state == TraceStateType::Begin {
                // Find first M state emission
                let mut first_i = 0i32;
                let mut first_k = 0i32;
                let mut scan_idx = trace_idx + 1;
                while scan_idx < n && self.steps[scan_idx].state != TraceStateType::End {
                    if self.steps[scan_idx].state == TraceStateType::Match && first_i == 0 {
                        first_i = self.steps[scan_idx].sequence_position;
                        first_k = self.steps[scan_idx].node_index;
                    }
                    scan_idx += 1;
                }

                // Find last M state emission
                let mut last_i = 0i32;
                let mut last_k = 0i32;
                if scan_idx < n {
                    let mut reverse_scan_idx = scan_idx;
                    while reverse_scan_idx > trace_idx {
                        reverse_scan_idx -= 1;
                        if self.steps[reverse_scan_idx].state == TraceStateType::Match {
                            last_i = self.steps[reverse_scan_idx].sequence_position;
                            last_k = self.steps[reverse_scan_idx].node_index;
                            break;
                        }
                    }
                }

                domains.push(TraceDomain {
                    sequence_start: first_i,
                    sequence_end: last_i,
                    hmm_start: first_k,
                    hmm_end: last_k,
                });

                trace_idx = scan_idx;
            }
            trace_idx += 1;
        }
        domains
    }

    /// Get the expected accuracy from posterior probabilities.
    pub fn get_expected_accuracy(&self) -> f32 {
        // OA traces annotate every residue-emitting state: M/I and N/J/C
        // self-loops. HMMER's p7_trace_GetExpectedAccuracy() sums all of these
        // annotations, while non-emitting steps carry 0.0.
        self.steps.iter().map(|s| s.posterior_prob).sum()
    }
}

// ---------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_domains() {
        let mut trace = Trace::new();

        // Create a simple trace with one domain
        trace.append(TraceStateType::Start, 0, 0, 0.0);
        trace.append(TraceStateType::NTerminal, 0, 0, 0.0);
        trace.append(TraceStateType::Begin, 0, 0, 0.0);
        trace.append(TraceStateType::Match, 1, 1, 0.0);
        trace.append(TraceStateType::Match, 2, 2, 0.0);
        trace.append(TraceStateType::Match, 3, 3, 0.0);
        trace.append(TraceStateType::End, 0, 0, 0.0);
        trace.append(TraceStateType::CTerminal, 0, 0, 0.0);
        trace.append(TraceStateType::Terminate, 0, 0, 0.0);

        let domains = trace.compute_domains();
        assert_eq!(domains.len(), 1);
        let d = &domains[0];
        assert_eq!(d.sequence_start, 1);
        assert_eq!(d.sequence_end, 3);
        assert_eq!(d.hmm_start, 1);
        assert_eq!(d.hmm_end, 3);
    }

    #[test]
    fn expected_accuracy_includes_special_state_residue_posteriors() {
        let mut trace = Trace::new();
        trace.append(TraceStateType::Match, 1, 1, 0.6);
        trace.append(TraceStateType::Insert, 1, 2, 0.2);
        trace.append(TraceStateType::NTerminal, 0, 3, 0.1);
        trace.append(TraceStateType::Jump, 0, 4, 0.05);
        trace.append(TraceStateType::CTerminal, 0, 5, 0.04);
        trace.append(TraceStateType::End, 0, 5, 0.0);

        assert!((trace.get_expected_accuracy() - 0.99).abs() < 1e-6);
    }
}
