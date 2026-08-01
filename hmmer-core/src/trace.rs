// p7_trace.rs - Traceback (alignment of seq to profile)
//
// Port of src/p7_trace.c
//
// A traceback path records a state sequence in a profile HMM alignment.

use crate::config::*;

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
    pub trace_start: i32,    // trace index of B state
    pub trace_end: i32,      // trace index of E state
    pub sequence_start: i32, // first M-emitted residue on sequence
    pub sequence_end: i32,   // last M-emitted residue on sequence
    pub hmm_start: i32,      // first M state on model
    pub hmm_end: i32,        // last M state on model
}

/// A traceback path — alignment of a sequence to a profile.
///
/// Corresponds to P7_TRACE in the C code.
#[derive(Debug, Clone)]
pub struct Trace {
    pub steps: Vec<TraceStep>,
    pub has_posterior_probs: bool,
    pub model_length: usize,
    pub sequence_length: usize,
}

impl Trace {
    /// Create a new growable, reusable traceback.
    pub fn new() -> Self {
        Trace {
            steps: Vec::with_capacity(256),
            has_posterior_probs: false,
            model_length: 0,
            sequence_length: 0,
        }
    }

    /// Create a traceback that includes posterior probabilities.
    pub fn with_pp() -> Self {
        Trace {
            steps: Vec::with_capacity(256),
            has_posterior_probs: true,
            model_length: 0,
            sequence_length: 0,
        }
    }

    /// Number of states in the trace.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Reinitialize for reuse.
    pub fn reuse(&mut self) {
        self.model_length = 0;
        self.sequence_length = 0;
        self.steps.clear();
    }

    /// Append a state to the trace.
    pub fn append(&mut self, st: TraceStateType, k: i32, i: i32) {
        self.steps.push(TraceStep {
            state: st,
            node_index: k,
            sequence_position: i,
            posterior_prob: 0.0,
        });
    }

    /// Append a state with posterior probability.
    pub fn append_with_pp(&mut self, st: TraceStateType, k: i32, i: i32, pp_val: f32) {
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
                let domain_trace_start = trace_idx as i32;

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

                let domain_trace_end = if scan_idx < n {
                    scan_idx as i32
                } else {
                    n as i32 - 1
                };

                domains.push(TraceDomain {
                    trace_start: domain_trace_start,
                    trace_end: domain_trace_end,
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

    /// Get state usage counts.
    pub fn get_state_use_counts(&self) -> [usize; NUM_TRACE_STATE_TYPES] {
        let mut counts = [0usize; NUM_TRACE_STATE_TYPES];
        for step in &self.steps {
            let idx = step.state.idx();
            if idx < NUM_TRACE_STATE_TYPES {
                counts[idx] += 1;
            }
        }
        counts
    }

    /// Dump the trace to a file for debugging.
    pub fn dump(&self, fp: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(
            fp,
            "Trace: N={} M={} L={}",
            self.steps.len(),
            self.model_length,
            self.sequence_length
        )?;
        for (z, step) in self.steps.iter().enumerate() {
            write!(
                fp,
                " {:4} {} {:4} {:4}",
                z,
                step.state.as_str(),
                step.node_index,
                step.sequence_position
            )?;
            if self.has_posterior_probs {
                write!(fp, " {:.4}", step.posterior_prob)?;
            }
            writeln!(fp)?;
        }
        Ok(())
    }

    /// Compare two traces for equality.
    pub fn compare(&self, other: &Trace, pp_tol: f32) -> bool {
        if self.steps.len() != other.steps.len() {
            return false;
        }
        for (a, b) in self.steps.iter().zip(other.steps.iter()) {
            if a.state != b.state {
                return false;
            }
            if a.node_index != b.node_index {
                return false;
            }
            if a.sequence_position != b.sequence_position {
                return false;
            }
        }
        if self.has_posterior_probs && other.has_posterior_probs {
            for (a, b) in self.steps.iter().zip(other.steps.iter()) {
                if (a.posterior_prob - b.posterior_prob).abs() > pp_tol {
                    return false;
                }
            }
        }
        true
    }

    /// Calculate the score of a trace given a digital sequence and profile.
    /// (Stub — full implementation requires P7_PROFILE)
    pub fn score(&self) -> f32 {
        0.0
    }

    /// Get the expected accuracy from posterior probabilities.
    pub fn get_expected_accuracy(&self) -> f32 {
        if !self.has_posterior_probs {
            return 0.0;
        }
        // OA traces annotate every residue-emitting state: M/I and N/J/C
        // self-loops. HMMER's p7_trace_GetExpectedAccuracy() sums all of these
        // annotations, while non-emitting steps carry 0.0.
        self.steps.iter().map(|s| s.posterior_prob).sum()
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_reuse() {
        let mut trace = Trace::new();
        assert_eq!(trace.len(), 0);

        // Append some states
        trace.append(TraceStateType::Start, 0, 0);
        trace.append(TraceStateType::NTerminal, 0, 0);
        trace.append(TraceStateType::Begin, 0, 0);
        trace.append(TraceStateType::Match, 1, 1);
        trace.append(TraceStateType::End, 0, 0);
        trace.append(TraceStateType::CTerminal, 0, 0);
        trace.append(TraceStateType::Terminate, 0, 0);
        assert_eq!(trace.len(), 7);

        // Reuse
        trace.reuse();
        assert_eq!(trace.len(), 0);
    }

    #[test]
    fn test_compute_domains() {
        let mut trace = Trace::new();

        // Create a simple trace with one domain
        trace.append(TraceStateType::Start, 0, 0);
        trace.append(TraceStateType::NTerminal, 0, 0);
        trace.append(TraceStateType::Begin, 0, 0);
        trace.append(TraceStateType::Match, 1, 1);
        trace.append(TraceStateType::Match, 2, 2);
        trace.append(TraceStateType::Match, 3, 3);
        trace.append(TraceStateType::End, 0, 0);
        trace.append(TraceStateType::CTerminal, 0, 0);
        trace.append(TraceStateType::Terminate, 0, 0);

        let domains = trace.compute_domains();
        assert_eq!(domains.len(), 1);
        let d = &domains[0];
        assert_eq!(d.sequence_start, 1);
        assert_eq!(d.sequence_end, 3);
        assert_eq!(d.hmm_start, 1);
        assert_eq!(d.hmm_end, 3);
    }

    #[test]
    fn test_compare() {
        let mut tr1 = Trace::new();
        tr1.append(TraceStateType::Start, 0, 0);
        tr1.append(TraceStateType::Match, 1, 1);

        let mut tr2 = Trace::new();
        tr2.append(TraceStateType::Start, 0, 0);
        tr2.append(TraceStateType::Match, 1, 1);

        assert!(tr1.compare(&tr2, 0.001));
    }

    #[test]
    fn expected_accuracy_includes_special_state_residue_posteriors() {
        let mut trace = Trace::with_pp();
        trace.append_with_pp(TraceStateType::Match, 1, 1, 0.6);
        trace.append_with_pp(TraceStateType::Insert, 1, 2, 0.2);
        trace.append_with_pp(TraceStateType::NTerminal, 0, 3, 0.1);
        trace.append_with_pp(TraceStateType::Jump, 0, 4, 0.05);
        trace.append_with_pp(TraceStateType::CTerminal, 0, 5, 0.04);
        trace.append_with_pp(TraceStateType::End, 0, 5, 0.0);

        assert!((trace.get_expected_accuracy() - 0.99).abs() < 1e-6);
    }
}
