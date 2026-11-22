// p7_hmm.rs - The Plan7 core HMM data structure
//
// Port of src/p7_hmm.c

use crate::hmmer_core::alphabet::Alphabet;
use crate::hmmer_core::config::*;

/// E-value statistical parameters estimated during HMM calibration.
///
/// Contains the Gumbel/exponential parameters for MSV, Viterbi, and Forward
/// filter scores, used to compute P-values and E-values in the search pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvParams {
    pub msv_mu: f32,
    pub msv_lambda: f32,
    pub viterbi_mu: f32,
    pub viterbi_lambda: f32,
    pub forward_tau: f32,
    pub forward_lambda: f32,
}

impl EvParams {
    /// Create from the legacy fixed-size array representation.
    pub fn from_array(arr: &[f32; NUM_EV_PARAMS]) -> Self {
        Self {
            msv_mu: arr[EvParam::MsvMu.idx()],
            msv_lambda: arr[EvParam::MsvLambda.idx()],
            viterbi_mu: arr[EvParam::ViterbiMu.idx()],
            viterbi_lambda: arr[EvParam::ViterbiLambda.idx()],
            forward_tau: arr[EvParam::ForwardTau.idx()],
            forward_lambda: arr[EvParam::ForwardLambda.idx()],
        }
    }

    /// Convert to the legacy fixed-size array representation.
    pub fn to_array(&self) -> [f32; NUM_EV_PARAMS] {
        let mut arr = [EV_PARAM_UNSET; NUM_EV_PARAMS];
        arr[EvParam::MsvMu.idx()] = self.msv_mu;
        arr[EvParam::MsvLambda.idx()] = self.msv_lambda;
        arr[EvParam::ViterbiMu.idx()] = self.viterbi_mu;
        arr[EvParam::ViterbiLambda.idx()] = self.viterbi_lambda;
        arr[EvParam::ForwardTau.idx()] = self.forward_tau;
        arr[EvParam::ForwardLambda.idx()] = self.forward_lambda;
        arr
    }
}

/// Pfam score cutoff thresholds.
///
/// Each pair (per-sequence, per-domain) may be absent if not defined for the HMM.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Cutoffs {
    pub gathering: Option<(f32, f32)>,
    pub trusted: Option<(f32, f32)>,
    pub noise: Option<(f32, f32)>,
}

/// The core Plan7 profile HMM data structure.
///
/// Corresponds to P7_HMM in the C code.
///
/// Key layout (flat arrays with row-view accessors):
///   - `transitions(node)`: transition probabilities at node (slice of 7)
///   - `match_emissions(node)`: match emission probabilities at node (slice of K)
///   - `insert_emissions(node)`: insert emission probabilities at node (slice of K)
///   - Nodes are numbered 0..M, where 0 is a special entry node
#[derive(Debug, Clone)]
pub struct Hmm {
    // Core model parameters (flat layouts)
    pub num_nodes: usize,
    transitions: Vec<f32>,      // transition probs: flat [(M+1) * NTRANSITIONS]
    match_emissions: Vec<f32>,  // match emissions:  flat [(M+1) * K]
    insert_emissions: Vec<f32>, // insert emissions: flat [(M+1) * K]

    // Annotation
    pub name: String,
    pub accession: Option<String>,
    pub description: Option<String>,
    pub reference_annotation: Option<Vec<u8>>,
    pub model_mask: Option<Vec<u8>>,
    pub consensus: Option<Vec<u8>>,
    pub consensus_structure: Option<Vec<u8>>,

    // Metadata
    pub command_log: Option<String>,
    pub num_sequences: Option<u32>,
    pub effective_num_seq_float: Option<f32>,
    pub effective_num_seq: f64,
    pub max_length: Option<usize>,
    pub map: Option<Vec<i32>>,
    pub checksum: Option<u32>,

    // Model parameters
    pub ev_params: Option<EvParams>,
    pub cutoffs: Cutoffs,
    pub model_composition: Option<Vec<f32>>,

    pub alphabet: Alphabet,
}

impl Hmm {
    /// Create a new HMM of M nodes for the given alphabet.
    pub fn new(m: usize, abc: &Alphabet) -> Self {
        let canonical_size = abc.canonical_size;
        let mut hmm = Hmm {
            num_nodes: m,
            transitions: vec![0.0; (m + 1) * HMM_NUM_TRANSITIONS],
            match_emissions: vec![0.0; (m + 1) * canonical_size],
            insert_emissions: vec![0.0; (m + 1) * canonical_size],
            name: String::new(),
            accession: None,
            description: None,
            reference_annotation: None,
            model_mask: None,
            consensus: None,
            consensus_structure: None,
            command_log: None,
            num_sequences: None,
            effective_num_seq_float: None,
            effective_num_seq: -1.0,
            max_length: None,
            map: None,
            checksum: None,
            ev_params: None,
            cutoffs: Cutoffs::default(),
            model_composition: None,
            alphabet: abc.clone(),
        };
        // Enforce conventions on node 0
        hmm.match_emissions_mut(0)[0] = 1.0;
        hmm.transitions_mut(0)[HTransition::DeleteToMatch.idx()] = 1.0;
        hmm
    }

    // ---------------------------------------------------------------
    // Row-view accessors for flat arrays
    // ---------------------------------------------------------------

    #[inline]
    pub fn transitions(&self, node: usize) -> &[f32] {
        let start = node * HMM_NUM_TRANSITIONS;
        &self.transitions[start..start + HMM_NUM_TRANSITIONS]
    }

    #[inline]
    pub fn transitions_mut(&mut self, node: usize) -> &mut [f32] {
        let start = node * HMM_NUM_TRANSITIONS;
        &mut self.transitions[start..start + HMM_NUM_TRANSITIONS]
    }

    #[inline]
    pub fn match_emissions(&self, node: usize) -> &[f32] {
        let ak = self.alphabet.canonical_size;
        let start = node * ak;
        &self.match_emissions[start..start + ak]
    }

    #[inline]
    pub fn match_emissions_mut(&mut self, node: usize) -> &mut [f32] {
        let ak = self.alphabet.canonical_size;
        let start = node * ak;
        &mut self.match_emissions[start..start + ak]
    }

    #[inline]
    pub fn insert_emissions(&self, node: usize) -> &[f32] {
        let ak = self.alphabet.canonical_size;
        let start = node * ak;
        &self.insert_emissions[start..start + ak]
    }

    #[inline]
    pub fn insert_emissions_mut(&mut self, node: usize) -> &mut [f32] {
        let ak = self.alphabet.canonical_size;
        let start = node * ak;
        &mut self.insert_emissions[start..start + ak]
    }

    // ---------------------------------------------------------------
    // Other routines
    // ---------------------------------------------------------------

    /// Calculate the occupancy of match and insert states.
    /// mocc\[1..M\] = probability of occupying match state canonical_size (mocc\[0\] = 0)
    /// iocc\[1..M\] = probability of occupying insert state canonical_size (iocc\[0\] = 0)
    pub fn calculate_occupancy(&self) -> (Vec<f32>, Vec<f32>) {
        let mut mocc = vec![0.0f32; self.num_nodes + 1];
        let mut iocc = vec![0.0f32; self.num_nodes + 1];

        let node_zero_transitions = self.transitions(0);
        mocc[1] = node_zero_transitions[HTransition::MatchToMatch.idx()]
            + node_zero_transitions[HTransition::MatchToInsert.idx()];
        // As `p7_hmm_CalculateOccupancy()`: the first product in single
        // precision, the second in double precision and fused with the sum
        // (HMMER's arm64 build; see `background::fused_mul_add`), the result
        // rounded to single precision. The begin scores of a profile, and
        // through them the words of the Viterbi filter, follow from these.
        for node in 2..=self.num_nodes {
            let node_transitions = self.transitions(node - 1);
            let stayed = mocc[node - 1]
                * (node_transitions[HTransition::MatchToMatch.idx()]
                    + node_transitions[HTransition::MatchToInsert.idx()]);
            mocc[node] = (1.0 - f64::from(mocc[node - 1])).mul_add(
                f64::from(node_transitions[HTransition::DeleteToMatch.idx()]),
                f64::from(stayed),
            ) as f32;
        }

        for canonical_size in 1..=self.num_nodes {
            let node_transitions = self.transitions(canonical_size);
            iocc[canonical_size] = mocc[canonical_size]
                * node_transitions[HTransition::MatchToInsert.idx()]
                / (1.0 - node_transitions[HTransition::InsertToInsert.idx()]).max(1e-10);
        }

        (mocc, iocc)
    }
}

// ---------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_destroy() {
        let abc = Alphabet::amino();
        let hmm = Hmm::new(100, &abc);

        assert_eq!(hmm.num_nodes, 100);
        assert_eq!(hmm.transitions(0).len(), HMM_NUM_TRANSITIONS);
        assert_eq!(hmm.match_emissions(1).len(), abc.canonical_size);
        assert_eq!(hmm.insert_emissions(1).len(), abc.canonical_size);
    }

    #[test]
    fn test_clone() {
        let abc = Alphabet::amino();
        let mut hmm = Hmm::new(10, &abc);
        hmm.name = "test_model".to_string();
        hmm.accession = Some("PF00001".to_string());
        hmm.match_emissions_mut(1)[0] = 0.5;
        hmm.transitions_mut(1)[0] = 0.9;

        let cloned = hmm.clone();
        assert_eq!(cloned.name, "test_model");
        assert_eq!(cloned.accession, Some("PF00001".to_string()));
        assert!((cloned.match_emissions(1)[0] - 0.5).abs() < 1e-6);
        assert!((cloned.transitions(1)[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_ev_params_roundtrip() {
        let ev = EvParams {
            msv_mu: 1.0,
            msv_lambda: 2.0,
            viterbi_mu: 3.0,
            viterbi_lambda: 4.0,
            forward_tau: 5.0,
            forward_lambda: 6.0,
        };
        let arr = ev.to_array();
        let ev2 = EvParams::from_array(&arr);
        assert_eq!(ev, ev2);
    }
}
