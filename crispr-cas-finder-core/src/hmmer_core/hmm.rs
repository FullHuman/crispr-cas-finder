// p7_hmm.rs - The Plan7 core HMM data structure
//
// Port of src/p7_hmm.c

use crate::hmmer_core::alphabet::Alphabet;
use crate::hmmer_core::config::*;

/// E-value statistical parameters estimated during HMM calibration.
///
/// Contains the Gumbel/exponential parameters for MSV, Viterbi, and Forward
/// filter scores, used to compute P-values and E-values in the search pipeline.
#[cfg_attr(feature = "hmmer-serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "hmmer-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Cutoffs {
    pub gathering: Option<(f32, f32)>,
    pub trusted: Option<(f32, f32)>,
    pub noise: Option<(f32, f32)>,
}

impl Cutoffs {
    /// Convert to the legacy fixed-size array representation.
    pub fn to_array(&self) -> [f32; NUM_CUTOFFS] {
        let mut arr = [CUTOFF_UNSET; NUM_CUTOFFS];
        if let Some((v1, v2)) = self.gathering {
            arr[Cutoff::GatheringSequence.idx()] = v1;
            arr[Cutoff::GatheringDomain.idx()] = v2;
        }
        if let Some((v1, v2)) = self.trusted {
            arr[Cutoff::TrustedSequence.idx()] = v1;
            arr[Cutoff::TrustedDomain.idx()] = v2;
        }
        if let Some((v1, v2)) = self.noise {
            arr[Cutoff::NoiseSequence.idx()] = v1;
            arr[Cutoff::NoiseDomain.idx()] = v2;
        }
        arr
    }
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
#[cfg_attr(feature = "hmmer-serde", derive(serde::Serialize, serde::Deserialize))]
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
    pub character_annotation: Option<Vec<u8>>,

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

    pub offset: i64,
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
            character_annotation: None,
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
            offset: 0,
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
    // Core operations
    // ---------------------------------------------------------------

    /// Copy parameters (t, mat, ins) from src to self.
    /// Both must have the same M and alphabet.
    pub fn copy_parameters(&mut self, src: &Hmm) {
        debug_assert_eq!(self.num_nodes, src.num_nodes);
        debug_assert_eq!(self.alphabet.canonical_size, src.alphabet.canonical_size);
        self.transitions.copy_from_slice(&src.transitions);
        self.match_emissions.copy_from_slice(&src.match_emissions);
        self.insert_emissions.copy_from_slice(&src.insert_emissions);
    }

    /// Set all parameters to zero.
    pub fn zero(&mut self) {
        self.transitions.fill(0.0);
        self.match_emissions.fill(0.0);
        self.insert_emissions.fill(0.0);
        self.model_composition = None;
    }

    // ---------------------------------------------------------------
    // Convenience routines for setting fields
    // ---------------------------------------------------------------

    /// Set or change the name of the HMM. Trailing whitespace is chopped.
    pub fn set_name(&mut self, name: &str) {
        self.name = name.trim_end().to_string();
    }

    /// Set or change the accession number.
    pub fn set_accession(&mut self, acc: Option<&str>) {
        self.accession = acc.map(|a| a.trim_end().to_string());
    }

    /// Set or change the description.
    pub fn set_description(&mut self, desc: Option<&str>) {
        self.description = desc.map(|d| d.trim_end().to_string());
    }

    /// Append a command line to the build log.
    pub fn append_comlog(&mut self, argv: &[String]) {
        let line = argv.join(" ");
        match &mut self.command_log {
            Some(log) => {
                log.push('\n');
                log.push_str(&line);
            }
            None => {
                self.command_log = Some(line);
            }
        }
    }
    // ---------------------------------------------------------------
    // Renormalization and rescaling
    // ---------------------------------------------------------------

    /// Scale all counts in the HMM by a factor.
    pub fn scale(&mut self, scale: f64) {
        let scale_factor = scale as f32;
        for v in self.transitions.iter_mut() {
            *v *= scale_factor;
        }
        for v in self.match_emissions.iter_mut() {
            *v *= scale_factor;
        }
        for v in self.insert_emissions.iter_mut() {
            *v *= scale_factor;
        }
    }

    /// Raise all parameters to an exponential power (for entropy weighting).
    pub fn scale_exponential(&mut self, exp: f64) {
        for v in self.transitions.iter_mut() {
            *v = (*v as f64).powf(exp) as f32;
        }
        for v in self.match_emissions.iter_mut() {
            *v = (*v as f64).powf(exp) as f32;
        }
        for v in self.insert_emissions.iter_mut() {
            *v = (*v as f64).powf(exp) as f32;
        }
    }

    /// Renormalize all probability parameters.
    pub fn renormalize(&mut self) {
        let canonical_size = self.alphabet.canonical_size;
        for node in 0..=self.num_nodes {
            let t = self.transitions_mut(node);
            vec_fnorm(&mut t[0..HMM_NUM_MATCH_TRANSITIONS]);
            vec_fnorm(
                &mut t[HTransition::InsertToMatch.idx()
                    ..HTransition::InsertToMatch.idx() + HMM_NUM_INSERT_TRANSITIONS],
            );
            vec_fnorm(
                &mut t[HTransition::DeleteToMatch.idx()
                    ..HTransition::DeleteToMatch.idx() + HMM_NUM_DELETE_TRANSITIONS],
            );
            vec_fnorm(self.match_emissions_mut(node));
            let ins = self.insert_emissions_mut(node);
            vec_fnorm(&mut ins[..canonical_size]);
        }
    }

    // ---------------------------------------------------------------
    // Debugging and development
    // ---------------------------------------------------------------

    /// Encode a state type string to internal code.
    pub fn encode_statetype(typestring: &str) -> TraceStateType {
        match typestring.to_uppercase().as_str() {
            "M" => TraceStateType::Match,
            "D" => TraceStateType::Delete,
            "I" => TraceStateType::Insert,
            "S" => TraceStateType::Start,
            "N" => TraceStateType::NTerminal,
            "B" => TraceStateType::Begin,
            "E" => TraceStateType::End,
            "C" => TraceStateType::CTerminal,
            "T" => TraceStateType::Terminate,
            "J" => TraceStateType::Jump,
            "X" => TraceStateType::Missing,
            _ => TraceStateType::Bogus,
        }
    }

    /// Decode a state type code to a string.
    pub fn decode_statetype(st: TraceStateType) -> &'static str {
        match st {
            TraceStateType::Match => "M",
            TraceStateType::Delete => "D",
            TraceStateType::Insert => "I",
            TraceStateType::Start => "S",
            TraceStateType::NTerminal => "N",
            TraceStateType::Begin => "B",
            TraceStateType::End => "E",
            TraceStateType::CTerminal => "C",
            TraceStateType::Terminate => "T",
            TraceStateType::Jump => "J",
            TraceStateType::Missing => "X",
            TraceStateType::Bogus => "?",
        }
    }

    /// Dump the HMM parameters to a file for debugging.
    pub fn dump(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        let canonical_size = self.alphabet.canonical_size;
        writeln!(writer, "HMM: {} M={}", self.name, self.num_nodes)?;

        for node in 0..=self.num_nodes {
            write!(writer, "  Node {:4}  ", node)?;
            write!(writer, "t: ")?;
            let t = self.transitions(node);
            for transition_value in &t[..HMM_NUM_TRANSITIONS] {
                write!(writer, "{:7.4} ", transition_value)?;
            }
            write!(writer, " mat: ")?;
            let mat = self.match_emissions(node);
            for &emission_value in &mat[..canonical_size.min(5)] {
                write!(writer, "{:7.4} ", emission_value)?;
            }
            if canonical_size > 5 {
                write!(writer, "...")?;
            }
            writeln!(writer)?;
        }
        Ok(())
    }

    /// Compare two HMMs for equality within tolerance.
    pub fn compare(&self, other: &Hmm, tol: f32) -> bool {
        if self.num_nodes != other.num_nodes {
            return false;
        }

        let canonical_size = self.alphabet.canonical_size;
        for node in 0..=self.num_nodes {
            let st = self.transitions(node);
            let ot = other.transitions(node);
            for j in 0..HMM_NUM_TRANSITIONS {
                if (st[j] - ot[j]).abs() > tol {
                    return false;
                }
            }
            let sm = self.match_emissions(node);
            let om = other.match_emissions(node);
            let si = self.insert_emissions(node);
            let oi = other.insert_emissions(node);
            for a in 0..canonical_size {
                if (sm[a] - om[a]).abs() > tol {
                    return false;
                }
                if (si[a] - oi[a]).abs() > tol {
                    return false;
                }
            }
        }
        true
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
        for canonical_size in 2..=self.num_nodes {
            let node_transitions = self.transitions(canonical_size - 1);
            mocc[canonical_size] = mocc[canonical_size - 1]
                * (node_transitions[HTransition::MatchToMatch.idx()]
                    + node_transitions[HTransition::MatchToInsert.idx()])
                + (1.0 - mocc[canonical_size - 1])
                    * node_transitions[HTransition::DeleteToMatch.idx()];
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

/// Normalize a float vector to sum to 1.0.
fn vec_fnorm(v: &mut [f32]) {
    let sum: f32 = v.iter().sum();
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
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
        hmm.set_name("test_model");
        hmm.set_accession(Some("PF00001"));
        hmm.match_emissions_mut(1)[0] = 0.5;
        hmm.transitions_mut(1)[0] = 0.9;

        let cloned = hmm.clone();
        assert_eq!(cloned.name, "test_model");
        assert_eq!(cloned.accession, Some("PF00001".to_string()));
        assert!((cloned.match_emissions(1)[0] - 0.5).abs() < 1e-6);
        assert!((cloned.transitions(1)[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_zero() {
        let abc = Alphabet::amino();
        let mut hmm = Hmm::new(5, &abc);
        hmm.match_emissions_mut(1)[0] = 1.0;
        hmm.transitions_mut(1)[0] = 1.0;

        hmm.zero();
        assert_eq!(hmm.match_emissions(1)[0], 0.0);
        assert_eq!(hmm.transitions(1)[0], 0.0);
    }

    #[test]
    fn test_encode_decode_statetype() {
        assert_eq!(Hmm::encode_statetype("M"), TraceStateType::Match);
        assert_eq!(Hmm::encode_statetype("d"), TraceStateType::Delete);
        assert_eq!(Hmm::decode_statetype(TraceStateType::Match), "M");
        assert_eq!(Hmm::decode_statetype(TraceStateType::Delete), "D");
    }

    #[test]
    fn test_compare() {
        let abc = Alphabet::amino();
        let hmm1 = Hmm::new(10, &abc);
        let hmm2 = hmm1.clone();
        assert!(hmm1.compare(&hmm2, 1e-6));
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

    #[test]
    fn test_cutoffs_roundtrip() {
        let c = Cutoffs {
            gathering: Some((25.0, 25.0)),
            trusted: None,
            noise: Some((20.0, 20.0)),
        };
        let arr = c.to_array();
        assert_eq!(arr[Cutoff::GatheringSequence.idx()], 25.0);
        assert_eq!(arr[Cutoff::GatheringDomain.idx()], 25.0);
        assert_eq!(arr[Cutoff::TrustedSequence.idx()], CUTOFF_UNSET);
        assert_eq!(arr[Cutoff::NoiseSequence.idx()], 20.0);
    }
}
