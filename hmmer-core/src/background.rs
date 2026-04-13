// p7_bg.rs - The null (background) model
//
// Port of src/p7_bg.c
//
// Contains three different things:
//   - "null1" model: one-state HMM of background frequencies + geometric length
//   - "bias filter" fhmm: two-state HMM from null1 background and model composition
//   - single omega term for null2 model balance

use crate::alphabet::{Alphabet, AlphabetKind, Dsq};
use crate::constants::background::{DEFAULT_OMEGA, DEFAULT_P1};
use crate::hmmer;

/// A simple two-state HMM used as the bias filter.
///
/// Previously `BiasFilterHmm` from Easel — only used here, so inlined.
/// Flat storage: `transitions` is row-major with stride `num_states + 1`,
/// `emissions` is row-major with stride `alphabet_size`.
#[derive(Debug, Clone)]
pub struct BiasFilterHmm {
    pub num_states: usize,             // number of states
    pub alphabet_size: usize,          // alphabet size (stride for emissions)
    pub transitions: Vec<f64>,         // t[i*(num_states+1) + j]
    pub emissions: Vec<f64>,           // e[i*alphabet_size + x]
    pub initial_state_probs: Vec<f64>, // initial state probs
}

impl BiasFilterHmm {
    pub fn new(num_states: usize, alphabet_size: usize) -> Self {
        BiasFilterHmm {
            num_states,
            alphabet_size,
            transitions: vec![0.0; num_states * (num_states + 1)],
            emissions: vec![0.0; num_states * alphabet_size],
            initial_state_probs: vec![0.0; num_states],
        }
    }

    #[inline]
    pub fn transition(&self, from: usize, to: usize) -> f64 {
        self.transitions[from * (self.num_states + 1) + to]
    }

    #[inline]
    pub fn set_transition(&mut self, from: usize, to: usize, val: f64) {
        self.transitions[from * (self.num_states + 1) + to] = val;
    }

    #[inline]
    pub fn emission(&self, state: usize, residue: usize) -> f64 {
        self.emissions[state * self.alphabet_size + residue]
    }

    #[inline]
    pub fn set_emission(&mut self, state: usize, residue: usize, val: f64) {
        self.emissions[state * self.alphabet_size + residue] = val;
    }
}

/// The null (background) model.
///
/// Corresponds to P7_BG in the C code.
#[derive(Debug, Clone)]
pub struct BackgroundModel {
    pub residue_frequencies: Vec<f32>, // null1 background residue frequencies [0..K-1]
    pub null1_transition_prob: f32,    // null1 transition prob; set by set_length()
    pub bias_filter_hmm: BiasFilterHmm, // bias filter HMM
    pub omega: f32,                    // prior on null2/null3
    pub alphabet: Alphabet,            // reference to alphabet
}

impl BackgroundModel {
    /// Create a P7_BG null model object for the given alphabet.
    ///
    /// For protein models, default iid background frequencies are set to
    /// average Swiss-Prot residue composition. For DNA/RNA and other
    /// alphabets, default frequencies are uniform.
    pub fn new(abc: &Alphabet) -> Self {
        let canonical_size = abc.canonical_size;
        let mut f = vec![0.0f32; canonical_size];

        if abc.kind == AlphabetKind::Amino {
            hmmer::amino_frequencies(&mut f);
        } else {
            let uniform = 1.0 / canonical_size as f32;
            f.iter_mut().for_each(|v| *v = uniform);
        }

        let fhmm = BiasFilterHmm::new(2, canonical_size);

        BackgroundModel {
            residue_frequencies: f,
            null1_transition_prob: DEFAULT_P1,
            bias_filter_hmm: fhmm,
            omega: DEFAULT_OMEGA,
            alphabet: abc.clone(),
        }
    }

    /// Create a background model with uniform frequencies.
    pub fn create_uniform(abc: &Alphabet) -> Self {
        let canonical_size = abc.canonical_size;
        let uniform = 1.0 / canonical_size as f32;
        let f = vec![uniform; canonical_size];
        let fhmm = BiasFilterHmm::new(2, canonical_size);

        BackgroundModel {
            residue_frequencies: f,
            null1_transition_prob: DEFAULT_P1,
            bias_filter_hmm: fhmm,
            omega: DEFAULT_OMEGA,
            alphabet: abc.clone(),
        }
    }

    /// Set the geometric null model length distribution to mean of L residues.
    pub fn set_length(&mut self, sequence_length: usize) {
        self.null1_transition_prob = sequence_length as f32 / (sequence_length as f32 + 1.0);
        self.bias_filter_hmm
            .set_transition(0, 0, self.null1_transition_prob as f64);
        self.bias_filter_hmm
            .set_transition(0, 1, 1.0 - self.null1_transition_prob as f64);
    }

    /// Calculate the null1 lod score for a digital sequence.
    ///
    /// Because the residue composition in null1 background is the same as the
    /// background used to calculate residue scores in profiles, all we
    /// have to do here is score null model transitions.
    pub fn null_one(&self, _digital_sequence: &[Dsq], sequence_length: usize) -> f32 {
        sequence_length as f32 * self.null1_transition_prob.ln()
            + (1.0 - self.null1_transition_prob).ln()
    }

    /// Set the bias filter model using model composition.
    ///
    /// Creates a two-state HMM: state 0 emits with model composition,
    /// state 1 emits with background composition.
    pub fn set_filter(&mut self, _num_nodes: usize, compo: &[f32]) {
        let canonical_size = self.alphabet.canonical_size;

        // State 0: model composition; State 1: background composition
        for a in 0..canonical_size {
            self.bias_filter_hmm
                .set_emission(0, a, compo[a.min(compo.len() - 1)] as f64);
            self.bias_filter_hmm
                .set_emission(1, a, self.residue_frequencies[a] as f64);
        }

        // Transitions:
        // state 0: stay in 0 with p1, go to end with 1-p1
        // state 1: same
        self.bias_filter_hmm
            .set_transition(0, 0, self.null1_transition_prob as f64);
        self.bias_filter_hmm
            .set_transition(0, 1, 1.0 - self.null1_transition_prob as f64);
        self.bias_filter_hmm
            .set_transition(1, 0, self.null1_transition_prob as f64);
        self.bias_filter_hmm
            .set_transition(1, 1, 1.0 - self.null1_transition_prob as f64);

        // Initial probs: start in state 0
        self.bias_filter_hmm.initial_state_probs[0] = 0.5;
        self.bias_filter_hmm.initial_state_probs[1] = 0.5;
    }

    /// Calculate the bias filter score for a digital sequence.
    pub fn filter_score(&self, digital_sequence: &[Dsq], sequence_length: usize) -> f32 {
        // Simple implementation: just return null1 score
        // (full implementation would run the two-state HMM forward algorithm)
        self.null_one(digital_sequence, sequence_length)
    }

    /// Dump the background model for debugging.
    pub fn dump(&self, fp: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(fp, "Background model:")?;
        for (i, &freq) in self.residue_frequencies.iter().enumerate() {
            write!(fp, "  f[{:2}] = {:.6}", i, freq)?;
            if i < self.alphabet.symbols.len() {
                write!(fp, " ({})", self.alphabet.symbols[i] as char)?;
            }
            writeln!(fp)?;
        }
        writeln!(fp, "  p1    = {:.6}", self.null1_transition_prob)?;
        writeln!(fp, "  omega = {:.6}", self.omega)?;
        Ok(())
    }
}

// ---------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create() {
        let abc = Alphabet::amino();
        let background = BackgroundModel::new(&abc);

        assert_eq!(background.residue_frequencies.len(), 20);
        // Check that frequencies sum to ~1
        let sum: f32 = background.residue_frequencies.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        assert!((background.null1_transition_prob - 350.0 / 351.0).abs() < 1e-6);
    }

    #[test]
    fn test_create_uniform() {
        let abc = Alphabet::dna();
        let background = BackgroundModel::create_uniform(&abc);

        assert_eq!(background.residue_frequencies.len(), 4);
        for &freq in &background.residue_frequencies {
            assert!((freq - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_set_length() {
        let abc = Alphabet::amino();
        let mut background = BackgroundModel::new(&abc);

        background.set_length(400);
        assert!((background.null1_transition_prob - 400.0 / 401.0).abs() < 1e-6);
    }
}
