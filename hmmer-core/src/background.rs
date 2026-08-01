use crate::alphabet::{Alphabet, AlphabetKind, Dsq};
use crate::constants::background::{DEFAULT_OMEGA, DEFAULT_P1};
use crate::errors::HmmerError;
use crate::hmmer;

const BIAS_FILTER_STATES: usize = 2;
const BIAS_FILTER_TRANSITION_STRIDE: usize = BIAS_FILTER_STATES + 1;

/// Two-state HMM used by HMMER's composition-bias filter.
///
/// State 0 emits the normal background composition and state 1 emits the
/// profile's average composition. Emissions are stored as odds relative to
/// the normal background, matching Easel's configured `ESL_HMM`.
#[derive(Debug, Clone)]
struct BiasFilterHmm {
    transitions: [f32; BIAS_FILTER_STATES * BIAS_FILTER_TRANSITION_STRIDE],
    emission_odds: Vec<f32>,
    initial_state_probs: [f32; BIAS_FILTER_STATES],
    full_alphabet_size: usize,
}

impl BiasFilterHmm {
    fn new(full_alphabet_size: usize) -> Self {
        Self {
            transitions: [0.0; BIAS_FILTER_STATES * BIAS_FILTER_TRANSITION_STRIDE],
            emission_odds: vec![0.0; BIAS_FILTER_STATES * full_alphabet_size],
            initial_state_probs: [0.0; BIAS_FILTER_STATES],
            full_alphabet_size,
        }
    }

    #[inline]
    fn transition(&self, from: usize, to: usize) -> f32 {
        self.transitions[from * BIAS_FILTER_TRANSITION_STRIDE + to]
    }

    #[inline]
    fn set_transition(&mut self, from: usize, to: usize, value: f32) {
        self.transitions[from * BIAS_FILTER_TRANSITION_STRIDE + to] = value;
    }

    #[inline]
    fn emission_odds(&self, state: usize, residue: Dsq) -> f32 {
        let residue = residue as usize;
        if residue < self.full_alphabet_size {
            self.emission_odds[state * self.full_alphabet_size + residue]
        } else {
            1.0
        }
    }

    #[inline]
    fn set_emission_odds(&mut self, state: usize, residue: usize, value: f32) {
        self.emission_odds[state * self.full_alphabet_size + residue] = value;
    }
}

/// The null (background) model.
///
/// Corresponds to P7_BG in the C code.
#[derive(Debug, Clone)]
pub struct BackgroundModel {
    pub residue_frequencies: Vec<f32>, // null1 background residue frequencies [0..K-1]
    pub null1_transition_prob: f32,    // null1 transition prob; set by set_length()
    bias_filter_hmm: BiasFilterHmm,
    pub omega: f32,         // prior on null2/null3
    pub alphabet: Alphabet, // reference to alphabet
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

        let fhmm = BiasFilterHmm::new(abc.full_size);

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
        let fhmm = BiasFilterHmm::new(abc.full_size);

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
            .set_transition(0, 0, self.null1_transition_prob);
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
    /// Creates the same two-state conditional-on-length model as
    /// `p7_bg_SetFilter`: state 0 emits the normal background and state 1
    /// emits the profile's average composition.
    pub fn set_filter(&mut self, num_nodes: usize, compo: &[f32]) -> Result<(), HmmerError> {
        let canonical_size = self.alphabet.canonical_size;
        if num_nodes == 0 {
            return Err(HmmerError::InvalidArgument(
                "bias filter requires a non-empty profile".into(),
            ));
        }
        if compo.len() < canonical_size {
            return Err(HmmerError::InvalidArgument(format!(
                "bias filter composition has {} residues; expected {}",
                compo.len(),
                canonical_size
            )));
        }

        let composition_sum: f32 = compo[..canonical_size].iter().sum();
        if !composition_sum.is_finite()
            || composition_sum <= 0.0
            || compo[..canonical_size]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(HmmerError::InvalidArgument(
                "bias filter composition must contain finite, non-negative probabilities".into(),
            ));
        }

        let normal_mean_length = 400.0f32;
        let biased_mean_length = num_nodes as f32 / 8.0;

        self.bias_filter_hmm
            .set_transition(0, 0, normal_mean_length / (normal_mean_length + 1.0));
        self.bias_filter_hmm
            .set_transition(0, 1, 1.0 / (normal_mean_length + 1.0));
        self.bias_filter_hmm.set_transition(0, 2, 1.0);
        self.bias_filter_hmm
            .set_transition(1, 0, 1.0 / (biased_mean_length + 1.0));
        self.bias_filter_hmm
            .set_transition(1, 1, biased_mean_length / (biased_mean_length + 1.0));
        self.bias_filter_hmm.set_transition(1, 2, 1.0);
        self.bias_filter_hmm.initial_state_probs = [0.999, 0.001];

        // Configure canonical emission odds relative to the null1 background.
        for (residue, &composition_probability) in compo.iter().take(canonical_size).enumerate() {
            let background = self.residue_frequencies[residue];
            let normal_odds = if background > 0.0 { 1.0 } else { 0.0 };
            let biased_probability = composition_probability / composition_sum;
            let biased_odds = if background > 0.0 {
                biased_probability / background
            } else {
                0.0
            };
            self.bias_filter_hmm
                .set_emission_odds(0, residue, normal_odds);
            self.bias_filter_hmm
                .set_emission_odds(1, residue, biased_odds);
        }

        // Easel treats gap, nonresidue, and missing data as X (odds 1).
        let gap = canonical_size;
        let nonresidue = self.alphabet.full_size - 2;
        let missing = self.alphabet.full_size - 1;
        for state in 0..BIAS_FILTER_STATES {
            self.bias_filter_hmm.set_emission_odds(state, gap, 1.0);
            self.bias_filter_hmm
                .set_emission_odds(state, nonresidue, 1.0);
            self.bias_filter_hmm.set_emission_odds(state, missing, 1.0);
        }

        // Configure ambiguous residues from their canonical expansions.
        for residue in (canonical_size + 1)..nonresidue {
            let degeneracy = self.alphabet.degen_row(residue);
            let denominator: f32 = degeneracy
                .iter()
                .zip(self.residue_frequencies.iter())
                .filter_map(|(&included, &frequency)| included.then_some(frequency))
                .sum();
            for state in 0..BIAS_FILTER_STATES {
                let numerator: f32 = degeneracy
                    .iter()
                    .enumerate()
                    .filter_map(|(canonical, &included)| {
                        included.then_some(
                            self.bias_filter_hmm.emission_odds(state, canonical as Dsq)
                                * self.residue_frequencies[canonical],
                        )
                    })
                    .sum();
                self.bias_filter_hmm.set_emission_odds(
                    state,
                    residue,
                    if denominator > 0.0 {
                        numerator / denominator
                    } else {
                        0.0
                    },
                );
            }
        }

        Ok(())
    }

    /// Calculate the two-state bias-filter Forward score in nats.
    ///
    /// The HMM itself is conditional on sequence length, so the same geometric
    /// length term as null1 is added after the Forward calculation.
    pub fn filter_score(&self, digital_sequence: &[Dsq], sequence_length: usize) -> f32 {
        if sequence_length == 0 {
            return self.null_one(digital_sequence, sequence_length);
        }
        assert!(
            sequence_length <= digital_sequence.len(),
            "sequence length exceeds digital sequence storage"
        );

        let first = digital_sequence[0];
        let mut previous = [
            self.bias_filter_hmm.initial_state_probs[0]
                * self.bias_filter_hmm.emission_odds(0, first),
            self.bias_filter_hmm.initial_state_probs[1]
                * self.bias_filter_hmm.emission_odds(1, first),
        ];
        let mut scale = previous[0].max(previous[1]);
        if scale <= 0.0 || !scale.is_finite() {
            return f32::NEG_INFINITY;
        }
        previous.iter_mut().for_each(|value| *value /= scale);
        let mut forward_score = scale.ln();

        for &residue in &digital_sequence[1..sequence_length] {
            let mut current = [0.0f32; BIAS_FILTER_STATES];
            for (state, value) in current.iter_mut().enumerate() {
                *value = (previous[0] * self.bias_filter_hmm.transition(0, state)
                    + previous[1] * self.bias_filter_hmm.transition(1, state))
                    * self.bias_filter_hmm.emission_odds(state, residue);
            }

            scale = current[0].max(current[1]);
            if scale <= 0.0 || !scale.is_finite() {
                return f32::NEG_INFINITY;
            }
            current.iter_mut().for_each(|value| *value /= scale);
            forward_score += scale.ln();
            previous = current;
        }

        let end_probability = previous[0] * self.bias_filter_hmm.transition(0, 2)
            + previous[1] * self.bias_filter_hmm.transition(1, 2);
        if end_probability <= 0.0 || !end_probability.is_finite() {
            return f32::NEG_INFINITY;
        }

        forward_score + end_probability.ln() + self.null_one(digital_sequence, sequence_length)
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

        background
            .set_filter(80, &background.residue_frequencies.clone())
            .unwrap();
        let switch_probability = background.bias_filter_hmm.transition(0, 1);
        background.set_length(400);
        assert!((background.null1_transition_prob - 400.0 / 401.0).abs() < 1e-6);
        assert!((background.bias_filter_hmm.transition(0, 0) - 400.0 / 401.0).abs() < 1e-6);
        assert_eq!(
            background.bias_filter_hmm.transition(0, 1),
            switch_probability
        );
    }

    #[test]
    fn test_set_filter_matches_upstream_parameters() {
        let abc = Alphabet::amino();
        let mut background = BackgroundModel::new(&abc);
        let composition = background.residue_frequencies.clone();

        background.set_filter(80, &composition).unwrap();

        let filter = &background.bias_filter_hmm;
        assert!((filter.transition(0, 0) - 400.0 / 401.0).abs() < 1e-6);
        assert!((filter.transition(0, 1) - 1.0 / 401.0).abs() < 1e-6);
        assert_eq!(filter.transition(0, 2), 1.0);
        assert!((filter.transition(1, 0) - 1.0 / 11.0).abs() < 1e-6);
        assert!((filter.transition(1, 1) - 10.0 / 11.0).abs() < 1e-6);
        assert_eq!(filter.transition(1, 2), 1.0);
        assert_eq!(filter.initial_state_probs, [0.999, 0.001]);

        for residue in 0..abc.full_size {
            assert!((filter.emission_odds(0, residue as Dsq) - 1.0).abs() < 1e-6);
            assert!((filter.emission_odds(1, residue as Dsq) - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_filter_score_matches_direct_two_state_forward_sum() {
        let abc = Alphabet::dna();
        let mut background = BackgroundModel::create_uniform(&abc);
        background.set_filter(80, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        background.set_length(2);

        let a = abc.digitize(b'A');
        let sequence = [a, a];
        let filter = &background.bias_filter_hmm;
        let mut direct_odds = 0.0f32;
        for first_state in 0..BIAS_FILTER_STATES {
            for second_state in 0..BIAS_FILTER_STATES {
                direct_odds += filter.initial_state_probs[first_state]
                    * filter.emission_odds(first_state, a)
                    * filter.transition(first_state, second_state)
                    * filter.emission_odds(second_state, a)
                    * filter.transition(second_state, 2);
            }
        }
        let expected = direct_odds.ln() + background.null_one(&sequence, sequence.len());
        let observed = background.filter_score(&sequence, sequence.len());

        assert!((observed - expected).abs() < 1e-6);

        let biased_sequence = vec![a; 64];
        background.set_length(biased_sequence.len());
        assert!(
            background.filter_score(&biased_sequence, biased_sequence.len())
                > background.null_one(&biased_sequence, biased_sequence.len())
        );
    }
}
