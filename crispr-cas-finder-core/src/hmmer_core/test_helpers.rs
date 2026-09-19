// test_helpers.rs - Testing utilities for HMMER Rust port
//
// Provides random HMM sampling, random sequence generation, and other
// utilities needed by the ported HMMER test suite. Corresponds to
// p7_hmm_Sample(), esl_rsq_xfIID(), and related C functions.

use crate::hmmer_core::alphabet::{Alphabet, Dsq};
use crate::hmmer_core::background::BackgroundModel;
use crate::hmmer_core::config::*;
use crate::hmmer_core::hmm::Hmm;
use crate::hmmer_core::profile::Profile;
use crate::hmmer_core::rng::XorShift64;

/// Sample a random HMM of length M for a given alphabet.
///
/// Corresponds to p7_hmm_Sample() in C.
/// Generates random transition and emission probabilities using
/// Dirichlet sampling, then normalizes.
pub fn hmm_sample(rng: &mut XorShift64, model_length: usize, alphabet: &Alphabet) -> Hmm {
    let canonical_size = alphabet.canonical_size;
    let mut hmm = Hmm::new(model_length, alphabet);

    // Sample emission probabilities for each node
    for node in 0..=model_length {
        // Match emissions: sample from Dirichlet(1,1,...,1) = uniform on simplex
        let match_row = sample_dirichlet(rng, canonical_size);
        hmm.match_emissions_mut(node)[..canonical_size].copy_from_slice(&match_row);

        // Insert emissions: sample from Dirichlet
        let insert_row = sample_dirichlet(rng, canonical_size);
        hmm.insert_emissions_mut(node)[..canonical_size].copy_from_slice(&insert_row);

        // Transition probabilities
        // Match transitions (MM, MI, MD) - 3 values that sum to 1
        let t_mat = sample_dirichlet(rng, 3);
        hmm.transitions_mut(node)[HTransition::MatchToMatch as usize] = t_mat[0];
        hmm.transitions_mut(node)[HTransition::MatchToInsert as usize] = t_mat[1];
        hmm.transitions_mut(node)[HTransition::MatchToDelete as usize] = t_mat[2];

        // Insert transitions (IM, II) - 2 values that sum to 1
        let t_ins = sample_dirichlet(rng, 2);
        hmm.transitions_mut(node)[HTransition::InsertToMatch as usize] = t_ins[0];
        hmm.transitions_mut(node)[HTransition::InsertToInsert as usize] = t_ins[1];

        // Delete transitions (DM, DD) - 2 values that sum to 1
        let t_del = sample_dirichlet(rng, 2);
        hmm.transitions_mut(node)[HTransition::DeleteToMatch as usize] = t_del[0];
        hmm.transitions_mut(node)[HTransition::DeleteToDelete as usize] = t_del[1];
    }

    // Enforce conventions on node 0 and node M
    // Node 0: no emissions from match state (entry node)
    hmm.match_emissions_mut(0).fill(0.0);
    hmm.match_emissions_mut(0)[0] = 1.0;

    // Node M: no transitions from M_M to I_M or D_M+1
    hmm.transitions_mut(model_length)[HTransition::MatchToInsert as usize] = 0.0;
    hmm.transitions_mut(model_length)[HTransition::MatchToDelete as usize] = 0.0;
    hmm.transitions_mut(model_length)[HTransition::MatchToMatch as usize] = 1.0;
    hmm.transitions_mut(model_length)[HTransition::DeleteToMatch as usize] = 1.0;
    hmm.transitions_mut(model_length)[HTransition::DeleteToDelete as usize] = 0.0;

    // Set name
    hmm.name = format!("sampled-hmm-{}", model_length);
    hmm.num_sequences = Some(0);
    hmm.effective_num_seq_float = Some(0.0);
    hmm.effective_num_seq = 0.0;

    hmm
}

/// Sample an "enumerable" HMM (no insert states) for sequence enumeration tests.
///
/// Corresponds to p7_hmm_SampleEnumerable() in C.
/// All insert transitions are 0, so the model can only generate
/// sequences of length 0..M.
pub fn hmm_sample_enumerable(
    rng: &mut XorShift64,
    model_length: usize,
    alphabet: &Alphabet,
) -> Hmm {
    let canonical_size = alphabet.canonical_size;
    let mut hmm = Hmm::new(model_length, alphabet);

    for node in 0..=model_length {
        // Match emissions: random
        let match_row = sample_dirichlet(rng, canonical_size);
        hmm.match_emissions_mut(node)[..canonical_size].copy_from_slice(&match_row);

        // Insert emissions: doesn't matter, but set uniform
        let uniform_probability = 1.0 / canonical_size as f32;
        for residue_index in 0..canonical_size {
            hmm.insert_emissions_mut(node)[residue_index] = uniform_probability;
        }

        // Transitions: no inserts
        // Match: MM + MD = 1, MI = 0
        let r = rng.random() as f32;
        hmm.transitions_mut(node)[HTransition::MatchToMatch as usize] = r;
        hmm.transitions_mut(node)[HTransition::MatchToInsert as usize] = 0.0;
        hmm.transitions_mut(node)[HTransition::MatchToDelete as usize] = 1.0 - r;

        // Insert transitions: doesn't matter
        hmm.transitions_mut(node)[HTransition::InsertToMatch as usize] = 1.0;
        hmm.transitions_mut(node)[HTransition::InsertToInsert as usize] = 0.0;

        // Delete: DM + DD = 1
        let r = rng.random() as f32;
        hmm.transitions_mut(node)[HTransition::DeleteToMatch as usize] = r;
        hmm.transitions_mut(node)[HTransition::DeleteToDelete as usize] = 1.0 - r;
    }

    // Node 0 conventions
    hmm.match_emissions_mut(0).fill(0.0);
    hmm.match_emissions_mut(0)[0] = 1.0;

    // Node M: end
    hmm.transitions_mut(model_length)[HTransition::MatchToMatch as usize] = 1.0;
    hmm.transitions_mut(model_length)[HTransition::MatchToInsert as usize] = 0.0;
    hmm.transitions_mut(model_length)[HTransition::MatchToDelete as usize] = 0.0;
    hmm.transitions_mut(model_length)[HTransition::DeleteToMatch as usize] = 1.0;
    hmm.transitions_mut(model_length)[HTransition::DeleteToDelete as usize] = 0.0;

    hmm.name = format!("enumerable-hmm-{}", model_length);
    hmm
}

/// Generate a random IID digital sequence of length L,
/// drawn from a frequency distribution f[0..K-1].
///
/// Corresponds to esl_rsq_xfIID() in C.
/// Returns dsq[0..L] with 0-based residue codes (no sentinels).
pub fn random_digital_seq(
    rng: &mut XorShift64,
    frequencies: &[f32],
    alphabet_size: usize,
    sequence_length: usize,
) -> Vec<Dsq> {
    let mut digitized_sequence = vec![0u8; sequence_length];

    // Build cumulative distribution
    let mut cumulative_distribution = vec![0.0f64; alphabet_size];
    cumulative_distribution[0] = frequencies[0] as f64;
    for index in 1..alphabet_size {
        cumulative_distribution[index] =
            cumulative_distribution[index - 1] + frequencies[index] as f64;
    }
    // Normalize
    let total_probability = cumulative_distribution[alphabet_size - 1];
    for cumulative_probability in &mut cumulative_distribution {
        *cumulative_probability /= total_probability;
    }

    for residue_code in &mut digitized_sequence {
        let random_value = rng.random();
        let mut residue_index = 0;
        while residue_index < alphabet_size - 1
            && random_value > cumulative_distribution[residue_index]
        {
            residue_index += 1;
        }
        *residue_code = residue_index as Dsq;
    }
    digitized_sequence
}

/// Sample from a Dirichlet(1,1,...,1) distribution, i.e. uniform on the simplex.
///
/// Uses the standard algorithm: sample K exponential(1) variables,
/// then normalize.
fn sample_dirichlet(rng: &mut XorShift64, category_count: usize) -> Vec<f32> {
    let mut sampled_values = Vec::with_capacity(category_count);
    for _ in 0..category_count {
        // Sample from exponential(1): -log(U)
        let uniform_value = rng.random();
        let exponential_sample = if uniform_value > 0.0 {
            -(uniform_value.ln())
        } else {
            20.0
        };
        sampled_values.push(exponential_sample as f32);
    }
    let total: f32 = sampled_values.iter().sum();
    if total > 0.0 {
        for value in &mut sampled_values {
            *value /= total;
        }
    }
    sampled_values
}

/// Validate that a probability vector sums to approximately 1.0.
pub fn vec_f_validate(v: &[f32], tol: f32) -> bool {
    let sum: f32 = v.iter().sum();
    (sum - 1.0).abs() <= tol
}

/// Compare two float vectors within tolerance.
pub fn vec_f_compare(v1: &[f32], v2: &[f32], tol: f32) -> bool {
    if v1.len() != v2.len() {
        return false;
    }
    v1.iter().zip(v2.iter()).all(|(a, b)| (a - b).abs() <= tol)
}

/// Validate that an HMM's probability distributions are properly normalized.
pub fn hmm_validate(hmm: &Hmm, tol: f32) -> bool {
    let canonical_size = hmm.alphabet.canonical_size;

    for node in 1..=hmm.num_nodes {
        // Match emissions should sum to ~1
        let match_sum: f32 = hmm.match_emissions(node)[..canonical_size].iter().sum();
        if (match_sum - 1.0).abs() > tol {
            return false;
        }
        // Insert emissions should sum to ~1
        let insert_sum: f32 = hmm.insert_emissions(node)[..canonical_size].iter().sum();
        if (insert_sum - 1.0).abs() > tol {
            return false;
        }
        // Match transitions (MM+MI+MD) should sum to ~1
        let t_mat = hmm.transitions(node)[HTransition::MatchToMatch as usize]
            + hmm.transitions(node)[HTransition::MatchToInsert as usize]
            + hmm.transitions(node)[HTransition::MatchToDelete as usize];
        if (t_mat - 1.0).abs() > tol {
            return false;
        }
        // Insert transitions (IM+II) should sum to ~1
        let t_ins = hmm.transitions(node)[HTransition::InsertToMatch as usize]
            + hmm.transitions(node)[HTransition::InsertToInsert as usize];
        if (t_ins - 1.0).abs() > tol {
            return false;
        }
        // Delete transitions (DM+DD) should sum to ~1
        let t_del = hmm.transitions(node)[HTransition::DeleteToMatch as usize]
            + hmm.transitions(node)[HTransition::DeleteToDelete as usize];
        if (t_del - 1.0).abs() > tol {
            return false;
        }
    }
    true
}

/// Set the composition of an HMM from its match emission probabilities.
///
/// Corresponds to p7_hmm_SetComposition() in C.
/// `hmm->compo[a]` = weighted average of match emissions across all positions.
#[allow(clippy::needless_range_loop)]
pub fn hmm_set_composition(hmm: &mut Hmm) {
    let k = hmm.alphabet.canonical_size;
    let (mocc, _) = hmm.calculate_occupancy();

    let mut sum_occ = 0.0f64;
    for &occ in &mocc[1..=hmm.num_nodes] {
        sum_occ += occ as f64;
    }

    let mut compo = vec![0.0f32; k.min(MAX_CANONICAL_ALPHABET)];
    for a in 0..k.min(MAX_CANONICAL_ALPHABET) {
        let mut weighted = 0.0f64;
        for node in 1..=hmm.num_nodes {
            weighted += mocc[node] as f64 * hmm.match_emissions(node)[a] as f64;
        }
        compo[a] = if sum_occ > 0.0 {
            (weighted / sum_occ) as f32
        } else {
            1.0 / k as f32
        };
    }
    hmm.model_composition = Some(compo);
}

/// Create a background model, configure a profile, and return all three.
///
/// Utility for tests that need a configured (hmm, bg, profile) triple.
pub fn setup_profile(
    rng: &mut XorShift64,
    model_length: usize,
    target_length: usize,
    alphabet: &Alphabet,
    mode: crate::hmmer_core::config::SearchMode,
) -> (Hmm, BackgroundModel, Profile) {
    let hmm = hmm_sample(rng, model_length, alphabet);
    let mut background = BackgroundModel::new(alphabet);
    background.set_length(target_length);
    let mut profile = Profile::new(hmm.num_nodes, alphabet);
    crate::hmmer_core::modelconfig::profile_config(
        &hmm,
        &background,
        &mut profile,
        target_length,
        mode,
    );
    (hmm, background, profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmmer_core::alphabet::Alphabet;

    #[test]
    fn test_hmm_sample() {
        let amino_alphabet = Alphabet::amino();
        let mut rng = XorShift64::new(42);
        let hmm = hmm_sample(&mut rng, 100, &amino_alphabet);

        assert_eq!(hmm.num_nodes, 100);
        assert!(hmm_validate(&hmm, 0.001));
    }

    #[test]
    fn test_hmm_sample_enumerable() {
        let amino_alphabet = Alphabet::amino();
        let mut rng = XorShift64::new(42);
        let hmm = hmm_sample_enumerable(&mut rng, 10, &amino_alphabet);

        assert_eq!(hmm.num_nodes, 10);
        // All insert transitions should be 0
        for node in 0..=10 {
            assert_eq!(
                hmm.transitions(node)[HTransition::MatchToInsert as usize],
                0.0
            );
            assert_eq!(
                hmm.transitions(node)[HTransition::InsertToInsert as usize],
                0.0
            );
        }
        assert!(hmm_validate(&hmm, 0.001));
    }

    #[test]
    fn test_random_digital_seq() {
        let amino_alphabet = Alphabet::amino();
        let background = BackgroundModel::new(&amino_alphabet);
        let mut rng = XorShift64::new(42);

        let digitized_sequence = random_digital_seq(
            &mut rng,
            &background.residue_frequencies,
            amino_alphabet.canonical_size,
            200,
        );
        assert_eq!(digitized_sequence.len(), 200);

        // All residues should be in range [0, K)
        for &residue_code in &digitized_sequence {
            assert!((residue_code as usize) < amino_alphabet.canonical_size);
        }
    }

    #[test]
    fn test_sample_dirichlet() {
        let mut rng = XorShift64::new(42);
        let v = sample_dirichlet(&mut rng, 20);
        assert_eq!(v.len(), 20);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        // All values should be positive
        assert!(v.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_vec_f_compare() {
        let v1 = vec![0.1, 0.2, 0.3, 0.4];
        let v2 = vec![0.1, 0.2, 0.3, 0.4];
        assert!(vec_f_compare(&v1, &v2, 0.001));

        let v3 = vec![0.11, 0.21, 0.31, 0.41];
        assert!(!vec_f_compare(&v1, &v3, 0.001));
        assert!(vec_f_compare(&v1, &v3, 0.02));
    }
}
