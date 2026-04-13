// test_helpers.rs - Testing utilities for HMMER Rust port
//
// Provides random HMM sampling, random sequence generation, and other
// utilities needed by the ported HMMER test suite. Corresponds to
// p7_hmm_Sample(), esl_rsq_xfIID(), and related C functions.

use crate::alphabet::{Alphabet, Dsq};
use crate::background::BackgroundModel;
use crate::config::*;
use crate::hmm::Hmm;
use crate::profile::Profile;
use crate::rng::XorShift64;

/// Sample a random HMM of length M for a given alphabet.
///
/// Corresponds to p7_hmm_Sample() in C.
/// Generates random transition and emission probabilities using
/// Dirichlet sampling, then normalizes.
pub fn hmm_sample(rng: &mut XorShift64, m: usize, abc: &Alphabet) -> Hmm {
    let k = abc.canonical_size;
    let mut hmm = Hmm::new(m, abc);

    // Sample emission probabilities for each node
    for node in 0..=m {
        // Match emissions: sample from Dirichlet(1,1,...,1) = uniform on simplex
        let mat_row = sample_dirichlet(rng, k);
        hmm.match_emissions_mut(node)[..k].copy_from_slice(&mat_row[..k]);

        // Insert emissions: sample from Dirichlet
        let ins_row = sample_dirichlet(rng, k);
        hmm.insert_emissions_mut(node)[..k].copy_from_slice(&ins_row[..k]);

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
    hmm.transitions_mut(m)[HTransition::MatchToInsert as usize] = 0.0;
    hmm.transitions_mut(m)[HTransition::MatchToDelete as usize] = 0.0;
    hmm.transitions_mut(m)[HTransition::MatchToMatch as usize] = 1.0;
    hmm.transitions_mut(m)[HTransition::DeleteToMatch as usize] = 1.0;
    hmm.transitions_mut(m)[HTransition::DeleteToDelete as usize] = 0.0;

    // Set name
    hmm.name = format!("sampled-hmm-{}", m);
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
pub fn hmm_sample_enumerable(rng: &mut XorShift64, m: usize, abc: &Alphabet) -> Hmm {
    let k = abc.canonical_size;
    let mut hmm = Hmm::new(m, abc);

    for node in 0..=m {
        // Match emissions: random
        let mat_row = sample_dirichlet(rng, k);
        hmm.match_emissions_mut(node)[..k].copy_from_slice(&mat_row[..k]);

        // Insert emissions: doesn't matter, but set uniform
        let uniform = 1.0 / k as f32;
        for a in 0..k {
            hmm.insert_emissions_mut(node)[a] = uniform;
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
    hmm.transitions_mut(m)[HTransition::MatchToMatch as usize] = 1.0;
    hmm.transitions_mut(m)[HTransition::MatchToInsert as usize] = 0.0;
    hmm.transitions_mut(m)[HTransition::MatchToDelete as usize] = 0.0;
    hmm.transitions_mut(m)[HTransition::DeleteToMatch as usize] = 1.0;
    hmm.transitions_mut(m)[HTransition::DeleteToDelete as usize] = 0.0;

    hmm.name = format!("enumerable-hmm-{}", m);
    hmm
}

/// Generate a random IID digital sequence of length L,
/// drawn from a frequency distribution f[0..K-1].
///
/// Corresponds to esl_rsq_xfIID() in C.
/// Returns dsq[0..L] with 0-based residue codes (no sentinels).
pub fn random_digital_seq(rng: &mut XorShift64, f: &[f32], k: usize, l: usize) -> Vec<Dsq> {
    let mut dsq = vec![0u8; l];

    // Build cumulative distribution
    let mut cdf = vec![0.0f64; k];
    cdf[0] = f[0] as f64;
    for i in 1..k {
        cdf[i] = cdf[i - 1] + f[i] as f64;
    }
    // Normalize
    let total = cdf[k - 1];
    for c in cdf.iter_mut() {
        *c /= total;
    }

    for dsq_val in dsq.iter_mut() {
        let r = rng.random();
        let mut a = 0;
        while a < k - 1 && r > cdf[a] {
            a += 1;
        }
        *dsq_val = a as Dsq;
    }
    dsq
}

/// Sample from a Dirichlet(1,1,...,1) distribution, i.e. uniform on the simplex.
///
/// Uses the standard algorithm: sample K exponential(1) variables,
/// then normalize.
fn sample_dirichlet(rng: &mut XorShift64, k: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(k);
    for _ in 0..k {
        // Sample from exponential(1): -log(U)
        let u = rng.random();
        let e = if u > 0.0 { -(u.ln()) } else { 20.0 };
        v.push(e as f32);
    }
    let sum: f32 = v.iter().sum();
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
    v
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
    let k = hmm.alphabet.canonical_size;

    for node in 1..=hmm.num_nodes {
        // Match emissions should sum to ~1
        let mat_sum: f32 = hmm.match_emissions(node)[..k].iter().sum();
        if (mat_sum - 1.0).abs() > tol {
            return false;
        }
        // Insert emissions should sum to ~1
        let ins_sum: f32 = hmm.insert_emissions(node)[..k].iter().sum();
        if (ins_sum - 1.0).abs() > tol {
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
/// hmm->compo[a] = weighted average of match emissions across all positions.
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
    m: usize,
    l: usize,
    abc: &Alphabet,
    mode: crate::config::SearchMode,
) -> (Hmm, BackgroundModel, Profile) {
    let hmm = hmm_sample(rng, m, abc);
    let mut bg = BackgroundModel::new(abc);
    bg.set_length(l);
    let mut gm = Profile::new(hmm.num_nodes, abc);
    crate::modelconfig::profile_config(&hmm, &bg, &mut gm, l, mode);
    (hmm, bg, gm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;

    #[test]
    fn test_hmm_sample() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(42);
        let hmm = hmm_sample(&mut rng, 100, &abc);

        assert_eq!(hmm.num_nodes, 100);
        assert!(hmm_validate(&hmm, 0.001));
    }

    #[test]
    fn test_hmm_sample_enumerable() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(42);
        let hmm = hmm_sample_enumerable(&mut rng, 10, &abc);

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
        let abc = Alphabet::amino();
        let bg = BackgroundModel::new(&abc);
        let mut rng = XorShift64::new(42);

        let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, 200);
        assert_eq!(dsq.len(), 200);

        // All residues should be in range [0, K)
        for &d in &dsq {
            assert!((d as usize) < abc.canonical_size);
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
