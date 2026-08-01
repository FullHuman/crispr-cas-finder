// generic_null2.rs - Null2 score calculation (generic)
//
// Port of src/generic_null2.c

use crate::config::*;
use crate::profile::Profile;
use crate::score_matrix::{C_STATE, INSERT_CELL, J_STATE, MATCH_CELL, N_STATE, ScoreMatrix};

/// Calculate null2 scores from posterior probabilities (expectation method).
///
/// Given a posterior decoding matrix `posterior_matrix` over envelope positions 1..Ld,
/// compute and return null2 odds ratios f'(x)/f(x) for each residue x.
pub fn null2_by_expectation(profile: &Profile, posterior_matrix: &ScoreMatrix) -> Vec<f32> {
    let m = profile.num_nodes;
    let ld = posterior_matrix.sequence_length;
    if ld == 0 {
        return vec![1.0; profile.alphabet.full_size];
    }
    let log_ld = (ld as f32).ln();

    // Sum state usage across rows 1..=ld into local accumulators.
    // Keep match and insert accumulators together: both are consumed as a
    // pair, and this requires only one allocation.
    let mut state_sums = vec![[0.0f32; 2]; m + 1];
    let mut n_sum = 0.0f32;
    let mut j_sum = 0.0f32;
    let mut c_sum = 0.0f32;

    for i in 1..=ld {
        // M_0 and I_0 are unused, as is I_M. Avoid reading and
        // accumulating posterior cells that cannot affect the result.
        for (k, sums) in state_sums.iter_mut().enumerate().take(m).skip(1) {
            sums[0] += posterior_matrix.main[[i, k, MATCH_CELL]];
            sums[1] += posterior_matrix.main[[i, k, INSERT_CELL]];
        }
        state_sums[m][0] += posterior_matrix.main[[i, m, MATCH_CELL]];

        n_sum += posterior_matrix.special[[i, N_STATE]];
        j_sum += posterior_matrix.special[[i, J_STATE]];
        c_sum += posterior_matrix.special[[i, C_STATE]];
    }

    // Preserve the original floating-point transformation exactly. Replacing
    // this with multiplication by 1/Ld changes rounded results.
    let normalize = |sum: f32| {
        if sum > 0.0 {
            (sum.ln() - log_ld).exp()
        } else {
            0.0
        }
    };
    for sums in state_sums.iter_mut().take(m).skip(1) {
        sums[0] = normalize(sums[0]);
        sums[1] = normalize(sums[1]);
    }
    state_sums[m][0] = normalize(state_sums[m][0]);

    let xfactor_prob = normalize(n_sum) + normalize(c_sum) + normalize(j_sum);

    // Calculate null2 odds ratios directly in probability space.
    // This avoids the sequential flogsum dependency chain, enabling
    // the compiler to vectorize the inner loop.
    let k_canonical = profile.alphabet.canonical_size;
    let full_size = profile.alphabet.full_size;
    let mut null2 = vec![1.0f32; full_size];

    for (x, null2_x) in null2[..k_canonical].iter_mut().enumerate() {
        let residue_scores = profile.residue_scores_for(x);
        let mut sum = 0.0f32;
        for k in 1..m {
            sum += state_sums[k][0]
                * residue_scores[k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize].exp();
            sum += state_sums[k][1]
                * residue_scores[k * PROFILE_NUM_EMISSIONS + PRsc::InsertScore as usize].exp();
        }
        sum += state_sums[m][0]
            * residue_scores[m * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize].exp();
        sum += xfactor_prob;
        *null2_x = sum;
    }

    set_degenerate_odds(&profile.alphabet, &mut null2);

    null2
}

/// Equivalent to Easel's `esl_abc_FAvgScVec()`: ambiguous symbols get the
/// arithmetic mean of the canonical odds they represent. Gap, nonresidue,
/// and missing-data symbols are neutral under null2.
fn set_degenerate_odds(alphabet: &crate::alphabet::Alphabet, odds: &mut [f32]) {
    let k = alphabet.canonical_size;
    let kp = alphabet.full_size;
    for x in (k + 1)..kp.saturating_sub(2) {
        let row = alphabet.degen_row(x);
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for (a, &included) in row.iter().enumerate() {
            if included {
                sum += odds[a];
                count += 1;
            }
        }
        odds[x] = if count > 0 { sum / count as f32 } else { 1.0 };
    }
    odds[k] = 1.0;
    odds[kp - 2] = 1.0;
    odds[kp - 1] = 1.0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;

    #[test]
    fn degenerate_odds_match_hmmer_averaging() {
        let alphabet = Alphabet::amino();
        let mut odds = vec![1.0; alphabet.full_size];
        odds[2] = 2.0; // D
        odds[11] = 4.0; // N
        odds[7] = 6.0; // I
        odds[9] = 10.0; // L
        set_degenerate_odds(&alphabet, &mut odds);

        assert_eq!(odds[21], 3.0); // B = mean(D, N)
        assert_eq!(odds[22], 8.0); // J = mean(I, L)
        assert_eq!(odds[alphabet.canonical_size], 1.0); // gap
        assert_eq!(odds[alphabet.full_size - 2], 1.0); // nonresidue
        assert_eq!(odds[alphabet.full_size - 1], 1.0); // missing
    }
}
