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
    let log_ld = (ld as f32).ln();

    // Sum state usage across rows 1..=ld into local accumulators.
    let mut match_sums = vec![0.0f32; m + 1];
    let mut insert_sums = vec![0.0f32; m + 1];
    let mut n_sum = 0.0f32;
    let mut j_sum = 0.0f32;
    let mut c_sum = 0.0f32;

    for i in 1..=ld {
        for k in 0..=m {
            match_sums[k] += posterior_matrix.main[[i, k, MATCH_CELL]];
            insert_sums[k] += posterior_matrix.main[[i, k, INSERT_CELL]];
        }
        n_sum += posterior_matrix.special[[i, N_STATE]];
        j_sum += posterior_matrix.special[[i, J_STATE]];
        c_sum += posterior_matrix.special[[i, C_STATE]];
    }

    // Convert summed counts to frequencies (exp of log-frequency)
    let mmx_freq: Vec<f32> = match_sums
        .iter()
        .map(|&v| {
            if v > 0.0 {
                (v.ln() - log_ld).exp()
            } else {
                0.0
            }
        })
        .collect();
    let imx_freq: Vec<f32> = insert_sums
        .iter()
        .map(|&v| {
            if v > 0.0 {
                (v.ln() - log_ld).exp()
            } else {
                0.0
            }
        })
        .collect();

    // xfactor in probability space
    let xfactor_prob = (if n_sum > 0.0 {
        (n_sum.ln() - log_ld).exp()
    } else {
        0.0
    }) + (if c_sum > 0.0 {
        (c_sum.ln() - log_ld).exp()
    } else {
        0.0
    }) + (if j_sum > 0.0 {
        (j_sum.ln() - log_ld).exp()
    } else {
        0.0
    });

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
            sum += mmx_freq[k]
                * residue_scores[k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize].exp();
            sum += imx_freq[k]
                * residue_scores[k * PROFILE_NUM_EMISSIONS + PRsc::InsertScore as usize].exp();
        }
        sum += mmx_freq[m]
            * residue_scores[m * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize].exp();
        sum += xfactor_prob;
        *null2_x = sum;
    }

    null2
}
