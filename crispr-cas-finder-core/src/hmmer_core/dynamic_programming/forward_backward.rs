use ndarray::Array2;

use super::simd::fwd_bck;
use super::simd::oprofile::OptimizedProfile;
use crate::hmmer_core::config::{PROFILE_NUM_EMISSIONS, PROFILE_NUM_TRANSITIONS, PRsc, PTsc};
use crate::hmmer_core::errors::HmmerError;
use crate::hmmer_core::profile::Profile;
use crate::hmmer_core::score_matrix::*;

#[inline]
fn add_logs(a: f32, b: f32) -> f32 {
    super::logsum::flogsum(a, b)
}

#[inline]
fn transition(tsc: &[f32], source_node: usize, kind: PTsc) -> f32 {
    tsc[source_node * PROFILE_NUM_TRANSITIONS + kind.idx()]
}

#[inline]
fn emission(profile: &Profile, residue: usize, node: usize, kind: PRsc) -> f32 {
    profile.residue_scores_for(residue)[node * PROFILE_NUM_EMISSIONS + kind.idx()]
}

// ─── Forward checkpoints ────────────────────────────────────────────────────

/// Checkpoint data saved during the forward pass.
///
/// Stores main-state rows at intervals of `ceil(sqrt(L))` plus special-state
/// scores for *every* row (only 5 floats each). This reduces memory from
/// O(LM) to O(M√L) while allowing on-demand forward row reconstruction
/// during the backward/decode pass.
pub struct ForwardCheckpoints {
    /// Checkpointed main-state rows, each shape (num_nodes+1, 3).
    /// Stored in order of ascending row index.
    pub checkpoint_main: Vec<Array2<f32>>,
    /// Row indices corresponding to each entry in `checkpoint_main`.
    pub checkpoint_indices: Vec<usize>,
    /// Special-state scores for every row 0..=L, shape (L+1, 5).
    pub all_specials: Array2<f32>,
    /// Overall forward score = fwd_special(L, C) + c_move.
    pub overall_score: f32,
    /// Sequence length L.
    pub sequence_length: usize,
    /// Number of model nodes M.
    pub num_nodes: usize,
}

impl ForwardCheckpoints {
    /// Get the special-state score at row `i`, state `s`.
    #[inline]
    pub fn special_score(&self, i: usize, s: usize) -> f32 {
        self.all_specials[[i, s]]
    }

    /// Find the largest checkpoint index at or before `row`.
    ///
    /// Returns the position in `checkpoint_main` and its row index, or `None`
    /// when the checkpoint list is empty or every checkpoint follows `row`.
    pub fn checkpoint_at_or_before(&self, row: usize) -> Option<(usize, usize)> {
        // Binary search: find rightmost index where checkpoint_indices[idx] <= row
        let pos = self.checkpoint_indices.partition_point(|&r| r <= row);
        let idx = pos.checked_sub(1)?;
        Some((idx, self.checkpoint_indices[idx]))
    }
}

// ─── Fused backward + decoding ──────────────────────────────────────────────

use crate::hmmer_core::domaindef::DomainDef;

fn generic_forward_backward_decode(
    digital_sequence: &[u8],
    profile: &Profile,
    expected_forward_score: f32,
    posterior: &mut ScoreMatrix,
    domain_def: Option<&mut DomainDef>,
) -> Result<f32, HmmerError> {
    let m = profile.num_nodes;
    let l = digital_sequence.len();
    if m == 0 || l == 0 {
        return Err(HmmerError::Internal(
            "Forward/Backward requires a non-empty model and target".into(),
        ));
    }

    posterior.resize(m, l)?;
    let neg_inf = f32::NEG_INFINITY;
    let tsc = profile.transition_scores_raw();
    let esc = if profile.mode.is_local() {
        0.0
    } else {
        neg_inf
    };
    let special = &profile.special_scores;

    // Generic Forward, matching p7_GForward(). The output matrix is used as
    // temporary Forward storage and is overwritten row-by-row by posteriors.
    posterior.fill_main_row(0, neg_inf);
    posterior.set_special_row(0, [neg_inf, 0.0, neg_inf, special.n_move, neg_inf]);

    for i in 1..=l {
        let x = digital_sequence[i - 1] as usize;
        if x >= profile.alphabet.full_size {
            return Err(HmmerError::Internal(format!(
                "digital residue code {x} is outside the profile alphabet"
            )));
        }
        posterior.fill_main_row(i, neg_inf);
        let mut x_e = neg_inf;

        for k in 1..m {
            let from_mi = add_logs(
                posterior.main[[i - 1, k - 1, MATCH_CELL]]
                    + transition(tsc, k - 1, PTsc::MatchToMatch),
                posterior.main[[i - 1, k - 1, INSERT_CELL]]
                    + transition(tsc, k - 1, PTsc::InsertToMatch),
            );
            let from_bd = add_logs(
                posterior.special[[i - 1, BEGIN_STATE]]
                    + transition(tsc, k - 1, PTsc::BeginToMatch),
                posterior.main[[i - 1, k - 1, DELETE_CELL]]
                    + transition(tsc, k - 1, PTsc::DeleteToMatch),
            );
            let mm = add_logs(from_mi, from_bd) + emission(profile, x, k, PRsc::MatchScore);
            posterior.main[[i, k, MATCH_CELL]] = mm;

            let im = add_logs(
                posterior.main[[i - 1, k, MATCH_CELL]] + transition(tsc, k, PTsc::MatchToInsert),
                posterior.main[[i - 1, k, INSERT_CELL]] + transition(tsc, k, PTsc::InsertToInsert),
            ) + emission(profile, x, k, PRsc::InsertScore);
            posterior.main[[i, k, INSERT_CELL]] = im;

            let dm = add_logs(
                posterior.main[[i, k - 1, MATCH_CELL]]
                    + transition(tsc, k - 1, PTsc::MatchToDelete),
                posterior.main[[i, k - 1, DELETE_CELL]]
                    + transition(tsc, k - 1, PTsc::DeleteToDelete),
            );
            posterior.main[[i, k, DELETE_CELL]] = dm;
            x_e = add_logs(x_e, add_logs(mm + esc, dm + esc));
        }

        let k = m;
        let from_mi = add_logs(
            posterior.main[[i - 1, k - 1, MATCH_CELL]] + transition(tsc, k - 1, PTsc::MatchToMatch),
            posterior.main[[i - 1, k - 1, INSERT_CELL]]
                + transition(tsc, k - 1, PTsc::InsertToMatch),
        );
        let from_bd = add_logs(
            posterior.special[[i - 1, BEGIN_STATE]] + transition(tsc, k - 1, PTsc::BeginToMatch),
            posterior.main[[i - 1, k - 1, DELETE_CELL]]
                + transition(tsc, k - 1, PTsc::DeleteToMatch),
        );
        posterior.main[[i, k, MATCH_CELL]] =
            add_logs(from_mi, from_bd) + emission(profile, x, k, PRsc::MatchScore);
        posterior.main[[i, k, INSERT_CELL]] = neg_inf;
        posterior.main[[i, k, DELETE_CELL]] = add_logs(
            posterior.main[[i, k - 1, MATCH_CELL]] + transition(tsc, k - 1, PTsc::MatchToDelete),
            posterior.main[[i, k - 1, DELETE_CELL]] + transition(tsc, k - 1, PTsc::DeleteToDelete),
        );
        x_e = add_logs(
            x_e,
            add_logs(
                posterior.main[[i, k, MATCH_CELL]],
                posterior.main[[i, k, DELETE_CELL]],
            ),
        );

        let x_j = add_logs(
            posterior.special[[i - 1, J_STATE]] + special.j_loop,
            x_e + special.e_loop,
        );
        let x_c = add_logs(
            posterior.special[[i - 1, C_STATE]] + special.c_loop,
            x_e + special.e_move,
        );
        let x_n = posterior.special[[i - 1, N_STATE]] + special.n_loop;
        let x_b = add_logs(x_n + special.n_move, x_j + special.j_move);
        posterior.set_special_row(i, [x_e, x_n, x_j, x_b, x_c]);
    }

    let forward_score = posterior.special[[l, C_STATE]] + special.c_move;
    if !forward_score.is_finite()
        || !expected_forward_score.is_finite()
        || (forward_score - expected_forward_score).abs() > 0.1
    {
        return Err(HmmerError::Internal(format!(
            "Forward implementations disagree: generic={forward_score}, SIMD={}",
            expected_forward_score
        )));
    }

    let forward_specials: Vec<[f32; NUM_SPECIAL_STATES]> = (0..=l)
        .map(|i| {
            let row = posterior.special_row(i);
            [row[0], row[1], row[2], row[3], row[4]]
        })
        .collect();
    let mut backward_specials = vec![[neg_inf; NUM_SPECIAL_STATES]; l + 1];
    let mut b_next = vec![[neg_inf; NUM_MAIN_STATES]; m + 1];
    let mut b_cur = vec![[neg_inf; NUM_MAIN_STATES]; m + 1];

    backward_specials[l][C_STATE] = special.c_move;
    backward_specials[l][EXIT_STATE] = special.c_move + special.e_move;
    b_next[m][MATCH_CELL] = backward_specials[l][EXIT_STATE];
    b_next[m][DELETE_CELL] = backward_specials[l][EXIT_STATE];
    for k in (1..m).rev() {
        b_next[k][MATCH_CELL] = add_logs(
            backward_specials[l][EXIT_STATE] + esc,
            b_next[k + 1][DELETE_CELL] + transition(tsc, k, PTsc::MatchToDelete),
        );
        b_next[k][DELETE_CELL] = add_logs(
            backward_specials[l][EXIT_STATE] + esc,
            b_next[k + 1][DELETE_CELL] + transition(tsc, k, PTsc::DeleteToDelete),
        );
    }
    decode_posterior_row(
        posterior,
        &forward_specials,
        &backward_specials,
        &b_next,
        forward_score,
        l,
        m,
        special,
    );

    for i in (1..l).rev() {
        let x_next = digital_sequence[i] as usize;
        b_cur.fill([neg_inf; NUM_MAIN_STATES]);

        let mut x_b = neg_inf;
        for (k, backward) in b_next.iter().enumerate().take(m + 1).skip(1) {
            x_b = add_logs(
                x_b,
                backward[MATCH_CELL]
                    + transition(tsc, k - 1, PTsc::BeginToMatch)
                    + emission(profile, x_next, k, PRsc::MatchScore),
            );
        }
        let x_j = add_logs(
            backward_specials[i + 1][J_STATE] + special.j_loop,
            x_b + special.j_move,
        );
        let x_c = backward_specials[i + 1][C_STATE] + special.c_loop;
        let x_e = add_logs(x_j + special.e_loop, x_c + special.e_move);
        let x_n = add_logs(
            backward_specials[i + 1][N_STATE] + special.n_loop,
            x_b + special.n_move,
        );
        backward_specials[i] = [x_e, x_n, x_j, x_b, x_c];

        b_cur[m][MATCH_CELL] = x_e;
        b_cur[m][DELETE_CELL] = x_e;
        for k in (1..m).rev() {
            let to_match =
                b_next[k + 1][MATCH_CELL] + emission(profile, x_next, k + 1, PRsc::MatchScore);
            let to_insert =
                b_next[k][INSERT_CELL] + emission(profile, x_next, k, PRsc::InsertScore);
            b_cur[k][MATCH_CELL] = add_logs(
                add_logs(
                    to_match + transition(tsc, k, PTsc::MatchToMatch),
                    to_insert + transition(tsc, k, PTsc::MatchToInsert),
                ),
                add_logs(
                    x_e + esc,
                    b_cur[k + 1][DELETE_CELL] + transition(tsc, k, PTsc::MatchToDelete),
                ),
            );
            b_cur[k][INSERT_CELL] = add_logs(
                to_match + transition(tsc, k, PTsc::InsertToMatch),
                to_insert + transition(tsc, k, PTsc::InsertToInsert),
            );
            b_cur[k][DELETE_CELL] = add_logs(
                to_match + transition(tsc, k, PTsc::DeleteToMatch),
                add_logs(
                    b_cur[k + 1][DELETE_CELL] + transition(tsc, k, PTsc::DeleteToDelete),
                    x_e + esc,
                ),
            );
        }

        decode_posterior_row(
            posterior,
            &forward_specials,
            &backward_specials,
            &b_cur,
            forward_score,
            i,
            m,
            special,
        );
        std::mem::swap(&mut b_cur, &mut b_next);
    }

    let x_first = digital_sequence[0] as usize;
    let mut b_zero = neg_inf;
    for (k, backward) in b_next.iter().enumerate().take(m + 1).skip(1) {
        b_zero = add_logs(
            b_zero,
            backward[MATCH_CELL]
                + transition(tsc, k - 1, PTsc::BeginToMatch)
                + emission(profile, x_first, k, PRsc::MatchScore),
        );
    }
    let backward_score = add_logs(
        backward_specials[1][N_STATE] + special.n_loop,
        b_zero + special.n_move,
    );
    backward_specials[0][BEGIN_STATE] = b_zero;
    backward_specials[0][N_STATE] = backward_score;

    if !backward_score.is_finite() || (backward_score - forward_score).abs() > 0.02 {
        return Err(HmmerError::Internal(format!(
            "Forward/Backward scores disagree: forward={forward_score}, backward={backward_score}"
        )));
    }

    if let Some(domain_def) = domain_def {
        domain_def.grow_to(l);
        domain_def.begin_totals[0] = 0.0;
        domain_def.exit_totals[0] = 0.0;
        domain_def.model_occupancy[0] = 0.0;
        for i in 1..=l {
            domain_def.begin_totals[i] = domain_def.begin_totals[i - 1]
                + (forward_specials[i - 1][BEGIN_STATE] + backward_specials[i - 1][BEGIN_STATE]
                    - forward_score)
                    .exp();
            domain_def.exit_totals[i] = domain_def.exit_totals[i - 1]
                + (forward_specials[i][EXIT_STATE] + backward_specials[i][EXIT_STATE]
                    - forward_score)
                    .exp();
            let outside =
                (forward_specials[i - 1][N_STATE] + special.n_loop + backward_specials[i][N_STATE]
                    - forward_score)
                    .exp()
                    + (forward_specials[i - 1][J_STATE]
                        + special.j_loop
                        + backward_specials[i][J_STATE]
                        - forward_score)
                        .exp()
                    + (forward_specials[i - 1][C_STATE]
                        + special.c_loop
                        + backward_specials[i][C_STATE]
                        - forward_score)
                        .exp();
            domain_def.model_occupancy[i] = 1.0 - outside;
        }
        domain_def.length = l;
    }

    Ok(backward_score)
}

#[allow(clippy::too_many_arguments)]
fn decode_posterior_row(
    posterior: &mut ScoreMatrix,
    forward_specials: &[[f32; NUM_SPECIAL_STATES]],
    backward_specials: &[[f32; NUM_SPECIAL_STATES]],
    backward_main: &[[f32; NUM_MAIN_STATES]],
    overall_score: f32,
    i: usize,
    m: usize,
    special: &crate::hmmer_core::profile::SpecialScores,
) {
    for (k, backward) in backward_main.iter().enumerate().take(m + 1).skip(1) {
        posterior.main[[i, k, MATCH_CELL]] =
            (posterior.main[[i, k, MATCH_CELL]] + backward[MATCH_CELL] - overall_score).exp();
        posterior.main[[i, k, INSERT_CELL]] =
            (posterior.main[[i, k, INSERT_CELL]] + backward[INSERT_CELL] - overall_score).exp();
        posterior.main[[i, k, DELETE_CELL]] = 0.0;
    }
    let n = (forward_specials[i - 1][N_STATE] + special.n_loop + backward_specials[i][N_STATE]
        - overall_score)
        .exp();
    let j = (forward_specials[i - 1][J_STATE] + special.j_loop + backward_specials[i][J_STATE]
        - overall_score)
        .exp();
    let c = (forward_specials[i - 1][C_STATE] + special.c_loop + backward_specials[i][C_STATE]
        - overall_score)
        .exp();
    posterior.set_special_row(i, [0.0, n, j, 0.0, c]);
}

/// Log-space Backward algorithm and posterior decoding.
///
/// A generic full Forward matrix is reconstructed here and checked against an
/// independently computed SIMD Forward score before posterior decoding.
pub fn backward_decode_log_space(
    digital_sequence: &[u8],
    profile: &Profile,
    expected_forward_score: f32,
    posterior_matrix: &mut ScoreMatrix,
    domain_def: Option<&mut DomainDef>,
) -> Result<f32, HmmerError> {
    generic_forward_backward_decode(
        digital_sequence,
        profile,
        expected_forward_score,
        posterior_matrix,
        domain_def,
    )
}

/// Compatibility wrapper for the former checkpoint-replay Backward API.
///
/// The current decoder is log-space and only needs the checkpointed Forward
/// score. New callers should use [`backward_decode_log_space`].
#[allow(clippy::too_many_arguments)]
pub fn backward_decode_prob_space(
    digital_sequence: &[u8],
    profile: &Profile,
    _om: &OptimizedProfile,
    checkpoints: &ForwardCheckpoints,
    _simd_data: &fwd_bck::SIMDCheckpointData,
    posterior_matrix: &mut ScoreMatrix,
    domain_def: Option<&mut DomainDef>,
    _seg_buf: &mut fwd_bck::OddsSegmentBuf,
) -> Result<f32, HmmerError> {
    if digital_sequence.len() < checkpoints.sequence_length {
        return Err(HmmerError::Internal(format!(
            "checkpoint sequence length {} exceeds target length {}",
            checkpoints.sequence_length,
            digital_sequence.len()
        )));
    }
    backward_decode_log_space(
        &digital_sequence[..checkpoints.sequence_length],
        profile,
        checkpoints.overall_score,
        posterior_matrix,
        domain_def,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoints(indices: Vec<usize>) -> ForwardCheckpoints {
        ForwardCheckpoints {
            checkpoint_main: indices
                .iter()
                .map(|_| Array2::zeros((1, NUM_MAIN_STATES)))
                .collect(),
            checkpoint_indices: indices,
            all_specials: Array2::zeros((1, NUM_SPECIAL_STATES)),
            overall_score: 0.0,
            sequence_length: 0,
            num_nodes: 0,
        }
    }

    #[test]
    fn checkpoint_lookup_handles_boundaries() {
        let checkpoints = checkpoints(vec![0, 4, 8]);

        assert_eq!(checkpoints.checkpoint_at_or_before(0), Some((0, 0)));
        assert_eq!(checkpoints.checkpoint_at_or_before(3), Some((0, 0)));
        assert_eq!(checkpoints.checkpoint_at_or_before(4), Some((1, 4)));
        assert_eq!(checkpoints.checkpoint_at_or_before(100), Some((2, 8)));
    }

    #[test]
    fn checkpoint_lookup_returns_none_without_a_preceding_checkpoint() {
        assert_eq!(checkpoints(Vec::new()).checkpoint_at_or_before(0), None);
        assert_eq!(checkpoints(vec![2, 4]).checkpoint_at_or_before(1), None);
    }
}
