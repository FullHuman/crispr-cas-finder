//! Posterior decoding in probability space.
//!
//! The same Forward, Backward and posterior computation as
//! [`super::forward_backward::backward_decode_log_space`], carried out on
//! odds ratios rather than log scores, as HMMER's production pipeline does:
//! sums and products in place of table-driven log-sums, with every row
//! rescaled when its exit odds grow large so that single precision never
//! overflows. Scores agree with the log-space decoder to about a hundredth
//! of a nat and posteriors to about a thousandth; the log-space decoder is
//! several times slower because each of its rows is a serial chain of
//! table look-ups, and it remains the reference in tests and in exact mode.

use super::score_matrix::*;
use crate::hmmer_core::config::{PROFILE_NUM_EMISSIONS, PROFILE_NUM_TRANSITIONS, PRsc, PTsc};
use crate::hmmer_core::domaindef::DomainDef;
use crate::hmmer_core::errors::HmmerError;
use crate::hmmer_core::profile::Profile;

/// Rows whose exit odds exceed this are rescaled.
const RESCALE_ABOVE: f32 = 1.0e4;

/// Transition odds of one node, indexed by `PTsc`.
type NodeTransitions = [f32; PROFILE_NUM_TRANSITIONS];

/// Odds ratios of a profile's emissions and transitions, shared by all the
/// sequences a worker searches with the profile.
#[derive(Debug, Default)]
pub struct OddsTables {
    num_nodes: usize,
    /// `[residue][node]` match and insert emission odds.
    emissions: Vec<[f32; 2]>,
    transitions: Vec<NodeTransitions>,
}

impl OddsTables {
    /// Build the tables of `profile`; its length model does not matter.
    pub fn new(profile: &Profile) -> Self {
        let m = profile.num_nodes;
        let kp = profile.alphabet.full_size;
        let mut emissions = vec![[0.0f32; 2]; kp * (m + 1)];
        for x in 0..kp {
            let scores = profile.residue_scores_for(x);
            for k in 0..=m {
                let base = k * PROFILE_NUM_EMISSIONS;
                emissions[x * (m + 1) + k] = [
                    scores[base + PRsc::MatchScore.idx()].exp(),
                    scores[base + PRsc::InsertScore.idx()].exp(),
                ];
            }
        }
        let (rows, _) = profile
            .transition_scores_raw()
            .as_chunks::<PROFILE_NUM_TRANSITIONS>();
        let transitions = rows[..=m.min(rows.len().saturating_sub(1))]
            .iter()
            .map(|row| row.map(f32::exp))
            .collect();
        Self {
            num_nodes: m,
            emissions,
            transitions,
        }
    }

    #[inline]
    fn emissions_of(&self, residue: usize) -> &[[f32; 2]] {
        &self.emissions[residue * (self.num_nodes + 1)..][..self.num_nodes + 1]
    }
}

/// Special transition odds of the configured length model.
struct Specials {
    n_loop: f32,
    n_move: f32,
    j_loop: f32,
    j_move: f32,
    c_loop: f32,
    c_move: f32,
    e_loop: f32,
    e_move: f32,
    /// Local exit odds of nodes before the last: one in local mode.
    exit: f32,
}

/// Forward, Backward and posterior decoding of `digital_sequence` against
/// `profile`, writing posterior probabilities into `posterior` and, when
/// given, the per-position totals `domain_def` needs. Returns the Backward
/// score in nats.
///
/// `tables` must have been built from `profile`. The Forward score is
/// checked against `expected_forward_score`, the SIMD filter's.
pub fn decode_odds_space(
    digital_sequence: &[u8],
    profile: &Profile,
    tables: &OddsTables,
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
    if tables.num_nodes != m || tables.transitions.len() < m {
        return Err(HmmerError::Internal(
            "odds tables do not belong to this profile".into(),
        ));
    }
    posterior.resize(m, l)?;
    let alphabet_size = profile.alphabet.full_size;
    for &x in digital_sequence {
        if x as usize >= alphabet_size {
            return Err(HmmerError::Internal(format!(
                "digital residue code {x} is outside the profile alphabet"
            )));
        }
    }
    let scores = &profile.special_scores;
    let special = Specials {
        n_loop: scores.n_loop.exp(),
        n_move: scores.n_move.exp(),
        j_loop: scores.j_loop.exp(),
        j_move: scores.j_move.exp(),
        c_loop: scores.c_loop.exp(),
        c_move: scores.c_move.exp(),
        e_loop: scores.e_loop.exp(),
        e_move: scores.e_move.exp(),
        exit: if profile.mode.is_local() { 1.0 } else { 0.0 },
    };
    let transitions = &tables.transitions[..m];
    let stride = posterior.main_row_stride();
    let row_cells = (m + 1) * NUM_MAIN_STATES;

    // Forward, stored in the posterior matrix until each row is decoded.
    // Row i is divided by `scales[i]` once computed; a row's values are thus
    // the true odds divided by the product of the scales up to it.
    let mut scales = vec![1.0f32; l + 1];
    let mut log_scale = 0.0f64;
    // [E, N, J, B, C] per row, scaled like the row.
    let mut forward_specials = vec![[0.0f32; NUM_SPECIAL_STATES]; l + 1];
    forward_specials[0] = [0.0, 1.0, 0.0, special.n_move, 0.0];
    {
        let main = posterior.main_as_slice_mut();
        main[..row_cells].fill(0.0);
        for i in 1..=l {
            let emissions = tables.emissions_of(digital_sequence[i - 1] as usize);
            let (before, from_here) = main.split_at_mut(i * stride);
            let (previous, _) =
                before[(i - 1) * stride..][..row_cells].as_chunks::<NUM_MAIN_STATES>();
            let (current, _) = from_here[..row_cells].as_chunks_mut::<NUM_MAIN_STATES>();
            let x_b_prev = forward_specials[i - 1][BEGIN_STATE];
            current[0] = [0.0; NUM_MAIN_STATES];
            let mut x_e = 0.0f32;
            let mut previous_left = previous[0];
            let mut delete_left = 0.0f32;
            let mut match_left = 0.0f32;
            let mut t_left = &transitions[0];
            for k in 1..=m {
                let t_here = &transitions[k.min(m - 1)];
                let emission = emissions[k];
                let mm = (previous_left[MATCH_CELL] * t_left[PTsc::MatchToMatch.idx()]
                    + previous_left[INSERT_CELL] * t_left[PTsc::InsertToMatch.idx()]
                    + previous_left[DELETE_CELL] * t_left[PTsc::DeleteToMatch.idx()]
                    + x_b_prev * t_left[PTsc::BeginToMatch.idx()])
                    * emission[0];
                let im = if k < m {
                    (previous[k][MATCH_CELL] * t_here[PTsc::MatchToInsert.idx()]
                        + previous[k][INSERT_CELL] * t_here[PTsc::InsertToInsert.idx()])
                        * emission[1]
                } else {
                    0.0
                };
                let dm = match_left * t_left[PTsc::MatchToDelete.idx()]
                    + delete_left * t_left[PTsc::DeleteToDelete.idx()];
                current[k] = [mm, im, dm];
                x_e += if k < m {
                    (mm + dm) * special.exit
                } else {
                    mm + dm
                };
                previous_left = previous[k];
                match_left = mm;
                delete_left = dm;
                t_left = t_here;
            }
            let previous_specials = forward_specials[i - 1];
            let mut x_j = previous_specials[J_STATE] * special.j_loop + x_e * special.e_loop;
            let mut x_c = previous_specials[C_STATE] * special.c_loop + x_e * special.e_move;
            let mut x_n = previous_specials[N_STATE] * special.n_loop;
            let mut x_b = x_n * special.n_move + x_j * special.j_move;
            if x_e > RESCALE_ABOVE {
                let inverse = 1.0 / x_e;
                for cell in current.iter_mut() {
                    cell[MATCH_CELL] *= inverse;
                    cell[INSERT_CELL] *= inverse;
                    cell[DELETE_CELL] *= inverse;
                }
                x_j *= inverse;
                x_c *= inverse;
                x_n *= inverse;
                x_b *= inverse;
                scales[i] = x_e;
                log_scale += f64::from(x_e).ln();
                x_e = 1.0;
            }
            forward_specials[i] = [x_e, x_n, x_j, x_b, x_c];
        }
    }
    let total = forward_specials[l][C_STATE] * special.c_move;
    let forward_score = (log_scale + f64::from(total).ln()) as f32;
    if !forward_score.is_finite()
        || !expected_forward_score.is_finite()
        || (forward_score - expected_forward_score).abs() > 0.1
    {
        return Err(HmmerError::Internal(format!(
            "Forward implementations disagree: odds={forward_score}, SIMD={expected_forward_score}"
        )));
    }
    let inverse_total = 1.0 / total;

    // Backward, row by row from the end, each row scaled like the Forward
    // row of the same index; a posterior is then forward × backward × the
    // row's scale / total. Two main-state rows suffice.
    let mut backward_specials = vec![[0.0f32; NUM_SPECIAL_STATES]; l + 1];
    let mut b_next = vec![[0.0f32; NUM_MAIN_STATES]; m + 1];
    let mut b_cur = vec![[0.0f32; NUM_MAIN_STATES]; m + 1];
    backward_specials[l][C_STATE] = special.c_move;
    backward_specials[l][EXIT_STATE] = special.c_move * special.e_move;
    {
        let exit_odds = backward_specials[l][EXIT_STATE];
        b_next[m][MATCH_CELL] = exit_odds;
        b_next[m][DELETE_CELL] = exit_odds;
        for k in (1..m).rev() {
            let t = &transitions[k];
            b_next[k][MATCH_CELL] = exit_odds * special.exit
                + b_next[k + 1][DELETE_CELL] * t[PTsc::MatchToDelete.idx()];
            b_next[k][DELETE_CELL] = exit_odds * special.exit
                + b_next[k + 1][DELETE_CELL] * t[PTsc::DeleteToDelete.idx()];
        }
        let inverse = 1.0 / scales[l];
        for cell in b_next.iter_mut() {
            cell[MATCH_CELL] *= inverse;
            cell[INSERT_CELL] *= inverse;
            cell[DELETE_CELL] *= inverse;
        }
        for value in backward_specials[l].iter_mut() {
            *value *= inverse;
        }
    }
    decode_row(
        posterior,
        &forward_specials,
        &backward_specials,
        &b_next,
        &scales,
        inverse_total,
        l,
        m,
        &special,
    );

    for i in (1..l).rev() {
        let emissions = tables.emissions_of(digital_sequence[i] as usize);
        let next = &b_next[..=m];
        let mut x_b = 0.0f32;
        for k in 1..=m {
            x_b += next[k][MATCH_CELL]
                * transitions[k - 1][PTsc::BeginToMatch.idx()]
                * emissions[k][0];
        }
        let x_j = backward_specials[i + 1][J_STATE] * special.j_loop + x_b * special.j_move;
        let x_c = backward_specials[i + 1][C_STATE] * special.c_loop;
        let x_e = x_j * special.e_loop + x_c * special.e_move;
        let x_n = backward_specials[i + 1][N_STATE] * special.n_loop + x_b * special.n_move;
        backward_specials[i] = [x_e, x_n, x_j, x_b, x_c];

        let current = &mut b_cur[..=m];
        current[0] = [0.0; NUM_MAIN_STATES];
        current[m] = [x_e, 0.0, x_e];
        let mut delete_right = x_e;
        for k in (1..m).rev() {
            let t = &transitions[k];
            let to_match = next[k + 1][MATCH_CELL] * emissions[k + 1][0];
            let to_insert = next[k][INSERT_CELL] * emissions[k][1];
            let match_cell = to_match * t[PTsc::MatchToMatch.idx()]
                + to_insert * t[PTsc::MatchToInsert.idx()]
                + x_e * special.exit
                + delete_right * t[PTsc::MatchToDelete.idx()];
            let insert_cell =
                to_match * t[PTsc::InsertToMatch.idx()] + to_insert * t[PTsc::InsertToInsert.idx()];
            let delete_cell = to_match * t[PTsc::DeleteToMatch.idx()]
                + delete_right * t[PTsc::DeleteToDelete.idx()]
                + x_e * special.exit;
            current[k] = [match_cell, insert_cell, delete_cell];
            delete_right = delete_cell;
        }
        if scales[i] != 1.0 {
            let inverse = 1.0 / scales[i];
            for cell in current.iter_mut() {
                cell[MATCH_CELL] *= inverse;
                cell[INSERT_CELL] *= inverse;
                cell[DELETE_CELL] *= inverse;
            }
            for value in backward_specials[i].iter_mut() {
                *value *= inverse;
            }
        }
        decode_row(
            posterior,
            &forward_specials,
            &backward_specials,
            &b_cur,
            &scales,
            inverse_total,
            i,
            m,
            &special,
        );
        std::mem::swap(&mut b_cur, &mut b_next);
    }

    let emissions = tables.emissions_of(digital_sequence[0] as usize);
    let mut b_zero = 0.0f32;
    for k in 1..=m {
        b_zero +=
            b_next[k][MATCH_CELL] * transitions[k - 1][PTsc::BeginToMatch.idx()] * emissions[k][0];
    }
    let backward_total = backward_specials[1][N_STATE] * special.n_loop + b_zero * special.n_move;
    let backward_score = (log_scale + f64::from(backward_total).ln()) as f32;
    backward_specials[0][BEGIN_STATE] = b_zero;
    backward_specials[0][N_STATE] = backward_total;
    posterior.set_special_row(
        0,
        [
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
            scores.n_move,
            f32::NEG_INFINITY,
        ],
    );

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
            // Row i-1 of Forward and row i-1 of Backward share the scale of
            // row i-1 exactly once between them, as do rows i and i.
            let begin = forward_specials[i - 1][BEGIN_STATE]
                * backward_specials[i - 1][BEGIN_STATE]
                * scales[i - 1]
                * inverse_total;
            let exit = forward_specials[i][EXIT_STATE]
                * backward_specials[i][EXIT_STATE]
                * scales[i]
                * inverse_total;
            domain_def.begin_totals[i] = domain_def.begin_totals[i - 1] + begin;
            domain_def.exit_totals[i] = domain_def.exit_totals[i - 1] + exit;
            let outside =
                (forward_specials[i - 1][N_STATE] * special.n_loop * backward_specials[i][N_STATE]
                    + forward_specials[i - 1][J_STATE]
                        * special.j_loop
                        * backward_specials[i][J_STATE]
                    + forward_specials[i - 1][C_STATE]
                        * special.c_loop
                        * backward_specials[i][C_STATE])
                    * inverse_total;
            domain_def.model_occupancy[i] = 1.0 - outside;
        }
        domain_def.length = l;
    }

    Ok(backward_score)
}

/// Overwrite Forward row `i` of `posterior` with posterior probabilities.
#[allow(clippy::too_many_arguments)]
fn decode_row(
    posterior: &mut ScoreMatrix,
    forward_specials: &[[f32; NUM_SPECIAL_STATES]],
    backward_specials: &[[f32; NUM_SPECIAL_STATES]],
    backward_main: &[[f32; NUM_MAIN_STATES]],
    scales: &[f32],
    inverse_total: f32,
    i: usize,
    m: usize,
    special: &Specials,
) {
    let factor = scales[i] * inverse_total;
    let stride = posterior.main_row_stride();
    let row = &mut posterior.main_as_slice_mut()[i * stride..][..(m + 1) * NUM_MAIN_STATES];
    let (cells, _) = row.as_chunks_mut::<NUM_MAIN_STATES>();
    for (cell, backward) in cells[1..=m].iter_mut().zip(&backward_main[1..=m]) {
        cell[MATCH_CELL] = cell[MATCH_CELL] * backward[MATCH_CELL] * factor;
        cell[INSERT_CELL] = cell[INSERT_CELL] * backward[INSERT_CELL] * factor;
        cell[DELETE_CELL] = 0.0;
    }
    // Rows i-1 of Forward and i of Backward together carry every scale once.
    let n = forward_specials[i - 1][N_STATE]
        * special.n_loop
        * backward_specials[i][N_STATE]
        * inverse_total;
    let j = forward_specials[i - 1][J_STATE]
        * special.j_loop
        * backward_specials[i][J_STATE]
        * inverse_total;
    let c = forward_specials[i - 1][C_STATE]
        * special.c_loop
        * backward_specials[i][C_STATE]
        * inverse_total;
    posterior.set_special_row(i, [0.0, n, j, 0.0, c]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmmer_core::alphabet::Alphabet;
    use crate::hmmer_core::config::SearchMode;
    use crate::hmmer_core::domaindef::DomainDef;
    use crate::hmmer_core::dynamic_programming::simd::forward_filter::forward_filter;
    use crate::hmmer_core::dynamic_programming::simd::oprofile::OptimizedProfile;
    use crate::hmmer_core::forward_backward::backward_decode_log_space;
    use crate::hmmer_core::rng::XorShift64;
    use crate::hmmer_core::test_helpers::{random_digital_seq, setup_profile};

    /// The probability-space decoder agrees with the log-space reference on
    /// scores, posteriors and the per-position totals, up to the rounding of
    /// the reference's log-sum table.
    #[test]
    fn odds_space_decoding_matches_the_log_space_reference() {
        let alphabet = Alphabet::amino();
        for (seed, m, l) in [
            (1u64, 7usize, 20usize),
            (2, 40, 90),
            (3, 120, 300),
            (4, 9, 400),
        ] {
            let mut rng = XorShift64::new(seed);
            let (_, background, profile) =
                setup_profile(&mut rng, m, l, &alphabet, SearchMode::Local);
            let om = OptimizedProfile::from_profile(&profile);
            // A consensus stretch inside a random sequence gives a real
            // alignment with sharp posteriors.
            let mut sequence = random_digital_seq(
                &mut rng,
                &background.residue_frequencies,
                alphabet.canonical_size,
                l,
            );
            for (offset, k) in (1..=m.min(l / 2)).enumerate() {
                let best = (0..alphabet.canonical_size)
                    .max_by(|&a, &b| {
                        profile.residue_scores_for(a)[k * 2]
                            .total_cmp(&profile.residue_scores_for(b)[k * 2])
                    })
                    .unwrap();
                sequence[l / 4 + offset] = best as u8;
            }
            let forward = forward_filter(&sequence, l, &om).score;

            let mut reference = ScoreMatrix::new(m, l).unwrap();
            let mut reference_def = DomainDef::new();
            let reference_score = backward_decode_log_space(
                &sequence,
                &profile,
                forward,
                &mut reference,
                Some(&mut reference_def),
            )
            .unwrap();

            let tables = OddsTables::new(&profile);
            let mut odds = ScoreMatrix::new(m, l).unwrap();
            let mut odds_def = DomainDef::new();
            let odds_score = decode_odds_space(
                &sequence,
                &profile,
                &tables,
                forward,
                &mut odds,
                Some(&mut odds_def),
            )
            .unwrap();

            assert!(
                (odds_score - reference_score).abs() < 0.02,
                "seed {seed}: scores {odds_score} vs {reference_score}"
            );
            let mut worst = 0.0f32;
            for i in 1..=l {
                for k in 1..=m {
                    for cell in [MATCH_CELL, INSERT_CELL] {
                        worst = worst
                            .max((odds.main[[i, k, cell]] - reference.main[[i, k, cell]]).abs());
                    }
                }
                for state in [N_STATE, J_STATE, C_STATE] {
                    worst =
                        worst.max((odds.special[[i, state]] - reference.special[[i, state]]).abs());
                }
                worst = worst.max((odds_def.begin_totals[i] - reference_def.begin_totals[i]).abs());
                worst = worst.max((odds_def.exit_totals[i] - reference_def.exit_totals[i]).abs());
                worst = worst
                    .max((odds_def.model_occupancy[i] - reference_def.model_occupancy[i]).abs());
            }
            assert!(
                worst < 1e-2,
                "seed {seed}: posteriors differ by up to {worst}"
            );
        }
    }
}
