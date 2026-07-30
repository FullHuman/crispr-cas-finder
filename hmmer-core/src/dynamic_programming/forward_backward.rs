use ndarray::Array2;

use super::simd::fwd_bck;
use super::simd::oprofile;
use super::simd::oprofile::OptimizedProfile;
use crate::constants::dynamic_programming::forward_backward::SCALE_THRESH;
use crate::errors::HmmerError;
use crate::profile::Profile;
use crate::score_matrix::*;

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

    /// Find the largest checkpoint index <= `row`.
    /// Returns the position in `checkpoint_main` and the row index.
    pub fn checkpoint_at_or_before(&self, row: usize) -> (usize, usize) {
        // Binary search: find rightmost index where checkpoint_indices[idx] <= row
        let pos = self.checkpoint_indices.partition_point(|&r| r <= row);
        // pos is the count of elements <= row, so idx is pos-1
        let idx = pos - 1;
        (idx, self.checkpoint_indices[idx])
    }
}

// ─── Fused backward + decoding ──────────────────────────────────────────────

use crate::domaindef::DomainDef;

/// Backward + posterior decoding in probability/odds space.
///
/// Replaces the log-space `flogsum` inner loop with multiply-add operations,
/// matching C HMMER's approach. Same interface as `backward_decode_fused_simd_odds`.
#[allow(clippy::too_many_arguments)]
pub fn backward_decode_prob_space(
    digital_sequence: &[u8],
    profile: &Profile,
    om: &OptimizedProfile,
    checkpoints: &ForwardCheckpoints,
    simd_data: &fwd_bck::SIMDCheckpointData,
    posterior_matrix: &mut ScoreMatrix,
    domain_def: Option<&mut DomainDef>,
    seg_buf: &mut fwd_bck::OddsSegmentBuf,
) -> Result<f32, HmmerError> {
    use std::simd::prelude::*;

    let m = checkpoints.num_nodes;
    let l = checkpoints.sequence_length;
    let overall_sc = checkpoints.overall_score;
    let q = om.q4;
    let dd_base = oprofile::NTSC_PER_Q * q;
    let zero_v = f32x4::splat(0.0);

    posterior_matrix.resize(m, l)?;
    let main_row_stride = (m + 1) * 3;

    // Special transition probabilities
    let sp_e_move = om.xf[oprofile::XST_E][oprofile::XTR_MOVE];
    let sp_e_loop = om.xf[oprofile::XST_E][oprofile::XTR_LOOP];
    let sp_n_move = om.xf[oprofile::XST_N][oprofile::XTR_MOVE];
    let sp_n_loop = om.xf[oprofile::XST_N][oprofile::XTR_LOOP];
    let sp_c_move = om.xf[oprofile::XST_C][oprofile::XTR_MOVE];
    let sp_c_loop = om.xf[oprofile::XST_C][oprofile::XTR_LOOP];
    let sp_j_move = om.xf[oprofile::XST_J][oprofile::XTR_MOVE];
    let sp_j_loop = om.xf[oprofile::XST_J][oprofile::XTR_LOOP];

    let exit_prob = if profile.mode.is_local() {
        1.0f32
    } else {
        0.0f32
    };

    // Node m position in striped layout
    let qi_m = (m - 1) % q;
    let z_m = (m - 1) / q;

    // --- Initialize backward row L (flat, then convert to striped) ---
    let e_l = sp_e_move * sp_c_move;
    let mut bm_flat = vec![0.0f32; m + 2];
    let mut bd_flat = vec![0.0f32; m + 2];
    bm_flat[m] = e_l;
    bd_flat[m] = e_l;
    for k in (1..m).rev() {
        let qi_k = (k - 1) % q;
        let z_k = (k - 1) / q;
        let tmd_k = om.tfv[(qi_k * oprofile::NTSC_PER_Q + oprofile::TSC_MD) * 4 + z_k];
        let tdd_k = om.tfv[(dd_base + qi_k) * 4 + z_k];
        bm_flat[k] = exit_prob * e_l + tmd_k * bd_flat[k + 1];
        bd_flat[k] = exit_prob * e_l + tdd_k * bd_flat[k + 1];
    }

    let mut bck_c = sp_c_move;
    let mut bck_j = 0.0f32;
    let mut bck_n = 0.0f32;
    let mut bck_b = 0.0f32;
    let mut bck_e = bck_c * sp_e_move;

    // Scale row L
    let mut row_max = bck_c;
    for k in 1..=m {
        row_max = row_max.max(bm_flat[k]).max(bd_flat[k]);
    }
    if row_max > 0.0 {
        let inv = 1.0 / row_max;
        for k in 1..=m {
            bm_flat[k] *= inv;
            bd_flat[k] *= inv;
        }
        bck_c *= inv;
        bck_j *= inv;
        bck_n *= inv;
        bck_e *= inv;
    }
    let mut bck_cumscale: f64 = if row_max > 0.0 {
        (row_max as f64).ln()
    } else {
        0.0
    };

    // === Posterior for row L (log-space forward × prob-space backward) ===
    {
        let fwd_main_l = &checkpoints.checkpoint_main[checkpoints.checkpoint_main.len() - 1];
        let fwd_cumscale_l = simd_data.per_row_totscale[l];
        let bck_sc_l = bck_cumscale as f32;
        let log_correction = fwd_cumscale_l as f32 + bck_sc_l - overall_sc;

        let mut denom = 0.0f32;
        posterior_matrix.main[[l, 0, MATCH_CELL]] = 0.0;
        posterior_matrix.main[[l, 0, INSERT_CELL]] = 0.0;
        posterior_matrix.main[[l, 0, DELETE_CELL]] = 0.0;

        for k in 1..=m {
            let mm = (fwd_main_l[[k, MATCH_CELL]]
                + if bm_flat[k] > 0.0 {
                    bm_flat[k].ln()
                } else {
                    f32::NEG_INFINITY
                }
                + log_correction)
                .exp();
            let im = 0.0f32; // bi_flat[k] = 0 at row L
            posterior_matrix.main[[l, k, MATCH_CELL]] = mm;
            posterior_matrix.main[[l, k, INSERT_CELL]] = im;
            posterior_matrix.main[[l, k, DELETE_CELL]] = 0.0;
            denom += mm;
        }

        let n_pp = (checkpoints.all_specials[[l - 1, N_STATE]] + profile.special_scores.n_loop
            - overall_sc
            + if bck_n > 0.0 {
                bck_n.ln() + bck_sc_l
            } else {
                f32::NEG_INFINITY
            })
        .exp();
        let j_pp = (checkpoints.all_specials[[l - 1, J_STATE]] + profile.special_scores.j_loop
            - overall_sc
            + if bck_j > 0.0 {
                bck_j.ln() + bck_sc_l
            } else {
                f32::NEG_INFINITY
            })
        .exp();
        let c_pp = (checkpoints.all_specials[[l - 1, C_STATE]] + profile.special_scores.c_loop
            - overall_sc
            + if bck_c > 0.0 {
                bck_c.ln() + bck_sc_l
            } else {
                f32::NEG_INFINITY
            })
        .exp();
        posterior_matrix.set_special_row(l, [0.0, n_pp, j_pp, 0.0, c_pp]);
        denom += n_pp + j_pp + c_pp;

        if denom > 0.0 {
            let inv = 1.0 / denom;
            for k in 1..=m {
                posterior_matrix.main[[l, k, MATCH_CELL]] *= inv;
                posterior_matrix.main[[l, k, INSERT_CELL]] *= inv;
            }
            posterior_matrix.special[[l, N_STATE]] *= inv;
            posterior_matrix.special[[l, J_STATE]] *= inv;
            posterior_matrix.special[[l, C_STATE]] *= inv;
        }
    }

    // Convert flat backward row L to striped SIMD vectors
    let mut bmx = vec![zero_v; q];
    let mut bix = vec![zero_v; q];
    let mut bdx = vec![zero_v; q];
    for qi in 0..q {
        let mut bm_arr = [0.0f32; 4];
        let mut bd_arr = [0.0f32; 4];
        for z in 0..4 {
            let k = qi + 1 + z * q;
            if k <= m {
                bm_arr[z] = bm_flat[k];
                bd_arr[z] = bd_flat[k];
            }
        }
        bmx[qi] = f32x4::from_array(bm_arr);
        bdx[qi] = f32x4::from_array(bd_arr);
    }
    // bix is all zeros at row L (no insert backward)

    let mut bmx2 = vec![zero_v; q];
    let mut bix2 = vec![zero_v; q];
    let mut bdx2 = vec![zero_v; q];

    let has_domain_def = domain_def.is_some();

    let mut bck_store_b = vec![0.0f32; l + 1];
    let mut bck_store_e = vec![0.0f32; l + 1];
    let mut bck_store_n = vec![0.0f32; l + 1];
    let mut bck_store_j = vec![0.0f32; l + 1];
    let mut bck_store_c = vec![0.0f32; l + 1];
    let mut bck_cs = vec![0.0f64; l + 1];

    bck_store_b[l] = bck_b;
    bck_store_e[l] = bck_e;
    bck_store_n[l] = bck_n;
    bck_store_j[l] = bck_j;
    bck_store_c[l] = bck_c;
    bck_cs[l] = bck_cumscale;

    // === Main backward sweep from L-1 to 1 (Farrar-striped SIMD) ===
    if l > 1 {
        let (mut curr_seg_pos, _) = checkpoints.checkpoint_at_or_before(l - 1);
        let mut seg_start = checkpoints.checkpoint_indices[curr_seg_pos];
        let seg_end = checkpoints.checkpoint_indices[curr_seg_pos + 1];
        fwd_bck::replay_segment_odds_buf(
            digital_sequence,
            om,
            simd_data,
            curr_seg_pos,
            seg_start,
            seg_end,
            seg_buf,
        );

        for i in (1..l).rev() {
            if i < seg_start {
                curr_seg_pos -= 1;
                seg_start = checkpoints.checkpoint_indices[curr_seg_pos];
                let seg_end = checkpoints.checkpoint_indices[curr_seg_pos + 1];
                fwd_bck::replay_segment_odds_buf(
                    digital_sequence,
                    om,
                    simd_data,
                    curr_seg_pos,
                    seg_start,
                    seg_end,
                    seg_buf,
                );
            }

            let x_next = digital_sequence[i] as usize;
            let rsc = if x_next < om.kp {
                &om.rfv[x_next]
            } else {
                &om.rfv[0]
            };
            let isc = if x_next < om.kp {
                &om.riv[x_next]
            } else {
                &om.riv[0]
            };

            // --- Begin sum (SIMD striped) ---
            let mut begin_v = zero_v;
            for qi in 0..q {
                let tbm_v = f32x4::from_slice(
                    &om.tfv[(qi * oprofile::NTSC_PER_Q + oprofile::TSC_BM) * 4
                        ..(qi * oprofile::NTSC_PER_Q + oprofile::TSC_BM) * 4 + 4],
                );
                let rsc_v = f32x4::from_slice(&rsc[qi * 4..qi * 4 + 4]);
                begin_v += tbm_v * rsc_v * bmx[qi];
            }
            bck_b = begin_v.reduce_sum();

            // --- Specials ---
            bck_j = sp_j_move * bck_b + sp_j_loop * bck_j;
            bck_c *= sp_c_loop;
            bck_e = sp_e_move * bck_c + sp_e_loop * bck_j;
            bck_n = sp_n_move * bck_b + sp_n_loop * bck_n;

            // --- Main backward states (SIMD Farrar-striped) ---
            // Per-stripe exit vectors: exit_prob*bck_e everywhere, bck_e at node m
            let exit_base = exit_prob * bck_e;
            let exit_base_v = f32x4::splat(exit_base);
            // For node m, always exit with bck_e regardless of exit_prob
            let exit_m_v = {
                let mut arr = [exit_base; 4];
                arr[z_m] = bck_e;
                f32x4::from_array(arr)
            };

            // Precompute shifted values for qi=q-1 wrap
            let bm_carry = fwd_bck::shl_f32x4(bmx[0], zero_v);
            let rsc_carry = fwd_bck::shl_f32x4(f32x4::from_slice(&rsc[0..4]), zero_v);
            let tmm_carry = fwd_bck::shl_f32x4(
                f32x4::from_slice(&om.tfv[oprofile::TSC_MM * 4..oprofile::TSC_MM * 4 + 4]),
                zero_v,
            );
            let tim_carry = fwd_bck::shl_f32x4(
                f32x4::from_slice(&om.tfv[oprofile::TSC_IM * 4..oprofile::TSC_IM * 4 + 4]),
                zero_v,
            );
            let tdm_carry = fwd_bck::shl_f32x4(
                f32x4::from_slice(&om.tfv[oprofile::TSC_DM * 4..oprofile::TSC_DM * 4 + 4]),
                zero_v,
            );

            // Main backward sweep: qi = q-1 downto 0
            for qi in (0..q).rev() {
                let exit_v = if qi == qi_m { exit_m_v } else { exit_base_v };

                // "Next node" values: bm[k+1], emission[k+1], and "into-match" transitions
                let (bm_next, rsc_next, tmm_v, tim_v, tdm_v) = if qi < q - 1 {
                    let tsc_next = (qi + 1) * oprofile::NTSC_PER_Q;
                    (
                        bmx[qi + 1],
                        f32x4::from_slice(&rsc[(qi + 1) * 4..(qi + 1) * 4 + 4]),
                        f32x4::from_slice(
                            &om.tfv[(tsc_next + oprofile::TSC_MM) * 4
                                ..(tsc_next + oprofile::TSC_MM) * 4 + 4],
                        ),
                        f32x4::from_slice(
                            &om.tfv[(tsc_next + oprofile::TSC_IM) * 4
                                ..(tsc_next + oprofile::TSC_IM) * 4 + 4],
                        ),
                        f32x4::from_slice(
                            &om.tfv[(tsc_next + oprofile::TSC_DM) * 4
                                ..(tsc_next + oprofile::TSC_DM) * 4 + 4],
                        ),
                    )
                } else {
                    (bm_carry, rsc_carry, tmm_carry, tim_carry, tdm_carry)
                };

                let mm_v = rsc_next * bm_next;

                // "Same node" values
                let isc_v = f32x4::from_slice(&isc[qi * 4..qi * 4 + 4]);
                let ii_v = isc_v * bix[qi];

                let tsc_base = qi * oprofile::NTSC_PER_Q;
                let tmi_v = f32x4::from_slice(
                    &om.tfv
                        [(tsc_base + oprofile::TSC_MI) * 4..(tsc_base + oprofile::TSC_MI) * 4 + 4],
                );
                let tmd_v = f32x4::from_slice(
                    &om.tfv
                        [(tsc_base + oprofile::TSC_MD) * 4..(tsc_base + oprofile::TSC_MD) * 4 + 4],
                );
                let tii_v = f32x4::from_slice(
                    &om.tfv
                        [(tsc_base + oprofile::TSC_II) * 4..(tsc_base + oprofile::TSC_II) * 4 + 4],
                );
                let tdd_v = f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);

                // bd_next = bd2 at k+1 from current row
                let bd_next = if qi < q - 1 { bdx2[qi + 1] } else { zero_v };

                bix2[qi] = tim_v * mm_v + tii_v * ii_v;
                bdx2[qi] = tdm_v * mm_v + tdd_v * bd_next + exit_v;
                bmx2[qi] = tmm_v * mm_v + tmi_v * ii_v + tmd_v * bd_next + exit_v;
            }

            // DD correction passes (handle wrap-around from stripe 0 to stripe q-1)
            for _pass in 0..3 {
                let carry = fwd_bck::shl_f32x4(bdx2[0], zero_v);
                if carry == zero_v {
                    break;
                }
                let mut dcv = carry;
                for qi in (0..q).rev() {
                    let tsc_base = qi * oprofile::NTSC_PER_Q;
                    let tdd_v =
                        f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
                    let tmd_v = f32x4::from_slice(
                        &om.tfv[(tsc_base + oprofile::TSC_MD) * 4
                            ..(tsc_base + oprofile::TSC_MD) * 4 + 4],
                    );
                    bmx2[qi] += tmd_v * dcv;
                    let new_delta = tdd_v * dcv;
                    bdx2[qi] += new_delta;
                    dcv = new_delta;
                }
            }

            // --- Scale (SIMD) ---
            let mut max_v = zero_v;
            for qi in 0..q {
                max_v = max_v
                    .simd_max(bmx2[qi])
                    .simd_max(bix2[qi])
                    .simd_max(bdx2[qi]);
            }
            row_max = max_v
                .reduce_max()
                .max(bck_c)
                .max(bck_j)
                .max(bck_n)
                .max(bck_e)
                .max(bck_b);

            if row_max < SCALE_THRESH && row_max > 0.0 {
                let inv = 1.0 / row_max;
                let inv_v = f32x4::splat(inv);
                for qi in 0..q {
                    bmx2[qi] *= inv_v;
                    bix2[qi] *= inv_v;
                    bdx2[qi] *= inv_v;
                }
                bck_c *= inv;
                bck_j *= inv;
                bck_n *= inv;
                bck_e *= inv;
                bck_b *= inv;
                bck_cumscale += (row_max as f64).ln();
            }

            // Store for domain_def
            if has_domain_def {
                bck_store_b[i] = bck_b;
                bck_store_e[i] = bck_e;
                bck_store_n[i] = bck_n;
                bck_store_j[i] = bck_j;
                bck_store_c[i] = bck_c;
                bck_cs[i] = bck_cumscale;
            }

            // --- Posterior (SIMD: fwd × bck in striped layout) ---
            let local_row = i - seg_start;
            let fwd_ts = simd_data.per_row_totscale[i];
            let fwd_ts_prev = simd_data.per_row_totscale[i - 1];
            let spec_corr = ((fwd_ts_prev - fwd_ts) as f32).exp();

            let mut denom_v = zero_v;

            // The old backward row is dead now, so use it as posterior scratch.
            for qi in 0..q {
                let mm_v = seg_buf.m_odds_vec(local_row, qi) * bmx2[qi];
                let im_v = seg_buf.i_odds_vec(local_row, qi) * bix2[qi];
                bmx[qi] = mm_v;
                bix[qi] = im_v;
                denom_v += mm_v + im_v;
            }
            let mut denom = denom_v.reduce_sum();

            let fwd_odds = &simd_data.odds_specials[i - 1];
            let n_pp = fwd_odds[N_STATE] * sp_n_loop * bck_n * spec_corr;
            let j_pp = fwd_odds[J_STATE] * sp_j_loop * bck_j * spec_corr;
            let c_pp = fwd_odds[C_STATE] * sp_c_loop * bck_c * spec_corr;
            denom += n_pp + j_pp + c_pp;

            let inv = if denom > 0.0 { 1.0 / denom } else { 1.0 };
            let inv_v = f32x4::splat(inv);

            let row_base = i * main_row_stride;
            let main = posterior_matrix
                .main
                .as_slice_mut()
                .expect("score matrix must be contiguous");
            let main_row = &mut main[row_base..row_base + main_row_stride];
            main_row[MATCH_CELL] = 0.0;
            main_row[INSERT_CELL] = 0.0;
            main_row[DELETE_CELL] = 0.0;

            for qi in 0..q {
                let mm_a = (bmx[qi] * inv_v).to_array();
                let im_a = (bix[qi] * inv_v).to_array();
                for z in 0..4usize {
                    let k = qi + 1 + z * q;
                    if k <= m {
                        let cell = k * 3;
                        main_row[cell + MATCH_CELL] = mm_a[z];
                        main_row[cell + INSERT_CELL] = im_a[z];
                        main_row[cell + DELETE_CELL] = 0.0;
                    }
                }
            }

            posterior_matrix.set_special_row(i, [0.0, n_pp * inv, j_pp * inv, 0.0, c_pp * inv]);

            std::mem::swap(&mut bmx, &mut bmx2);
            std::mem::swap(&mut bix, &mut bix2);
            std::mem::swap(&mut bdx, &mut bdx2);
        }
    }

    // === Backward row 0 (SIMD begin sum) ===
    let bck_score = if l > 0 {
        let x_next = digital_sequence[0] as usize;
        let rsc = if x_next < om.kp {
            &om.rfv[x_next]
        } else {
            &om.rfv[0]
        };

        let mut begin_v = zero_v;
        for qi in 0..q {
            let tbm_v = f32x4::from_slice(
                &om.tfv[(qi * oprofile::NTSC_PER_Q + oprofile::TSC_BM) * 4
                    ..(qi * oprofile::NTSC_PER_Q + oprofile::TSC_BM) * 4 + 4],
            );
            let rsc_v = f32x4::from_slice(&rsc[qi * 4..qi * 4 + 4]);
            begin_v += tbm_v * rsc_v * bmx[qi];
        }
        bck_b = begin_v.reduce_sum();
        bck_n = sp_n_move * bck_b + sp_n_loop * bck_n;

        bck_store_b[0] = bck_b;
        bck_store_n[0] = bck_n;

        if bck_n > 0.0 {
            bck_n.ln() + bck_cumscale as f32
        } else {
            f32::NEG_INFINITY
        }
    } else {
        0.0
    };
    bck_cs[0] = bck_cumscale;

    // === Domain decoding (optional) ===
    if let Some(domain_def) = domain_def {
        domain_def.grow_to(l);
        if domain_def.begin_totals.len() <= l {
            domain_def.begin_totals.resize(l + 1, 0.0);
            domain_def.exit_totals.resize(l + 1, 0.0);
            domain_def.model_occupancy.resize(l + 1, 0.0);
        }
        domain_def.begin_totals[0] = 0.0;
        domain_def.exit_totals[0] = 0.0;

        for i in 1..=l {
            let fwd_log_b = checkpoints.all_specials[[i - 1, BEGIN_STATE]];
            let bck_cs_im1 = bck_cs[i - 1];
            let corr_begin = (fwd_log_b + bck_cs_im1 as f32 - overall_sc).exp();
            domain_def.begin_totals[i] =
                domain_def.begin_totals[i - 1] + corr_begin * bck_store_b[i - 1];

            let fwd_log_e = checkpoints.all_specials[[i, EXIT_STATE]];
            let bck_cs_i = bck_cs[i];
            let corr_exit = (fwd_log_e + bck_cs_i as f32 - overall_sc).exp();
            domain_def.exit_totals[i] = domain_def.exit_totals[i - 1] + corr_exit * bck_store_e[i];

            let fwd_log_n = checkpoints.all_specials[[i - 1, N_STATE]];
            let fwd_log_j = checkpoints.all_specials[[i - 1, J_STATE]];
            let fwd_log_c = checkpoints.all_specials[[i - 1, C_STATE]];
            let nn = (fwd_log_n + bck_cs_i as f32 - overall_sc).exp() * bck_store_n[i] * sp_n_loop;
            let jj = (fwd_log_j + bck_cs_i as f32 - overall_sc).exp() * bck_store_j[i] * sp_j_loop;
            let cc = (fwd_log_c + bck_cs_i as f32 - overall_sc).exp() * bck_store_c[i] * sp_c_loop;
            let njcp = nn + jj + cc;
            domain_def.model_occupancy[i] = 1.0 - njcp;
        }
        domain_def.length = l;
    }

    Ok(bck_score)
}
