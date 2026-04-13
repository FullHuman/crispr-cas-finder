// simd/fwd_bck.rs — SIMD checkpointed Forward + segment recomputation
//
// Replaces the scalar forward_checkpointed() with an odds-space f32x4 Farrar-striped
// implementation. Produces both a standard log-space ForwardCheckpoints (for the
// backward pass) and an odds-space SIMDCheckpointData (for fast segment replay).

use super::oprofile::*;
use crate::dynamic_programming::forward_backward::ForwardCheckpoints;
use crate::dynamic_programming::score_matrix;
use ndarray::Array2;
use std::simd::prelude::*;

/// Shift right by 1 lane (MSB lane gets fill).
#[inline(always)]
fn shr_f32x4(fill: f32x4, v: f32x4) -> f32x4 {
    use std::simd::simd_swizzle;
    simd_swizzle!(fill, v, [3, 4, 5, 6])
}

/// Shift left by 1 lane: result = [v[1], v[2], v[3], fill[0]].
#[inline(always)]
pub fn shl_f32x4(v: f32x4, fill: f32x4) -> f32x4 {
    use std::simd::simd_swizzle;
    simd_swizzle!(v, fill, [1, 2, 3, 4])
}

/// Odds-space SIMD checkpoint data for fast segment replay.
///
/// Stored alongside ForwardCheckpoints so that segment recomputation
/// during backward can use SIMD forward replay in native odds-space.
pub struct SIMDCheckpointData {
    pub cp_mmx: Vec<Vec<f32x4>>,
    pub cp_imx: Vec<Vec<f32x4>>,
    pub cp_dmx: Vec<Vec<f32x4>>,
    pub cp_specials: Vec<[f32; 5]>,   // [E, N, J, B, C] per checkpoint
    pub cp_totscale: Vec<f64>,        // cumulative log-scale at each checkpoint
    pub odds_specials: Vec<[f32; 5]>, // [E, N, J, B, C] per row 0..=L
    pub per_row_totscale: Vec<f64>,   // cumulative log-scale per row
    pub q4: usize,
}

/// Run SIMD forward in odds-space with checkpoint storage.
///
/// Returns both a standard ForwardCheckpoints (log-space, for backward)
/// and SIMDCheckpointData (odds-space, for fast segment replay).
pub fn forward_checkpointed_simd(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
) -> (ForwardCheckpoints, SIMDCheckpointData) {
    let q = om.q4;
    let m = om.m;
    let zero_v = f32x4::splat(0.0);

    let interval = if l == 0 {
        1
    } else {
        ((l as f64).sqrt().ceil() as usize).max(1)
    };

    let mut checkpoint_indices: Vec<usize> = (0..=l).step_by(interval).collect();
    if *checkpoint_indices.last().unwrap() != l {
        checkpoint_indices.push(l);
    }
    let mut next_cp_pos: usize;

    let mut mmx = vec![f32x4::splat(0.0); q];
    let mut imx = vec![f32x4::splat(0.0); q];
    let mut dmx = vec![f32x4::splat(0.0); q];

    let mut totscale: f64 = 0.0;
    let mut odds_specials = vec![[0.0f32; 5]; l + 1];
    let mut per_row_totscale = vec![0.0f64; l + 1];

    let mut x_e: f32 = 0.0;
    let mut x_n: f32 = 1.0;
    let mut x_j: f32 = 0.0;
    let mut x_b: f32 = om.xf[XST_N][XTR_MOVE];
    let mut x_c: f32 = 0.0;
    odds_specials[0] = [x_e, x_n, x_j, x_b, x_c];

    let mut cp_mmx: Vec<Vec<f32x4>> = Vec::with_capacity(checkpoint_indices.len());
    let mut cp_imx: Vec<Vec<f32x4>> = Vec::with_capacity(checkpoint_indices.len());
    let mut cp_dmx: Vec<Vec<f32x4>> = Vec::with_capacity(checkpoint_indices.len());
    let mut cp_totscale: Vec<f64> = Vec::with_capacity(checkpoint_indices.len());
    let mut cp_specials: Vec<[f32; 5]> = Vec::with_capacity(checkpoint_indices.len());

    // Save row 0 checkpoint
    cp_mmx.push(mmx.clone());
    cp_imx.push(imx.clone());
    cp_dmx.push(dmx.clone());
    cp_totscale.push(0.0);
    cp_specials.push([x_e, x_n, x_j, x_b, x_c]);
    next_cp_pos = 1;

    for i in 0..l {
        let xi = dsq[i] as usize;
        let rsc = if xi < om.kp { &om.rfv[xi] } else { &om.rfv[0] };
        let isc = if xi < om.kp { &om.riv[xi] } else { &om.riv[0] };

        let mut dcv = f32x4::splat(0.0);
        let mut xe_v = f32x4::splat(0.0);
        let xb_v = f32x4::splat(x_b);

        let mut mpv = shr_f32x4(zero_v, mmx[q - 1]);
        let mut dpv = shr_f32x4(zero_v, dmx[q - 1]);
        let mut ipv = shr_f32x4(zero_v, imx[q - 1]);

        let mut tsc_idx = 0usize;

        for qi in 0..q {
            let rsc_v = f32x4::from_slice(&rsc[qi * 4..qi * 4 + 4]);

            let tbm_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tmm_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tim_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tdm_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;

            let mut sv = xb_v * tbm_v;
            sv += mpv * tmm_v;
            sv += ipv * tim_v;
            sv += dpv * tdm_v;
            sv *= rsc_v;

            xe_v += sv;

            mpv = mmx[qi];
            dpv = dmx[qi];
            ipv = imx[qi];

            mmx[qi] = sv;
            dmx[qi] = dcv;

            let tmd_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            dcv = sv * tmd_v;

            let tmi_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tii_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let isc_v = f32x4::from_slice(&isc[qi * 4..qi * 4 + 4]);
            imx[qi] = (mpv * tmi_v + ipv * tii_v) * isc_v;
        }

        // DD passes
        dcv = shr_f32x4(zero_v, dcv);
        dmx[0] = f32x4::splat(0.0);

        let dd_base = NTSC_PER_Q * q;
        for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
            let dd_v = f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
            *dmx_qi += dcv;
            dcv = *dmx_qi * dd_v;
        }

        if om.m < 100 {
            for _pass in 1..4 {
                dcv = shr_f32x4(zero_v, dcv);
                for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
                    let dd_v =
                        f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
                    *dmx_qi += dcv;
                    dcv *= dd_v;
                }
            }
        } else {
            for _pass in 1..4 {
                dcv = shr_f32x4(zero_v, dcv);
                let mut any_change = false;
                for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
                    let dd_v =
                        f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
                    let old = *dmx_qi;
                    let new_val = old + dcv;
                    if new_val.simd_gt(old).any() {
                        any_change = true;
                    }
                    *dmx_qi = new_val;
                    dcv *= dd_v;
                }
                if !any_change {
                    break;
                }
            }
        }

        for &v in &dmx {
            xe_v += v;
        }
        x_e = xe_v.reduce_sum();

        x_n *= om.xf[XST_N][XTR_LOOP];
        x_c = x_c * om.xf[XST_C][XTR_LOOP] + x_e * om.xf[XST_E][XTR_MOVE];
        x_j = x_j * om.xf[XST_J][XTR_LOOP] + x_e * om.xf[XST_E][XTR_LOOP];
        x_b = x_j * om.xf[XST_J][XTR_MOVE] + x_n * om.xf[XST_N][XTR_MOVE];

        // Sparse rescaling
        if x_e > 1.0e4 {
            let inv = 1.0 / x_e;
            x_n *= inv;
            x_c *= inv;
            x_j *= inv;
            x_b *= inv;

            let inv_v = f32x4::splat(inv);
            for qi in 0..q {
                mmx[qi] *= inv_v;
                dmx[qi] *= inv_v;
                imx[qi] *= inv_v;
            }
            totscale += (x_e as f64).ln();
            x_e = 1.0;
        }

        odds_specials[i + 1] = [x_e, x_n, x_j, x_b, x_c];
        per_row_totscale[i + 1] = totscale;

        if next_cp_pos < checkpoint_indices.len() && i + 1 == checkpoint_indices[next_cp_pos] {
            cp_mmx.push(mmx.clone());
            cp_imx.push(imx.clone());
            cp_dmx.push(dmx.clone());
            cp_totscale.push(totscale);
            cp_specials.push([x_e, x_n, x_j, x_b, x_c]);
            next_cp_pos += 1;
        }
    }

    let overall_score = if l > 0 && x_c > 0.0 {
        totscale as f32 + (x_c * om.xf[XST_C][XTR_MOVE]).ln()
    } else {
        f32::NEG_INFINITY
    };

    let checkpoint_main = convert_checkpoints_to_log(&cp_mmx, &cp_imx, &cp_dmx, &cp_totscale, q, m);
    let all_specials = convert_specials_to_log(&odds_specials, &per_row_totscale, l);

    let fc = ForwardCheckpoints {
        checkpoint_main,
        checkpoint_indices: checkpoint_indices.clone(),
        all_specials,
        overall_score,
        sequence_length: l,
        num_nodes: m,
    };

    let sd = SIMDCheckpointData {
        cp_mmx,
        cp_imx,
        cp_dmx,
        cp_specials,
        cp_totscale,
        odds_specials,
        per_row_totscale,
        q4: q,
    };

    (fc, sd)
}

fn convert_checkpoints_to_log(
    cp_mmx: &[Vec<f32x4>],
    cp_imx: &[Vec<f32x4>],
    cp_dmx: &[Vec<f32x4>],
    cp_totscale: &[f64],
    q: usize,
    m: usize,
) -> Vec<Array2<f32>> {
    let neg_inf = f32::NEG_INFINITY;
    let mut result = Vec::with_capacity(cp_mmx.len());
    for (cp_idx, ((mm, im), dm)) in cp_mmx
        .iter()
        .zip(cp_imx.iter())
        .zip(cp_dmx.iter())
        .enumerate()
    {
        let offset = cp_totscale[cp_idx] as f32;
        let mut row = Array2::from_elem((m + 1, score_matrix::NUM_MAIN_STATES), neg_inf);
        for qi in 0..q {
            let ma = mm[qi].to_array();
            let ia = im[qi].to_array();
            let da = dm[qi].to_array();
            for z in 0..4 {
                let k = qi + 1 + z * q;
                if k <= m {
                    row[[k, score_matrix::MATCH_CELL]] = if ma[z] > 0.0 {
                        ma[z].ln() + offset
                    } else {
                        neg_inf
                    };
                    row[[k, score_matrix::INSERT_CELL]] = if ia[z] > 0.0 {
                        ia[z].ln() + offset
                    } else {
                        neg_inf
                    };
                    row[[k, score_matrix::DELETE_CELL]] = if da[z] > 0.0 {
                        da[z].ln() + offset
                    } else {
                        neg_inf
                    };
                }
            }
        }
        result.push(row);
    }
    result
}

fn convert_specials_to_log(
    xmx_odds: &[[f32; 5]],
    per_row_totscale: &[f64],
    l: usize,
) -> Array2<f32> {
    let neg_inf = f32::NEG_INFINITY;
    let mut specials = Array2::from_elem((l + 1, score_matrix::NUM_SPECIAL_STATES), neg_inf);
    for i in 0..=l {
        let offset = per_row_totscale[i] as f32;
        for s in 0..5 {
            let v = xmx_odds[i][s];
            specials[[i, s]] = if v > 0.0 { v.ln() + offset } else { neg_inf };
        }
    }
    specials
}

/// SIMD forward segment recomputation using native odds-space checkpoints.
pub fn recompute_forward_segment_simd(
    dsq: &[u8],
    om: &OptimizedProfile,
    checkpoints: &ForwardCheckpoints,
    simd_data: &SIMDCheckpointData,
    cp_index: usize,
    start_row: usize,
    end_row: usize,
) -> Vec<(usize, Array2<f32>)> {
    let q = om.q4;
    let m = om.m;
    let zero_v = f32x4::splat(0.0);
    let neg_inf = f32::NEG_INFINITY;

    // Start from odds-space checkpoint directly
    let mut mmx = simd_data.cp_mmx[cp_index].clone();
    let mut imx = simd_data.cp_imx[cp_index].clone();
    let mut dmx = simd_data.cp_dmx[cp_index].clone();
    let base_totscale = simd_data.cp_totscale[cp_index];

    let mut rows = Vec::with_capacity(end_row - start_row + 1);
    // Convert start row to log-space
    rows.push((start_row, checkpoints.checkpoint_main[cp_index].clone()));

    let mut current_totscale = base_totscale;

    for i in (start_row + 1)..=end_row {
        let xi = dsq[i - 1] as usize;
        let rsc = if xi < om.kp { &om.rfv[xi] } else { &om.rfv[0] };
        let isc = if xi < om.kp { &om.riv[xi] } else { &om.riv[0] };

        // Get x_b from odds-space specials (same scale context)
        let row_totscale = simd_data.per_row_totscale[i - 1];
        let scale_diff = (row_totscale - current_totscale) as f32;
        let x_b_abs = simd_data.odds_specials[i - 1][3]; // B state
        let x_b = x_b_abs * (-scale_diff).exp(); // adjust to current scale context

        let mut dcv = f32x4::splat(0.0);
        let xb_v = f32x4::splat(x_b);

        let mut mpv = shr_f32x4(zero_v, mmx[q - 1]);
        let mut dpv = shr_f32x4(zero_v, dmx[q - 1]);
        let mut ipv = shr_f32x4(zero_v, imx[q - 1]);

        let mut tsc_idx = 0usize;

        for qi in 0..q {
            let rsc_v = f32x4::from_slice(&rsc[qi * 4..qi * 4 + 4]);

            let tbm_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tmm_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tim_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tdm_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;

            let mut sv = xb_v * tbm_v;
            sv += mpv * tmm_v;
            sv += ipv * tim_v;
            sv += dpv * tdm_v;
            sv *= rsc_v;

            mpv = mmx[qi];
            dpv = dmx[qi];
            ipv = imx[qi];

            mmx[qi] = sv;
            dmx[qi] = dcv;

            let tmd_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            dcv = sv * tmd_v;

            let tmi_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tii_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let isc_v = f32x4::from_slice(&isc[qi * 4..qi * 4 + 4]);
            imx[qi] = (mpv * tmi_v + ipv * tii_v) * isc_v;
        }

        // DD passes
        dcv = shr_f32x4(zero_v, dcv);
        dmx[0] = f32x4::splat(0.0);

        let dd_base = NTSC_PER_Q * q;
        for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
            let dd_v = f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
            *dmx_qi += dcv;
            dcv = *dmx_qi * dd_v;
        }
        for _pass in 1..4 {
            dcv = shr_f32x4(zero_v, dcv);
            for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
                let dd_v = f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
                *dmx_qi += dcv;
                dcv *= dd_v;
            }
        }

        // Sparse rescaling to keep values in reasonable range
        let mut x_e_local: f32 = 0.0;
        for qi in 0..q {
            x_e_local += (mmx[qi] + dmx[qi]).reduce_sum();
        }
        if x_e_local > 1.0e4 {
            let inv = 1.0 / x_e_local;
            let inv_v = f32x4::splat(inv);
            for qi in 0..q {
                mmx[qi] *= inv_v;
                dmx[qi] *= inv_v;
                imx[qi] *= inv_v;
            }
            current_totscale += (x_e_local as f64).ln();
        }

        // Convert to log-space row
        let offset = current_totscale as f32;
        let mut row = Array2::from_elem((m + 1, score_matrix::NUM_MAIN_STATES), neg_inf);
        for qi in 0..q {
            let ma = mmx[qi].to_array();
            let ia = imx[qi].to_array();
            let da = dmx[qi].to_array();
            for z in 0..4 {
                let k = qi + 1 + z * q;
                if k <= m {
                    row[[k, score_matrix::MATCH_CELL]] = if ma[z] > 0.0 {
                        ma[z].ln() + offset
                    } else {
                        neg_inf
                    };
                    row[[k, score_matrix::INSERT_CELL]] = if ia[z] > 0.0 {
                        ia[z].ln() + offset
                    } else {
                        neg_inf
                    };
                    row[[k, score_matrix::DELETE_CELL]] = if da[z] > 0.0 {
                        da[z].ln() + offset
                    } else {
                        neg_inf
                    };
                }
            }
        }
        rows.push((i, row));
    }

    rows
}

/// Pre-allocated buffer for odds-space segment replay.
/// Avoids per-row Vec allocations during backward sweep.
#[derive(Debug, Default)]
pub struct OddsSegmentBuf {
    /// Flat storage: [row_offset * q * 4 * 3 + state * q * 4 + qi * 4 + z]
    /// state: 0=M, 1=I, 2=D
    pub data: Vec<f32>,
    pub totscales: Vec<f64>,
    pub q: usize,
    pub num_rows: usize,
    pub start_row: usize,
}

impl OddsSegmentBuf {
    pub fn new(max_segment: usize, q: usize) -> Self {
        let cap = (max_segment + 1) * q * 4 * 3;
        OddsSegmentBuf {
            data: vec![0.0f32; cap],
            totscales: vec![0.0f64; max_segment + 1],
            q,
            num_rows: 0,
            start_row: 0,
        }
    }

    fn ensure_capacity(&mut self, num_rows: usize) {
        let need = num_rows * self.q * 4 * 3;
        if self.data.len() < need {
            self.data.resize(need, 0.0);
        }
        if self.totscales.len() < num_rows {
            self.totscales.resize(num_rows, 0.0);
        }
    }

    #[inline(always)]
    fn row_offset(&self, local_row: usize, state: usize, qi: usize) -> usize {
        (local_row * 3 + state) * self.q * 4 + qi * 4
    }

    fn store_row(
        &mut self,
        local_row: usize,
        mmx: &[f32x4],
        imx: &[f32x4],
        dmx: &[f32x4],
        totscale: f64,
    ) {
        for qi in 0..self.q {
            let m_off = self.row_offset(local_row, 0, qi);
            let i_off = self.row_offset(local_row, 1, qi);
            let d_off = self.row_offset(local_row, 2, qi);
            let ma = mmx[qi].to_array();
            let ia = imx[qi].to_array();
            let da = dmx[qi].to_array();
            self.data[m_off..m_off + 4].copy_from_slice(&ma);
            self.data[i_off..i_off + 4].copy_from_slice(&ia);
            self.data[d_off..d_off + 4].copy_from_slice(&da);
        }
        self.totscales[local_row] = totscale;
    }

    /// Get odds-space M value for node k at local row offset.
    #[inline(always)]
    pub fn m_odds(&self, local_row: usize, k: usize) -> f32 {
        let qi = (k - 1) % self.q;
        let z = (k - 1) / self.q;
        let off = self.row_offset(local_row, 0, qi);
        self.data[off + z]
    }

    /// Get odds-space I value for node k at local row offset.
    #[inline(always)]
    pub fn i_odds(&self, local_row: usize, k: usize) -> f32 {
        let qi = (k - 1) % self.q;
        let z = (k - 1) / self.q;
        let off = self.row_offset(local_row, 1, qi);
        self.data[off + z]
    }

    /// Get totscale for a local row.
    #[inline(always)]
    pub fn totscale(&self, local_row: usize) -> f64 {
        self.totscales[local_row]
    }

    /// De-stripe match and insert odds for a local row into flat arrays.
    ///
    /// Writes `fwd_m[k]` and `fwd_i[k]` for k = 1..=m, where m is the model size.
    /// The flat arrays must have at least m+1 entries (index 0 is untouched).
    #[inline]
    pub fn destripe_row_into(
        &self,
        local_row: usize,
        m: usize,
        fwd_m: &mut [f32],
        fwd_i: &mut [f32],
    ) {
        let q = self.q;
        for qi in 0..q {
            let m_off = self.row_offset(local_row, 0, qi);
            let i_off = self.row_offset(local_row, 1, qi);
            for z in 0..4 {
                let k = qi + 1 + z * q;
                if k <= m {
                    fwd_m[k] = self.data[m_off + z];
                    fwd_i[k] = self.data[i_off + z];
                }
            }
        }
    }

    /// Load match odds as an f32x4 vector for stripe qi at the given local row.
    #[inline(always)]
    pub fn m_odds_vec(&self, local_row: usize, qi: usize) -> f32x4 {
        let off = self.row_offset(local_row, 0, qi);
        f32x4::from_slice(&self.data[off..off + 4])
    }

    /// Load insert odds as an f32x4 vector for stripe qi at the given local row.
    #[inline(always)]
    pub fn i_odds_vec(&self, local_row: usize, qi: usize) -> f32x4 {
        let off = self.row_offset(local_row, 1, qi);
        f32x4::from_slice(&self.data[off..off + 4])
    }
}

/// Replay a segment into a pre-allocated buffer. No heap allocations.
pub fn replay_segment_odds_buf(
    dsq: &[u8],
    om: &OptimizedProfile,
    simd_data: &SIMDCheckpointData,
    cp_index: usize,
    start_row: usize,
    end_row: usize,
    buf: &mut OddsSegmentBuf,
) {
    let q = om.q4;
    let zero_v = f32x4::splat(0.0);
    let num_rows = end_row - start_row + 1;

    buf.ensure_capacity(num_rows);
    buf.num_rows = num_rows;
    buf.start_row = start_row;

    let mut mmx = simd_data.cp_mmx[cp_index].clone();
    let mut imx = simd_data.cp_imx[cp_index].clone();
    let mut dmx = simd_data.cp_dmx[cp_index].clone();
    let base_totscale = simd_data.cp_totscale[cp_index];

    buf.store_row(0, &mmx, &imx, &dmx, base_totscale);

    let mut current_totscale = base_totscale;

    for i in (start_row + 1)..=end_row {
        let xi = dsq[i - 1] as usize;
        let rsc = if xi < om.kp { &om.rfv[xi] } else { &om.rfv[0] };
        let isc = if xi < om.kp { &om.riv[xi] } else { &om.riv[0] };

        let row_totscale = simd_data.per_row_totscale[i - 1];
        let scale_diff = (row_totscale - current_totscale) as f32;
        let x_b_abs = simd_data.odds_specials[i - 1][3];
        let x_b = x_b_abs * (-scale_diff).exp();

        let mut dcv = f32x4::splat(0.0);
        let xb_v = f32x4::splat(x_b);

        let mut mpv = shr_f32x4(zero_v, mmx[q - 1]);
        let mut dpv = shr_f32x4(zero_v, dmx[q - 1]);
        let mut ipv = shr_f32x4(zero_v, imx[q - 1]);

        let mut tsc_idx = 0usize;

        for qi in 0..q {
            let rsc_v = f32x4::from_slice(&rsc[qi * 4..qi * 4 + 4]);
            let tbm_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tmm_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tim_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tdm_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;

            let mut sv = xb_v * tbm_v;
            sv += mpv * tmm_v;
            sv += ipv * tim_v;
            sv += dpv * tdm_v;
            sv *= rsc_v;

            mpv = mmx[qi];
            dpv = dmx[qi];
            ipv = imx[qi];

            mmx[qi] = sv;
            dmx[qi] = dcv;

            let tmd_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            dcv = sv * tmd_v;

            let tmi_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tii_v = f32x4::from_slice(&om.tfv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let isc_v = f32x4::from_slice(&isc[qi * 4..qi * 4 + 4]);
            imx[qi] = (mpv * tmi_v + ipv * tii_v) * isc_v;
        }

        dcv = shr_f32x4(zero_v, dcv);
        dmx[0] = f32x4::splat(0.0);
        let dd_base = NTSC_PER_Q * q;
        for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
            let dd_v = f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
            *dmx_qi += dcv;
            dcv = *dmx_qi * dd_v;
        }
        for _pass in 1..4 {
            dcv = shr_f32x4(zero_v, dcv);
            for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
                let dd_v = f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
                *dmx_qi += dcv;
                dcv *= dd_v;
            }
        }

        let mut x_e_local: f32 = 0.0;
        for qi in 0..q {
            x_e_local += (mmx[qi] + dmx[qi]).reduce_sum();
        }
        if x_e_local > 1.0e4 {
            let inv = 1.0 / x_e_local;
            let inv_v = f32x4::splat(inv);
            for qi in 0..q {
                mmx[qi] *= inv_v;
                dmx[qi] *= inv_v;
                imx[qi] *= inv_v;
            }
            current_totscale += (x_e_local as f64).ln();
        }

        buf.store_row(i - start_row, &mmx, &imx, &dmx, current_totscale);
    }
}
