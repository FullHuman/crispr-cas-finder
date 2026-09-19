// simd/fwd_bck.rs — optional SIMD checkpointed Forward + segment replay
//
// Provides an odds-space f32x4 Farrar-striped checkpoint/replay implementation.
// The current production posterior decoder uses a generic log-space Forward/
// Backward implementation; these utilities remain useful for verification,
// benchmarking, and a future memory-bounded SIMD Backward decoder.

use super::oprofile::*;
use crate::hmmer_core::dynamic_programming::forward_backward::ForwardCheckpoints;
use crate::hmmer_core::dynamic_programming::score_matrix;
use ndarray::Array2;
use std::simd::prelude::*;

// One initial DD pass plus three wraparound passes propagate a delete chain
// through all four SIMD lanes, matching HMMER 3.4's SSE Forward algorithm.
const DD_PROPAGATION_PASSES: usize = 4;

/// Shift right by one lane: `[fill[3], v[0], v[1], v[2]]`.
#[inline(always)]
fn shr_f32x4(fill: f32x4, v: f32x4) -> f32x4 {
    use std::simd::simd_swizzle;
    simd_swizzle!(fill, v, [3, 4, 5, 6])
}

/// Shift left by one lane: `[v[1], v[2], v[3], fill[0]]`.
#[inline(always)]
pub fn shl_f32x4(v: f32x4, fill: f32x4) -> f32x4 {
    use std::simd::simd_swizzle;
    simd_swizzle!(v, fill, [1, 2, 3, 4])
}

/// Odds-space SIMD checkpoint data for fast segment replay.
///
/// Stored alongside `ForwardCheckpoints` for SIMD replay in native odds space.
pub struct SIMDCheckpointData {
    pub cp_mmx: Vec<Vec<f32x4>>,
    pub cp_imx: Vec<Vec<f32x4>>,
    pub cp_dmx: Vec<Vec<f32x4>>,
    pub cp_totscale: Vec<f64>, // cumulative log-scale at each checkpoint
    pub odds_specials: Vec<[f32; 5]>, // [E, N, J, B, C] per row 0..=L
    /// Cumulative log scale for each row. Segment replay needs this to put
    /// stored B-state odds and a replayed main-state row on the same scale.
    pub per_row_totscale: Vec<f64>,
    pub q4: usize,
}

fn checkpoint_interval(sequence_length: usize) -> usize {
    if sequence_length == 0 {
        1
    } else {
        ((sequence_length as f64).sqrt().ceil() as usize).max(1)
    }
}

fn checkpoint_indices(sequence_length: usize) -> Vec<usize> {
    let interval = checkpoint_interval(sequence_length);
    let mut indices: Vec<usize> = (0..=sequence_length).step_by(interval).collect();
    if *indices.last().expect("row zero is always checkpointed") != sequence_length {
        indices.push(sequence_length);
    }
    indices
}

/// Run SIMD forward in odds-space with checkpoint storage.
///
/// Returns log-space reconstructed checkpoints plus the odds-space state
/// required to resume fast segment replay.
pub fn forward_checkpointed_simd(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
) -> (ForwardCheckpoints, SIMDCheckpointData) {
    assert!(
        l <= dsq.len(),
        "requested Forward length {l} exceeds sequence length {}",
        dsq.len()
    );
    let q = om.q4;
    let m = om.m;
    let zero_v = f32x4::splat(0.0);

    // With s = ceil(sqrt(L)), both the number of checkpoints and the maximum
    // rows in a replayed segment are <= s + 1, for O(M*sqrt(L)) storage.
    let checkpoint_indices = checkpoint_indices(l);
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

    // Save row 0 checkpoint
    cp_mmx.push(mmx.clone());
    cp_imx.push(imx.clone());
    cp_dmx.push(dmx.clone());
    cp_totscale.push(0.0);
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

        let score_len = q * 4;
        let transition_len = q * NTSC_PER_Q * 4;
        for (qi, ((rsc4, isc4), tsc)) in rsc[..score_len]
            .as_chunks::<4>()
            .0
            .iter()
            .zip(isc[..score_len].as_chunks::<4>().0.iter())
            .zip(
                om.tfv[..transition_len]
                    .as_chunks::<{ NTSC_PER_Q * 4 }>()
                    .0
                    .iter(),
            )
            .enumerate()
        {
            let rsc_v = f32x4::from_slice(rsc4);
            let tbm_v = f32x4::from_slice(&tsc[0..4]);
            let tmm_v = f32x4::from_slice(&tsc[4..8]);
            let tim_v = f32x4::from_slice(&tsc[8..12]);
            let tdm_v = f32x4::from_slice(&tsc[12..16]);

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

            let tmd_v = f32x4::from_slice(&tsc[16..20]);
            dcv = sv * tmd_v;

            let tmi_v = f32x4::from_slice(&tsc[20..24]);
            let tii_v = f32x4::from_slice(&tsc[24..28]);
            let isc_v = f32x4::from_slice(isc4);
            imx[qi] = (mpv * tmi_v + ipv * tii_v) * isc_v;
        }

        // DD passes
        dcv = shr_f32x4(zero_v, dcv);
        dmx[0] = f32x4::splat(0.0);

        let dd_base = NTSC_PER_Q * q;
        let dd_tfv = &om.tfv[dd_base * 4..(dd_base + q) * 4];
        for (dmx_qi, dd) in dmx.iter_mut().zip(dd_tfv.as_chunks::<4>().0.iter()) {
            let dd_v = f32x4::from_slice(dd);
            *dmx_qi += dcv;
            dcv = *dmx_qi * dd_v;
        }

        if om.m < 100 {
            for _pass in 1..DD_PROPAGATION_PASSES {
                dcv = shr_f32x4(zero_v, dcv);
                for (dmx_qi, dd) in dmx.iter_mut().zip(dd_tfv.as_chunks::<4>().0.iter()) {
                    let dd_v = f32x4::from_slice(dd);
                    *dmx_qi += dcv;
                    dcv *= dd_v;
                }
            }
        } else {
            for _pass in 1..DD_PROPAGATION_PASSES {
                dcv = shr_f32x4(zero_v, dcv);
                let mut any_change = false;
                for (dmx_qi, dd) in dmx.iter_mut().zip(dd_tfv.as_chunks::<4>().0.iter()) {
                    let dd_v = f32x4::from_slice(dd);
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
        let row_data = row
            .as_slice_mut()
            .expect("newly allocated Array2 is contiguous");
        for qi in 0..q {
            let ma = mm[qi].to_array();
            let ia = im[qi].to_array();
            let da = dm[qi].to_array();
            for z in 0..4 {
                let k = qi + 1 + z * q;
                if k <= m {
                    let base = k * score_matrix::NUM_MAIN_STATES;
                    row_data[base + score_matrix::MATCH_CELL] = if ma[z] > 0.0 {
                        ma[z].ln() + offset
                    } else {
                        neg_inf
                    };
                    row_data[base + score_matrix::INSERT_CELL] = if ia[z] > 0.0 {
                        ia[z].ln() + offset
                    } else {
                        neg_inf
                    };
                    row_data[base + score_matrix::DELETE_CELL] = if da[z] > 0.0 {
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
    let data = specials
        .as_slice_mut()
        .expect("newly allocated Array2 is contiguous");
    for i in 0..=l {
        let offset = per_row_totscale[i] as f32;
        let base = i * score_matrix::NUM_SPECIAL_STATES;
        for s in 0..5 {
            let v = xmx_odds[i][s];
            data[base + s] = if v > 0.0 { v.ln() + offset } else { neg_inf };
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
    assert!(start_row <= end_row, "segment start follows segment end");
    assert!(
        end_row <= dsq.len(),
        "segment end {end_row} exceeds sequence length {}",
        dsq.len()
    );
    assert_eq!(
        simd_data.q4, om.q4,
        "checkpoint stripe count does not match optimized profile"
    );
    assert_eq!(
        checkpoints.checkpoint_indices.get(cp_index),
        Some(&start_row),
        "checkpoint index does not identify the requested segment start"
    );
    assert!(
        cp_index < simd_data.cp_mmx.len()
            && cp_index < simd_data.cp_imx.len()
            && cp_index < simd_data.cp_dmx.len()
            && cp_index < simd_data.cp_totscale.len(),
        "checkpoint index {cp_index} is missing SIMD state"
    );
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
        let x_b = if scale_diff == 0.0 {
            x_b_abs
        } else {
            x_b_abs * (-scale_diff).exp()
        };

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
        for _pass in 1..DD_PROPAGATION_PASSES {
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
/// Avoids per-row allocations across repeated segment replays.
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
        assert!(q > 0, "segment replay requires at least one SIMD stripe");
        let cap = (max_segment + 1) * q * 4 * 3;
        OddsSegmentBuf {
            data: vec![0.0f32; cap],
            totscales: vec![0.0f64; max_segment + 1],
            q,
            num_rows: 0,
            start_row: 0,
        }
    }

    fn ensure_capacity(&mut self, num_rows: usize, q: usize) {
        assert!(q > 0, "segment replay requires at least one SIMD stripe");
        self.q = q;
        self.num_rows = num_rows;
        let need = num_rows * q * 4 * 3;
        if self.data.len() < need {
            self.data.resize(need, 0.0);
        }
        if self.totscales.len() < num_rows {
            self.totscales.resize(num_rows, 0.0);
        }
    }

    #[inline(always)]
    fn row_offset(&self, local_row: usize, state: usize, qi: usize) -> usize {
        assert!(
            local_row < self.num_rows,
            "local row {local_row} is outside replay buffer with {} rows",
            self.num_rows
        );
        assert!(state < 3, "main-state index {state} is outside 0..3");
        assert!(
            qi < self.q,
            "stripe index {qi} is outside replay buffer with {} stripes",
            self.q
        );
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
        assert_eq!(mmx.len(), self.q, "match row has the wrong stripe count");
        assert_eq!(imx.len(), self.q, "insert row has the wrong stripe count");
        assert_eq!(dmx.len(), self.q, "delete row has the wrong stripe count");
        assert!(
            local_row < self.num_rows,
            "local row {local_row} is outside replay buffer with {} rows",
            self.num_rows
        );
        let state_width = self.q * 4;
        let row_start = local_row * 3 * state_width;
        let row_data = &mut self.data[row_start..row_start + 3 * state_width];
        let (m_data, rest) = row_data.split_at_mut(state_width);
        let (i_data, d_data) = rest.split_at_mut(state_width);

        for (dst, &src) in m_data.as_chunks_mut::<4>().0.iter_mut().zip(mmx.iter()) {
            dst.copy_from_slice(&src.to_array());
        }
        for (dst, &src) in i_data.as_chunks_mut::<4>().0.iter_mut().zip(imx.iter()) {
            dst.copy_from_slice(&src.to_array());
        }
        for (dst, &src) in d_data.as_chunks_mut::<4>().0.iter_mut().zip(dmx.iter()) {
            dst.copy_from_slice(&src.to_array());
        }

        self.totscales[local_row] = totscale;
    }

    /// Get odds-space M value for node k at local row offset.
    #[inline(always)]
    pub fn m_odds(&self, local_row: usize, k: usize) -> f32 {
        assert!(
            (1..=self.q * 4).contains(&k),
            "model node {k} is outside replay buffer capacity 1..={}",
            self.q * 4
        );
        let qi = (k - 1) % self.q;
        let z = (k - 1) / self.q;
        let off = self.row_offset(local_row, 0, qi);
        self.data[off + z]
    }

    /// Get odds-space I value for node k at local row offset.
    #[inline(always)]
    pub fn i_odds(&self, local_row: usize, k: usize) -> f32 {
        assert!(
            (1..=self.q * 4).contains(&k),
            "model node {k} is outside replay buffer capacity 1..={}",
            self.q * 4
        );
        let qi = (k - 1) % self.q;
        let z = (k - 1) / self.q;
        let off = self.row_offset(local_row, 1, qi);
        self.data[off + z]
    }

    /// Get totscale for a local row.
    #[inline(always)]
    pub fn totscale(&self, local_row: usize) -> f64 {
        assert!(
            local_row < self.num_rows,
            "local row {local_row} is outside replay buffer with {} rows",
            self.num_rows
        );
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
        assert!(
            m <= self.q * 4,
            "model length exceeds replay buffer stripes"
        );
        assert!(fwd_m.len() > m, "match output must contain m + 1 entries");
        assert!(fwd_i.len() > m, "insert output must contain m + 1 entries");
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
    assert!(start_row <= end_row, "segment start follows segment end");
    assert!(
        end_row <= dsq.len(),
        "segment end {end_row} exceeds sequence length {}",
        dsq.len()
    );
    assert_eq!(
        simd_data.q4, om.q4,
        "checkpoint stripe count does not match optimized profile"
    );
    assert!(
        cp_index < simd_data.cp_mmx.len()
            && cp_index < simd_data.cp_imx.len()
            && cp_index < simd_data.cp_dmx.len()
            && cp_index < simd_data.cp_totscale.len(),
        "checkpoint index {cp_index} is missing SIMD state"
    );
    let q = om.q4;
    let zero_v = f32x4::splat(0.0);
    let num_rows = end_row - start_row + 1;

    buf.ensure_capacity(num_rows, q);
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
        let x_b = if scale_diff == 0.0 {
            x_b_abs
        } else {
            x_b_abs * (-scale_diff).exp()
        };

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
        for _pass in 1..DD_PROPAGATION_PASSES {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmmer_core::alphabet::Alphabet;
    use crate::hmmer_core::background::BackgroundModel;
    use crate::hmmer_core::config::SearchMode;
    use crate::hmmer_core::profile::Profile;
    use crate::hmmer_core::rng::XorShift64;
    use crate::hmmer_core::test_helpers::{hmm_sample, random_digital_seq};

    #[test]
    fn lane_shifts_have_explicit_direction_and_fill_lane() {
        let values = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
        let fill = f32x4::from_array([10.0, 20.0, 30.0, 40.0]);

        assert_eq!(shr_f32x4(fill, values).to_array(), [40.0, 1.0, 2.0, 3.0]);
        assert_eq!(shl_f32x4(values, fill).to_array(), [2.0, 3.0, 4.0, 10.0]);
    }

    #[test]
    fn checkpoint_schedule_has_sqrt_space_bounds() {
        for sequence_length in [0, 1, 2, 3, 15, 16, 17, 1_000, 1_000_000] {
            let interval = checkpoint_interval(sequence_length);
            let indices = checkpoint_indices(sequence_length);

            assert_eq!(indices.first(), Some(&0));
            assert_eq!(indices.last(), Some(&sequence_length));
            assert!(indices.len() <= interval + 1);
            assert!(indices.windows(2).all(|rows| rows[1] - rows[0] <= interval));
        }
    }

    #[test]
    fn odds_segment_buffer_reconfigures_and_destripes_rows() {
        let mut buffer = OddsSegmentBuf::default();
        buffer.ensure_capacity(2, 2);
        let matches = [
            f32x4::from_array([1.0, 3.0, 5.0, 7.0]),
            f32x4::from_array([2.0, 4.0, 6.0, 8.0]),
        ];
        let inserts = [
            f32x4::from_array([11.0, 13.0, 15.0, 17.0]),
            f32x4::from_array([12.0, 14.0, 16.0, 18.0]),
        ];
        let deletes = [f32x4::splat(0.0); 2];
        buffer.store_row(1, &matches, &inserts, &deletes, 2.5);

        let mut flat_matches = vec![0.0; 9];
        let mut flat_inserts = vec![0.0; 9];
        buffer.destripe_row_into(1, 8, &mut flat_matches, &mut flat_inserts);

        assert_eq!(
            &flat_matches[1..],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            &flat_inserts[1..],
            &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
        );
        assert_eq!(buffer.m_odds(1, 8), 8.0);
        assert_eq!(buffer.i_odds(1, 8), 18.0);
        assert_eq!(buffer.m_odds_vec(1, 0).to_array(), [1.0, 3.0, 5.0, 7.0]);
        assert_eq!(buffer.i_odds_vec(1, 1).to_array(), [12.0, 14.0, 16.0, 18.0]);
        assert_eq!(buffer.totscale(1), 2.5);

        buffer.ensure_capacity(1, 3);
        assert_eq!(buffer.q, 3);
        assert_eq!(buffer.num_rows, 1);
        assert!(buffer.data.len() >= 3 * 4 * 3);
    }

    #[test]
    #[should_panic(expected = "local row 1 is outside replay buffer with 1 rows")]
    fn odds_segment_buffer_reports_invalid_row() {
        let mut buffer = OddsSegmentBuf::default();
        buffer.ensure_capacity(1, 1);
        let _ = buffer.m_odds(1, 1);
    }

    #[test]
    fn buffered_replay_matches_log_space_segment_recomputation() {
        let alphabet = Alphabet::amino();
        let mut rng = XorShift64::new(91);
        let background = BackgroundModel::new(&alphabet);
        let sequence_length = 17;
        let hmm = hmm_sample(&mut rng, 9, &alphabet);
        let mut profile = Profile::new(hmm.num_nodes, &alphabet);
        crate::hmmer_core::modelconfig::profile_config(
            &hmm,
            &background,
            &mut profile,
            sequence_length,
            SearchMode::Local,
        );
        let optimized = OptimizedProfile::from_profile(&profile);
        let sequence = random_digital_seq(
            &mut rng,
            &background.residue_frequencies,
            alphabet.canonical_size,
            sequence_length,
        );
        let (checkpoints, simd_data) =
            forward_checkpointed_simd(&sequence, sequence_length, &optimized);
        let cp_index = 0;
        let start_row = checkpoints.checkpoint_indices[cp_index];
        let end_row = checkpoints.checkpoint_indices[cp_index + 1];
        let expected = recompute_forward_segment_simd(
            &sequence,
            &optimized,
            &checkpoints,
            &simd_data,
            cp_index,
            start_row,
            end_row,
        );
        let mut buffer = OddsSegmentBuf::default();

        replay_segment_odds_buf(
            &sequence,
            &optimized,
            &simd_data,
            cp_index,
            start_row,
            end_row,
            &mut buffer,
        );

        assert_eq!(buffer.q, optimized.q4);
        assert_eq!(buffer.num_rows, expected.len());
        for (local_row, (_, expected_row)) in expected.iter().enumerate() {
            let scale = buffer.totscale(local_row) as f32;
            for node in 1..=profile.num_nodes {
                for (odds, state) in [
                    (buffer.m_odds(local_row, node), score_matrix::MATCH_CELL),
                    (buffer.i_odds(local_row, node), score_matrix::INSERT_CELL),
                ] {
                    let actual = if odds > 0.0 {
                        odds.ln() + scale
                    } else {
                        f32::NEG_INFINITY
                    };
                    let wanted = expected_row[[node, state]];
                    assert!(
                        (actual - wanted).abs() < 1e-4
                            || (actual.is_infinite() && wanted.is_infinite()),
                        "row={local_row}, node={node}, state={state}: {actual} != {wanted}"
                    );
                }
            }
        }
    }
}
