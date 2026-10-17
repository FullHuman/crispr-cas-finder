// simd/f32_msv.rs — f32x4 MSV filter using portable SIMD
//
// Uses Simd<f32, 4> in log-space (nats). Produces scores identical to
// the generic scalar MSV, but with 4-way SIMD parallelism via Farrar striping.
//
// This avoids the u8 quantization and -3.0 approximation of the integer MSV.

use super::oprofile::OptimizedProfile;
use std::simd::prelude::*;

/// Four single-precision lanes.
pub type F32x4 = Simd<f32, 4>;

/// Number of independent running maxima in the inner loop.
const XE_ACCUMULATORS: usize = 4;

/// Run the f32x4 MSV filter on a digitized sequence.
///
/// `dsq` is 0-indexed: residues at dsq[0..l].
/// Returns the MSV score in nats (same as generic MSV).
pub fn msv_filter_f32(dsq: &[u8], l: usize, om: &OptimizedProfile) -> f32 {
    msv_filter_f32_with(dsq, l, om, &mut Vec::new())
}

/// [`msv_filter_f32`] with a caller-owned DP row, so that a worker searching
/// many sequences does not allocate one per call.
#[inline(never)]
pub fn msv_filter_f32_with(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    m_row: &mut Vec<F32x4>,
) -> f32 {
    let q = om.q4;
    let m = om.m;
    let neg_inf = f32::NEG_INFINITY;

    // MSV-specific transition parameters (same as generic)
    let tloop = (l as f32 / (l as f32 + 3.0)).ln();
    let tmove = (3.0 / (l as f32 + 3.0)).ln();
    let tbmk = (2.0 / ((m * (m + 1)) as f32)).ln();
    let tec = 0.5f32.ln();

    // Single DP row: only M-states
    m_row.clear();
    m_row.resize(q, F32x4::splat(neg_inf));

    // Special states (exact f32)
    let mut xn: f32 = 0.0;
    let mut xb: f32 = tmove;
    let mut xj: f32 = neg_inf;
    let mut xc: f32 = neg_inf;

    // B(i) = max(N(i) + tmove, J(i) + tmove). Until a hit lifts J above N,
    // the first term wins and depends on the row index alone; computing B
    // from it, and only checking the second term, lets the processor start a
    // row before the previous one has finished. When the check fires, B is
    // the maximum of the two terms, exactly as below, and the row-by-row
    // recurrence takes over. The scores are the same either way.
    let mut row = 0;
    while row < l {
        let exit_score = msv_row(m_row, residue_scores(om, dsq[row]), xb + tbmk);
        xj = f32::max(xj + tloop, exit_score + tec);
        xc = f32::max(xc + tloop, exit_score + tec);
        xn += tloop;
        let from_n = xn + tmove;
        let from_j = xj + tmove;
        xb = from_n;
        row += 1;
        if from_j > from_n {
            xb = from_j;
            break;
        }
    }
    for &residue in &dsq[row..l] {
        let exit_score = msv_row(m_row, residue_scores(om, residue), xb + tbmk);
        xj = f32::max(xj + tloop, exit_score + tec);
        xc = f32::max(xc + tloop, exit_score + tec);
        xn += tloop;
        xb = f32::max(xn + tmove, xj + tmove);
    }

    xc + tmove
}

/// Striped match emission log-scores of `residue`.
#[inline(always)]
fn residue_scores(om: &OptimizedProfile, residue: u8) -> &[[f32; 4]] {
    let xi = residue as usize;
    let rsc = if xi < om.kp { &om.rlv[xi] } else { &om.rlv[0] };
    // One [f32; 4] per striped vector; iterating in lockstep with the DP
    // row lets the compiler drop every bounds check from the inner loop.
    rsc.as_chunks::<4>().0
}

/// One row of the recurrence: updates `m_row` in place and returns the
/// maximum of the new match cells.
#[inline(always)]
fn msv_row(m_row: &mut [F32x4], rsc_vectors: &[[f32; 4]], xb_tbmk: f32) -> f32 {
    let q = m_row.len();
    let neg_inf_v = F32x4::splat(f32::NEG_INFINITY);
    let xb_tbmk_v = F32x4::splat(xb_tbmk);

    // Right-shift previous M row by 1 lane (Farrar shift)
    let mut mpv = shr_f32x4(m_row[q - 1]);

    // The row maximum is the only value carried from one vector to the
    // next. Keeping several independent running maxima stops that max
    // from serialising the loop; max is exact, so the result is the same.
    let mut xe_lanes = [neg_inf_v; XE_ACCUMULATORS];
    let (cell_groups, cell_rest) = m_row.as_chunks_mut::<XE_ACCUMULATORS>();
    let (rsc_groups, rsc_rest) = rsc_vectors.as_chunks::<XE_ACCUMULATORS>();
    for (cell_group, rsc_group) in cell_groups.iter_mut().zip(rsc_groups) {
        for index in 0..XE_ACCUMULATORS {
            // M(i,q) = max(M(i-1,q-1), xB+tBMk) + rsc
            let sv = mpv.simd_max(xb_tbmk_v) + F32x4::from_array(rsc_group[index]);
            xe_lanes[index] = xe_lanes[index].simd_max(sv);
            mpv = cell_group[index];
            cell_group[index] = sv;
        }
    }
    for (cell, rsc_vector) in cell_rest.iter_mut().zip(rsc_rest) {
        let sv = mpv.simd_max(xb_tbmk_v) + F32x4::from_array(*rsc_vector);
        xe_lanes[0] = xe_lanes[0].simd_max(sv);
        mpv = *cell;
        *cell = sv;
    }
    xe_lanes
        .into_iter()
        .fold(neg_inf_v, |maximum, lane| maximum.simd_max(lane))
        .reduce_max()
}

/// Right-shift a f32x4 by 1 lane: `[NEG_INF, v[0], v[1], v[2]]`.
#[inline(always)]
fn shr_f32x4(v: Simd<f32, 4>) -> Simd<f32, 4> {
    let fill = Simd::<f32, 4>::splat(f32::NEG_INFINITY);
    simd_swizzle!(v, fill, [4, 0, 1, 2])
}
