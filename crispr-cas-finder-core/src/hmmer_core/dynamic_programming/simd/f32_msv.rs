// simd/f32_msv.rs — f32x4 MSV filter using portable SIMD
//
// Uses Simd<f32, 4> in log-space (nats). Produces scores identical to
// the generic scalar MSV, but with 4-way SIMD parallelism via Farrar striping.
//
// This avoids the u8 quantization and -3.0 approximation of the integer MSV.

use super::oprofile::OptimizedProfile;
use std::simd::prelude::*;

type F32x4 = Simd<f32, 4>;

/// Number of independent running maxima in the inner loop.
const XE_ACCUMULATORS: usize = 4;

/// Run the f32x4 MSV filter on a digitized sequence.
///
/// `dsq` is 0-indexed: residues at dsq[0..l].
/// Returns the MSV score in nats (same as generic MSV).
pub fn msv_filter_f32(dsq: &[u8], l: usize, om: &OptimizedProfile) -> f32 {
    let q = om.q4;
    let m = om.m;
    let neg_inf = f32::NEG_INFINITY;
    let neg_inf_v = F32x4::splat(neg_inf);

    // MSV-specific transition parameters (same as generic)
    let tloop = (l as f32 / (l as f32 + 3.0)).ln();
    let tmove = (3.0 / (l as f32 + 3.0)).ln();
    let tbmk = (2.0 / ((m * (m + 1)) as f32)).ln();
    let tec = 0.5f32.ln();

    // Single DP row: only M-states
    let mut m_row = vec![F32x4::splat(neg_inf); q];

    // Special states (exact f32)
    let mut xn: f32 = 0.0;
    let mut xb: f32 = tmove;
    let mut xj: f32 = neg_inf;
    let mut xc: f32 = neg_inf;

    for &residue in &dsq[..l] {
        let xi = residue as usize;
        let rsc = if xi < om.kp { &om.rlv[xi] } else { &om.rlv[0] };
        // One [f32; 4] per striped vector; iterating in lockstep with the DP
        // row lets the compiler drop every bounds check from the inner loop.
        let (rsc_vectors, _) = rsc.as_chunks::<4>();

        let xb_tbmk_v = F32x4::splat(xb + tbmk);

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
        let xe_v = xe_lanes
            .into_iter()
            .fold(neg_inf_v, |maximum, lane| maximum.simd_max(lane));

        // Special states (exact, matching generic MSV)
        let exit_score = xe_v.reduce_max();
        xj = f32::max(xj + tloop, exit_score + tec);
        xc = f32::max(xc + tloop, exit_score + tec);
        xn += tloop;
        xb = f32::max(xn + tmove, xj + tmove);
    }

    xc + tmove
}

/// Right-shift a f32x4 by 1 lane: `[NEG_INF, v[0], v[1], v[2]]`.
#[inline(always)]
fn shr_f32x4(v: Simd<f32, 4>) -> Simd<f32, 4> {
    let fill = Simd::<f32, 4>::splat(f32::NEG_INFINITY);
    simd_swizzle!(v, fill, [4, 0, 1, 2])
}
