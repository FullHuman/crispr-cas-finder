// simd/f32_msv.rs — f32x4 MSV filter using portable SIMD
//
// Uses Simd<f32, 4> in log-space (nats). Produces scores identical to
// the generic scalar MSV, but with 4-way SIMD parallelism via Farrar striping.
//
// This avoids the u8 quantization and -3.0 approximation of the integer MSV.

use super::oprofile::OptimizedProfile;
use std::simd::prelude::*;

type F32x4 = Simd<f32, 4>;

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

        let xb_tbmk_v = F32x4::splat(xb + tbmk);
        let mut xe_v = neg_inf_v;

        // Right-shift previous M row by 1 lane (Farrar shift)
        let mut mpv = shr_f32x4(m_row[q - 1]);

        for qi in 0..q {
            let rsc_v = F32x4::from_slice(&rsc[qi * 4..qi * 4 + 4]);

            // M(i,q) = max(M(i-1,q-1), xB+tBMk) + rsc
            let mut sv = mpv.simd_max(xb_tbmk_v);
            sv += rsc_v;

            xe_v = xe_v.simd_max(sv);

            // Save prev, overwrite
            mpv = m_row[qi];
            m_row[qi] = sv;
        }

        // Special states (exact, matching generic MSV)
        let exit_score = xe_v.reduce_max();
        xj = f32::max(xj + tloop, exit_score + tec);
        xc = f32::max(xc + tloop, exit_score + tec);
        xn += tloop;
        xb = f32::max(xn + tmove, xj + tmove);
    }

    xc + tmove
}

/// Right-shift a f32x4 by 1 lane: [NEG_INF, v[0], v[1], v[2]].
#[inline(always)]
fn shr_f32x4(v: Simd<f32, 4>) -> Simd<f32, 4> {
    let fill = Simd::<f32, 4>::splat(f32::NEG_INFINITY);
    simd_swizzle!(v, fill, [4, 0, 1, 2])
}
