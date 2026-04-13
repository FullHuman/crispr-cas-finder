// simd/f32_viterbi.rs — f32x4 Viterbi filter using portable SIMD
//
// Uses Simd<f32, 4> in log-space (nats). Produces scores identical to
// the generic Viterbi, but with 4-way SIMD parallelism via Farrar striping.
//
// Full 3-state DP (M, I, D) with lazy-F-style DD propagation.

use super::oprofile::*;
use std::simd::prelude::*;

type F32x4 = Simd<f32, 4>;

/// Run the f32x4 Viterbi filter on a digitized sequence.
///
/// `dsq` is 0-indexed: residues at dsq[0..l].
/// Returns the Viterbi score in nats (same as generic Viterbi).
pub fn viterbi_filter_f32(dsq: &[u8], l: usize, om: &OptimizedProfile) -> f32 {
    let q = om.q4;
    let neg_inf = f32::NEG_INFINITY;
    let neg_inf_v = F32x4::splat(neg_inf);

    let ss = &om.xf; // We'll use the profile's special_scores directly below

    // Single DP row: M, I, D states
    let mut mmx = vec![neg_inf_v; q];
    let mut imx = vec![neg_inf_v; q];
    let mut dmx = vec![neg_inf_v; q];

    // Special states (exact f32, matching generic Viterbi)
    let mut xn: f32 = 0.0;
    let mut xb: f32;
    let mut xj: f32 = neg_inf;
    let mut xc: f32 = neg_inf;

    // Read special scores from xf (stored as exp() probabilities, convert back to log nats)
    let e_loop = ss[XST_E][XTR_LOOP].ln();
    let e_move_f = ss[XST_E][XTR_MOVE].ln();
    let n_loop_f = ss[XST_N][XTR_LOOP].ln();
    let n_move_f = ss[XST_N][XTR_MOVE].ln();
    let c_loop_f = ss[XST_C][XTR_LOOP].ln();
    let c_move_f = ss[XST_C][XTR_MOVE].ln();
    let j_loop_f = ss[XST_J][XTR_LOOP].ln();
    let j_move_f = ss[XST_J][XTR_MOVE].ln();

    xb = n_move_f; // B(0) = N(0) + n_move = 0 + n_move

    for &residue in &dsq[..l] {
        let xi = residue as usize;
        let rsc = if xi < om.kp { &om.rlv[xi] } else { &om.rlv[0] };

        let mut dcv = neg_inf_v;
        let mut xe_v = neg_inf_v;
        let xb_v = F32x4::splat(xb);

        // Right-shift previous row by 1 lane (Farrar shift)
        let mut mpv = shr_f32x4(mmx[q - 1]);
        let mut dpv = shr_f32x4(dmx[q - 1]);
        let mut ipv = shr_f32x4(imx[q - 1]);

        let mut tsc_idx = 0usize;

        for qi in 0..q {
            let rsc_v = F32x4::from_slice(&rsc[qi * 4..qi * 4 + 4]);

            // Transitions (log-space)
            let tbm_v = F32x4::from_slice(&om.tlv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tmm_v = F32x4::from_slice(&om.tlv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tim_v = F32x4::from_slice(&om.tlv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tdm_v = F32x4::from_slice(&om.tlv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;

            // M(i,q) = max(B+tBM, M(i-1,q-1)+tMM, I(i-1,q-1)+tIM, D(i-1,q-1)+tDM) + rsc
            let mut sv = (xb_v + tbm_v).simd_max(mpv + tmm_v);
            sv = sv.simd_max(ipv + tim_v);
            sv = sv.simd_max(dpv + tdm_v);
            sv += rsc_v;

            // Update E from M-states (local mode: all M positions exit to E)
            xe_v = xe_v.simd_max(sv);

            // Save prev, overwrite
            mpv = mmx[qi];
            dpv = dmx[qi];
            ipv = imx[qi];

            mmx[qi] = sv;
            dmx[qi] = dcv;

            // D carry: M(i,k) + tMD
            let tmd_v = F32x4::from_slice(&om.tlv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            dcv = sv + tmd_v;

            // I(i,q) = max(M(i-1,q)+tMI, I(i-1,q)+tII)
            let tmi_v = F32x4::from_slice(&om.tlv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            let tii_v = F32x4::from_slice(&om.tlv[tsc_idx * 4..tsc_idx * 4 + 4]);
            tsc_idx += 1;
            imx[qi] = (mpv + tmi_v).simd_max(ipv + tii_v);
        }

        // DD propagation: shift dcv and propagate D->D across stripes
        dcv = shr_f32x4(dcv);
        dmx[0] = neg_inf_v; // D(i, k=1) = -inf always

        let dd_base = NTSC_PER_Q * q;
        for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
            let dd_v = F32x4::from_slice(&om.tlv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
            *dmx_qi = (*dmx_qi).simd_max(dcv);
            dcv = *dmx_qi + dd_v;
        }

        // Additional DD passes for cross-stripe propagation
        for _pass in 1..4 {
            dcv = shr_f32x4(dcv);
            let mut any_change = false;
            for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
                let dd_v = F32x4::from_slice(&om.tlv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
                let old = *dmx_qi;
                let new_val = old.simd_max(dcv);
                if new_val.simd_gt(old).any() {
                    any_change = true;
                }
                *dmx_qi = new_val;
                dcv = new_val + dd_v;
            }
            if !any_change {
                break;
            }
        }

        // Add D contributions to xE (local mode: D exits to E too)
        for &v in &dmx {
            xe_v = xe_v.simd_max(v);
        }

        let exit_score = xe_v.reduce_max();

        // Special states (exact f32, matching generic)
        xj = f32::max(xj + j_loop_f, exit_score + e_loop);
        xc = f32::max(xc + c_loop_f, exit_score + e_move_f);
        xn += n_loop_f;
        xb = f32::max(xn + n_move_f, xj + j_move_f);
    }

    xc + c_move_f
}

/// Right-shift a f32x4 by 1 lane: [NEG_INF, v[0], v[1], v[2]].
#[inline(always)]
fn shr_f32x4(v: Simd<f32, 4>) -> Simd<f32, 4> {
    let fill = Simd::<f32, 4>::splat(f32::NEG_INFINITY);
    simd_swizzle!(v, fill, [4, 0, 1, 2])
}
