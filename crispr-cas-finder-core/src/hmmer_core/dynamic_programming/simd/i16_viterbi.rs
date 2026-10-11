// simd/i16_viterbi.rs — i16x8 over-estimate of the Viterbi filter score
//
// Same idea as i16_msv.rs, applied to the full match/insert/delete
// recurrence of f32_viterbi.rs: 16-bit cells, eight lanes, every score
// rounded up and every clamp going upward, so the result is never below the
// exact Viterbi filter score. A sequence whose over-estimate fails the
// Viterbi threshold is dropped without running the exact filter.
//
// The N, C and J self-loops share one score, tloop. Subtracting i·tloop from
// every state of row i removes them and leaves delta = -tloop to add for
// each residue a match or insert state consumes. As in i16_msv.rs, cells
// store their value minus the rounded-up cumulative delta D(i); it is
// subtracted from the begin state on entry and added back to the row
// maximum on exit, and one extra unit on entry covers the rounding of that
// difference. Transitions between rows then need no delta at all.

use super::i16_msv::{CEIL_GUARD, LANES, SAFETY_MARGIN, emission_units, units_above};
use super::oprofile::{
    NTSC_PER_Q, OptimizedProfile, TSC_BM, TSC_DM, TSC_II, TSC_IM, TSC_MD, TSC_MI, TSC_MM, XST_C,
    XST_E, XST_J, XST_N, XTR_LOOP, XTR_MOVE,
};
use std::simd::prelude::*;

type I16x8 = Simd<i16, 8>;

/// Round a transition score up to integer units, saturating to i16.
#[inline]
pub fn transition_units(scale: f64, nats: f32) -> i16 {
    emission_units(scale, nats)
}

/// Units for a special-state score, where minus infinity must stay far
/// below every reachable value without overflowing 64-bit sums.
fn special_units(scale: f64, nats: f64) -> i64 {
    if nats == f64::NEG_INFINITY {
        i64::MIN / 4
    } else {
        (scale * nats + CEIL_GUARD).ceil() as i64
    }
}

/// Upper bound, in nats, on the score `viterbi_filter_f32` returns.
///
/// `om` must have been reconfigured for length `l`, as for the exact filter.
/// Returns `f32::INFINITY` when no bound is known: a cell saturated, or the
/// profile's N, C and J loops do not share one score.
pub fn viterbi_upper_bound(dsq: &[u8], l: usize, om: &OptimizedProfile) -> f32 {
    if l == 0 {
        return f32::INFINITY;
    }
    let ln = |probability: f32| f64::from(probability).ln();
    let tloop = ln(om.xf[XST_N][XTR_LOOP]);
    let tmove = ln(om.xf[XST_N][XTR_MOVE]);
    if om.xf[XST_C] != om.xf[XST_N] || om.xf[XST_J] != om.xf[XST_N] || tloop.is_nan() || tloop > 0.0
    {
        return f32::INFINITY;
    }

    let scale = om.msv_scale;
    let q = om.q8;
    let length = l as f64;
    let tmove_units = special_units(scale, tmove);
    let e_loop_units = special_units(scale, ln(om.xf[XST_E][XTR_LOOP]));
    let e_move_units = special_units(scale, ln(om.xf[XST_E][XTR_MOVE]));
    let cumulative_delta = |row: usize| units_above(scale, row as f64 * -tloop);

    // Cells hold units above `base`, offset by i16::MIN. The begin state is
    // never below tmove, so no path enters a row below `base`; saturating at
    // the bottom only raises values that could not win anyway.
    let base = tmove_units + i64::from(om.vit_min_begin_units) - cumulative_delta(l) - 1;
    let floor_v = I16x8::splat(i16::MIN);
    let mut mmx = vec![floor_v; q];
    let mut imx = vec![floor_v; q];
    let mut dmx = vec![floor_v; q];

    let mut xj = i64::MIN / 2;
    let mut xc = i64::MIN / 2;
    let mut xb = tmove_units;

    let (transitions, _) = om.vit_units.as_chunks::<LANES>();
    let (main_transitions, dd_transitions) = transitions.split_at(NTSC_PER_Q * q);
    let (main_transitions, _) = main_transitions.as_chunks::<NTSC_PER_Q>();

    for (row, &residue) in dsq[..l].iter().enumerate() {
        let xi = residue as usize;
        let rsc = if xi < om.kp {
            &om.msv_units[xi]
        } else {
            &om.msv_units[0]
        };
        let (rsc_vectors, _) = rsc.as_chunks::<LANES>();

        // One extra unit covers the shortfall of the cumulative delta.
        let entry = xb + 1 - cumulative_delta(row) - base;
        if entry > 65_535 {
            return f32::INFINITY;
        }
        let xb_v = I16x8::splat((entry.max(0) - 32_768) as i16);

        let mut dcv = floor_v;
        let mut xe_v = floor_v;

        // Right-shift previous row by 1 lane (Farrar shift)
        let mut mpv = shift_right(mmx[q - 1]);
        let mut dpv = shift_right(dmx[q - 1]);
        let mut ipv = shift_right(imx[q - 1]);

        for (qi, (tsc, rsc_vector)) in main_transitions.iter().zip(rsc_vectors).enumerate() {
            let t = |kind: usize| I16x8::from_array(tsc[kind]);

            // M(i,q) = max(B+tBM, M(i-1,q-1)+tMM, I(i-1,q-1)+tIM, D(i-1,q-1)+tDM) + rsc
            let mut sv = xb_v
                .saturating_add(t(TSC_BM))
                .simd_max(mpv.saturating_add(t(TSC_MM)));
            sv = sv.simd_max(ipv.saturating_add(t(TSC_IM)));
            sv = sv.simd_max(dpv.saturating_add(t(TSC_DM)));
            sv = sv.saturating_add(I16x8::from_array(*rsc_vector));

            xe_v = xe_v.simd_max(sv);

            mpv = mmx[qi];
            dpv = dmx[qi];
            ipv = imx[qi];

            mmx[qi] = sv;
            dmx[qi] = dcv;

            // D carry: M(i,k) + tMD
            dcv = sv.saturating_add(t(TSC_MD));

            // I(i,q) = max(M(i-1,q)+tMI, I(i-1,q)+tII)
            imx[qi] = mpv
                .saturating_add(t(TSC_MI))
                .simd_max(ipv.saturating_add(t(TSC_II)));
        }

        // D->D propagation across stripes. One full pass, then passes that
        // stop at the first vector the carry does not improve; a lane can
        // need as many passes as there are lanes.
        dcv = shift_right(dcv);
        for (dmx_q, dd) in dmx.iter_mut().zip(dd_transitions) {
            *dmx_q = (*dmx_q).simd_max(dcv);
            xe_v = xe_v.simd_max(*dmx_q);
            dcv = (*dmx_q).saturating_add(I16x8::from_array(*dd));
        }
        for _pass in 1..LANES {
            dcv = shift_right(dcv);
            let mut reached_end = true;
            for (dmx_q, dd) in dmx.iter_mut().zip(dd_transitions) {
                if !dcv.simd_gt(*dmx_q).any() {
                    reached_end = false;
                    break;
                }
                *dmx_q = (*dmx_q).simd_max(dcv);
                xe_v = xe_v.simd_max(*dmx_q);
                dcv = (*dmx_q).saturating_add(I16x8::from_array(*dd));
            }
            if !reached_end {
                break;
            }
        }

        let xe = xe_v.reduce_max();
        if xe == i16::MAX {
            return f32::INFINITY;
        }

        let exit = base + i64::from(xe) + 32_768 + cumulative_delta(row + 1);
        xj = xj.max(exit + e_loop_units);
        xc = xc.max(exit + e_move_units);
        xb = xj.max(0) + tmove_units;
    }

    let bound = xc as f64 / scale + length * tloop + tmove + SAFETY_MARGIN;
    // Round toward +inf when narrowing to f32.
    let narrowed = bound as f32;
    if f64::from(narrowed) < bound {
        narrowed.next_up()
    } else {
        narrowed
    }
}

/// Right-shift an i16x8 by 1 lane, filling with the floor value.
#[inline(always)]
fn shift_right(v: I16x8) -> I16x8 {
    let fill = I16x8::splat(i16::MIN);
    simd_swizzle!(v, fill, [8, 0, 1, 2, 3, 4, 5, 6])
}
