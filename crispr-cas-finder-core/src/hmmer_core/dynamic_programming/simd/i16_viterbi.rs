// simd/i16_viterbi.rs — 16-bit over-estimate of the Viterbi filter score
//
// Same idea as i16_msv.rs, applied to the full match/insert/delete
// recurrence of f32_viterbi.rs: 16-bit cells, every score rounded up and
// every clamp going upward, so the result is never below the exact Viterbi
// filter score. A sequence whose over-estimate fails the Viterbi threshold
// is dropped without running the exact filter. By default a finite
// over-estimate also stands in for the score of a sequence that passes; the
// exact filter runs when the bound saturates and, for every sequence the
// bound lets through, with `FilterPolicy::exact`.
//
// The N, C and J self-loops share one score, tloop. Subtracting i·tloop from
// every state of row i removes them and leaves delta = -tloop to add for
// each residue a match or insert state consumes. As in i16_msv.rs, cells
// store their value minus the rounded-up cumulative delta D(i); it is
// subtracted from the begin state on entry and added back to the row
// maximum on exit, and one extra unit on entry covers the rounding of that
// difference. Transitions between rows then need no delta at all.
//
// As for the MSV bounds (int_msv.rs), the kernel runs with the widest
// vectors the processor has, eight to thirty-two lanes, compiled once per
// instruction set; the result does not depend on the striping.

use super::i16_msv::{CEIL_GUARD, SAFETY_MARGIN, emission_units};
#[cfg(target_arch = "x86_64")]
use super::int_msv::{Avx2Kernels, Avx512Kernels, Sse41Kernels};
use super::int_msv::{DELTA_FRACTION_BITS, Isa, Kernels, PortableKernels};
use super::oprofile::{
    NTSC_PER_Q, OptimizedProfile, TSC_BM, TSC_DM, TSC_II, TSC_IM, TSC_MD, TSC_MI, TSC_MM, XST_C,
    XST_E, XST_J, XST_N, XTR_LOOP, XTR_MOVE,
};
use std::simd::prelude::*;

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

/// DP rows of the Viterbi over-estimate, kept by a worker across sequences:
/// match, insert and delete cells, vector by vector.
#[derive(Debug, Default)]
pub struct Rows {
    mmx: Vec<i16>,
    imx: Vec<i16>,
    dmx: Vec<i16>,
}

/// Upper bound, in nats, on the score `viterbi_filter_f32` returns.
///
/// `om` must have been reconfigured for length `l`, as for the exact filter.
/// Returns `f32::INFINITY` when no bound is known: a cell saturated, or the
/// profile's N, C and J loops do not share one score.
pub fn viterbi_upper_bound(dsq: &[u8], l: usize, om: &OptimizedProfile) -> f32 {
    viterbi_upper_bound_with(dsq, l, om, &mut Rows::default())
}

/// [`viterbi_upper_bound`] with caller-owned DP rows, so that a worker
/// searching many sequences does not allocate them per call.
#[inline(never)]
pub fn viterbi_upper_bound_with(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    rows: &mut Rows,
) -> f32 {
    // The kernels of an instruction set run only where it is available,
    // which `OptimizedProfile` guarantees of the one it records (see
    // `int_msv::Kernels`).
    match om.isa() {
        Isa::Portable => {
            portable::viterbi_upper_bound::<{ Isa::Portable.lanes::<i16>() }>(dsq, l, om, rows)
        }
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => unsafe {
            sse41::viterbi_upper_bound::<{ Isa::Sse41.lanes::<i16>() }>(dsq, l, om, rows)
        },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe {
            avx2::viterbi_upper_bound::<{ Isa::Avx2.lanes::<i16>() }>(dsq, l, om, rows)
        },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe {
            avx512::viterbi_upper_bound::<{ Isa::Avx512.lanes::<i16>() }>(dsq, l, om, rows)
        },
    }
}

/// The kernel compiled for one instruction set.
macro_rules! isa_kernel {
    ($(#[$meta:meta])* $module:ident, $kernels:ty $(, $feature:literal)*) => {
        $(#[$meta])*
        mod $module {
            use super::*;

            $(#[target_feature(enable = $feature)])*
            pub(super) fn viterbi_upper_bound<const N: usize>(
                dsq: &[u8],
                l: usize,
                om: &OptimizedProfile,
                rows: &mut Rows,
            ) -> f32 {
                viterbi_upper_bound_impl::<N, $kernels>(dsq, l, om, rows)
            }
        }
    };
}

isa_kernel!(portable, PortableKernels);
isa_kernel!(
    #[cfg(target_arch = "x86_64")]
    sse41,
    Sse41Kernels,
    "sse4.1"
);
isa_kernel!(
    #[cfg(target_arch = "x86_64")]
    avx2,
    Avx2Kernels,
    "avx2"
);
isa_kernel!(
    #[cfg(target_arch = "x86_64")]
    avx512,
    Avx512Kernels,
    "avx512bw"
);

/// [`viterbi_upper_bound_with`] on vectors of `N` lanes with the shift of
/// `K`; `om`'s Viterbi tables must be striped for `N` lanes.
#[inline(always)]
fn viterbi_upper_bound_impl<const N: usize, K: Kernels>(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    rows: &mut Rows,
) -> f32 {
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
    let q = om.q_vit;
    debug_assert_eq!(om.isa().lanes::<i16>(), N);
    let length = l as f64;
    let tmove_units = special_units(scale, tmove);
    let e_loop_units = special_units(scale, ln(om.xf[XST_E][XTR_LOOP]));
    let e_move_units = special_units(scale, ln(om.xf[XST_E][XTR_MOVE]));
    // The cumulative delta in fixed point, as in int_msv.rs.
    let delta_step = (scale * -tloop * f64::from(1u32 << DELTA_FRACTION_BITS)).ceil() as u64;
    let delta_of = |fixed_point: u64| {
        ((fixed_point + ((1 << DELTA_FRACTION_BITS) - 1)) >> DELTA_FRACTION_BITS) as i64
    };

    // Cells hold units above `base`, offset by i16::MIN. The begin state is
    // never below tmove, so no path enters a row below `base`; saturating at
    // the bottom only raises values that could not win anyway.
    let base =
        tmove_units + i64::from(om.vit_min_begin_units) - delta_of(l as u64 * delta_step) - 1;
    for row in [&mut rows.mmx, &mut rows.imx, &mut rows.dmx] {
        row.clear();
        row.resize(q * N, i16::MIN);
    }
    let (mmx, _) = rows.mmx.as_chunks_mut::<N>();
    let (imx, _) = rows.imx.as_chunks_mut::<N>();
    let (dmx, _) = rows.dmx.as_chunks_mut::<N>();

    let mut xj = i64::MIN / 2;
    let mut xc = i64::MIN / 2;
    let mut xb = tmove_units;

    let (transitions, _) = om.vit_units.as_chunks::<N>();
    let (main_transitions, dd_transitions) = transitions.split_at(NTSC_PER_Q * q);
    let (main_transitions, _) = main_transitions.as_chunks::<NTSC_PER_Q>();
    let (emissions, _) = om.msv_units.as_chunks::<N>();
    assert!(emissions.len() >= om.kp * q, "emission table too short");
    let residue_vectors = |residue: u8| -> &[[i16; N]] {
        let xi = residue as usize;
        let xi = if xi < om.kp { xi } else { 0 };
        &emissions[xi * q..][..q]
    };
    let exit_base = base + 32_768;
    // Cumulative delta of the current row, as a fixed-point number.
    let mut delta_acc = 0u64;

    // While no row has lifted J above zero, B stays at tmove and the entry
    // score of a row depends on the row index alone, so the processor can
    // start a row before the previous one has finished (see int_msv.rs).
    let mut row = 0;
    while row < l {
        // One extra unit covers the shortfall of the cumulative delta.
        let entry = tmove_units + 1 - delta_of(delta_acc) - base;
        if entry > 65_535 {
            return f32::INFINITY;
        }
        let xb_v = Simd::<i16, N>::splat((entry.max(0) - 32_768) as i16);
        let xe = viterbi_row::<N, K>(
            mmx,
            imx,
            dmx,
            main_transitions,
            dd_transitions,
            residue_vectors(dsq[row]),
            xb_v,
        );
        if xe == i16::MAX {
            return f32::INFINITY;
        }
        delta_acc += delta_step;
        let exit = exit_base + i64::from(xe) + delta_of(delta_acc);
        xj = xj.max(exit + e_loop_units);
        xc = xc.max(exit + e_move_units);
        row += 1;
        if xj > 0 {
            xb = xj + tmove_units;
            break;
        }
    }

    // From here on B depends on J, row by row.
    for &residue in &dsq[row..l] {
        let entry = xb + 1 - delta_of(delta_acc) - base;
        if entry > 65_535 {
            return f32::INFINITY;
        }
        let xb_v = Simd::<i16, N>::splat((entry.max(0) - 32_768) as i16);
        let xe = viterbi_row::<N, K>(
            mmx,
            imx,
            dmx,
            main_transitions,
            dd_transitions,
            residue_vectors(residue),
            xb_v,
        );
        if xe == i16::MAX {
            return f32::INFINITY;
        }
        delta_acc += delta_step;
        let exit = exit_base + i64::from(xe) + delta_of(delta_acc);
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

/// One row of the recurrence: updates the rows in place and returns the
/// maximum over the new match and delete cells.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn viterbi_row<const N: usize, K: Kernels>(
    mmx: &mut [[i16; N]],
    imx: &mut [[i16; N]],
    dmx: &mut [[i16; N]],
    main_transitions: &[[[i16; N]; NTSC_PER_Q]],
    dd_transitions: &[[i16; N]],
    rsc_vectors: &[[i16; N]],
    xb_v: Simd<i16, N>,
) -> i16 {
    let q = mmx.len();
    let floor_v = Simd::<i16, N>::splat(i16::MIN);
    let mut dcv = floor_v;
    let mut xe_v = floor_v;

    // Right-shift previous row by 1 lane (Farrar shift)
    let mut mpv = K::shift_right_one(Simd::from_array(mmx[q - 1]));
    let mut dpv = K::shift_right_one(Simd::from_array(dmx[q - 1]));
    let mut ipv = K::shift_right_one(Simd::from_array(imx[q - 1]));

    for (qi, (tsc, rsc_vector)) in main_transitions.iter().zip(rsc_vectors).enumerate() {
        let t = |kind: usize| Simd::<i16, N>::from_array(tsc[kind]);

        // M(i,q) = max(B+tBM, M(i-1,q-1)+tMM, I(i-1,q-1)+tIM, D(i-1,q-1)+tDM) + rsc
        let mut sv = xb_v
            .saturating_add(t(TSC_BM))
            .simd_max(mpv.saturating_add(t(TSC_MM)));
        sv = sv.simd_max(ipv.saturating_add(t(TSC_IM)));
        sv = sv.simd_max(dpv.saturating_add(t(TSC_DM)));
        sv = sv.saturating_add(Simd::from_array(*rsc_vector));

        xe_v = xe_v.simd_max(sv);

        mpv = Simd::from_array(mmx[qi]);
        dpv = Simd::from_array(dmx[qi]);
        ipv = Simd::from_array(imx[qi]);

        mmx[qi] = sv.to_array();
        dmx[qi] = dcv.to_array();

        // D carry: M(i,k) + tMD
        dcv = sv.saturating_add(t(TSC_MD));

        // I(i,q) = max(M(i-1,q)+tMI, I(i-1,q)+tII)
        imx[qi] = mpv
            .saturating_add(t(TSC_MI))
            .simd_max(ipv.saturating_add(t(TSC_II)))
            .to_array();
    }

    // D->D propagation across stripes. One full pass, then passes that
    // stop at the first vector the carry does not improve; a lane can
    // need as many passes as there are lanes.
    dcv = K::shift_right_one(dcv);
    for (dmx_q, dd) in dmx.iter_mut().zip(dd_transitions) {
        let cell = Simd::from_array(*dmx_q).simd_max(dcv);
        *dmx_q = cell.to_array();
        xe_v = xe_v.simd_max(cell);
        dcv = cell.saturating_add(Simd::from_array(*dd));
    }
    for _pass in 1..N {
        dcv = K::shift_right_one(dcv);
        let mut reached_end = true;
        for (dmx_q, dd) in dmx.iter_mut().zip(dd_transitions) {
            let cell = Simd::from_array(*dmx_q);
            if !dcv.simd_gt(cell).any() {
                reached_end = false;
                break;
            }
            let cell = cell.simd_max(dcv);
            *dmx_q = cell.to_array();
            xe_v = xe_v.simd_max(cell);
            dcv = cell.saturating_add(Simd::from_array(*dd));
        }
        if !reached_end {
            break;
        }
    }

    xe_v.reduce_max()
}
