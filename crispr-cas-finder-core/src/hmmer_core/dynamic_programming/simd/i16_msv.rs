// simd/i16_msv.rs — i16x8 over-estimate of the MSV score
//
// The exact MSV filter (f32_msv.rs) decides which sequences continue down
// the pipeline. This filter runs the same recurrence on 16-bit integers,
// twice as many lanes per vector, with every rounding and every clamp going
// upward. Its result is therefore never below the exact score, so a sequence
// whose over-estimate fails the MSV threshold would fail the exact filter
// too and can be dropped without computing it. Everything else is rescored
// exactly, which keeps search results identical.
//
// Derivation. With tloop the N/C/J self-loop score, subtracting i·tloop from
// every state of row i removes the loops from the exact recurrence:
//
//   M'(i,k) = max(M'(i-1,k-1), B'(i-1) + tBMk) + rsc(x_i,k) + delta
//   J'(i)   = max(J'(i-1), max_k M'(i,k) + tEC)
//   B'(i)   = max(0, J'(i)) + tmove
//   score   = J'(L) + L·tloop + tmove
//
// where delta = -tloop > 0. delta is far smaller than one integer unit, so
// it is not added to each cell: cells store M'' = M' - D(i), with D(i) the
// rounded-up cumulative delta, and D(i) is added back when a row maximum
// leaves for J'. A path entering after row a and leaving at row b collects
// D(b) - D(a), which can fall short of the true (b - a)·delta by less than
// one unit; one extra unit on the begin transition covers that.

use super::oprofile::OptimizedProfile;
use std::simd::prelude::*;

type I16x8 = Simd<i16, 8>;

/// Lanes per vector.
pub const LANES: usize = 8;

/// Number of independent running maxima in the inner loop.
const XE_ACCUMULATORS: usize = 4;

/// Added to the over-estimate, in nats, to cover the rounding of the exact
/// filter's own single-precision arithmetic.
const SAFETY_MARGIN: f64 = 0.05;

/// Guards `ceil` against a double-precision product landing just below the
/// real value it approximates.
const CEIL_GUARD: f64 = 1e-6;

/// Number of i16x8 vectors for a model of `m` nodes.
#[inline]
pub fn vector_count(m: usize) -> usize {
    2.max(m.saturating_sub(1) / LANES + 1)
}

/// Integer units per nat for a model of `m` nodes.
///
/// Cells cover 65,535 units above the lowest score a path can enter with.
/// That entry score is dominated by the begin transition, ln(2 / (m(m+1))),
/// and passing scores sit a couple of dozen nats above it; a cell reaching
/// the top is reported as "may pass".
pub fn scale_for(m: usize) -> f64 {
    let begin_score = (2.0 / (m as f64 * (m as f64 + 1.0))).ln();
    (65_535.0 / (26.0 - begin_score)).floor()
}

/// Round a score in nats up to integer units.
#[inline]
fn units_above(scale: f64, nats: f64) -> i64 {
    (scale * nats + CEIL_GUARD).ceil() as i64
}

/// Round an emission score up to integer units, saturating to i16.
#[inline]
pub fn emission_units(scale: f64, nats: f32) -> i16 {
    if nats == f32::NEG_INFINITY {
        return i16::MIN;
    }
    (scale * f64::from(nats) + CEIL_GUARD)
        .ceil()
        .clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16
}

/// Upper bound, in nats, on the score `msv_filter_f32` returns.
///
/// Returns `f32::INFINITY` when a cell saturates, meaning no bound is known.
pub fn msv_upper_bound(dsq: &[u8], l: usize, om: &OptimizedProfile) -> f32 {
    if l == 0 {
        return f32::INFINITY;
    }
    let scale = om.msv_scale;
    let q = om.q8;
    let m = om.m as f64;
    let length = l as f64;

    let tloop = (length / (length + 3.0)).ln();
    let tmove = (3.0 / (length + 3.0)).ln();
    let tmove_units = units_above(scale, tmove);
    let tec_units = units_above(scale, 0.5f64.ln());
    // One extra unit covers the shortfall of the cumulative delta (see above).
    let tbmk_units = units_above(scale, (2.0 / (m * (m + 1.0))).ln()) + 1;
    let cumulative_delta = |row: usize| units_above(scale, row as f64 * -tloop);

    // Cells hold units above `base`, offset by i16::MIN. No path enters a
    // row below `base`, so saturating at the bottom loses nothing.
    let base = tmove_units + tbmk_units - cumulative_delta(l) - 1;
    let floor_v = I16x8::splat(i16::MIN);
    let mut m_row = vec![floor_v; q];

    let mut xj = i64::MIN / 2;
    let mut xb = tmove_units;

    for (row, &residue) in dsq[..l].iter().enumerate() {
        let xi = residue as usize;
        let rsc = if xi < om.kp {
            &om.msv_units[xi]
        } else {
            &om.msv_units[0]
        };
        let (rsc_vectors, _) = rsc.as_chunks::<LANES>();

        let entry = (xb + tbmk_units - cumulative_delta(row) - base).clamp(0, 65_535);
        let xb_v = I16x8::splat((entry - 32_768) as i16);

        // Right-shift previous row by 1 lane (Farrar shift)
        let mut mpv = shift_right(m_row[q - 1]);

        let mut xe_lanes = [floor_v; XE_ACCUMULATORS];
        let (cell_groups, cell_rest) = m_row.as_chunks_mut::<XE_ACCUMULATORS>();
        let (rsc_groups, rsc_rest) = rsc_vectors.as_chunks::<XE_ACCUMULATORS>();
        for (cell_group, rsc_group) in cell_groups.iter_mut().zip(rsc_groups) {
            for index in 0..XE_ACCUMULATORS {
                let sv = mpv
                    .simd_max(xb_v)
                    .saturating_add(I16x8::from_array(rsc_group[index]));
                xe_lanes[index] = xe_lanes[index].simd_max(sv);
                mpv = cell_group[index];
                cell_group[index] = sv;
            }
        }
        for (cell, rsc_vector) in cell_rest.iter_mut().zip(rsc_rest) {
            let sv = mpv
                .simd_max(xb_v)
                .saturating_add(I16x8::from_array(*rsc_vector));
            xe_lanes[0] = xe_lanes[0].simd_max(sv);
            mpv = *cell;
            *cell = sv;
        }
        let xe = xe_lanes
            .into_iter()
            .fold(floor_v, |maximum, lane| maximum.simd_max(lane))
            .reduce_max();
        if xe == i16::MAX {
            return f32::INFINITY;
        }

        let exit = base + i64::from(xe) + 32_768 + cumulative_delta(row + 1) + tec_units;
        xj = xj.max(exit);
        xb = xj.max(0) + tmove_units;
    }

    let bound = xj as f64 / scale + length * tloop + tmove + SAFETY_MARGIN;
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
