// simd/i16_msv.rs — i16x8 over-estimate of the MSV score
//
// The integer MSV bound of int_msv.rs on 16-bit cells, eight lanes per
// vector. With thousands of units per nat it sits within a few hundredths
// of a nat of the exact score, so nearly everything it lets through is a
// real MSV hit. The 8-bit bound of i8_msv.rs runs before it.

use super::int_msv::{self, Blocks, Table, Unit};
use super::oprofile::OptimizedProfile;
use std::simd::Simd;

pub use super::int_msv::{CEIL_GUARD, SAFETY_MARGIN, units_above};

/// Eight 16-bit lanes.
pub type I16x8 = Simd<i16, 8>;

/// Lanes per vector.
pub const LANES: usize = 8;

/// Number of i16x8 vectors for a model of `m` nodes.
#[inline]
pub fn vector_count(m: usize) -> usize {
    int_msv::vector_count(m, LANES)
}

/// Integer units per nat for a model of `m` nodes, shared with the Viterbi
/// over-estimate and its transition table.
///
/// Cells cover 65,535 units above the lowest score a path can enter with.
/// That entry score is dominated by the begin transition, ln(2 / (m(m+1))),
/// and the scores that pass the Viterbi threshold sit within about thirty
/// nats above it; a cell reaching the top is reported as "may pass".
pub fn scale_for(m: usize) -> f64 {
    let begin_score = (2.0 / (m as f64 * (m as f64 + 1.0))).ln();
    (65_535.0 / (32.0 - begin_score)).floor()
}

/// Integer units per nat of the MSV bound.
///
/// Its cells count from the begin state and never need to hold much more
/// than the seventeen nats a segment needs to be worth including (see
/// int_msv.rs), so they can be finer than the Viterbi bound's.
pub const SCALE: f64 = 2_600.0;

/// Round an emission score up to integer units, saturating to i16.
#[inline]
pub fn emission_units(scale: f64, nats: f32) -> i16 {
    <i16 as Unit>::emission_units(scale, nats)
}

/// Upper bound, in nats, on the score `msv_filter_f32` returns.
///
/// Returns `f32::INFINITY` when a cell saturates, meaning no bound is known.
pub fn msv_upper_bound(dsq: &[u8], l: usize, om: &OptimizedProfile) -> f32 {
    msv_upper_bound_with(dsq, l, om, &mut Vec::new())
}

/// [`msv_upper_bound`] over the blocks of `selected` only, each started from
/// the row the 8-bit bound recorded in `blocks` (see
/// `int_msv::msv_upper_bound_within`).
#[inline(never)]
pub fn msv_upper_bound_within(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    m_row: &mut Vec<I16x8>,
    blocks: &Blocks<i8, { super::i8_msv::LANES }>,
    selected: u64,
) -> f32 {
    int_msv::msv_upper_bound_within(
        dsq,
        l,
        om,
        table(om),
        m_row,
        blocks,
        om.msv_scale8,
        selected,
    )
}

/// The emission table of `om` for the upper bound.
fn table(om: &OptimizedProfile) -> Table<'_, i16> {
    Table {
        units: &om.ssv_units,
        units_plus_one: None,
        kp: om.kp,
        q: om.q8,
        scale: SCALE,
    }
}

/// Lower bound, in nats, on the score `msv_filter_f32` returns, or `None`
/// when none is known (see `int_msv::msv_lower_bound`).
pub fn msv_lower_bound(dsq: &[u8], l: usize, om: &OptimizedProfile) -> Option<f32> {
    msv_lower_bound_with(dsq, l, om, &mut Vec::new())
}

/// [`msv_lower_bound`] with a caller-owned DP row.
#[inline(never)]
pub fn msv_lower_bound_with(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    m_row: &mut Vec<I16x8>,
) -> Option<f32> {
    let table = Table {
        units: &om.ssv_units_down,
        units_plus_one: None,
        kp: om.kp,
        q: om.q8,
        scale: SCALE,
    };
    int_msv::msv_lower_bound(dsq, l, om, table, m_row)
}

/// [`msv_upper_bound`] with a caller-owned DP row, so that a worker searching
/// many sequences does not allocate one per call.
#[inline(never)]
pub fn msv_upper_bound_with(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    m_row: &mut Vec<I16x8>,
) -> f32 {
    int_msv::msv_upper_bound(dsq, l, om, table(om), m_row, None)
}
