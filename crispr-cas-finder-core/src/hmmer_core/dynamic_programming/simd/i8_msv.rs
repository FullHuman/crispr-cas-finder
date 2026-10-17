// simd/i8_msv.rs — i8x16 over-estimate of the MSV score
//
// The integer MSV bound of int_msv.rs on 8-bit cells, sixteen lanes per
// vector. Its coarse scale makes it looser than the 16-bit bound, so what it
// lets through is checked by that one next; it costs half as much per cell.

use super::int_msv::{self, Blocks, Table, Unit};
use super::oprofile::OptimizedProfile;
use std::simd::Simd;

/// Sixteen 8-bit lanes.
pub type I8x16 = Simd<i8, 16>;

/// Lanes per vector.
pub const LANES: usize = 16;

/// Integer units per nat.
///
/// A cell has 255 units above the begin state. A single segment worth
/// including needs about seventeen nats of emissions, and the bound gives
/// up on a sequence as soon as one segment gets there, so the cells only
/// ever need to hold a little more than that; fourteen units per nat leaves
/// eighteen nats while keeping the rounding of each emission under a
/// fourteenth of a nat.
pub const SCALE: f64 = 14.0;

/// Number of i8x16 vectors for a model of `m` nodes.
#[inline]
pub fn vector_count(m: usize) -> usize {
    int_msv::vector_count(m, LANES)
}

/// Round an emission score up to integer units, saturating to i8.
#[inline]
pub fn emission_units(scale: f64, nats: f32) -> i8 {
    <i8 as Unit>::emission_units(scale, nats)
}

/// Upper bound, in nats, on the score `msv_filter_f32` returns.
///
/// Returns `f32::INFINITY` when a cell saturates, meaning no bound is known.
pub fn msv_upper_bound(dsq: &[u8], l: usize, om: &OptimizedProfile) -> f32 {
    msv_upper_bound_with(dsq, l, om, &mut Vec::new(), None)
}

/// [`msv_upper_bound`] with a caller-owned DP row, so that a worker searching
/// many sequences does not allocate one per call, and optionally the
/// block-wise record a finer bound can continue from.
#[inline(never)]
pub fn msv_upper_bound_with(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    m_row: &mut Vec<I8x16>,
    blocks: Option<&mut Blocks<i8, LANES>>,
) -> f32 {
    int_msv::msv_upper_bound(dsq, l, om, table(om), m_row, blocks)
}

/// Lower bound, in nats, on the score `msv_filter_f32` returns, or `None`
/// when none is known (see `int_msv::msv_lower_bound`). Within about half a
/// nat of the exact score on average; the fast pipeline decides the MSV
/// filter with it.
pub fn msv_lower_bound(dsq: &[u8], l: usize, om: &OptimizedProfile) -> Option<f32> {
    msv_lower_bound_with(dsq, l, om, &mut Vec::new())
}

/// [`msv_lower_bound`] with a caller-owned DP row.
#[inline(never)]
pub fn msv_lower_bound_with(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    m_row: &mut Vec<I8x16>,
) -> Option<f32> {
    let table = Table {
        units: &om.msv_units8_down,
        units_plus_one: Some(&om.msv_units8_down_plus_one),
        kp: om.kp,
        q: om.q16,
        scale: om.msv_scale8,
    };
    int_msv::msv_lower_bound(dsq, l, om, table, m_row)
}

/// The emission tables of `om` for the upper bound.
fn table(om: &OptimizedProfile) -> Table<'_, i8> {
    Table {
        units: &om.msv_units8,
        units_plus_one: Some(&om.msv_units8_plus_one),
        kp: om.kp,
        q: om.q16,
        scale: om.msv_scale8,
    }
}
