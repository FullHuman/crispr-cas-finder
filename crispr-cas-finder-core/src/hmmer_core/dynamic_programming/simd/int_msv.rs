// simd/int_msv.rs — integer bounds on the MSV score
//
// This kernel bounds the score of the exact MSV filter (f32_msv.rs) on
// integer cells, 16-bit or 8-bit lanes packed into one vector. With every
// rounding and every clamp going upward the result is an upper bound: a
// sequence whose bound fails the MSV threshold would fail the exact filter
// too. With them going downward it is a lower bound: a sequence whose bound
// passes would pass the exact filter too. The search pipeline
// (pipeline/search.rs) uses the bounds two ways. By default the 8-bit lower
// bound stands in for the score and decides the MSV and bias filters, as
// HMMER's own rounded filter does, and the exact filter never runs. With
// `FilterPolicy::exact` the bounds settle only the sequences they can, and
// the exact filter scores the rest, so the decisions are those of the exact
// score.
//
// Single segments. An MSV path visits the model in one or more segments,
// diagonal runs of match states. With tloop the N/C/J self-loop score,
// tmove the N->B, J->B and C->T score, tBMk the begin transition and tEC the
// exit, a path with segments of lengths len_j and emission sums emis_j
// scores
//
//   L·tloop + tmove + Σ_j S_j,   S_j = tBMk + emis_j + tEC + tmove - len_j·tloop.
//
// Only segments with S_j > 0 can raise the score, and they need about
// seventeen nats of emissions to do so. When the best single segment has
// S* <= 0, every set of segments scores at most S*, so the MSV score is
// exactly L·tloop + tmove + S*: the single-segment recurrence
//
//   U(i,k) = max(U(i-1,k-1), 0) + rsc(x_i,k) + delta,   delta = -tloop,
//
// whose maximum cell is emis* - len*·tloop, gives the MSV score without the
// J state. When S* > 0 the sequence is a strong candidate anyway and the
// bound saturates: it passes the MSV filters, or, with exact filters, the
// exact filter scores it.
//
// Units. delta is not a whole number of units, so it is not added to each
// cell as such: D(i), the rounded-up cumulative delta, grows by a whole
// number of units from one row to the next, and that growth is added to
// every cell of the row. Over a segment this collects D(b) - D(a-1), which
// can fall short of the true len·delta by less than one unit; one extra
// unit on the result covers that. With a coarse scale the growth is zero or
// one unit and comes from a second emission table one unit higher; with a
// fine scale it is added as a vector. The floor of a cell is the begin
// state: saturating addition at the bottom is the max with zero, so a cell
// costs one saturating add (two with the growth vector) and one max for the
// running maximum, and the row stays in registers when it is short enough.
// Nothing is carried between rows except the cells themselves, so the
// processor overlaps consecutive rows freely.
//
// Vector width. The recurrence is the same whatever the number of lanes:
// each cell depends on one cell of the previous row, and the result is the
// largest cell, so the bounds come out identical for any striping. The
// kernels therefore run with the widest vectors the processor has (`Isa`):
// 128-bit vectors everywhere, 256-bit with AVX2 and 512-bit with AVX-512BW,
// chosen once when the optimized profile is built and its emission tables
// striped. Each kernel is compiled once per instruction set, so that the
// portable `std::simd` operations lower to the native instructions of that
// set (on the x86-64 baseline, for instance, there is no 8-bit signed
// maximum, and every running maximum costs four instructions instead of
// one).

use super::oprofile::OptimizedProfile;
use std::ops::Range;
use std::simd::cmp::SimdOrd;
use std::simd::num::SimdInt;
use std::simd::{Simd, SimdElement};

/// Added to the over-estimate, in nats, to cover the rounding of the exact
/// filter's own single-precision arithmetic.
pub const SAFETY_MARGIN: f64 = 0.05;

/// Guards `ceil` against a double-precision product landing just below the
/// real value it approximates.
pub const CEIL_GUARD: f64 = 1e-6;

/// Fractional bits of the fixed-point cumulative delta.
///
/// A step is at most a few thousand units, a protein at most a few hundred
/// thousand residues; their product stays far below 2^64 with 24 fractional
/// bits, and the rounding of the step adds less than a thousandth of a unit
/// over the longest protein, which `SAFETY_MARGIN` covers.
pub const DELTA_FRACTION_BITS: u32 = 24;

/// Number of independent running maxima within a row.
const ACCUMULATORS: usize = 4;

/// Round a score in nats up to integer units.
#[inline]
pub fn units_above(scale: f64, nats: f64) -> i64 {
    (scale * nats + CEIL_GUARD).ceil() as i64
}

/// Number of vectors of `lanes` lanes for a model of `m` nodes.
#[inline]
pub fn vector_count(m: usize, lanes: usize) -> usize {
    2.max(m.saturating_sub(1) / lanes + 1)
}

// ---------- instruction sets ----------

/// The instruction set the integer kernels run with, and with it the width
/// of their vectors. Detected once per process with [`Isa::detect`]; the
/// optimized profile records the one its emission tables are striped for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Isa {
    /// The target's baseline: 128-bit vectors through portable `std::simd`.
    #[default]
    Portable,
    /// SSE4.1: 128-bit vectors with a native 8-bit maximum.
    #[cfg(target_arch = "x86_64")]
    Sse41,
    /// AVX2: 256-bit vectors.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512BW: 512-bit vectors.
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Isa {
    /// Every instruction set the kernels are compiled for, slowest first.
    pub const ALL: &'static [Isa] = &[
        Isa::Portable,
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512,
    ];

    /// The fastest instruction set this processor supports.
    pub fn detect() -> Isa {
        Self::ALL
            .iter()
            .rev()
            .copied()
            .find(|isa| isa.is_available())
            .unwrap_or_default()
    }

    /// The instruction sets this processor supports.
    pub fn available() -> Vec<Isa> {
        Self::ALL
            .iter()
            .copied()
            .filter(|isa| isa.is_available())
            .collect()
    }

    /// Whether this processor supports the instruction set.
    pub fn is_available(self) -> bool {
        match self {
            Isa::Portable => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Sse41 => std::arch::is_x86_feature_detected!("sse4.1"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => std::arch::is_x86_feature_detected!("avx512bw"),
        }
    }

    /// Bytes per vector.
    pub const fn vector_bytes(self) -> usize {
        match self {
            Isa::Portable => 16,
            #[cfg(target_arch = "x86_64")]
            Isa::Sse41 => 16,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => 32,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => 64,
        }
    }

    /// Lanes per vector for cells of type `T`.
    pub const fn lanes<T>(self) -> usize {
        self.vector_bytes() / size_of::<T>()
    }

    /// Number of vectors per DP row for a model of `m` nodes and cells of
    /// type `T`.
    pub const fn vector_count<T>(self, m: usize) -> usize {
        let lanes = self.lanes::<T>();
        let count = if m == 0 { 0 } else { (m - 1) / lanes + 1 };
        if count < 2 { 2 } else { count }
    }
}

/// The kernels compiled for one instruction set (see the `Kernels` marker
/// types), as the generic entry points below reach them.
pub trait Kernels {
    /// The instruction set the kernels are compiled for.
    const ISA: Isa;

    /// The largest cell of the single-segment recurrence over `rows` of
    /// `dsq` (see `best_segment`).
    fn best_segment_of<T, const N: usize>(
        dsq: &[u8],
        rows: Range<usize>,
        start: Option<&[i64]>,
        emissions: &Emissions<'_, T, N>,
        m_row: &mut Vec<T>,
        recorder: Option<&mut Recorder<'_, T>>,
    ) -> Option<i64>
    where
        T: Unit,
        Simd<T, N>: SimdInt<Scalar = T> + SimdOrd;
}

/// Calls `$f::<$t, N, K>($args)` with the lane count `N` and the kernels `K`
/// of the instruction set `$isa`: the way the typed entry points of i8_msv.rs
/// and i16_msv.rs reach the generic ones here (`$f` names one of them, in
/// scope).
macro_rules! dispatch {
    ($isa:expr, $t:ty, $f:ident ( $($arg:expr),* $(,)? )) => {
        match $isa {
            $crate::hmmer_core::dynamic_programming::simd::int_msv::Isa::Portable => $f::<
                $t,
                { $crate::hmmer_core::dynamic_programming::simd::int_msv::Isa::Portable.lanes::<$t>() },
                $crate::hmmer_core::dynamic_programming::simd::int_msv::PortableKernels,
            >($($arg),*),
            #[cfg(target_arch = "x86_64")]
            $crate::hmmer_core::dynamic_programming::simd::int_msv::Isa::Sse41 => $f::<
                $t,
                { $crate::hmmer_core::dynamic_programming::simd::int_msv::Isa::Sse41.lanes::<$t>() },
                $crate::hmmer_core::dynamic_programming::simd::int_msv::Sse41Kernels,
            >($($arg),*),
            #[cfg(target_arch = "x86_64")]
            $crate::hmmer_core::dynamic_programming::simd::int_msv::Isa::Avx2 => $f::<
                $t,
                { $crate::hmmer_core::dynamic_programming::simd::int_msv::Isa::Avx2.lanes::<$t>() },
                $crate::hmmer_core::dynamic_programming::simd::int_msv::Avx2Kernels,
            >($($arg),*),
            #[cfg(target_arch = "x86_64")]
            $crate::hmmer_core::dynamic_programming::simd::int_msv::Isa::Avx512 => $f::<
                $t,
                { $crate::hmmer_core::dynamic_programming::simd::int_msv::Isa::Avx512.lanes::<$t>() },
                $crate::hmmer_core::dynamic_programming::simd::int_msv::Avx512Kernels,
            >($($arg),*),
        }
    };
}
pub(crate) use dispatch;

// ---------- cells ----------

/// An integer cell type of the recurrence.
pub trait Unit: SimdElement + PartialEq + Default {
    const MIN: Self;
    const MAX: Self;
    /// Units of this cell above the floor.
    fn units(self) -> i64;
    /// The cell `units` above the floor, saturating.
    fn from_units(units: i64) -> Self;
    /// A whole number of units as a cell value, saturating.
    fn from_count(units: i64) -> Self;
    /// Round an emission score up to units, saturating.
    fn emission_units(scale: f64, nats: f32) -> Self;
    /// Round an emission score down to units, saturating.
    fn emission_units_down(scale: f64, nats: f32) -> Self;
    /// The emission one unit higher, for rows where the cumulative delta
    /// grows; an impossible emission stays impossible.
    fn one_unit_higher(self) -> Self;
}

macro_rules! unit {
    ($t:ty, $offset:expr) => {
        impl Unit for $t {
            const MIN: Self = <$t>::MIN;
            const MAX: Self = <$t>::MAX;
            #[inline]
            fn units(self) -> i64 {
                i64::from(self) + $offset
            }
            #[inline]
            fn from_units(units: i64) -> Self {
                (units - $offset).clamp(i64::from(<$t>::MIN), i64::from(<$t>::MAX)) as $t
            }
            #[inline]
            fn from_count(units: i64) -> Self {
                units.clamp(i64::from(<$t>::MIN), i64::from(<$t>::MAX)) as $t
            }
            fn emission_units(scale: f64, nats: f32) -> Self {
                if nats == f32::NEG_INFINITY {
                    return <$t>::MIN;
                }
                (scale * f64::from(nats) + CEIL_GUARD)
                    .ceil()
                    .clamp(f64::from(<$t>::MIN), f64::from(<$t>::MAX)) as $t
            }
            fn emission_units_down(scale: f64, nats: f32) -> Self {
                if nats == f32::NEG_INFINITY {
                    return <$t>::MIN;
                }
                (scale * f64::from(nats) - CEIL_GUARD)
                    .floor()
                    .clamp(f64::from(<$t>::MIN), f64::from(<$t>::MAX)) as $t
            }
            #[inline]
            fn one_unit_higher(self) -> Self {
                if self == <$t>::MIN {
                    self
                } else {
                    self.saturating_add(1)
                }
            }
        }
    };
}
unit!(i8, 128);
unit!(i16, 32_768);

/// Striped match emissions of a profile in integer units: `kp` rows of
/// `q` vectors of `lanes` lanes, lane `z` of vector `qi` holding node
/// `qi + 1 + z·q`, and optionally the same one unit higher (see the module
/// comment).
pub struct Table<'a, T: Unit> {
    pub units: &'a [T],
    pub units_plus_one: Option<&'a [T]>,
    pub kp: usize,
    pub q: usize,
    pub scale: f64,
}

/// Which way scores are rounded to integer units.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    /// Every score up: the result is an upper bound.
    Up,
    /// Every score down: the result is a lower bound.
    Down,
}

/// Build the emission table of a profile for `scale` units per nat.
pub fn emission_table<T: Unit>(
    gm: &crate::hmmer_core::profile::Profile,
    kp: usize,
    q: usize,
    lanes: usize,
    scale: f64,
    rounding: Rounding,
) -> Vec<T> {
    use crate::hmmer_core::config::{PROFILE_NUM_EMISSIONS, PRsc};
    let m = gm.num_nodes;
    let width = q * lanes;
    let mut units = vec![T::MIN; kp * width];
    for x in 0..kp {
        let rsc = gm.residue_scores_for(x);
        let row = &mut units[x * width..][..width];
        for qi in 0..q {
            for z in 0..lanes {
                let k = qi + 1 + z * q;
                let score_idx = k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize;
                if k <= m && score_idx < rsc.len() {
                    row[qi * lanes + z] = match rounding {
                        Rounding::Up => T::emission_units(scale, rsc[score_idx]),
                        Rounding::Down => T::emission_units_down(scale, rsc[score_idx]),
                    };
                }
            }
        }
    }
    units
}

/// The same table with every possible emission one unit higher.
pub fn one_unit_higher<T: Unit>(units: &[T]) -> Vec<T> {
    units.iter().map(|&unit| unit.one_unit_higher()).collect()
}

/// Rows per block of the block-wise record (see [`Blocks`]).
pub const BLOCK_ROWS: usize = 64;

/// What an upper-bound pass records about each block of [`BLOCK_ROWS`] rows:
/// an upper bound on the MSV score of a path whose best segment ends in the
/// block, and the DP row as it stood after the block's last row. A finer
/// bound can then start from that row and look at the blocks that matter
/// (see [`msv_upper_bound_within`]), instead of at the whole sequence.
#[derive(Debug, Default)]
pub struct Blocks<T: Unit> {
    /// Vectors per row.
    pub q: usize,
    /// Lanes per vector.
    pub lanes: usize,
    /// Per block, in nats; infinite when no bound is known for the block.
    pub bounds: Vec<f32>,
    /// `q · lanes` cells per block, vector by vector.
    pub states: Vec<T>,
    /// A cell saturated somewhere: the states cannot be trusted.
    pub saturated: bool,
}

impl<T: Unit> Blocks<T> {
    fn clear(&mut self, q: usize, lanes: usize) {
        self.q = q;
        self.lanes = lanes;
        self.bounds.clear();
        self.states.clear();
        self.saturated = false;
    }

    /// The row state after block `block`, as units above the begin state
    /// per model node (node 1 first), or `None` for the first block.
    pub fn state_by_node(&self, block: usize, m: usize) -> Option<Vec<i64>> {
        if block == 0 {
            return None;
        }
        let width = self.q * self.lanes;
        let state = &self.states[(block - 1) * width..][..width];
        Some(
            (1..=m)
                .map(|k| {
                    // Lane `z` of vector `qi` holds node `qi + 1 + z·q`.
                    let (qi, z) = ((k - 1) % self.q, (k - 1) / self.q);
                    state[qi * self.lanes + z].units()
                })
                .collect(),
        )
    }
}

/// Vector `qi` of a row of `q` vectors loaded from `units` above the begin
/// state per node; nodes beyond the units given sit at the floor.
#[inline]
fn restriped<T: Unit, const N: usize>(units: &[i64], q: usize, qi: usize) -> [T; N] {
    let mut lanes = [T::MIN; N];
    for (z, lane) in lanes.iter_mut().enumerate() {
        let k = qi + 1 + z * q;
        if let Some(&value) = units.get(k - 1) {
            *lane = T::from_units(value);
        }
    }
    lanes
}

/// The special transition scores of a sequence of length `l`, in nats.
pub struct Specials {
    tloop: f64,
    tmove: f64,
    /// tBMk + tEC + tmove: what a segment pays besides its emissions.
    segment_cost: f64,
}

impl Specials {
    fn new(l: usize, m: usize) -> Self {
        let length = l as f64;
        let m = m as f64;
        let tloop = (length / (length + 3.0)).ln();
        let tmove = (3.0 / (length + 3.0)).ln();
        let tbmk = (2.0 / (m * (m + 1.0))).ln();
        let tec = 0.5f64.ln();
        Self {
            tloop,
            tmove,
            segment_cost: tbmk + tec + tmove,
        }
    }

    /// The MSV score of a sequence whose best single segment has emissions
    /// and deltas worth `best_segment` nats, when no segment is worth more
    /// than the J state costs (see the module comment).
    fn msv_score(&self, best_segment: f64, l: usize) -> f64 {
        best_segment + self.segment_cost + l as f64 * self.tloop + self.tmove
    }
}

/// Upper bound, in nats, on the score `msv_filter_f32` returns.
///
/// Returns `f32::INFINITY` when no bound is known: a cell saturated, or a
/// single segment scores above the J-state costs, so that a multi-hit path
/// could score higher still. `m_row` is the DP row, kept by the caller so
/// that a worker searching many sequences does not allocate one per call;
/// short rows live in registers instead and leave it untouched. With
/// `blocks`, the pass also records its block-wise results. `N` is the lane
/// count `table` is striped for and `K` the kernels of its instruction set
/// (see [`dispatch`]).
pub fn msv_upper_bound<T, const N: usize, K>(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    table: Table<'_, T>,
    m_row: &mut Vec<T>,
    blocks: Option<&mut Blocks<T>>,
) -> f32
where
    T: Unit,
    K: Kernels,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    if l == 0 {
        if let Some(blocks) = blocks {
            blocks.clear(table.q, N);
        }
        return f32::INFINITY;
    }
    let specials = Specials::new(l, om.m);
    let mut recorder = blocks.map(|blocks| {
        blocks.clear(table.q, N);
        Recorder {
            blocks,
            maxima: Vec::new(),
        }
    });
    let best = best_segment_of::<T, N, K>(
        dsq,
        0..l,
        None,
        &table,
        Rounding::Up,
        &specials,
        m_row,
        recorder.as_mut(),
    );
    if let Some(recorder) = recorder {
        let scale = table.scale;
        recorder.blocks.bounds = recorder
            .maxima
            .iter()
            .map(|&maximum| upper_bound_of(maximum, scale, &specials, l))
            .collect();
    }
    upper_bound_of(best, table.scale, &specials, l)
}

/// The upper bound on the MSV score from a largest cell of `maximum` units,
/// or `None` when a cell saturated.
fn upper_bound_of(maximum: Option<i64>, scale: f64, specials: &Specials, l: usize) -> f32 {
    let Some(best) = maximum else {
        return f32::INFINITY;
    };
    // One extra unit covers the shortfall of the cumulative delta.
    let best_segment = (best + 1) as f64 / scale;
    if best_segment + specials.segment_cost > 0.0 {
        // A multi-hit path may beat the best single segment.
        return f32::INFINITY;
    }
    let bound = specials.msv_score(best_segment, l) + SAFETY_MARGIN;
    // Round toward +inf when narrowing to f32.
    let narrowed = bound as f32;
    if f64::from(narrowed) < bound {
        narrowed.next_up()
    } else {
        narrowed
    }
}

/// [`msv_upper_bound`] restricted to the blocks selected in `selected` (bit
/// `b` for block `b`), each started from the row state a coarser pass
/// recorded in `from` at the end of the block before it.
///
/// The recorded state is itself an upper bound on the true row, and the
/// recurrence only ever raises a result when its inputs rise, so the bound
/// holds. The result is the largest bound over all blocks: the finer one
/// for the selected blocks, the coarser one for the rest.
#[allow(clippy::too_many_arguments)]
pub fn msv_upper_bound_within<T, const N: usize, K, S>(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    table: Table<'_, T>,
    m_row: &mut Vec<T>,
    from: &Blocks<S>,
    from_scale: f64,
    selected: u64,
) -> f32
where
    T: Unit,
    K: Kernels,
    S: Unit,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    debug_assert!(!from.saturated && l.div_ceil(BLOCK_ROWS) == from.bounds.len());
    let specials = Specials::new(l, om.m);
    // A cell of the coarse pass may stand almost one coarse unit above the
    // value it rounds up, which the finer units must keep covering.
    let ratio = table.scale / from_scale;
    let coarse_unit = ratio.ceil() as i64;
    // The coarse row carries the coarse pass's slack into every segment that
    // begins before the block; starting one block earlier leaves it only to
    // segments longer than a block. Runs of adjacent blocks go together.
    let covered = selected | (selected >> 1);
    let blocks = from.bounds.len();
    let mut bound = f32::NEG_INFINITY;
    let mut block = 0;
    while block < blocks {
        if covered & (1 << block) == 0 {
            bound = bound.max(from.bounds[block]);
            block += 1;
            continue;
        }
        let first = block;
        while block < blocks && covered & (1 << block) != 0 {
            block += 1;
        }
        let start = from.state_by_node(first, om.m).map(|units| {
            units
                .into_iter()
                .map(|value| (value as f64 * ratio).ceil() as i64 + coarse_unit)
                .collect::<Vec<i64>>()
        });
        let rows = first * BLOCK_ROWS..(block * BLOCK_ROWS).min(l);
        let best = best_segment_of::<T, N, K>(
            dsq,
            rows,
            start.as_deref(),
            &table,
            Rounding::Up,
            &specials,
            m_row,
            None,
        );
        bound = bound.max(upper_bound_of(best, table.scale, &specials, l));
    }
    bound
}

/// Lower bound, in nats, on the score `msv_filter_f32` returns, from a
/// table rounded down. `None` when no cell rose above the begin state, so
/// that the best segment's score is unknown; `Some(f32::INFINITY)` when a
/// cell saturated, so that the score is at least as high as the cells can
/// count and no finite bound is known.
///
/// The best single segment never scores above the whole multi-hit path, so
/// this holds whatever the J state does.
pub fn msv_lower_bound<T, const N: usize, K>(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    table: Table<'_, T>,
    m_row: &mut Vec<T>,
) -> Option<f32>
where
    T: Unit,
    K: Kernels,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    if l == 0 {
        return None;
    }
    let specials = Specials::new(l, om.m);
    let Some(best) = best_segment_of::<T, N, K>(
        dsq,
        0..l,
        None,
        &table,
        Rounding::Down,
        &specials,
        m_row,
        None,
    ) else {
        return Some(f32::INFINITY);
    };
    if best == 0 {
        // Cells never go below the begin state, so a best of zero may hide
        // a negative segment score.
        return None;
    }
    // One unit covers the excess of the rounded-down cumulative delta.
    let best_segment = (best - 1) as f64 / table.scale;
    let bound = specials.msv_score(best_segment, l) - SAFETY_MARGIN;
    // Round toward -inf when narrowing to f32.
    let narrowed = bound as f32;
    Some(if f64::from(narrowed) > bound {
        narrowed.next_down()
    } else {
        narrowed
    })
}

/// Collects the block-wise results of a pass.
pub struct Recorder<'a, T: Unit> {
    blocks: &'a mut Blocks<T>,
    /// Largest cell of each block, `None` when a cell saturated.
    maxima: Vec<Option<i64>>,
}

/// The largest cell of the single-segment recurrence over `rows` of `dsq`,
/// in units above the begin state, or `None` when a cell saturated. The row
/// starts from `start` (units above the begin state per node) or at the
/// begin state.
#[allow(clippy::too_many_arguments)]
fn best_segment_of<T, const N: usize, K>(
    dsq: &[u8],
    rows: Range<usize>,
    start: Option<&[i64]>,
    table: &Table<'_, T>,
    rounding: Rounding,
    specials: &Specials,
    m_row: &mut Vec<T>,
    recorder: Option<&mut Recorder<'_, T>>,
) -> Option<i64>
where
    T: Unit,
    K: Kernels,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    let q = table.q;
    let (vectors, _) = table.units.as_chunks::<N>();
    assert!(vectors.len() >= table.kp * q, "emission table too short");
    let vectors_plus_one = table.units_plus_one.map(|units| {
        let (vectors_plus_one, _) = units.as_chunks::<N>();
        assert!(
            vectors_plus_one.len() >= table.kp * q,
            "emission table too short"
        );
        vectors_plus_one
    });
    let delta_step = table.scale * -specials.tloop * f64::from(1u32 << DELTA_FRACTION_BITS);
    let emissions = Emissions {
        vectors,
        vectors_plus_one: vectors_plus_one.unwrap_or(vectors),
        kp: table.kp,
        q,
        delta_step: match rounding {
            Rounding::Up => delta_step.ceil(),
            Rounding::Down => delta_step.floor(),
        } as u64,
        raise_after: rounding == Rounding::Up,
        growth_vector: vectors_plus_one.is_none(),
    };
    K::best_segment_of::<T, N>(dsq, rows, start, &emissions, m_row, recorder)
}

/// Striped match emissions of a profile, `kp` rows of `q` vectors, and the
/// per-row growth of the cumulative delta.
pub struct Emissions<'a, T: Unit, const N: usize> {
    vectors: &'a [[T; N]],
    vectors_plus_one: &'a [[T; N]],
    kp: usize,
    q: usize,
    /// The delta of one row in units, as a fixed-point number rounded up.
    delta_step: u64,
    /// Whether a row's extra growth is added after its emissions (upper
    /// bound) or before them (lower bound).
    raise_after: bool,
    /// Whether the growth is added to every cell as a vector (fine scales)
    /// rather than taken from the table one unit higher (coarse scales).
    growth_vector: bool,
}

impl<'a, T: Unit, const N: usize> Emissions<'a, T, N> {
    /// The emissions of `residue`, one unit higher when `plus_one`.
    #[inline(always)]
    fn of(&self, residue: u8, plus_one: bool) -> &'a [[T; N]] {
        let xi = residue as usize;
        let xi = if xi < self.kp { xi } else { 0 };
        let vectors = if plus_one {
            self.vectors_plus_one
        } else {
            self.vectors
        };
        &vectors[xi * self.q..][..self.q]
    }
}

/// Picks the kernel for the row width `$emissions.q`: rows of up to 24
/// vectors are kept in registers rather than memory, as the row, the running
/// maxima and a few temporaries must fit the 32 vector registers of AArch64
/// NEON and of x86-64 with AVX-512. `T` and `N` are the enclosing function's
/// cell type and lane count.
macro_rules! by_row_width {
    ($growth:literal; $in_registers:ident, $in_memory:ident; $dsq:ident, $rows:ident, $start:ident, $emissions:ident, $m_row:ident, $recorder:ident) => {
        by_row_width!(
            @widths $growth; $in_registers, $in_memory; $dsq, $rows, $start, $emissions, $m_row, $recorder;
            2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        )
    };
    (@widths $growth:literal; $in_registers:ident, $in_memory:ident; $dsq:ident, $rows:ident, $start:ident, $emissions:ident, $m_row:ident, $recorder:ident; $($width:literal)*) => {
        match $emissions.q {
            $($width => $in_registers::<T, N, $width, $growth>($dsq, $rows, $start, $emissions, $recorder),)*
            _ => $in_memory::<T, N, $growth>($dsq, $rows, $start, $emissions, $m_row, $recorder),
        }
    };
}

/// The kernels of one instruction set: the row recurrence compiled with
/// that set's features, for rows in registers (one function per row width,
/// fully unrolled) and for rows in memory.
///
/// A kernel compiled for an instruction set must only run on a processor
/// that has it; the `Kernels` marker types are reached through an [`Isa`]
/// only (see [`dispatch`]), and an `OptimizedProfile` only ever records an
/// available one.
macro_rules! isa_kernels {
    ($(#[$meta:meta])* $module:ident, $kernels:ident, $isa:expr $(, $feature:literal)*) => {
        $(#[$meta])*
        mod $module {
            use super::*;

            /// `best_segment` with a row of `Q` vectors held in registers.
            #[inline(never)]
            $(#[target_feature(enable = $feature)])*
            fn in_registers<T, const N: usize, const Q: usize, const GROWTH_VECTOR: bool>(
                dsq: &[u8],
                rows: Range<usize>,
                start: Option<&[i64]>,
                emissions: &Emissions<'_, T, N>,
                recorder: Option<&mut Recorder<'_, T>>,
            ) -> Option<i64>
            where
                T: Unit,
                Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
            {
                let mut cells = [Simd::<T, N>::splat(T::MIN); Q];
                if let Some(start) = start {
                    for (qi, cell) in cells.iter_mut().enumerate() {
                        *cell = Simd::from_array(restriped(start, Q, qi));
                    }
                }
                best_segment::<T, N, _, GROWTH_VECTOR>(dsq, rows, emissions, &mut cells, recorder)
            }

            /// `best_segment` with a row of `q` vectors in memory.
            #[inline(never)]
            $(#[target_feature(enable = $feature)])*
            fn in_memory<T, const N: usize, const GROWTH_VECTOR: bool>(
                dsq: &[u8],
                rows: Range<usize>,
                start: Option<&[i64]>,
                emissions: &Emissions<'_, T, N>,
                m_row: &mut Vec<T>,
                recorder: Option<&mut Recorder<'_, T>>,
            ) -> Option<i64>
            where
                T: Unit,
                Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
            {
                let q = emissions.q;
                m_row.clear();
                m_row.resize(q * N, T::MIN);
                let (cells, _) = m_row.as_chunks_mut::<N>();
                if let Some(start) = start {
                    for (qi, cell) in cells.iter_mut().enumerate() {
                        *cell = restriped(start, q, qi);
                    }
                }
                best_segment::<T, N, _, GROWTH_VECTOR>(dsq, rows, emissions, cells, recorder)
            }

            /// The largest cell of the recurrence over `rows` (see
            /// `Kernels::best_segment_of`).
            $(#[target_feature(enable = $feature)])*
            pub(super) fn best_segment_of<T, const N: usize>(
                dsq: &[u8],
                rows: Range<usize>,
                start: Option<&[i64]>,
                emissions: &Emissions<'_, T, N>,
                m_row: &mut Vec<T>,
                recorder: Option<&mut Recorder<'_, T>>,
            ) -> Option<i64>
            where
                T: Unit,
                Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
            {
                if emissions.growth_vector {
                    by_row_width!(true; in_registers, in_memory; dsq, rows, start, emissions, m_row, recorder)
                } else {
                    by_row_width!(false; in_registers, in_memory; dsq, rows, start, emissions, m_row, recorder)
                }
            }
        }

        $(#[$meta])*
        #[doc = concat!("The kernels compiled for `", stringify!($isa), "`.")]
        pub struct $kernels;

        $(#[$meta])*
        impl Kernels for $kernels {
            const ISA: Isa = $isa;

            #[inline(always)]
            #[allow(unused_unsafe)]
            fn best_segment_of<T, const N: usize>(
                dsq: &[u8],
                rows: Range<usize>,
                start: Option<&[i64]>,
                emissions: &Emissions<'_, T, N>,
                m_row: &mut Vec<T>,
                recorder: Option<&mut Recorder<'_, T>>,
            ) -> Option<i64>
            where
                T: Unit,
                Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
            {
                debug_assert!(
                    Self::ISA.is_available(),
                    "{:?} kernels on a processor without them",
                    Self::ISA
                );
                // SAFETY: these kernels are reached through `dispatch!` on an
                // `Isa` that `OptimizedProfile` recorded after checking it is
                // available on this processor (see `Isa::detect` and
                // `OptimizedProfile::from_profile_with_isa`).
                unsafe { $module::best_segment_of::<T, N>(dsq, rows, start, emissions, m_row, recorder) }
            }
        }
    };
}

isa_kernels!(portable, PortableKernels, Isa::Portable);
isa_kernels!(
    #[cfg(target_arch = "x86_64")]
    sse41,
    Sse41Kernels,
    Isa::Sse41,
    "sse4.1"
);
isa_kernels!(
    #[cfg(target_arch = "x86_64")]
    avx2,
    Avx2Kernels,
    Isa::Avx2,
    "avx2"
);
isa_kernels!(
    #[cfg(target_arch = "x86_64")]
    avx512,
    Avx512Kernels,
    Isa::Avx512,
    "avx512bw"
);

/// The largest cell of the single-segment recurrence over `rows` of `dsq`,
/// in units above the begin state, or `None` when a cell saturated.
///
/// With `GROWTH_VECTOR` the growth of the cumulative delta is added to every
/// cell as a vector; without it, rows that grow by one unit take their
/// emissions from the table one unit higher, and the rare rows that grow by
/// more get the rest in a second pass. `recorder`, if any, receives each
/// block's largest cell and the row after it.
#[inline(always)]
fn best_segment<T, const N: usize, R, const GROWTH_VECTOR: bool>(
    dsq: &[u8],
    rows: Range<usize>,
    emissions: &Emissions<'_, T, N>,
    cells: &mut R,
    mut recorder: Option<&mut Recorder<'_, T>>,
) -> Option<i64>
where
    T: Unit,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
    R: Row<T, N> + ?Sized,
{
    let mut best = Simd::<T, N>::splat(T::MIN);
    let floor_v = Simd::<T, N>::splat(T::MIN);
    // Cumulative delta, as a fixed-point number, and its integer part.
    let mut delta_acc = rows.start as u64 * emissions.delta_step;
    let mut delta = (delta_acc >> DELTA_FRACTION_BITS) as i64;
    // The emissions of a row and the growth to add to its cells.
    let mut row_inputs = |residue: u8| -> (&[[T; N]], Simd<T, N>, i64) {
        delta_acc += emissions.delta_step;
        let delta_next = (delta_acc >> DELTA_FRACTION_BITS) as i64;
        let growth = delta_next - delta;
        delta = delta_next;
        if GROWTH_VECTOR {
            let growth_v = Simd::splat(T::from_count(growth));
            (emissions.of(residue, false), growth_v, 0)
        } else {
            (emissions.of(residue, growth > 0), floor_v, growth - 1)
        }
    };
    let mut saturated = false;
    let mut block_start = rows.start;
    while block_start < rows.end {
        let block_end = (block_start + BLOCK_ROWS).min(rows.end);
        let mut block_best = floor_v;
        // Rows go in pairs: a row in memory is then loaded and stored once
        // for two rows of work. A row whose delta grows by more than one unit
        // (short sequences only) gets the rest in a second pass: after the
        // emissions for an upper bound, where the raised cells are the row's
        // cells and join its maximum, and before them for a lower bound,
        // where a raise after the floor's clamp would overstate cells, while
        // the begin state entering node 1 goes without the raise.
        let extra_v = |extra: i64| Simd::<T, N>::splat(T::from_count(extra));
        let (pairs, rest) = dsq[block_start..block_end].as_chunks::<2>();
        for &[first, second] in pairs {
            let (rsc0, growth0, extra0) = row_inputs(first);
            let (rsc1, growth1, extra1) = row_inputs(second);
            if extra0 > 0 || extra1 > 0 {
                for (rsc, growth, extra) in [(rsc0, growth0, extra0), (rsc1, growth1, extra1)] {
                    if extra > 0 && !emissions.raise_after {
                        cells.raise(extra_v(extra), None);
                    }
                    cells.step::<GROWTH_VECTOR>(rsc, growth, &mut block_best);
                    if extra > 0 && emissions.raise_after {
                        cells.raise(extra_v(extra), Some(&mut block_best));
                    }
                }
                continue;
            }
            cells.step_pair::<GROWTH_VECTOR>(rsc0, growth0, rsc1, growth1, &mut block_best);
        }
        for &residue in rest {
            let (rsc, growth_v, extra) = row_inputs(residue);
            if extra > 0 && !emissions.raise_after {
                cells.raise(extra_v(extra), None);
            }
            cells.step::<GROWTH_VECTOR>(rsc, growth_v, &mut block_best);
            if extra > 0 && emissions.raise_after {
                cells.raise(extra_v(extra), Some(&mut block_best));
            }
        }
        best = best.simd_max(block_best);
        if let Some(recorder) = recorder.as_deref_mut() {
            let maximum = block_best.reduce_max();
            if maximum == T::MAX {
                saturated = true;
            }
            recorder
                .maxima
                .push((maximum != T::MAX).then(|| maximum.units()));
            cells.save(&mut recorder.blocks.states);
        }
        block_start = block_end;
    }
    if let Some(recorder) = recorder {
        recorder.blocks.saturated = saturated;
    }
    let best = best.reduce_max();
    (best != T::MAX).then(|| best.units())
}

/// The lane-wise maximum of the accumulators.
#[inline(always)]
fn fold<T, const N: usize>(lanes: [Simd<T, N>; ACCUMULATORS]) -> Simd<T, N>
where
    T: Unit,
    Simd<T, N>: SimdOrd,
{
    lanes[0]
        .simd_max(lanes[1])
        .simd_max(lanes[2].simd_max(lanes[3]))
}

/// A new cell from the shifted previous one, its emission and the growth.
#[inline(always)]
fn new_cell<T, const N: usize, const GROWTH: bool>(
    mpv: Simd<T, N>,
    rsc: [T; N],
    growth_v: Simd<T, N>,
) -> Simd<T, N>
where
    T: Unit,
    Simd<T, N>: SimdInt<Scalar = T>,
{
    // U(i,q) = max(U(i-1,q-1), 0) + growth + rsc: the floor is the begin
    // state. The growth goes first: a negative sum clamps to the floor, and
    // with the growth added afterwards a lower bound would overstate it.
    let sv = if GROWTH {
        mpv.saturating_add(growth_v)
    } else {
        mpv
    };
    sv.saturating_add(Simd::from_array(rsc))
}

/// A DP row of match cells, in memory or in registers.
trait Row<T: Unit, const N: usize>
where
    T: Unit,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    /// One row of the recurrence: updates the cells in place and folds the
    /// new cells into `best`. With `GROWTH`, `growth_v` is added to each cell.
    fn step<const GROWTH: bool>(
        &mut self,
        rsc_vectors: &[[T; N]],
        growth_v: Simd<T, N>,
        best: &mut Simd<T, N>,
    );
    /// Two consecutive rows of the recurrence.
    #[inline(always)]
    fn step_pair<const GROWTH: bool>(
        &mut self,
        rsc0: &[[T; N]],
        growth0: Simd<T, N>,
        rsc1: &[[T; N]],
        growth1: Simd<T, N>,
        best: &mut Simd<T, N>,
    ) {
        self.step::<GROWTH>(rsc0, growth0, best);
        self.step::<GROWTH>(rsc1, growth1, best);
    }
    /// Add `extra` to every cell, folding the results into `best` if given.
    fn raise(&mut self, extra: Simd<T, N>, best: Option<&mut Simd<T, N>>);
    /// Append the cells to `out`, vector by vector.
    fn save(&self, out: &mut Vec<T>);
}

/// A row in memory, one array per vector; vectors are loaded and stored
/// with unaligned accesses.
impl<T, const N: usize> Row<T, N> for [[T; N]]
where
    T: Unit,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    #[inline(always)]
    fn step<const GROWTH: bool>(
        &mut self,
        rsc_vectors: &[[T; N]],
        growth_v: Simd<T, N>,
        best: &mut Simd<T, N>,
    ) {
        let q = self.len();
        // Right-shift previous row by 1 lane (Farrar shift); the begin
        // state enters node 1.
        let mut mpv = Simd::from_array(self[q - 1]).shift_elements_right::<1>(T::MIN);
        // The row's maxima start afresh and join `best` with one op, so
        // one row's chain of maxima never waits for the previous row's.
        let mut row_best = [Simd::splat(T::MIN); ACCUMULATORS];
        let (cell_groups, cell_rest) = self.as_chunks_mut::<ACCUMULATORS>();
        let (rsc_groups, rsc_rest) = rsc_vectors.as_chunks::<ACCUMULATORS>();
        for (cell_group, rsc_group) in cell_groups.iter_mut().zip(rsc_groups) {
            for index in 0..ACCUMULATORS {
                let sv = new_cell::<T, N, GROWTH>(mpv, rsc_group[index], growth_v);
                row_best[index] = row_best[index].simd_max(sv);
                mpv = Simd::from_array(cell_group[index]);
                cell_group[index] = sv.to_array();
            }
        }
        for (cell, rsc_vector) in cell_rest.iter_mut().zip(rsc_rest) {
            let sv = new_cell::<T, N, GROWTH>(mpv, *rsc_vector, growth_v);
            row_best[0] = row_best[0].simd_max(sv);
            mpv = Simd::from_array(*cell);
            *cell = sv.to_array();
        }
        *best = best.simd_max(fold(row_best));
    }

    /// Vector `q` of row i+1 needs vector `q - 1` of row i, which has just
    /// been computed, and vector 0 needs the last one; so row i+1 trails
    /// row i by one vector and its first vector comes last. Each cell is
    /// loaded and stored once for the two rows.
    #[inline(always)]
    fn step_pair<const GROWTH: bool>(
        &mut self,
        rsc0: &[[T; N]],
        growth0: Simd<T, N>,
        rsc1: &[[T; N]],
        growth1: Simd<T, N>,
        best: &mut Simd<T, N>,
    ) {
        let q = self.len();
        let floor_v = Simd::<T, N>::splat(T::MIN);
        let mut best0 = [floor_v; 2];
        let mut best1 = [floor_v; 2];
        // Vector 0 of row i; row i+1 has nothing to do yet.
        let mut previous = Simd::from_array(self[0]);
        let mut new0_previous = new_cell::<T, N, GROWTH>(
            Simd::from_array(self[q - 1]).shift_elements_right::<1>(T::MIN),
            rsc0[0],
            growth0,
        );
        best0[0] = new0_previous;
        let (cell_pairs, cell_rest) = self[1..].as_chunks_mut::<2>();
        let (rsc0_pairs, rsc0_rest) = rsc0[1..].as_chunks::<2>();
        let (rsc1_pairs, rsc1_rest) = rsc1[1..].as_chunks::<2>();
        for ((cell_pair, rsc0_pair), rsc1_pair) in
            cell_pairs.iter_mut().zip(rsc0_pairs).zip(rsc1_pairs)
        {
            for index in 0..2 {
                let old = Simd::from_array(cell_pair[index]);
                let new0 = new_cell::<T, N, GROWTH>(previous, rsc0_pair[index], growth0);
                let new1 = new_cell::<T, N, GROWTH>(new0_previous, rsc1_pair[index], growth1);
                best0[index] = best0[index].simd_max(new0);
                best1[index] = best1[index].simd_max(new1);
                cell_pair[index] = new1.to_array();
                previous = old;
                new0_previous = new0;
            }
        }
        for ((cell, rsc0_vector), rsc1_vector) in cell_rest.iter_mut().zip(rsc0_rest).zip(rsc1_rest)
        {
            let old = Simd::from_array(*cell);
            let new0 = new_cell::<T, N, GROWTH>(previous, *rsc0_vector, growth0);
            let new1 = new_cell::<T, N, GROWTH>(new0_previous, *rsc1_vector, growth1);
            best0[0] = best0[0].simd_max(new0);
            best1[1] = best1[1].simd_max(new1);
            *cell = new1.to_array();
            previous = old;
            new0_previous = new0;
        }
        // Vector 0 of row i+1, from the last vector of row i.
        let new1_first = new_cell::<T, N, GROWTH>(
            new0_previous.shift_elements_right::<1>(T::MIN),
            rsc1[0],
            growth1,
        );
        self[0] = new1_first.to_array();
        best1[0] = best1[0].simd_max(new1_first);
        *best = best
            .simd_max(best0[0].simd_max(best0[1]))
            .simd_max(best1[0].simd_max(best1[1]));
    }

    #[inline(always)]
    fn raise(&mut self, extra: Simd<T, N>, mut best: Option<&mut Simd<T, N>>) {
        for cell in self.iter_mut() {
            let raised = Simd::from_array(*cell).saturating_add(extra);
            *cell = raised.to_array();
            if let Some(best) = best.as_deref_mut() {
                *best = best.simd_max(raised);
            }
        }
    }

    #[inline(always)]
    fn save(&self, out: &mut Vec<T>) {
        for cell in self {
            out.extend_from_slice(cell);
        }
    }
}

impl<T, const N: usize, const Q: usize> Row<T, N> for [Simd<T, N>; Q]
where
    T: Unit,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    #[inline(always)]
    fn step<const GROWTH: bool>(
        &mut self,
        rsc_vectors: &[[T; N]],
        growth_v: Simd<T, N>,
        best: &mut Simd<T, N>,
    ) {
        let rsc_vectors: &[[T; N]; Q] = rsc_vectors[..Q].try_into().expect("row width");
        let mut mpv = self[Q - 1].shift_elements_right::<1>(T::MIN);
        let mut row_best = [Simd::splat(T::MIN); ACCUMULATORS];
        // Fully unrolled: every index is a constant, so the row stays in
        // registers and no cell is loaded or stored.
        for index in 0..Q {
            let sv = new_cell::<T, N, GROWTH>(mpv, rsc_vectors[index], growth_v);
            row_best[index % ACCUMULATORS] = row_best[index % ACCUMULATORS].simd_max(sv);
            mpv = self[index];
            self[index] = sv;
        }
        *best = best.simd_max(fold(row_best));
    }

    #[inline(always)]
    fn raise(&mut self, extra: Simd<T, N>, mut best: Option<&mut Simd<T, N>>) {
        for cell in self.iter_mut() {
            *cell = cell.saturating_add(extra);
            if let Some(best) = best.as_deref_mut() {
                *best = best.simd_max(*cell);
            }
        }
    }

    #[inline(always)]
    fn save(&self, out: &mut Vec<T>) {
        for cell in self {
            out.extend_from_slice(cell.as_array());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detected_isa_is_available_and_lane_counts_match_vector_bytes() {
        let isa = Isa::detect();
        assert!(isa.is_available());
        assert!(Isa::available().contains(&isa));
        for &isa in Isa::ALL {
            assert_eq!(isa.lanes::<i8>(), isa.vector_bytes());
            assert_eq!(isa.lanes::<i16>() * 2, isa.vector_bytes());
            for m in [0, 1, 2, 15, 16, 17, 100, 1000] {
                assert_eq!(
                    isa.vector_count::<i8>(m),
                    vector_count(m, isa.lanes::<i8>())
                );
                assert_eq!(
                    isa.vector_count::<i16>(m),
                    vector_count(m, isa.lanes::<i16>())
                );
            }
        }
    }
}
