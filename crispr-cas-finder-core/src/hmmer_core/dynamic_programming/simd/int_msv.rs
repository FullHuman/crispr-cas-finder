// simd/int_msv.rs — the single-segment recurrence on integer cells
//
// HMMER's MSV filter tries single segments first (`p7_SSVFilter()`, see
// msv_filter.rs). An MSV path visits the model in one or more segments,
// diagonal runs of match states, and the recurrence of one segment is
//
//   U(i,k) = U(i-1,k-1) + units(x_i,k),
//
// on signed integer cells, 8-bit or 16-bit lanes packed into one vector.
// The floor of a cell is the begin state: saturating addition at the bottom
// is the maximum with the begin state, so a cell costs one saturating add
// and one max for the running maximum. `largest_cell` returns the largest
// cell of the recurrence, which is all the filter needs of it. Nothing is
// carried between rows except the cells themselves, so the processor
// overlaps consecutive rows freely.
//
// Rows. A row of up to 24 vectors is held in registers and updated from the
// last vector down, so that no cell has to be moved aside. A longer row
// stays in memory and goes four rows per pass along chains of cells, each
// loaded and stored once for four rows of work (`chained_rows`). Both find
// the emissions of a row without checking its residue, which is checked
// once for the whole sequence instead; a sequence with a residue outside
// the table goes one row at a time with every residue checked
// (`checked_rows`). The maxima are the same whatever the order the cells
// are taken in, so the three return the same number.
//
// Vector width. The recurrence is the same whatever the number of lanes:
// each cell depends on one cell of the previous row, and the result is the
// largest cell, so it comes out identical for any striping. The kernels
// therefore run with the widest vectors the processor has (`Isa`): 128-bit
// vectors everywhere, 256-bit with AVX2 and 512-bit with AVX-512BW, chosen
// once when the optimized profile is built and its tables striped. Each
// kernel is compiled once per instruction set, so that the portable
// `std::simd` operations lower to the native instructions of that set (on
// the x86-64 baseline, for instance, there is no 8-bit signed maximum, and
// every running maximum costs four instructions instead of one). The
// Viterbi filter (viterbi_filter.rs) takes its instruction sets and the
// shift between two rows from here (`Isa`, `Kernels`).

use std::simd::cmp::SimdOrd;
use std::simd::num::SimdInt;
use std::simd::{Simd, SimdElement};

/// Number of independent running maxima within a row.
const ACCUMULATORS: usize = 4;

// ---------- instruction sets ----------

/// The instruction set the integer kernels run with, and with it the width
/// of their vectors. Detected once per process with [`Isa::detect`]; the
/// optimized profile records the one its tables are striped for.
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

    /// The instruction set this processor runs a model of `m` nodes best
    /// with. 512-bit vectors pay off from about four vectors per row; below
    /// that, every row costs a fixed amount besides its vectors, and the
    /// shorter rows of 256-bit vectors come out ahead.
    pub fn for_model(m: usize) -> Isa {
        let isa = Self::detect();
        #[cfg(target_arch = "x86_64")]
        if isa == Isa::Avx512 && isa.vector_count::<i8>(m) < 4 {
            return Isa::Avx2;
        }
        let _ = m;
        isa
    }

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
/// types), as the generic entry point below reaches them.
pub trait Kernels {
    /// The instruction set the kernels are compiled for.
    const ISA: Isa;

    /// The largest cell of the single-segment recurrence over `dsq` (see
    /// [`largest_cell`]).
    fn largest_cell_of<T, const N: usize>(
        dsq: &[u8],
        emissions: &Emissions<'_, T, N>,
        m_row: &mut Vec<T>,
    ) -> Option<i64>
    where
        T: Unit,
        Simd<T, N>: SimdInt<Scalar = T> + SimdOrd;

    /// `v` shifted right by one lane, the floor entering lane 0: the Farrar
    /// shift between two rows. The portable lowering of a one-lane shift
    /// across a 256- or 512-bit vector takes several shuffles, where two
    /// instructions do; the wide instruction sets override this.
    #[inline(always)]
    fn shift_right_one<T, const N: usize>(v: Simd<T, N>) -> Simd<T, N>
    where
        T: Unit,
        Simd<T, N>: SimdInt<Scalar = T>,
    {
        v.shift_elements_right::<1>(T::MIN)
    }
}

/// Calls `$f::<$t, N, K>($args)` with the lane count `N` and the kernels `K`
/// of the instruction set `$isa`: the way msv_filter.rs reaches
/// [`largest_cell`] (`$f` names a generic function of that shape, in scope).
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
pub trait Unit: SimdElement + PartialEq {
    const MIN: Self;
    const MAX: Self;
    /// Units of this cell above the floor.
    fn units(self) -> i64;
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
        }
    };
}
unit!(i8, 128);
unit!(i16, 32_768);

/// The largest cell of the recurrence
///
///   U(i,k) = U(i-1,k-1) + units(x_i,k),
///
/// saturating, the floor being the begin state: in units above the begin
/// state, or `None` when a cell reached the top of its range. `units` holds
/// `kp` rows of `q` vectors of `N` lanes, lane `z` of vector `qi` holding
/// node `qi + 1 + z·q`. `m_row` is the DP row, kept by the caller so that a
/// worker searching many sequences does not allocate one per call; short
/// rows live in registers instead and leave it untouched. `N` is the lane
/// count `units` is striped for and `K` the kernels of its instruction set
/// (see `dispatch`).
///
/// This is the recurrence of HMMER's SSV filter (msv_filter.rs), which adds
/// one signed byte to each cell and lets the saturation at the bottom stand
/// for the begin state.
pub fn largest_cell<T, const N: usize, K>(
    dsq: &[u8],
    units: &[T],
    kp: usize,
    q: usize,
    m_row: &mut Vec<T>,
) -> Option<i64>
where
    T: Unit,
    K: Kernels,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    let (vectors, _) = units.as_chunks::<N>();
    assert!(vectors.len() >= kp * q, "emission table too short");
    let emissions = Emissions { vectors, kp, q };
    K::largest_cell_of::<T, N>(dsq, &emissions, m_row)
}

/// Striped match emissions of a profile: `kp` rows of `q` vectors, which
/// [`largest_cell`] has checked `vectors` to hold.
pub struct Emissions<'a, T: Unit, const N: usize> {
    vectors: &'a [[T; N]],
    kp: usize,
    q: usize,
}

/// Calls `$macro!` with its arguments followed by the row widths, in vectors,
/// that have a kernel holding the row in registers.
macro_rules! with_register_widths {
    ($macro:ident ! ( $($arguments:tt)* )) => {
        $macro!($($arguments)*; 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)
    };
}

/// Returns from the enclosing function with `$in_registers` for the row
/// width `$emissions.q`, when a row that wide is kept in registers: rows of
/// up to 24 vectors are, as the row, the running maxima and a few
/// temporaries must fit the 32 vector registers of AArch64 NEON and of
/// x86-64 with AVX-512. `T` and `N` are the enclosing function's cell type
/// and lane count; the call is unsafe, and the caller's to justify.
macro_rules! by_row_width {
    ($in_registers:ident; $dsq:ident, $emissions:ident) => {
        with_register_widths!(by_row_width!(
            @widths $in_registers; $dsq, $emissions
        ))
    };
    (@widths $in_registers:ident; $dsq:ident, $emissions:ident; $($width:literal)*) => {
        match $emissions.q {
            $($width => return $in_registers::<T, N, $width>($dsq, $emissions),)*
            _ => {}
        }
    };
}

/// The kernels of one instruction set: the row recurrence compiled with
/// that set's features, for rows in registers (one function per row width,
/// fully unrolled) and for rows in memory.
///
/// A kernel compiled for an instruction set must only run on a processor
/// that has it; the `Kernels` marker types are reached through an [`Isa`]
/// only (see [`dispatch`]), and the tables of an optimized profile only ever
/// record an available one.
macro_rules! isa_kernels {
    ($(#[$meta:meta])* $module:ident, $kernels:ident, $isa:expr $(, $feature:literal)*) => {
        isa_kernels!(@define $(#[$meta])* $module, $kernels, $isa $(, $feature)*; {});
    };
    (@define $(#[$meta:meta])* $module:ident, $kernels:ident, $isa:expr $(, $feature:literal)*; { $($shift:item)* }) => {
        $(#[$meta])*
        mod $module {
            use super::*;

            /// `rows_in_registers` compiled for this instruction set.
            ///
            /// # Safety
            ///
            /// As for `rows_in_registers`.
            #[inline(never)]
            $(#[target_feature(enable = $feature)])*
            unsafe fn in_registers<T, const N: usize, const Q: usize>(
                dsq: &[u8],
                emissions: &Emissions<'_, T, N>,
            ) -> Option<i64>
            where
                T: Unit,
                Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
            {
                // SAFETY: the caller's contract is this function's.
                unsafe { rows_in_registers::<T, N, Q, $kernels>(dsq, emissions) }
            }

            /// `chained_rows` compiled for this instruction set.
            ///
            /// # Safety
            ///
            /// As for `chained_rows`.
            #[inline(never)]
            $(#[target_feature(enable = $feature)])*
            unsafe fn in_chains<T, const N: usize>(
                dsq: &[u8],
                emissions: &Emissions<'_, T, N>,
                m_row: &mut Vec<T>,
            ) -> Option<i64>
            where
                T: Unit,
                Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
            {
                // SAFETY: the caller's contract is this function's.
                unsafe { chained_rows::<T, N, $kernels>(dsq, emissions, m_row) }
            }

            /// `checked_rows` compiled for this instruction set.
            #[inline(never)]
            $(#[target_feature(enable = $feature)])*
            fn checked<T, const N: usize>(
                dsq: &[u8],
                emissions: &Emissions<'_, T, N>,
                m_row: &mut Vec<T>,
            ) -> Option<i64>
            where
                T: Unit,
                Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
            {
                checked_rows::<T, N, $kernels>(dsq, emissions, m_row)
            }

            /// The largest cell of the recurrence over `dsq` (see
            /// `Kernels::largest_cell_of`).
            $(#[target_feature(enable = $feature)])*
            pub(super) fn largest_cell_of<T, const N: usize>(
                dsq: &[u8],
                emissions: &Emissions<'_, T, N>,
                m_row: &mut Vec<T>,
            ) -> Option<i64>
            where
                T: Unit,
                Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
            {
                if emissions.covers(dsq) {
                    // SAFETY: `covers` has just checked the residues of
                    // `dsq`.
                    unsafe {
                        by_row_width!(in_registers; dsq, emissions);
                        if emissions.q >= 2 * CHAIN_ROWS {
                            return in_chains::<T, N>(dsq, emissions, m_row);
                        }
                    }
                }
                checked::<T, N>(dsq, emissions, m_row)
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
            fn largest_cell_of<T, const N: usize>(
                dsq: &[u8],
                emissions: &Emissions<'_, T, N>,
                m_row: &mut Vec<T>,
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
                // `Isa` that the tables of an optimized profile recorded
                // after checking it is available on this processor (see
                // `Isa::detect` and `MsvBytes::from_costs`).
                unsafe { $module::largest_cell_of::<T, N>(dsq, emissions, m_row) }
            }

            $($shift)*
        }
    };
}

isa_kernels!(
    @define
    portable,
    PortableKernels,
    Isa::Portable;
    {
        /// One instruction on AArch64: the vector taken with the last lane
        /// of the floor before it. The portable lowering is a two-register
        /// table lookup, with its index vector loaded from memory.
        #[inline(always)]
        fn shift_right_one<T, const N: usize>(v: Simd<T, N>) -> Simd<T, N>
        where
            T: Unit,
            Simd<T, N>: SimdInt<Scalar = T>,
        {
            // Lanes are bytes in memory order on little-endian targets only.
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            {
                use core::arch::aarch64::{uint8x16_t, vextq_u8};
                if size_of::<Simd<T, N>>() == size_of::<uint8x16_t>() && size_of::<T>() <= 2 {
                    // SAFETY: the vector and `uint8x16_t` have the same size
                    // and both are plain data; NEON is part of the AArch64
                    // baseline.
                    unsafe {
                        let vector: uint8x16_t = core::mem::transmute_copy(&v);
                        let floor: uint8x16_t =
                            core::mem::transmute_copy(&Simd::<T, N>::splat(T::MIN));
                        let shifted = match size_of::<T>() {
                            1 => vextq_u8::<15>(floor, vector),
                            _ => vextq_u8::<14>(floor, vector),
                        };
                        return core::mem::transmute_copy(&shifted);
                    }
                }
            }
            v.shift_elements_right::<1>(T::MIN)
        }
    }
);
isa_kernels!(
    #[cfg(target_arch = "x86_64")]
    sse41,
    Sse41Kernels,
    Isa::Sse41,
    "sse4.1"
);
isa_kernels!(
    @define
    #[cfg(target_arch = "x86_64")]
    avx2,
    Avx2Kernels,
    Isa::Avx2,
    "avx2";
    {
        /// Two instructions: the vector with its low half moved up and the
        /// floor below it, then each 128-bit lane taken with the last lane
        /// of the previous one.
        #[inline(always)]
        fn shift_right_one<T, const N: usize>(v: Simd<T, N>) -> Simd<T, N>
        where
            T: Unit,
            Simd<T, N>: SimdInt<Scalar = T>,
        {
            use core::arch::x86_64::{__m256i, _mm256_alignr_epi8, _mm256_permute2x128_si256};
            if size_of::<Simd<T, N>>() != size_of::<__m256i>() {
                return v.shift_elements_right::<1>(T::MIN);
            }
            // SAFETY: the vector and `__m256i` have the same size and both
            // are plain data, and these kernels run only where AVX2 is
            // available (see `Kernels`).
            unsafe {
                let vector: __m256i = core::mem::transmute_copy(&v);
                let floor: __m256i = core::mem::transmute_copy(&Simd::<T, N>::splat(T::MIN));
                let carried = _mm256_permute2x128_si256::<0x20>(floor, vector);
                let shifted = match size_of::<T>() {
                    1 => _mm256_alignr_epi8::<15>(vector, carried),
                    _ => _mm256_alignr_epi8::<14>(vector, carried),
                };
                core::mem::transmute_copy(&shifted)
            }
        }
    }
);
isa_kernels!(
    @define
    #[cfg(target_arch = "x86_64")]
    avx512,
    Avx512Kernels,
    Isa::Avx512,
    "avx512bw";
    {
        /// Two instructions: the vector moved up one 128-bit lane with the
        /// floor below it, then each lane taken with the last lane of the
        /// previous one.
        #[inline(always)]
        fn shift_right_one<T, const N: usize>(v: Simd<T, N>) -> Simd<T, N>
        where
            T: Unit,
            Simd<T, N>: SimdInt<Scalar = T>,
        {
            use core::arch::x86_64::{__m512i, _mm512_alignr_epi8, _mm512_alignr_epi64};
            if size_of::<Simd<T, N>>() != size_of::<__m512i>() {
                return v.shift_elements_right::<1>(T::MIN);
            }
            // SAFETY: the vector and `__m512i` have the same size and both
            // are plain data, and these kernels run only where AVX-512BW is
            // available (see `Kernels`).
            unsafe {
                let vector: __m512i = core::mem::transmute_copy(&v);
                let floor: __m512i = core::mem::transmute_copy(&Simd::<T, N>::splat(T::MIN));
                let carried = _mm512_alignr_epi64::<6>(vector, floor);
                let shifted = match size_of::<T>() {
                    1 => _mm512_alignr_epi8::<15>(vector, carried),
                    _ => _mm512_alignr_epi8::<14>(vector, carried),
                };
                core::mem::transmute_copy(&shifted)
            }
        }
    }
);

/// Rows per pass of [`chained_rows`].
const CHAIN_ROWS: usize = 4;

impl<T: Unit, const N: usize> Emissions<'_, T, N> {
    /// Whether every residue of `dsq` has a row in the table. Those of a
    /// digital sequence of the profile's alphabet all have;
    /// [`UncheckedRows`] then finds a row's emissions without a check.
    fn covers(&self, dsq: &[u8]) -> bool {
        // The largest residue code, not `all`: this runs for every sequence
        // and every profile, and a reduction is vectorised where a search
        // that stops at the first match is not.
        let largest = dsq
            .iter()
            .fold(0u8, |largest, &residue| largest.max(residue));
        usize::from(largest) < self.kp
    }
}

/// Gives the emissions of a row without checking its residue (see
/// [`Emissions::covers`]).
struct UncheckedRows<'a, T: Unit, const N: usize> {
    table: *const [T; N],
    /// Vectors per row.
    q: usize,
    emissions: std::marker::PhantomData<&'a [[T; N]]>,
}

impl<'a, T: Unit, const N: usize> UncheckedRows<'a, T, N> {
    fn new(emissions: &Emissions<'a, T, N>) -> Self {
        assert!(
            emissions.vectors.len() >= emissions.kp * emissions.q,
            "emission table too short"
        );
        Self {
            table: emissions.vectors.as_ptr(),
            q: emissions.q,
            emissions: std::marker::PhantomData,
        }
    }

    /// The `q` emission vectors of a row whose residue is `residue`.
    ///
    /// # Safety
    ///
    /// `residue` must be under the `kp` of the emissions.
    #[inline(always)]
    unsafe fn of(&self, residue: u8) -> *const [T; N] {
        // SAFETY: the table holds `kp` rows of `q` vectors (checked in
        // `new`), and `residue < kp`.
        unsafe { self.table.add(usize::from(residue) * self.q) }
    }
}

/// The largest cell of the single-segment recurrence over `dsq`, starting
/// at the begin state, with the row of `Q` vectors held in registers.
///
/// # Safety
///
/// [`Emissions::covers`] must hold for `dsq`. The residues are not checked
/// again, and one outside the table would be read past its end.
#[inline(always)]
unsafe fn rows_in_registers<T, const N: usize, const Q: usize, K>(
    dsq: &[u8],
    emissions: &Emissions<'_, T, N>,
) -> Option<i64>
where
    T: Unit,
    K: Kernels,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    assert!(emissions.q == Q);
    let floor_v = Simd::<T, N>::splat(T::MIN);
    let mut cells = [floor_v; Q];
    let mut best = [floor_v; ACCUMULATORS];
    let rows = UncheckedRows::new(emissions);
    for &residue in dsq {
        // SAFETY: the residue is under `kp` by the caller's contract, and a
        // row has `Q` vectors.
        let rsc = unsafe { &*rows.of(residue).cast::<[[T; N]; Q]>() };
        cells.step::<K>(rsc, &mut best);
    }
    let best = fold(best).reduce_max();
    (best != T::MAX).then(|| best.units())
}

/// The largest cell of the single-segment recurrence over `dsq`, starting
/// at the begin state, four rows per pass over the row in memory.
///
/// # Safety
///
/// [`Emissions::covers`] must hold for `dsq`. The residues are not checked
/// again, and one outside the table would be read past its end.
#[inline(always)]
unsafe fn chained_rows<T, const N: usize, K>(
    dsq: &[u8],
    emissions: &Emissions<'_, T, N>,
    m_row: &mut Vec<T>,
) -> Option<i64>
where
    T: Unit,
    K: Kernels,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    let q = emissions.q;
    assert!(q >= 2 * CHAIN_ROWS);
    m_row.clear();
    m_row.resize(q * N, T::MIN);
    let (cells, _) = m_row.as_chunks_mut::<N>();
    let floor_v = Simd::<T, N>::splat(T::MIN);
    let mut best = [floor_v; ACCUMULATORS];
    let rows = UncheckedRows::new(emissions);
    let (passes, rest) = dsq.as_chunks::<CHAIN_ROWS>();
    for &[r0, r1, r2, r3] in passes {
        // SAFETY: the residues are under `kp` by the caller's contract; each
        // pointer is then the start of a table row of `q` vectors, and
        // `cells` has `q >= 2 * CHAIN_ROWS` vectors.
        unsafe {
            let rows = [rows.of(r0), rows.of(r1), rows.of(r2), rows.of(r3)];
            chain_pass::<T, N, K>(cells, rows, &mut best);
        }
    }
    for &residue in rest {
        // SAFETY: as above.
        let rsc = unsafe { std::slice::from_raw_parts(rows.of(residue), q) };
        cells.step::<K>(rsc, &mut best);
    }
    let best = fold(best).reduce_max();
    (best != T::MAX).then(|| best.units())
}

/// The largest cell of the single-segment recurrence over `dsq`, starting
/// at the begin state, one row at a time over the row in memory, with every
/// residue checked: one outside the table is taken for residue 0, as
/// `p7_MSVFilter()`'s own recurrence takes it (msv_filter.rs).
#[inline(always)]
fn checked_rows<T, const N: usize, K>(
    dsq: &[u8],
    emissions: &Emissions<'_, T, N>,
    m_row: &mut Vec<T>,
) -> Option<i64>
where
    T: Unit,
    K: Kernels,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    let q = emissions.q;
    m_row.clear();
    m_row.resize(q * N, T::MIN);
    let (cells, _) = m_row.as_chunks_mut::<N>();
    let mut best = [Simd::<T, N>::splat(T::MIN); ACCUMULATORS];
    for &residue in dsq {
        let xi = usize::from(residue);
        let xi = if xi < emissions.kp { xi } else { 0 };
        cells.step::<K>(&emissions.vectors[xi * q..][..q], &mut best);
    }
    let best = fold(best).reduce_max();
    (best != T::MAX).then(|| best.units())
}

/// Four rows of the recurrence in one pass over the row in memory.
///
/// A cell depends on one cell only, the one before it in the row above, so
/// the four rows come apart into chains: the cell of vector `c` gives, one
/// row after the other, those of vectors `c + 1` to `c + 4`, and the last
/// one is the new cell of vector `c + 4`. A chain loads one cell and stores
/// one for four cells of work, where a row at a time loads and stores each
/// of them, and no chain waits for another. The chains from the last four
/// vectors cross the end of the row, where the lanes shift (Farrar's
/// striping); they go first, as they read the cells the last chains of the
/// pass overwrite. Cells are updated in place: the old cell of a vector is
/// read just before the chain four vectors back stores the new one.
///
/// # Safety
///
/// Each of `rows` must point to as many vectors as `cells` has, and `cells`
/// must have at least `2 * CHAIN_ROWS` of them.
#[inline(always)]
unsafe fn chain_pass<T, const N: usize, K>(
    cells: &mut [[T; N]],
    rows: [*const [T; N]; CHAIN_ROWS],
    best: &mut [Simd<T, N>; ACCUMULATORS],
) where
    T: Unit,
    K: Kernels,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    let q = cells.len();
    debug_assert!(q >= 2 * CHAIN_ROWS);
    let [e0, e1, e2, e3] = rows;
    let cells = cells.as_mut_ptr();
    // The running maxima stay in registers: a store through `cells` cannot
    // be taken for one into them.
    let mut maxima = *best;
    // SAFETY: every vector index below is under `q` (see the comments), for
    // `cells` and, by the caller's contract, for the rows.
    unsafe {
        let load = |cell: *const [T; N]| Simd::<T, N>::from_array(*cell);
        let add = |cell: Simd<T, N>, rsc: *const [T; N]| cell.saturating_add(load(rsc));
        // The old cells of the first four vectors, which the crossing
        // chains are about to overwrite.
        let mut window = [
            load(cells),
            load(cells.add(1)),
            load(cells.add(2)),
            load(cells.add(3)),
        ];
        // The chains from vectors q-4 to q-1, each crossing to vector 0
        // one row earlier than the one before.
        let a0 = add(load(cells.add(q - 4)), e0.add(q - 3));
        let a1 = add(a0, e1.add(q - 2));
        let a2 = add(a1, e2.add(q - 1));
        let a3 = add(K::shift_right_one(a2), e3);
        let b0 = add(load(cells.add(q - 3)), e0.add(q - 2));
        let b1 = add(b0, e1.add(q - 1));
        let b2 = add(K::shift_right_one(b1), e2);
        let b3 = add(b2, e3.add(1));
        let c0 = add(load(cells.add(q - 2)), e0.add(q - 1));
        let c1 = add(K::shift_right_one(c0), e1);
        let c2 = add(c1, e2.add(1));
        let c3 = add(c2, e3.add(2));
        let d0 = add(K::shift_right_one(load(cells.add(q - 1))), e0);
        let d1 = add(d0, e1.add(1));
        let d2 = add(d1, e2.add(2));
        let d3 = add(d2, e3.add(3));
        *cells = a3.to_array();
        *cells.add(1) = b3.to_array();
        *cells.add(2) = c3.to_array();
        *cells.add(3) = d3.to_array();
        maxima[0] = maxima[0].simd_max(fold([a0, a1, a2, a3]));
        maxima[1] = maxima[1].simd_max(fold([b0, b1, b2, b3]));
        maxima[2] = maxima[2].simd_max(fold([c0, c1, c2, c3]));
        maxima[3] = maxima[3].simd_max(fold([d0, d1, d2, d3]));
        // The chains from vectors 0 to q-5, four at a time; chain `c` ends
        // at vector `c + 4 <= q - 1`.
        let chains = q - CHAIN_ROWS;
        // Chain `c` from the old cell of its vector: the new cell of vector
        // `c + 4` and the largest of the four cells on the way.
        let chain_from = |old: Simd<T, N>, c: usize| {
            let v0 = add(old, e0.add(c + 1));
            let v1 = add(v0, e1.add(c + 2));
            let v2 = add(v1, e2.add(c + 3));
            let v3 = add(v2, e3.add(c + 4));
            (v3, fold([v0, v1, v2, v3]))
        };
        let mut chain = 0;
        while chain + ACCUMULATORS <= chains {
            for index in 0..ACCUMULATORS {
                let c = chain + index;
                let next = load(cells.add(c + 4));
                let (last, largest) = chain_from(window[index], c);
                *cells.add(c + 4) = last.to_array();
                maxima[index] = maxima[index].simd_max(largest);
                window[index] = next;
            }
            chain += ACCUMULATORS;
        }
        for index in 0..chains - chain {
            let c = chain + index;
            let (last, largest) = chain_from(window[index], c);
            *cells.add(c + 4) = last.to_array();
            maxima[index] = maxima[index].simd_max(largest);
        }
    }
    *best = maxima;
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

/// A new cell from the shifted previous one and its emission.
#[inline(always)]
fn new_cell<T, const N: usize>(mpv: Simd<T, N>, rsc: [T; N]) -> Simd<T, N>
where
    T: Unit,
    Simd<T, N>: SimdInt<Scalar = T>,
{
    // U(i,q) = max(U(i-1,q-1), 0) + rsc: the floor is the begin state.
    mpv.saturating_add(Simd::from_array(rsc))
}

/// A DP row of match cells, in memory or in registers.
trait Row<T: Unit, const N: usize>
where
    T: Unit,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    /// One row of the recurrence: updates the cells in place and folds the
    /// new cells into the running maxima `best`. `K` provides the shift
    /// between rows.
    fn step<K: Kernels>(&mut self, rsc_vectors: &[[T; N]], best: &mut [Simd<T, N>; ACCUMULATORS]);
}

/// A row in memory, one array per vector; vectors are loaded and stored
/// with unaligned accesses.
impl<T, const N: usize> Row<T, N> for [[T; N]]
where
    T: Unit,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    #[inline(always)]
    fn step<K: Kernels>(&mut self, rsc_vectors: &[[T; N]], best: &mut [Simd<T, N>; ACCUMULATORS]) {
        let q = self.len();
        // Right-shift previous row by 1 lane (Farrar shift); the begin
        // state enters node 1.
        let mut mpv = K::shift_right_one(Simd::from_array(self[q - 1]));
        let (cell_groups, cell_rest) = self.as_chunks_mut::<ACCUMULATORS>();
        let (rsc_groups, rsc_rest) = rsc_vectors.as_chunks::<ACCUMULATORS>();
        for (cell_group, rsc_group) in cell_groups.iter_mut().zip(rsc_groups) {
            for index in 0..ACCUMULATORS {
                let sv = new_cell(mpv, rsc_group[index]);
                best[index] = best[index].simd_max(sv);
                mpv = Simd::from_array(cell_group[index]);
                cell_group[index] = sv.to_array();
            }
        }
        for (index, (cell, rsc_vector)) in cell_rest.iter_mut().zip(rsc_rest).enumerate() {
            let sv = new_cell(mpv, *rsc_vector);
            best[index] = best[index].simd_max(sv);
            mpv = Simd::from_array(*cell);
            *cell = sv.to_array();
        }
    }
}

impl<T, const N: usize, const Q: usize> Row<T, N> for [Simd<T, N>; Q]
where
    T: Unit,
    Simd<T, N>: SimdInt<Scalar = T> + SimdOrd,
{
    #[inline(always)]
    fn step<K: Kernels>(&mut self, rsc_vectors: &[[T; N]], best: &mut [Simd<T, N>; ACCUMULATORS]) {
        let rsc_vectors: &[[T; N]; Q] = rsc_vectors[..Q].try_into().expect("row width");
        let mpv = K::shift_right_one(self[Q - 1]);
        // Fully unrolled: every index is a constant, so the row stays in
        // registers and no cell is loaded or stored.
        // From the last vector down, each new cell can take the register
        // of the old one: the cell it is computed from, one vector lower,
        // has not been replaced yet. Going up, every cell would have to be
        // moved aside first.
        for index in (1..Q).rev() {
            self[index] = new_cell(self[index - 1], rsc_vectors[index]);
        }
        self[0] = new_cell(mpv, rsc_vectors[0]);
        // The new cells join the running maxima four at a time, as a tree:
        // a running maximum then takes one value in sixteen cells, and the
        // chain of maxima from one row to the next is no longer than the
        // row's additions.
        let (groups, rest) = self.as_chunks::<ACCUMULATORS>();
        for (group_index, group) in groups.iter().enumerate() {
            let slot = group_index % ACCUMULATORS;
            best[slot] = best[slot].simd_max(fold(*group));
        }
        if let Some((&first, others)) = rest.split_first() {
            let slot = groups.len() % ACCUMULATORS;
            let rest_max = others.iter().fold(first, |max, &cell| max.simd_max(cell));
            best[slot] = best[slot].simd_max(rest_max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmmer_core::rng::XorShift64;

    const LANES: usize = 16;
    const KP: usize = 29;

    /// The recurrence cell by cell on the unstriped table: node `k`, from
    /// 1, of residue `x` is lane `(k - 1) / q` of vector `(k - 1) % q`. A
    /// residue outside the table is taken for residue 0.
    fn cell_by_cell(dsq: &[u8], units: &[i8], q: usize) -> Option<i64> {
        let m = q * LANES;
        let mut row = vec![i8::MIN; m + 1];
        let mut largest = i8::MIN;
        for &residue in dsq {
            let xi = usize::from(residue);
            let xi = if xi < KP { xi } else { 0 };
            for k in (1..=m).rev() {
                let (qi, z) = ((k - 1) % q, (k - 1) / q);
                row[k] = row[k - 1].saturating_add(units[(xi * q + qi) * LANES + z]);
                largest = largest.max(row[k]);
            }
        }
        (largest != i8::MAX).then(|| i64::from(largest) + 128)
    }

    /// A random table of `q` vectors per row and a random sequence of `l`
    /// residues, one of them outside the table when `unknown_residue`.
    fn table_and_sequence(
        q: usize,
        l: usize,
        unknown_residue: bool,
        seed: u64,
    ) -> (Vec<i8>, Vec<u8>) {
        let mut rng = XorShift64::new(seed);
        let mut draw = |below: usize| rng.roll(below);
        let units: Vec<i8> = (0..KP * q * LANES)
            .map(|_| match draw(40) {
                0 => i8::MIN,
                // Mostly below zero, as emissions are, with a few high ones.
                value if value < 30 => -(value as i8) - 1,
                value => (value as i8 - 30) * 3,
            })
            .collect();
        let mut dsq: Vec<u8> = (0..l).map(|_| draw(KP) as u8).collect();
        if unknown_residue {
            dsq[draw(l)] = KP as u8;
        }
        (units, dsq)
    }

    #[test]
    fn kernels_return_the_largest_cell_of_the_recurrence() {
        let mut seed = 1;
        // Rows of up to 24 vectors are held in registers and longer ones go
        // four rows per pass; a row of one vector, and any row of a sequence
        // with a residue outside the table, goes one row at a time.
        for q in [
            1, 2, 3, 4, 7, 8, 9, 15, 16, 23, 24, 25, 26, 27, 28, 29, 33, 40, 64, 101,
        ] {
            // The lengths leave zero to three rows after the last pass.
            for l in [1, 5, 42, 43, 44, 45, 46, 47, 96, 257, 600] {
                for unknown_residue in [false, true] {
                    seed += 1;
                    let (units, dsq) = table_and_sequence(q, l, unknown_residue, seed);
                    let largest = largest_cell::<i8, LANES, PortableKernels>(
                        &dsq,
                        &units,
                        KP,
                        q,
                        &mut Vec::new(),
                    );
                    assert_eq!(
                        largest,
                        cell_by_cell(&dsq, &units, q),
                        "q={q} l={l} seed={seed}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_cell_at_the_top_of_its_range_gives_no_largest_cell() {
        for q in [1, 2, 24, 25] {
            let units = vec![100i8; KP * q * LANES];
            let largest_of = |dsq: &[u8]| {
                largest_cell::<i8, LANES, PortableKernels>(dsq, &units, KP, q, &mut Vec::new())
            };
            // One residue lifts a cell 100 units above the begin state; the
            // third takes it past the top of a signed byte.
            assert_eq!(largest_of(&[0]), Some(100), "q={q}");
            assert_eq!(largest_of(&[0, 1, 2]), None, "q={q}");
        }
    }

    #[test]
    fn only_sequences_within_the_table_go_unchecked() {
        let vectors = vec![[0i8; LANES]; 2 * KP];
        let emissions = Emissions {
            vectors: &vectors,
            kp: KP,
            q: 2,
        };
        assert!(emissions.covers(&[]));
        assert!(emissions.covers(&[0, 28]));
        assert!(!emissions.covers(&[0, 29]));
    }

    #[test]
    fn detected_isa_is_available_and_lane_counts_match_vector_bytes() {
        let isa = Isa::detect();
        assert!(isa.is_available());
        assert!(Isa::available().contains(&isa));
        for &isa in Isa::ALL {
            assert_eq!(isa.lanes::<i8>(), isa.vector_bytes());
            assert_eq!(isa.lanes::<i16>() * 2, isa.vector_bytes());
            for m in [0usize, 1, 2, 15, 16, 17, 100, 1000] {
                // At least two vectors, and enough lanes for `m` nodes.
                let vectors = |lanes: usize| 2.max(m.saturating_sub(1) / lanes + 1);
                assert_eq!(isa.vector_count::<i8>(m), vectors(isa.lanes::<i8>()));
                assert_eq!(isa.vector_count::<i16>(m), vectors(isa.lanes::<i16>()));
            }
        }
    }
}
