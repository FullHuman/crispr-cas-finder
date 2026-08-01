// p7_gmx.rs - Generic dynamic programming matrix
//
// Port of src/p7_gmx.c

use crate::errors::HmmerError;
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, s};

/// Number of main DP cells per node: M, I, D
pub const NUM_MAIN_STATES: usize = 3;
pub const MATCH_CELL: usize = 0;
pub const INSERT_CELL: usize = 1;
pub const DELETE_CELL: usize = 2;

/// Number of special state cells: E, N, J, B, C
pub const NUM_SPECIAL_STATES: usize = 5;
pub const EXIT_STATE: usize = 0;
pub const N_STATE: usize = 1;
pub const J_STATE: usize = 2;
pub const BEGIN_STATE: usize = 3;
pub const C_STATE: usize = 4;

/// Generic DP matrix backed by ndarray.
///
/// `main[[i, k, s]]` = score for sequence position i, node k, state s (M/I/D).  
/// `special[[i, s]]` = score for sequence position i, special state s (E/N/J/B/C).
///
/// Shapes are `(alloc_l+1, alloc_m+1, 3)` and `(alloc_l+1, 5)` respectively.
/// Active dimensions are tracked in `num_nodes` (M) and `sequence_length` (L).
/// The arrays grow but never shrink (`resize()` re-allocates only when capacity
/// would be exceeded).
///
/// Main-state cells use [`f32::NEG_INFINITY`] for an unreachable log/max-plus
/// DP state. A dense `Option<f32>` representation is intentionally avoided:
/// it would increase every cell's size and add branching to the hottest DP
/// loops, while `-∞` is the algebraically correct identity used by HMMER.
#[derive(Debug)]
pub struct ScoreMatrix {
    pub num_nodes: usize,
    pub sequence_length: usize,
    alloc_m: usize,
    alloc_l: usize,
    pub(crate) main: Array3<f32>,    // shape: (alloc_l+1, alloc_m+1, 3)
    pub(crate) special: Array2<f32>, // shape: (alloc_l+1, 5)
}

impl ScoreMatrix {
    /// Create a DP matrix for a model of `num_nodes` nodes and a sequence of
    /// length `sequence_length`.
    pub fn new(num_nodes: usize, sequence_length: usize) -> Result<Self, HmmerError> {
        let (rows, columns) = checked_shape(num_nodes, sequence_length)?;

        let main = Array3::from_elem((rows, columns, NUM_MAIN_STATES), f32::NEG_INFINITY);
        let special = Array2::zeros((rows, NUM_SPECIAL_STATES));

        Ok(ScoreMatrix {
            num_nodes,
            sequence_length,
            alloc_m: num_nodes,
            alloc_l: sequence_length,
            main,
            special,
        })
    }

    /// Resize to fit a model of `m` nodes and a sequence of length `l`.
    ///
    /// Grows the arrays if the requested dimensions exceed current capacity;
    /// otherwise just updates the active dimensions. As with HMMER's
    /// `p7_gmx_GrowTo`, previously allocated contents are invalid after this
    /// call: callers must initialize every cell they will read. A newly
    /// allocated main-state array starts at [`f32::NEG_INFINITY`], but retained
    /// cells are deliberately not cleared because that would add an otherwise
    /// redundant O(LM) pass before each DP fill.
    pub fn resize(&mut self, m: usize, l: usize) -> Result<(), HmmerError> {
        checked_shape(m, l)?;

        if l > self.alloc_l || m > self.alloc_m {
            let new_l = l.max(self.alloc_l);
            let new_m = m.max(self.alloc_m);
            let (rows, columns) = checked_shape(new_m, new_l)?;
            self.main = Array3::from_elem((rows, columns, NUM_MAIN_STATES), f32::NEG_INFINITY);
            self.special = Array2::zeros((rows, NUM_SPECIAL_STATES));
            self.alloc_l = new_l;
            self.alloc_m = new_m;
        }

        self.num_nodes = m;
        self.sequence_length = l;
        Ok(())
    }

    /// Mark the retained allocation as an empty matrix for reuse.
    ///
    /// Call [`Self::resize`] before reading or writing a new active matrix.
    pub fn reuse(&mut self) {
        self.num_nodes = 0;
        self.sequence_length = 0;
    }

    /// Maximum model width currently retained by this allocation.
    #[inline]
    pub fn model_capacity(&self) -> usize {
        self.alloc_m
    }

    /// Maximum sequence length currently retained by this allocation.
    #[inline]
    pub fn sequence_capacity(&self) -> usize {
        self.alloc_l
    }

    // ── Named cell accessors ────────────────────────────────────────────────

    #[inline]
    pub fn match_score(&self, i: usize, k: usize) -> f32 {
        debug_assert!(i <= self.sequence_length && k <= self.num_nodes);
        self.main[[i, k, MATCH_CELL]]
    }

    #[inline]
    pub fn set_match_score(&mut self, i: usize, k: usize, val: f32) {
        debug_assert!(i <= self.sequence_length && k <= self.num_nodes);
        self.main[[i, k, MATCH_CELL]] = val;
    }

    #[inline]
    pub fn insert_score(&self, i: usize, k: usize) -> f32 {
        debug_assert!(i <= self.sequence_length && k <= self.num_nodes);
        self.main[[i, k, INSERT_CELL]]
    }

    #[inline]
    pub fn set_insert_score(&mut self, i: usize, k: usize, val: f32) {
        debug_assert!(i <= self.sequence_length && k <= self.num_nodes);
        self.main[[i, k, INSERT_CELL]] = val;
    }

    #[inline]
    pub fn delete_score(&self, i: usize, k: usize) -> f32 {
        debug_assert!(i <= self.sequence_length && k <= self.num_nodes);
        self.main[[i, k, DELETE_CELL]]
    }

    #[inline]
    pub fn set_delete_score(&mut self, i: usize, k: usize, val: f32) {
        debug_assert!(i <= self.sequence_length && k <= self.num_nodes);
        self.main[[i, k, DELETE_CELL]] = val;
    }

    #[inline]
    pub fn special_score(&self, i: usize, s: usize) -> f32 {
        debug_assert!(i <= self.sequence_length && s < NUM_SPECIAL_STATES);
        self.special[[i, s]]
    }

    #[inline]
    pub fn set_special_score(&mut self, i: usize, s: usize, val: f32) {
        debug_assert!(i <= self.sequence_length && s < NUM_SPECIAL_STATES);
        self.special[[i, s]] = val;
    }

    // ── Row-level slice accessors ───────────────────────────────────────────

    /// Returns a 2D view of main states at row `i`: shape `(num_nodes+1, 3)`.
    #[inline]
    pub fn main_row(&self, i: usize) -> ArrayView2<'_, f32> {
        debug_assert!(i <= self.sequence_length);
        self.main.slice(s![i, ..=self.num_nodes, ..])
    }

    /// Returns a mutable 2D view of main states at row `i`.
    #[inline]
    pub fn main_row_mut(&mut self, i: usize) -> ArrayViewMut2<'_, f32> {
        debug_assert!(i <= self.sequence_length);
        self.main.slice_mut(s![i, ..=self.num_nodes, ..])
    }

    /// Returns a 1D view of special states at row `i`: shape `(5,)`.
    #[inline]
    pub fn special_row(&self, i: usize) -> ArrayView1<'_, f32> {
        debug_assert!(i <= self.sequence_length);
        self.special.slice(s![i, ..])
    }

    /// Returns a mutable 1D view of special states at row `i`.
    #[inline]
    pub fn special_row_mut(&mut self, i: usize) -> ArrayViewMut1<'_, f32> {
        debug_assert!(i <= self.sequence_length);
        self.special.slice_mut(s![i, ..])
    }

    /// Fill all main cells (M/I/D) at row `i` with `val`.
    #[inline]
    pub fn fill_main_row(&mut self, i: usize, val: f32) {
        debug_assert!(i <= self.sequence_length);
        self.main.slice_mut(s![i, ..=self.num_nodes, ..]).fill(val);
    }

    /// Set all 5 special states at row `i` in one call: `[E, N, J, B, C]`.
    #[inline]
    pub fn set_special_row(&mut self, i: usize, vals: [f32; NUM_SPECIAL_STATES]) {
        debug_assert!(i <= self.sequence_length);
        for (cell, value) in self.special.slice_mut(s![i, ..]).iter_mut().zip(vals) {
            *cell = value;
        }
    }
}

/// Validate dimensions before passing them to `ndarray` allocation routines.
///
/// HMMER limits a generic matrix to half the address space on 32-bit systems.
/// Applying that bound to the combined main and special arrays both preserves
/// the intent and ensures every shape/byte calculation is checked explicitly.
fn checked_shape(m: usize, l: usize) -> Result<(usize, usize), HmmerError> {
    let rows = l
        .checked_add(1)
        .ok_or_else(|| HmmerError::Range("DP matrix sequence dimension overflows".into()))?;
    let columns = m
        .checked_add(1)
        .ok_or_else(|| HmmerError::Range("DP matrix model dimension overflows".into()))?;
    let main_cells = rows
        .checked_mul(columns)
        .and_then(|cells| cells.checked_mul(NUM_MAIN_STATES))
        .ok_or_else(|| HmmerError::Range("DP matrix cell count overflows".into()))?;
    let special_cells = rows
        .checked_mul(NUM_SPECIAL_STATES)
        .ok_or_else(|| HmmerError::Range("DP matrix special-state count overflows".into()))?;
    let total_cells = main_cells
        .checked_add(special_cells)
        .ok_or_else(|| HmmerError::Range("DP matrix total cell count overflows".into()))?;
    let total_bytes = total_cells
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HmmerError::Range("DP matrix byte size overflows".into()))?;

    if total_bytes > usize::MAX / 2 {
        return Err(HmmerError::Range(
            "DP matrix exceeds half of the address space".into(),
        ));
    }

    Ok((rows, columns))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{E, PI};

    #[test]
    fn test_create() {
        let gx = ScoreMatrix::new(100, 200).unwrap();
        assert_eq!(gx.num_nodes, 100);
        assert_eq!(gx.sequence_length, 200);
        assert_eq!(gx.model_capacity(), 100);
        assert_eq!(gx.sequence_capacity(), 200);
        assert_eq!(gx.main.dim(), (201, 101, NUM_MAIN_STATES));
        assert_eq!(gx.special.dim(), (201, NUM_SPECIAL_STATES));
    }

    #[test]
    fn test_resize() {
        let mut gx = ScoreMatrix::new(10, 20).unwrap();
        assert!(gx.resize(50, 100).is_ok());
        assert_eq!(gx.num_nodes, 50);
        assert_eq!(gx.sequence_length, 100);
        assert!(gx.main.dim().0 >= 101);
        assert!(gx.main.dim().1 >= 51);
    }

    #[test]
    fn growth_initializes_main_cells_as_unreachable() {
        let mut gx = ScoreMatrix::new(1, 1).unwrap();
        gx.main.fill(42.0);

        gx.resize(2, 2).unwrap();

        assert!(gx.main.iter().all(|score| *score == f32::NEG_INFINITY));
    }

    #[test]
    fn reuse_clears_active_dimensions_but_retains_capacity() {
        let mut gx = ScoreMatrix::new(10, 20).unwrap();
        gx.reuse();

        assert_eq!(gx.num_nodes, 0);
        assert_eq!(gx.sequence_length, 0);
        assert_eq!(gx.model_capacity(), 10);
        assert_eq!(gx.sequence_capacity(), 20);
    }

    #[test]
    fn oversized_dimensions_return_range_errors() {
        assert!(matches!(
            ScoreMatrix::new(usize::MAX, 0),
            Err(HmmerError::Range(_))
        ));

        let mut gx = ScoreMatrix::new(1, 1).unwrap();
        assert!(matches!(
            gx.resize(0, usize::MAX),
            Err(HmmerError::Range(_))
        ));
    }

    #[test]
    fn test_accessors() {
        let mut gx = ScoreMatrix::new(10, 10).unwrap();
        gx.set_match_score(1, 2, PI);
        assert!((gx.match_score(1, 2) - PI).abs() < 1e-6);
        gx.set_special_score(1, EXIT_STATE, E);
        assert!((gx.special_score(1, EXIT_STATE) - E).abs() < 1e-6);
    }

    #[test]
    fn test_row_slices() {
        let mut gx = ScoreMatrix::new(5, 5).unwrap();
        gx.set_match_score(2, 3, 42.0);
        assert!((gx.match_score(2, 3) - 42.0).abs() < 1e-6);
        gx.set_insert_score(2, 4, 99.0);
        assert!((gx.insert_score(2, 4) - 99.0).abs() < 1e-6);
    }

    #[test]
    fn test_fill_and_special_row() {
        let mut gx = ScoreMatrix::new(5, 5).unwrap();
        gx.fill_main_row(2, 42.0);
        assert!((gx.match_score(2, 0) - 42.0).abs() < 1e-6);
        assert!((gx.insert_score(2, 3) - 42.0).abs() < 1e-6);
        assert!((gx.delete_score(2, 5) - 42.0).abs() < 1e-6);

        gx.set_special_row(1, [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((gx.special_score(1, EXIT_STATE) - 1.0).abs() < 1e-6);
        assert!((gx.special_score(1, N_STATE) - 2.0).abs() < 1e-6);
        assert!((gx.special_score(1, C_STATE) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_main_row_view() {
        let mut gx = ScoreMatrix::new(5, 5).unwrap();
        gx.set_match_score(1, 2, 10.0);
        let row = gx.main_row(1);
        assert!((row[[2, MATCH_CELL]] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn main_row_views_expose_only_the_active_model_width() {
        let mut gx = ScoreMatrix::new(10, 5).unwrap();
        gx.resize(3, 2).unwrap();

        assert_eq!(gx.main_row(1).dim(), (4, NUM_MAIN_STATES));
        assert_eq!(gx.main_row_mut(1).dim(), (4, NUM_MAIN_STATES));
    }
}
