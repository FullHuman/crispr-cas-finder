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
#[derive(Debug)]
pub struct ScoreMatrix {
    pub num_nodes: usize,
    pub sequence_length: usize,
    alloc_m: usize,
    alloc_l: usize,
    pub main: Array3<f32>,    // shape: (alloc_l+1, alloc_m+1, 3)
    pub special: Array2<f32>, // shape: (alloc_l+1, 5)
}

impl ScoreMatrix {
    /// Create a new DP matrix with capacity for models up to `alloc_m` nodes and
    /// sequences up to length `alloc_l`.
    pub fn new(alloc_m: usize, alloc_l: usize) -> Option<Self> {
        let ncells = ((alloc_m + 1) as u64) * ((alloc_l + 1) as u64);
        if ncells > (usize::MAX / (NUM_MAIN_STATES * std::mem::size_of::<f32>()) / 2) as u64 {
            return None;
        }

        let main = Array3::from_elem(
            (alloc_l + 1, alloc_m + 1, NUM_MAIN_STATES),
            f32::NEG_INFINITY,
        );
        let special = Array2::zeros((alloc_l + 1, NUM_SPECIAL_STATES));

        Some(ScoreMatrix {
            num_nodes: alloc_m,
            sequence_length: 0,
            alloc_m,
            alloc_l,
            main,
            special,
        })
    }

    /// Resize to fit a model of `m` nodes and a sequence of length `l`.
    ///
    /// Grows the arrays if the requested dimensions exceed current capacity;
    /// otherwise just updates the active dimensions.
    pub fn resize(&mut self, m: usize, l: usize) -> Result<(), HmmerError> {
        let ncells = ((m + 1) as u64) * ((l + 1) as u64);
        if ncells > (usize::MAX / (NUM_MAIN_STATES * std::mem::size_of::<f32>()) / 2) as u64 {
            return Err(HmmerError::Internal(
                "DP matrix too large to allocate".into(),
            ));
        }

        if l > self.alloc_l || m > self.alloc_m {
            let new_l = l.max(self.alloc_l);
            let new_m = m.max(self.alloc_m);
            self.main = Array3::from_elem((new_l + 1, new_m + 1, NUM_MAIN_STATES), 0.0);
            self.special = Array2::zeros((new_l + 1, NUM_SPECIAL_STATES));
            self.alloc_l = new_l;
            self.alloc_m = new_m;
        }

        self.num_nodes = m;
        self.sequence_length = l;
        Ok(())
    }

    /// Recycle for reuse; must call `resize()` before indexing again.
    pub fn reuse(&mut self) {
        self.sequence_length = 0;
    }

    // ── Named cell accessors ────────────────────────────────────────────────

    #[inline]
    pub fn match_score(&self, i: usize, k: usize) -> f32 {
        self.main[[i, k, MATCH_CELL]]
    }

    #[inline]
    pub fn set_match_score(&mut self, i: usize, k: usize, val: f32) {
        self.main[[i, k, MATCH_CELL]] = val;
    }

    #[inline]
    pub fn insert_score(&self, i: usize, k: usize) -> f32 {
        self.main[[i, k, INSERT_CELL]]
    }

    #[inline]
    pub fn set_insert_score(&mut self, i: usize, k: usize, val: f32) {
        self.main[[i, k, INSERT_CELL]] = val;
    }

    #[inline]
    pub fn delete_score(&self, i: usize, k: usize) -> f32 {
        self.main[[i, k, DELETE_CELL]]
    }

    #[inline]
    pub fn set_delete_score(&mut self, i: usize, k: usize, val: f32) {
        self.main[[i, k, DELETE_CELL]] = val;
    }

    #[inline]
    pub fn special_score(&self, i: usize, s: usize) -> f32 {
        self.special[[i, s]]
    }

    #[inline]
    pub fn set_special_score(&mut self, i: usize, s: usize, val: f32) {
        self.special[[i, s]] = val;
    }

    // ── Row-level slice accessors ───────────────────────────────────────────

    /// Returns a 2D view of main states at row `i`: shape `(num_nodes+1, 3)`.
    #[inline]
    pub fn main_row(&self, i: usize) -> ArrayView2<'_, f32> {
        self.main.slice(s![i, .., ..])
    }

    /// Returns a mutable 2D view of main states at row `i`.
    #[inline]
    pub fn main_row_mut(&mut self, i: usize) -> ArrayViewMut2<'_, f32> {
        self.main.slice_mut(s![i, .., ..])
    }

    /// Returns a 1D view of special states at row `i`: shape `(5,)`.
    #[inline]
    pub fn special_row(&self, i: usize) -> ArrayView1<'_, f32> {
        self.special.slice(s![i, ..])
    }

    /// Returns a mutable 1D view of special states at row `i`.
    #[inline]
    pub fn special_row_mut(&mut self, i: usize) -> ArrayViewMut1<'_, f32> {
        self.special.slice_mut(s![i, ..])
    }

    /// Fill all main cells (M/I/D) at row `i` with `val`.
    #[inline]
    pub fn fill_main_row(&mut self, i: usize, val: f32) {
        self.main.slice_mut(s![i, .., ..]).fill(val);
    }

    /// Set all 5 special states at row `i` in one call: `[E, N, J, B, C]`.
    #[inline]
    pub fn set_special_row(&mut self, i: usize, vals: [f32; NUM_SPECIAL_STATES]) {
        self.special
            .slice_mut(s![i, ..])
            .assign(&ArrayView1::from(&vals[..]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{E, PI};

    #[test]
    fn test_create() {
        let gx = ScoreMatrix::new(100, 200).unwrap();
        assert_eq!(gx.num_nodes, 100);
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
}
