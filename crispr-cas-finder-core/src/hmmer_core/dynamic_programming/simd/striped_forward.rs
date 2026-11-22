// simd/striped_forward.rs — full-matrix Forward in HMMER's striped layout
//
// Port of `forward_engine(do_full = TRUE, ...)` (`p7_Forward()`) in
// src/impl_neon/fwdback.c and src/impl_sse/fwdback.c of HMMER 3.4, and of
// the stochastic traceback of that matrix in src/impl_*/stotrace.c.
//
// `forward_filter.rs` computes the same recurrence and keeps one row. This
// one keeps every row, because a stochastic traceback reads them, and it
// performs HMMER's floating-point operations in HMMER's order: which path a
// traceback takes is decided by comparing a random number with ratios of
// these cells, and HMMER's envelopes of a multi-domain region only come out
// if the same comparisons are made on the same numbers. Three things differ
// from the filter for that reason: the horizontal sum of the exit vector is
// pairwise, a rescaled row divides its special states by the scale instead
// of multiplying by the inverse, and the special states are updated with
// fused multiply-adds (see `FUSED_SPECIALS`).
//
// The cells can still differ from HMMER's in their last bit, because the
// profile's odds ratios can: HMMER exponentiates the profile's scores with a
// vectorised approximation of its own, this port with `f32::exp`. Of the
// 2,172 hits with a multi-domain region in the comparison with `hmmsearch`
// 3.4 (22 genomes, the CasFinder 3.1 profiles), one comes out with an
// envelope two residues away, a hit of -36.7 bits with seven domains.

use super::oprofile::*;
use crate::hmmer_core::rng::{EaselFastRng, normalize_probabilities};
use std::simd::prelude::*;

/// Whether `xC`, `xJ` and `xB` of a row are computed as `a * b + (c * d)`
/// with one rounding, as a compiler contracting floating-point expressions
/// does to `forward_engine()`.
///
/// HMMER's source writes these as plain C expressions, so its results depend
/// on the build: the arm64 build of HMMER 3.4 (Homebrew) fuses them, three
/// `fmadd` in the disassembly of its `forward_engine`, while an x86-64 build
/// without FMA cannot. The two forms differ in the last bit of some
/// special-state cells, which is enough for a traceback to take another
/// path now and then: in the comparison with that `hmmsearch`, the plain
/// form left three hits of 1,989 with a different envelope and the fused
/// form two (one of 2,172 since the profile's scores are computed in
/// HMMER's arithmetic too). This port follows the arm64 build, the one its regression
/// fixtures were generated with; `f32::mul_add` rounds once on every target,
/// so the port's own results do not depend on the platform.
const FUSED_SPECIALS: bool = true;

/// Cells per striped vector.
const LANES: usize = 4;
/// Match, delete and insert vectors of one stripe, in HMMER's order.
const MATCH: usize = 0;
const DELETE: usize = 1;
const INSERT: usize = 2;
const STATES: usize = 3;

/// Special states of a row: `[E, N, J, B, C, scale]`, as `P7_OMX.xmx`.
pub const X_E: usize = 0;
pub const X_N: usize = 1;
pub const X_J: usize = 2;
pub const X_B: usize = 3;
pub const X_C: usize = 4;
pub const X_SCALE: usize = 5;

/// A state of a sampled path, in the order a traceback visits them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathState {
    Match,
    Delete,
    Insert,
    N,
    B,
    E,
    C,
    J,
    S,
    T,
}

/// One step of a sampled path. `node` is the model node of a match, delete
/// or insert state and `residue` the row of a match or insert state; both
/// are zero otherwise.
#[derive(Debug, Clone, Copy)]
pub struct PathStep {
    pub state: PathState,
    pub node: u32,
    pub residue: u32,
}

/// A stochastic traceback could not be sampled: the Forward matrix
/// underflowed, overflowed or holds a value that is not a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingFailed;

/// A Forward matrix in the striped layout, every row kept.
#[derive(Debug, Default)]
pub struct StripedForward {
    /// Striped vectors per row.
    q: usize,
    /// Model nodes.
    m: usize,
    /// Rows after row 0.
    length: usize,
    /// `[row][stripe][MATCH | DELETE | INSERT][lane]`.
    cells: Vec<f32>,
    /// `[E, N, J, B, C, scale]` of each row.
    specials: Vec<[f32; 6]>,
}

#[inline(always)]
fn load(row: &[f32], stripe: usize, state: usize) -> f32x4 {
    f32x4::from_slice(&row[(stripe * STATES + state) * LANES..][..LANES])
}

#[inline(always)]
fn store(row: &mut [f32], stripe: usize, state: usize, value: f32x4) {
    value.copy_to_slice(&mut row[(stripe * STATES + state) * LANES..][..LANES]);
}

/// Shift the lanes up by one, a zero entering lane 0: node `k - 1` of the
/// last stripe is the left neighbour of node `k` of the first.
#[inline(always)]
fn shift_right(v: f32x4) -> f32x4 {
    simd_swizzle!(v, f32x4::splat(0.0), [4, 0, 1, 2])
}

/// `vaddvq_f32()` and `esl_sse_hsum_ps()`: `(v0 + v1) + (v2 + v3)`.
#[inline(always)]
fn horizontal_sum(v: f32x4) -> f32 {
    let v = v.to_array();
    (v[0] + v[1]) + (v[2] + v[3])
}

#[inline(always)]
fn multiply_add(a: f32, b: f32, c: f32) -> f32 {
    if FUSED_SPECIALS {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

impl StripedForward {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rows after row 0 of the last matrix filled.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[inline(always)]
    fn row(&self, i: usize) -> &[f32] {
        let stride = self.q * STATES * LANES;
        &self.cells[i * stride..][..stride]
    }

    /// The cell of `state` at node `k` (1-based) of `row`.
    #[inline(always)]
    fn cell(&self, row: &[f32], state: usize, k: usize) -> f32 {
        let stripe = (k - 1) % self.q;
        let lane = (k - 1) / self.q;
        row[(stripe * STATES + state) * LANES + lane]
    }

    /// The cell of `state` at node `k - 1` of `row`, zero for `k == 1`.
    #[inline(always)]
    fn left_cell(&self, row: &[f32], state: usize, k: usize) -> f32 {
        if k > 1 {
            self.cell(row, state, k - 1)
        } else {
            0.0
        }
    }

    /// Fill the matrix for `dsq` against `om`, as `p7_Forward()` does.
    ///
    /// The Forward score itself is not computed: no caller needs it, and
    /// the filter has it. Fails when the last row's C state is zero, infinite
    /// or not a number, HMMER's range errors.
    #[inline(never)]
    pub fn fill(&mut self, dsq: &[u8], om: &OptimizedProfile) -> Result<(), SamplingFailed> {
        let q = om.q4;
        let l = dsq.len();
        let stride = q * STATES * LANES;
        self.q = q;
        self.m = om.m;
        self.length = l;
        if self.cells.len() < (l + 1) * stride {
            self.cells.resize((l + 1) * stride, 0.0);
        }
        self.cells[..stride].fill(0.0);
        self.specials.clear();
        self.specials.reserve(l + 1);

        let dd_base = NTSC_PER_Q * q * LANES;
        let (regular_tfv, dd_tfv) = om.tfv.split_at(dd_base);
        let (transitions, _) = regular_tfv.as_chunks::<{ NTSC_PER_Q * LANES }>();
        let (dd, _) = dd_tfv.as_chunks::<LANES>();
        let dd = &dd[..q];
        let serialize = om.m < 100;

        let mut x_n = 1.0f32;
        let mut x_j = 0.0f32;
        let mut x_b = om.xf[XST_N][XTR_MOVE];
        let mut x_c = 0.0f32;
        self.specials.push([0.0, x_n, x_j, x_b, x_c, 1.0]);

        for (i, &residue) in dsq.iter().enumerate() {
            let i = i + 1;
            let (before, from_here) = self.cells.split_at_mut(i * stride);
            let previous = &before[(i - 1) * stride..];
            let current = &mut from_here[..stride];
            let residue = residue as usize;
            let rsc = if residue < om.kp {
                &om.rfv[residue]
            } else {
                &om.rfv[0]
            };

            let mut dcv = f32x4::splat(0.0);
            let mut xe_v = f32x4::splat(0.0);
            let xb_v = f32x4::splat(x_b);
            let mut mpv = shift_right(load(previous, q - 1, MATCH));
            let mut dpv = shift_right(load(previous, q - 1, DELETE));
            let mut ipv = shift_right(load(previous, q - 1, INSERT));

            for (stripe, (emission, t)) in rsc
                .as_chunks::<LANES>()
                .0
                .iter()
                .zip(transitions)
                .enumerate()
            {
                let transition = |index: usize| f32x4::from_slice(&t[index * LANES..][..LANES]);
                let mut sv = xb_v * transition(TSC_BM);
                sv += mpv * transition(TSC_MM);
                sv += ipv * transition(TSC_IM);
                sv += dpv * transition(TSC_DM);
                sv *= f32x4::from_array(*emission);
                xe_v += sv;

                mpv = load(previous, stripe, MATCH);
                dpv = load(previous, stripe, DELETE);
                ipv = load(previous, stripe, INSERT);

                store(current, stripe, MATCH, sv);
                store(current, stripe, DELETE, dcv);
                dcv = sv * transition(TSC_MD);

                let sv = mpv * transition(TSC_MI);
                store(current, stripe, INSERT, sv + ipv * transition(TSC_II));
            }

            // The delete paths. One complete pass adds M->D and D->D...
            dcv = shift_right(dcv);
            store(current, 0, DELETE, f32x4::splat(0.0));
            for (stripe, dd_t) in dd.iter().enumerate() {
                let delete = dcv + load(current, stripe, DELETE);
                store(current, stripe, DELETE, delete);
                dcv = delete * f32x4::from_array(*dd_t);
            }
            // ...and up to three more extend the D->D paths across stripes:
            // all three on short models, as long as a cell changes otherwise.
            for _pass in 1..4 {
                dcv = shift_right(dcv);
                let mut changed = serialize;
                for (stripe, dd_t) in dd.iter().enumerate() {
                    let old = load(current, stripe, DELETE);
                    let delete = dcv + old;
                    changed |= delete.simd_gt(old).any();
                    store(current, stripe, DELETE, delete);
                    dcv *= f32x4::from_array(*dd_t);
                }
                if !changed {
                    break;
                }
            }
            for stripe in 0..q {
                xe_v = load(current, stripe, DELETE) + xe_v;
            }

            let mut x_e = horizontal_sum(xe_v);
            x_n *= om.xf[XST_N][XTR_LOOP];
            x_c = multiply_add(x_c, om.xf[XST_C][XTR_LOOP], x_e * om.xf[XST_E][XTR_MOVE]);
            x_j = multiply_add(x_j, om.xf[XST_J][XTR_LOOP], x_e * om.xf[XST_E][XTR_LOOP]);
            x_b = multiply_add(x_j, om.xf[XST_J][XTR_MOVE], x_n * om.xf[XST_N][XTR_MOVE]);

            let mut scale = 1.0f32;
            if x_e > 1.0e4 {
                x_n /= x_e;
                x_c /= x_e;
                x_j /= x_e;
                x_b /= x_e;
                let inverse = f32x4::splat(1.0 / x_e);
                for cells in current.as_chunks_mut::<LANES>().0 {
                    *cells = (f32x4::from_array(*cells) * inverse).to_array();
                }
                scale = x_e;
                x_e = 1.0;
            }
            self.specials.push([x_e, x_n, x_j, x_b, x_c, scale]);
        }

        if x_c.is_nan() || x_c.is_infinite() || (l > 0 && x_c == 0.0) {
            return Err(SamplingFailed);
        }
        Ok(())
    }

    /// Sample one path from the matrix, as `p7_StochasticTrace()` does, into
    /// `path`, from its end (`T`) back to its start (`S`): reversed, that
    /// is, with respect to the order of the sequence.
    ///
    /// `om` must be the profile the matrix was filled with, unchanged.
    #[inline(never)]
    pub fn sample_path(
        &self,
        rng: &mut EaselFastRng,
        om: &OptimizedProfile,
        path: &mut Vec<PathStep>,
    ) -> Result<(), SamplingFailed> {
        use PathState::*;
        path.clear();
        let mut i = self.length;
        let mut k = 0usize;
        path.push(PathStep {
            state: T,
            node: 0,
            residue: 0,
        });
        path.push(PathStep {
            state: C,
            node: 0,
            residue: 0,
        });
        // A path of a valid matrix visits each row and each node of a
        // domain once; the bound only stops a matrix that is not one.
        let limit = 4 * (self.length + 2) * (self.m + 2) + 16;
        let mut current = C;
        while current != S {
            if path.len() > limit {
                return Err(SamplingFailed);
            }
            let next = match current {
                Match => {
                    let next = self.select_match(rng, om, i, k)?;
                    k -= 1;
                    i -= 1;
                    next
                }
                Delete => {
                    let next = self.select_delete(rng, om, i, k)?;
                    k -= 1;
                    next
                }
                Insert => {
                    let next = self.select_insert(rng, om, i, k)?;
                    i -= 1;
                    next
                }
                N => {
                    if i == 0 {
                        S
                    } else {
                        N
                    }
                }
                C => self.select_c(rng, om, i)?,
                J => self.select_j(rng, om, i)?,
                E => {
                    let (next, node) = self.select_exit(rng, i)?;
                    k = node;
                    next
                }
                B => self.select_begin(rng, om, i)?,
                S | T => return Err(SamplingFailed),
            };
            let (node, residue) = match next {
                Match | Insert => (k as u32, i as u32),
                Delete => (k as u32, 0),
                _ => (0, 0),
            };
            path.push(PathStep {
                state: next,
                node,
                residue,
            });
            if matches!(next, N | J | C) && next == current {
                i = i.checked_sub(1).ok_or(SamplingFailed)?;
            }
            current = next;
        }
        Ok(())
    }

    /// M(i,k) is reached from B(i-1), M(i-1,k-1), I(i-1,k-1) or D(i-1,k-1).
    #[inline]
    fn select_match(
        &self,
        rng: &mut EaselFastRng,
        om: &OptimizedProfile,
        i: usize,
        k: usize,
    ) -> Result<PathState, SamplingFailed> {
        if i == 0 || k == 0 {
            return Err(SamplingFailed);
        }
        let stripe = (k - 1) % self.q;
        let lane = (k - 1) / self.q;
        let previous = self.row(i - 1);
        let t = &om.tfv[stripe * NTSC_PER_Q * LANES..];
        let mut paths = [
            self.specials[i - 1][X_B] * t[TSC_BM * LANES + lane],
            self.left_cell(previous, MATCH, k) * t[TSC_MM * LANES + lane],
            self.left_cell(previous, INSERT, k) * t[TSC_IM * LANES + lane],
            self.left_cell(previous, DELETE, k) * t[TSC_DM * LANES + lane],
        ];
        normalize_probabilities(&mut paths);
        const STATES_OF: [PathState; 4] = [
            PathState::B,
            PathState::Match,
            PathState::Insert,
            PathState::Delete,
        ];
        Ok(STATES_OF[rng.choose(&paths).ok_or(SamplingFailed)?])
    }

    /// D(i,k) is reached from M(i,k-1) or D(i,k-1).
    #[inline]
    fn select_delete(
        &self,
        rng: &mut EaselFastRng,
        om: &OptimizedProfile,
        i: usize,
        k: usize,
    ) -> Result<PathState, SamplingFailed> {
        if k == 0 {
            return Err(SamplingFailed);
        }
        let row = self.row(i);
        // The transitions out of node k - 1, which sit in its stripe; a
        // node 0 has none.
        let (t_md, t_dd) = if k > 1 {
            let stripe = (k - 2) % self.q;
            let lane = (k - 2) / self.q;
            (
                om.tfv[(stripe * NTSC_PER_Q + TSC_MD) * LANES + lane],
                om.tfv[(self.q * NTSC_PER_Q + stripe) * LANES + lane],
            )
        } else {
            (0.0, 0.0)
        };
        let mut paths = [
            self.left_cell(row, MATCH, k) * t_md,
            self.left_cell(row, DELETE, k) * t_dd,
        ];
        normalize_probabilities(&mut paths);
        const STATES_OF: [PathState; 2] = [PathState::Match, PathState::Delete];
        Ok(STATES_OF[rng.choose(&paths).ok_or(SamplingFailed)?])
    }

    /// I(i,k) is reached from M(i-1,k) or I(i-1,k).
    #[inline]
    fn select_insert(
        &self,
        rng: &mut EaselFastRng,
        om: &OptimizedProfile,
        i: usize,
        k: usize,
    ) -> Result<PathState, SamplingFailed> {
        if i == 0 || k == 0 {
            return Err(SamplingFailed);
        }
        let stripe = (k - 1) % self.q;
        let lane = (k - 1) / self.q;
        let previous = self.row(i - 1);
        let t = &om.tfv[stripe * NTSC_PER_Q * LANES..];
        let mut paths = [
            self.cell(previous, MATCH, k) * t[TSC_MI * LANES + lane],
            self.cell(previous, INSERT, k) * t[TSC_II * LANES + lane],
        ];
        normalize_probabilities(&mut paths);
        const STATES_OF: [PathState; 2] = [PathState::Match, PathState::Insert];
        Ok(STATES_OF[rng.choose(&paths).ok_or(SamplingFailed)?])
    }

    /// C(i) is reached from C(i-1) or E(i).
    #[inline]
    fn select_c(
        &self,
        rng: &mut EaselFastRng,
        om: &OptimizedProfile,
        i: usize,
    ) -> Result<PathState, SamplingFailed> {
        if i == 0 {
            return Err(SamplingFailed);
        }
        let mut paths = [
            self.specials[i - 1][X_C] * om.xf[XST_C][XTR_LOOP],
            self.specials[i][X_E] * om.xf[XST_E][XTR_MOVE] * self.specials[i][X_SCALE],
        ];
        normalize_probabilities(&mut paths);
        const STATES_OF: [PathState; 2] = [PathState::C, PathState::E];
        Ok(STATES_OF[rng.choose(&paths).ok_or(SamplingFailed)?])
    }

    /// J(i) is reached from J(i-1) or E(i).
    #[inline]
    fn select_j(
        &self,
        rng: &mut EaselFastRng,
        om: &OptimizedProfile,
        i: usize,
    ) -> Result<PathState, SamplingFailed> {
        if i == 0 {
            return Err(SamplingFailed);
        }
        let mut paths = [
            self.specials[i - 1][X_J] * om.xf[XST_J][XTR_LOOP],
            self.specials[i][X_E] * om.xf[XST_E][XTR_LOOP] * self.specials[i][X_SCALE],
        ];
        normalize_probabilities(&mut paths);
        const STATES_OF: [PathState; 2] = [PathState::J, PathState::E];
        Ok(STATES_OF[rng.choose(&paths).ok_or(SamplingFailed)?])
    }

    /// E(i) is reached from any M(i,k) or D(i,k); E(i) itself normalizes
    /// them. Returns the state and its node.
    #[inline]
    fn select_exit(
        &self,
        rng: &mut EaselFastRng,
        i: usize,
    ) -> Result<(PathState, usize), SamplingFailed> {
        let roll = rng.random();
        let norm = 1.0f32 / self.specials[i][X_E];
        let row = self.row(i);
        let mut sum = 0.0f64;
        // HMMER repeats the scan until the roll is reached; a second scan
        // can only succeed where rounding left the first just short of it.
        for _scan in 0..2 {
            for stripe in 0..self.q {
                for state in [MATCH, DELETE] {
                    let cells = &row[(stripe * STATES + state) * LANES..][..LANES];
                    for (lane, &cell) in cells.iter().enumerate() {
                        sum += f64::from(cell * norm);
                        if roll < sum {
                            let node = lane * self.q + stripe + 1;
                            let state = if state == MATCH {
                                PathState::Match
                            } else {
                                PathState::Delete
                            };
                            return Ok((state, node));
                        }
                    }
                }
            }
            // Written to be false for a sum that is not a number, too.
            let usable = sum > 0.0 && sum.is_finite();
            if !usable {
                break;
            }
        }
        Err(SamplingFailed)
    }

    /// B(i) is reached from N(i) or J(i).
    #[inline]
    fn select_begin(
        &self,
        rng: &mut EaselFastRng,
        om: &OptimizedProfile,
        i: usize,
    ) -> Result<PathState, SamplingFailed> {
        let mut paths = [
            self.specials[i][X_N] * om.xf[XST_N][XTR_MOVE],
            self.specials[i][X_J] * om.xf[XST_J][XTR_MOVE],
        ];
        normalize_probabilities(&mut paths);
        const STATES_OF: [PathState; 2] = [PathState::N, PathState::J];
        Ok(STATES_OF[rng.choose(&paths).ok_or(SamplingFailed)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmmer_core::alphabet::Alphabet;
    use crate::hmmer_core::dynamic_programming::simd::forward_filter::forward_filter;
    use crate::hmmer_core::rng::XorShift64;
    use crate::hmmer_core::test_helpers::{random_digital_seq, setup_profile};

    /// A profile of `m` nodes configured multihit for a target of `l`
    /// residues, and a random target with two copies of a stretch of the
    /// model's consensus, so that sampled paths hold one or two domains.
    fn model_and_target(seed: u64, m: usize, l: usize) -> (OptimizedProfile, Vec<u8>) {
        let alphabet = Alphabet::amino();
        let mut rng = XorShift64::new(seed);
        let (_, background, profile) = setup_profile(&mut rng, m, l, &alphabet);
        let mut om = OptimizedProfile::from_profile(&profile);
        om.reconfig_multihit(l);
        let mut sequence = random_digital_seq(
            &mut rng,
            &background.residue_frequencies,
            alphabet.canonical_size,
            l,
        );
        let copy = m.min(l / 3);
        for start in [l / 10, l / 10 + copy + l / 10] {
            for (offset, k) in (1..=copy).enumerate() {
                let best = (0..alphabet.canonical_size)
                    .max_by(|&a, &b| {
                        profile.residue_scores_for(a)[k * 2]
                            .total_cmp(&profile.residue_scores_for(b)[k * 2])
                    })
                    .unwrap();
                if start + offset < l {
                    sequence[start + offset] = best as u8;
                }
            }
        }
        (om, sequence)
    }

    /// The matrix's last row gives the Forward score of the filter, which
    /// computes the same recurrence and differs in its last bits only.
    #[test]
    fn the_full_matrix_has_the_score_of_the_filter() {
        for (seed, m, l) in [(1u64, 7usize, 40usize), (2, 60, 150), (3, 150, 400)] {
            let (om, sequence) = model_and_target(seed, m, l);
            let mut forward = StripedForward::new();
            forward.fill(&sequence, &om).unwrap();
            assert_eq!(forward.len(), l);
            let scales: f64 = forward
                .specials
                .iter()
                .map(|row| f64::from(row[X_SCALE]).ln())
                .sum();
            let score = scales + f64::from(forward.specials[l][X_C] * om.xf[XST_C][XTR_MOVE]).ln();
            let filter = forward_filter(&sequence, l, &om);
            assert!(
                (score as f32 - filter).abs() < 1e-3,
                "m={m} l={l}: {score} vs {filter}"
            );
        }
    }

    /// Every sampled path is a path of the model over the whole target, and
    /// the same seed samples the same paths.
    #[test]
    fn sampled_paths_are_valid_and_reproducible() {
        for (seed, m, l) in [(4u64, 9usize, 60usize), (5, 60, 200), (6, 130, 300)] {
            let (om, sequence) = model_and_target(seed, m, l);
            let mut forward = StripedForward::new();
            forward.fill(&sequence, &om).unwrap();
            let mut rng = EaselFastRng::new(42);
            let mut path = Vec::new();
            let mut path_lengths = Vec::new();
            let mut domains = 0usize;
            for _ in 0..50 {
                forward.sample_path(&mut rng, &om, &mut path).unwrap();
                // Read from the start of the sequence.
                let steps: Vec<PathStep> = path.iter().rev().copied().collect();
                assert_eq!(steps.first().unwrap().state, PathState::S);
                assert_eq!(steps[1].state, PathState::N);
                assert_eq!(steps.last().unwrap().state, PathState::T);
                let mut emitted = 0usize;
                let mut last_residue = 0u32;
                let mut in_domain = false;
                let mut last_node = 0u32;
                for pair in steps.windows(2) {
                    let (from, to) = (pair[0], pair[1]);
                    match to.state {
                        PathState::Match | PathState::Insert => {
                            assert!(in_domain);
                            emitted += 1;
                            assert_eq!(to.residue, last_residue + 1);
                            last_residue = to.residue;
                            assert!(to.node >= 1 && to.node as usize <= m);
                            if from.state != PathState::B {
                                let step = u32::from(to.state == PathState::Match);
                                assert_eq!(to.node, last_node + step);
                            }
                            last_node = to.node;
                        }
                        PathState::Delete => {
                            assert!(in_domain);
                            assert_eq!(to.node, last_node + 1);
                            last_node = to.node;
                        }
                        PathState::B => {
                            in_domain = true;
                            domains += 1;
                        }
                        PathState::E => in_domain = false,
                        PathState::N | PathState::C | PathState::J => {
                            if from.state == to.state {
                                emitted += 1;
                                last_residue += 1;
                            }
                        }
                        PathState::S | PathState::T => {}
                    }
                }
                assert_eq!(emitted, l, "every residue is emitted once");
                path_lengths.push(steps.len());
            }
            assert!(domains >= 50, "m={m}: {domains} domains in 50 paths");

            let mut rng = EaselFastRng::new(42);
            for &length in &path_lengths {
                forward.sample_path(&mut rng, &om, &mut path).unwrap();
                assert_eq!(path.len(), length);
            }
        }
    }

    /// A target the model cannot emit has no Forward mass to sample from.
    #[test]
    fn an_impossible_target_fails_instead_of_looping() {
        let (mut om, sequence) = model_and_target(7, 20, 50);
        for emissions in om.rfv.iter_mut() {
            emissions.fill(0.0);
        }
        let mut forward = StripedForward::new();
        assert_eq!(forward.fill(&sequence, &om), Err(SamplingFailed));
    }
}
