// simd/msv_filter.rs — the score of HMMER's MSV filter
//
// Port of `p7_MSVFilter()` (src/impl_*/msvfilter.c), of the `p7_SSVFilter()`
// it tries first (ssvfilter.c) and of the byte tables both read
// (`mf_conversion()` and `sf_conversion()` in p7_oprofile.c), HMMER 3.4.
//
// HMMER does not decide its MSV and bias filters on the MSV score of the
// profile but on what this filter returns: match emissions rounded to the
// nearest third of a bit and stored as byte costs, saturating byte
// arithmetic from a base of 190, constant costs for the begin, exit and
// J transitions, a constant -3 nats for the N, C and J loops, and plus
// infinity as soon as a cell reaches the top of the byte. The result differs
// from the MSV score of the profile by up to a nat or so, and many real hits
// sit at the thresholds, so the search pipeline (pipeline/search.rs) decides
// on this number too: the same sequences then pass as in `hmmsearch`.
//
// The arithmetic is on integers, so the result does not depend on how the
// cells are striped or on the width of the vectors.
//
// Single segments first. `p7_SSVFilter()` drops the J state: cells are signed
// bytes whose floor, -128, is the begin state, and one saturating addition
// per cell replaces the maximum with the begin state, the bias and the cost.
// That is the recurrence of int_msv.rs, and it runs on those kernels with
// HMMER's table. Its result stands when no segment scores high enough
// to pay for a second one through J; otherwise, and whenever the constants
// of the profile and the sequence length leave the shortcut unproven,
// `p7_MSVFilter()`'s own recurrence runs, as in HMMER.

use super::int_msv::{Isa, dispatch, largest_cell};
use super::oprofile::OptimizedProfile;
use crate::hmmer_core::config::{PROFILE_NUM_EMISSIONS, PRsc};
use crate::hmmer_core::profile::Profile;
use std::simd::prelude::*;

/// Lanes of the unsigned byte vectors of `p7_MSVFilter()`'s recurrence.
const LANES: usize = 16;

/// The byte tables of `P7_OPROFILE` the MSV filter reads.
#[derive(Debug, Clone, Default)]
pub struct MsvBytes {
    /// `scale_b`: byte units per nat, three per bit.
    pub scale: f32,
    /// `base_b`: the byte that stands for a score of zero.
    pub base: u8,
    /// `bias_b`: the cost that stands for the highest emission score.
    pub bias: u8,
    /// `tbm_b`: cost of entering any match state from B.
    pub tbm: u8,
    /// `tec_b`: cost of E to C, and of E to J.
    pub tec: u8,
    /// `tjb_b`: cost of N, J or C moving on, for the current target length.
    pub tjb: u8,
    /// Alphabet size of the tables.
    kp: usize,
    /// Vectors per row of `costs`: `p7O_NQB(M)`.
    q: usize,
    /// `rbv`: biased match emission costs, `kp` rows of `q` vectors of
    /// sixteen lanes; lane `z` of vector `qi` holds node `qi + 1 + z·q`.
    costs: Vec<u8>,
    /// The instruction set `gains` is striped for; an available one.
    isa: Isa,
    /// Vectors per row of `gains`.
    gain_q: usize,
    /// What a match emission adds to a cell of the single-segment
    /// recurrence: `bias - cost`, the negative of `sbv`, striped for the
    /// signed byte lanes of `isa`.
    gains: Vec<i8>,
}

/// `unbiased_byteify()`: a score in nats as a rounded, saturated byte cost.
#[inline]
pub fn unbiased_byteify(scale: f32, score: f32) -> u8 {
    let cost = -(scale * score).round();
    if cost > 255.0 { 255 } else { cost as u8 }
}

/// `biased_byteify()`: a match emission score as a byte cost above the bias.
#[inline]
pub fn biased_byteify(scale: f32, bias: u8, score: f32) -> u8 {
    let cost = -(scale * score).round();
    if f64::from(cost) > 255.0 - f64::from(bias) {
        255
    } else {
        (cost + f32::from(bias)) as u8
    }
}

impl MsvBytes {
    /// `mf_conversion()` and `sf_conversion()` for the profile `gm`, with
    /// the single-segment gains striped for the kernels of `isa`.
    pub fn from_profile(gm: &Profile, kp: usize, isa: Isa) -> Self {
        let m = gm.num_nodes;
        let scale = (3.0 / std::f64::consts::LN_2) as f32;
        let canonical = gm.alphabet.canonical_size;
        let scores = (m + 1) * PROFILE_NUM_EMISSIONS;
        let mut max = 0.0f32;
        for x in 0..canonical {
            for &score in &gm.residue_scores_for(x)[..scores] {
                if score > max {
                    max = score;
                }
            }
        }
        let bias = unbiased_byteify(scale, -max);
        let mut bytes = Self::from_costs(m, kp, bias, isa, |x, k| {
            let score =
                gm.residue_scores_for(x)[k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize];
            biased_byteify(scale, bias, score)
        });
        bytes.set_length(gm.target_length);
        bytes
    }

    /// The tables of a model of `m` nodes over `kp` symbols whose residue
    /// `x` costs `cost(x, k)` at node `k`, from 1, with HMMER's scale, base
    /// and transition costs, for a target of length zero. The single-segment
    /// gains are striped for the kernels of `isa`, or for the portable ones
    /// when this processor lacks it.
    pub fn from_costs(
        m: usize,
        kp: usize,
        bias: u8,
        isa: Isa,
        cost: impl Fn(usize, usize) -> u8,
    ) -> Self {
        let isa = if isa.is_available() {
            isa
        } else {
            Isa::Portable
        };
        let scale = (3.0 / std::f64::consts::LN_2) as f32;
        let q = 2.max(m.saturating_sub(1) / LANES + 1);
        let mut costs = vec![255u8; kp * q * LANES];
        let gain_lanes = isa.lanes::<i8>();
        let gain_q = isa.vector_count::<i8>(m);
        // A cost of 255 gives the lowest gain: the cell falls to the floor.
        let mut gains = vec![-127i8; kp * gain_q * gain_lanes];
        for x in 0..kp {
            for k in 1..=m {
                let cost = cost(x, k);
                let (qi, z) = ((k - 1) % q, (k - 1) / q);
                costs[(x * q + qi) * LANES + z] = cost;
                // `sbv` is `min(127, cost - bias)`, by an exclusive or on
                // `(bias + 127) - cost`; the cell loses that much.
                let gain = (i32::from(bias) - i32::from(cost)).clamp(-127, 127) as i8;
                let (qi, z) = ((k - 1) % gain_q, (k - 1) / gain_q);
                gains[(x * gain_q + qi) * gain_lanes + z] = gain;
            }
        }
        let mf = m as f32;
        Self {
            scale,
            base: 190,
            bias,
            tbm: unbiased_byteify(scale, (2.0f32 / (mf * (m + 1) as f32)).ln()),
            tec: unbiased_byteify(scale, 0.5f32.ln()),
            tjb: 0,
            kp,
            q,
            costs,
            isa,
            gain_q,
            gains,
        }
    }

    /// `p7_oprofile_ReconfigMSVLength()`: the cost of the N, J and C moves
    /// for a target of `l` residues.
    #[inline]
    pub fn set_length(&mut self, l: usize) {
        self.tjb = unbiased_byteify(self.scale, (3.0f32 / (l + 3) as f32).ln());
    }

    /// The instruction set the single-segment pass runs with.
    #[inline]
    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// The cost of residue `x` at node `k`, from 1: an entry of `rbv`.
    #[inline]
    pub fn cost(&self, x: usize, k: usize) -> u8 {
        let (qi, z) = ((k - 1) % self.q, (k - 1) / self.q);
        self.costs[(x * self.q + qi) * LANES + z]
    }

    /// The score, in nats, of a path whose J state ends on the byte `xj`.
    #[inline]
    fn score(&self, xj: i32) -> f32 {
        ((xj - i32::from(self.tjb)) as f32 - f32::from(self.base)) / self.scale - 3.0
    }
}

/// What `p7_MSVFilter()` returns for the sequence `dsq[..l]`: its score in
/// nats, or plus infinity when the bytes overflowed (`eslERANGE`), which
/// HMMER takes for a high-scoring hit.
///
/// `om` must have been reconfigured for length `l`.
pub fn msv_filter(dsq: &[u8], l: usize, om: &OptimizedProfile) -> f32 {
    msv_filter_with(dsq, l, om, &mut Vec::new(), &mut Vec::new())
}

/// [`msv_filter`] with caller-owned DP rows, so that a worker searching many
/// sequences does not allocate them per call.
#[inline(never)]
pub fn msv_filter_with(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    segment_row: &mut Vec<i8>,
    row: &mut Vec<u8>,
) -> f32 {
    msv_filter_of(&dsq[..l], &om.msv_bytes, segment_row, row)
}

/// [`msv_filter_with`] on the tables `bytes`, whose `tjb` must be the one of
/// the length of `dsq`.
pub fn msv_filter_of(
    dsq: &[u8],
    bytes: &MsvBytes,
    segment_row: &mut Vec<i8>,
    row: &mut Vec<u8>,
) -> f32 {
    let (base, bias) = (i32::from(bytes.base), i32::from(bytes.bias));
    let (tjb, tbm, tec) = (
        i32::from(bytes.tjb),
        i32::from(bytes.tbm),
        i32::from(bytes.tec),
    );

    // `p7_SSVFilter()`. Its cells run from the begin state at -128 up; one
    // within `bias` of the top may have wrapped, and the shortcut only holds
    // when that happens no sooner than a segment pays for the J state.
    if tjb + tbm + tec + bias < 127 {
        // The kernels of an instruction set run only where it is available,
        // which `MsvBytes::from_costs` has checked of the one it records.
        let largest = dispatch!(
            bytes.isa,
            i8,
            largest_cell(dsq, &bytes.gains, bytes.kp, bytes.gain_q, segment_row)
        );
        // The begin state in the bytes of `p7_MSVFilter()`.
        let begin = base - tjb - tbm;
        match largest {
            Some(units) if units < 127 - i64::from(bias) => {
                let xe = units as i32 + begin;
                if xe >= 255 - bias {
                    return f32::INFINITY;
                }
                let xj = xe - tec;
                if xj <= base {
                    return bytes.score(xj);
                }
                // A segment lifts J above the base: a second one may add to
                // it, which only the full recurrence tells.
            }
            _ => {
                if begin >= 128 {
                    // The full recurrence starts no lower and overflows too.
                    return f32::INFINITY;
                }
            }
        }
    }
    msv_rows(dsq, bytes, row)
}

/// `p7_MSVFilter()`'s own recurrence, with the J state.
fn msv_rows(dsq: &[u8], bytes: &MsvBytes, row: &mut Vec<u8>) -> f32 {
    type Bytes = Simd<u8, LANES>;
    let q = bytes.q;
    row.clear();
    row.resize(q * LANES, 0);
    let (cells, _) = row.as_chunks_mut::<LANES>();
    let (costs, _) = bytes.costs.as_chunks::<LANES>();
    assert!(costs.len() >= bytes.kp * q, "cost table too short");

    let bias_v = Bytes::splat(bytes.bias);
    // HMMER adds the two costs as integers and keeps the low byte.
    let tjbm = bytes.tjb.wrapping_add(bytes.tbm);
    let mut xj = 0u8;
    let mut xb = bytes.base.saturating_sub(tjbm);
    for &residue in dsq {
        let xi = usize::from(residue);
        let xi = if xi < bytes.kp { xi } else { 0 };
        let xb_v = Bytes::splat(xb);
        let mut xe_v = Bytes::splat(0);
        // The previous row shifted by one lane; zero, the floor, enters.
        let mut mpv = Bytes::from_array(cells[q - 1]).shift_elements_right::<1>(0);
        for (cell, cost) in cells.iter_mut().zip(&costs[xi * q..][..q]) {
            let sv = mpv
                .simd_max(xb_v)
                .saturating_add(bias_v)
                .saturating_sub(Bytes::from_array(*cost));
            xe_v = xe_v.simd_max(sv);
            mpv = Bytes::from_array(*cell);
            *cell = sv.to_array();
        }
        let xe = xe_v.reduce_max();
        if xe.saturating_add(bytes.bias) == u8::MAX {
            return f32::INFINITY;
        }
        xj = xj.max(xe.saturating_sub(bytes.tec));
        xb = bytes.base.max(xj).saturating_sub(tjbm);
    }
    bytes.score(i32::from(xj))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_costs_round_to_the_nearest_third_of_a_bit() {
        let scale = (3.0 / std::f64::consts::LN_2) as f32;
        // The example of p7_oprofile.c: +3.2 at scale 3 and bias 12 costs 2.
        assert_eq!(biased_byteify(3.0, 12, 3.2), 2);
        assert_eq!(unbiased_byteify(3.0, -2.1), 6);
        assert_eq!(unbiased_byteify(scale, 0.5f32.ln()), 3);
        assert_eq!(unbiased_byteify(scale, f32::NEG_INFINITY), 255);
        assert_eq!(biased_byteify(scale, 17, f32::NEG_INFINITY), 255);
        // A positive score has no unbiased cost.
        assert_eq!(unbiased_byteify(scale, 1.0), 0);
    }
}
