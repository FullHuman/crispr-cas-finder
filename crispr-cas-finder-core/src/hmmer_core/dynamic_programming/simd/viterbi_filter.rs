// simd/viterbi_filter.rs — the score of HMMER's Viterbi filter
//
// Port of `p7_ViterbiFilter()` (src/impl_*/vitfilter.c) and of the word
// tables it reads (`vf_conversion()` in p7_oprofile.c), HMMER 3.4.
//
// As for the MSV filter (msv_filter.rs), HMMER decides its Viterbi filter on
// what this function returns and not on the Viterbi score of the profile:
// scores rounded to the nearest five-hundredth of a bit in signed 16-bit
// words, saturating arithmetic from a base of 12000, no cost for the N, C
// and J loops but a constant -3 nats, and plus infinity as soon as a match
// cell reaches the top of the word.
//
// The recurrence is Farrar's striped one, with HMMER's lazy evaluation of
// the delete-to-delete paths: a row skips them when no delete state could
// beat the begin state on the next row, and then keeps the delete cells it
// has, reached from a match state only. Which rows skip is decided on
// integers that do not depend on the striping, and so is every cell of a
// model node, so the result is the same for any vector width; the kernel
// runs with the widest vectors the processor has, as the bounds of
// int_msv.rs do.

#[cfg(target_arch = "x86_64")]
use super::int_msv::{Avx2Kernels, Avx512Kernels, Sse41Kernels};
use super::int_msv::{Isa, Kernels, PortableKernels};
use super::oprofile::{
    NTSC_PER_Q, OptimizedProfile, TSC_BM, TSC_DM, TSC_II, TSC_IM, TSC_MD, TSC_MI, TSC_MM,
};
use crate::hmmer_core::config::{PROFILE_NUM_EMISSIONS, PROFILE_NUM_TRANSITIONS, PRsc, PTsc};
use crate::hmmer_core::profile::Profile;
use std::simd::prelude::*;

/// The word tables of `P7_OPROFILE` the Viterbi filter reads.
#[derive(Debug, Clone, Default)]
pub struct ViterbiWords {
    /// `scale_w`: word units per nat, five hundred per bit.
    pub scale: f32,
    /// `base_w`: the word that stands for a score of zero.
    pub base: i16,
    /// Alphabet size of the tables.
    pub kp: usize,
    /// Vectors per row.
    pub q: usize,
    /// `rwv`: match emission scores, `kp` rows of `q` vectors; lane `z` of
    /// vector `qi` holds node `qi + 1 + z·q`.
    pub emissions: Vec<i16>,
    /// `twv`: seven transitions per vector, in the order of the `TSC_*`
    /// constants of oprofile.rs, then the delete-to-delete transitions.
    pub transitions: Vec<i16>,
    /// `xw[E][LOOP]` and `xw[E][MOVE]`.
    pub e_loop: i16,
    pub e_move: i16,
    /// `ddbound_w`: the most a delete-to-delete path can gain on the begin
    /// transition into the node it leads to.
    pub dd_bound: i16,
}

/// `wordify()`: a score in nats as a rounded, saturated word.
#[inline]
pub fn wordify(scale: f32, score: f32) -> i16 {
    let word = (scale * score).round();
    if word >= 32767.0 {
        32767
    } else if word <= -32768.0 {
        -32768
    } else {
        word as i16
    }
}

impl ViterbiWords {
    /// `vf_conversion()` for the profile `gm`, striped for `lanes` lanes.
    /// The E transitions are those of the profile's mode as it stands.
    pub fn from_profile(gm: &Profile, kp: usize, lanes: usize) -> Self {
        let m = gm.num_nodes;
        let scale = (500.0 / std::f64::consts::LN_2) as f32;
        let q = 2.max(m.saturating_sub(1) / lanes + 1);
        let tsc = gm.transition_scores_raw();
        // `p7P_TSC(gm, k, kind)`. A profile has no node 0: only the begin
        // transition into node 1 is stored there.
        let transition = |k: usize, kind: PTsc| -> f32 {
            if k == 0 && kind != PTsc::BeginToMatch {
                return f32::NEG_INFINITY;
            }
            tsc.get(k * PROFILE_NUM_TRANSITIONS + kind.idx())
                .copied()
                .unwrap_or(f32::NEG_INFINITY)
        };

        let mut emissions = vec![i16::MIN; kp * q * lanes];
        for x in 0..kp {
            let rsc = gm.residue_scores_for(x);
            for k in 1..=m {
                let score = rsc[k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize];
                let (qi, z) = ((k - 1) % q, (k - 1) / q);
                emissions[(x * q + qi) * lanes + z] = wordify(scale, score);
            }
        }

        // Transitions into node k are stored at node k - 1; an
        // insert-to-insert transition may not cost nothing, "or hell may
        // occur".
        let kinds: [(PTsc, usize, i16); NTSC_PER_Q] = [
            (PTsc::BeginToMatch, 1, 0),
            (PTsc::MatchToMatch, 1, 0),
            (PTsc::InsertToMatch, 1, 0),
            (PTsc::DeleteToMatch, 1, 0),
            (PTsc::MatchToDelete, 0, 0),
            (PTsc::MatchToInsert, 0, 0),
            (PTsc::InsertToInsert, 0, -1),
        ];
        debug_assert!(
            [TSC_BM, TSC_MM, TSC_IM, TSC_DM, TSC_MD, TSC_MI, TSC_II] == [0, 1, 2, 3, 4, 5, 6]
        );
        let mut transitions = vec![i16::MIN; (NTSC_PER_Q + 1) * q * lanes];
        for qi in 0..q {
            for (slot, &(kind, back, highest)) in kinds.iter().enumerate() {
                for z in 0..lanes {
                    let kb = qi + 1 + z * q - back;
                    if kb < m {
                        transitions[(qi * NTSC_PER_Q + slot) * lanes + z] =
                            wordify(scale, transition(kb, kind)).min(highest);
                    }
                }
            }
            for z in 0..lanes {
                let k = qi + 1 + z * q;
                if k < m {
                    transitions[(NTSC_PER_Q * q + qi) * lanes + z] =
                        wordify(scale, transition(k, PTsc::DeleteToDelete));
                }
            }
        }

        let mut dd_bound = i32::from(i16::MIN);
        for k in 2..m.saturating_sub(1) {
            let bound = i32::from(wordify(scale, transition(k, PTsc::DeleteToDelete)))
                + i32::from(wordify(scale, transition(k + 1, PTsc::DeleteToMatch)))
                - i32::from(wordify(scale, transition(k + 1, PTsc::BeginToMatch)));
            dd_bound = dd_bound.max(bound);
        }

        Self {
            scale,
            base: 12000,
            kp,
            q,
            emissions,
            transitions,
            e_loop: wordify(scale, gm.special_scores.e_loop),
            e_move: wordify(scale, gm.special_scores.e_move),
            // HMMER keeps the bound in a 16-bit field.
            dd_bound: dd_bound as i16,
        }
    }

    /// `xw[N][MOVE]`, `xw[C][MOVE]` and `xw[J][MOVE]`, one score, as
    /// `p7_oprofile_ReconfigRestLength()` sets them for a target of `l`
    /// residues and `expected_j_uses` uses of J. The loops of N, C and J
    /// cost nothing. The filter runs on a small share of the targets, so
    /// the word is worked out there and not at every change of length.
    #[inline]
    pub fn ncj_move(&self, l: usize, expected_j_uses: f32) -> i16 {
        let pmove = (2.0f32 + expected_j_uses) / (l as f32 + 2.0f32 + expected_j_uses);
        wordify(self.scale, pmove.ln())
    }
}

/// DP rows of the Viterbi filter, kept by a worker across sequences: match,
/// insert and delete cells, vector by vector.
#[derive(Debug, Default)]
pub struct Rows {
    mmx: Vec<i16>,
    imx: Vec<i16>,
    dmx: Vec<i16>,
}

/// What `p7_ViterbiFilter()` returns for the sequence `dsq[..l]`: its score
/// in nats, plus infinity when a match cell overflowed (`eslERANGE`), which
/// HMMER takes for a high-scoring hit, or minus infinity when no path
/// reached the end.
///
/// `om` must have been reconfigured for length `l`.
pub fn viterbi_filter(dsq: &[u8], l: usize, om: &OptimizedProfile) -> f32 {
    viterbi_filter_with(dsq, l, om, &mut Rows::default())
}

/// [`viterbi_filter`] with caller-owned DP rows, so that a worker searching
/// many sequences does not allocate them per call.
#[inline(never)]
pub fn viterbi_filter_with(dsq: &[u8], l: usize, om: &OptimizedProfile, rows: &mut Rows) -> f32 {
    // The kernels of an instruction set run only where it is available,
    // which `OptimizedProfile` guarantees of the one it records (see
    // `int_msv::Kernels`).
    match om.isa() {
        Isa::Portable => {
            portable::viterbi_filter::<{ Isa::Portable.lanes::<i16>() }>(dsq, l, om, rows)
        }
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => unsafe {
            sse41::viterbi_filter::<{ Isa::Sse41.lanes::<i16>() }>(dsq, l, om, rows)
        },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe {
            avx2::viterbi_filter::<{ Isa::Avx2.lanes::<i16>() }>(dsq, l, om, rows)
        },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe {
            avx512::viterbi_filter::<{ Isa::Avx512.lanes::<i16>() }>(dsq, l, om, rows)
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
            pub(super) fn viterbi_filter<const N: usize>(
                dsq: &[u8],
                l: usize,
                om: &OptimizedProfile,
                rows: &mut Rows,
            ) -> f32 {
                viterbi_filter_impl::<N, $kernels>(dsq, l, om, rows)
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

/// [`viterbi_filter_with`] on vectors of `N` lanes with the shift of `K`;
/// `om`'s word tables must be striped for `N` lanes.
#[inline(always)]
fn viterbi_filter_impl<const N: usize, K: Kernels>(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    rows: &mut Rows,
) -> f32 {
    let words = &om.viterbi_words;
    let q = words.q;
    debug_assert_eq!(om.isa().lanes::<i16>(), N);
    for row in [&mut rows.mmx, &mut rows.imx, &mut rows.dmx] {
        row.clear();
        row.resize(q * N, i16::MIN);
    }
    let (mmx, _) = rows.mmx.as_chunks_mut::<N>();
    let (imx, _) = rows.imx.as_chunks_mut::<N>();
    let (dmx, _) = rows.dmx.as_chunks_mut::<N>();

    let (transitions, _) = words.transitions.as_chunks::<N>();
    let (main_transitions, dd_transitions) = transitions.split_at(NTSC_PER_Q * q);
    let (main_transitions, _) = main_transitions.as_chunks::<NTSC_PER_Q>();
    let (emissions, _) = words.emissions.as_chunks::<N>();
    assert!(emissions.len() >= words.kp * q, "emission table too short");
    assert!(dd_transitions.len() >= q, "transition table too short");

    // The special states are 16-bit integers in HMMER, computed as `int`
    // and stored back; none of the stores below can wrap, as B never falls
    // below N's move from the base and E stays under the top of the word.
    let (e_loop, e_move) = (i32::from(words.e_loop), i32::from(words.e_move));
    let ncj_move_word = words.ncj_move(l, om.nj);
    let ncj_move = i32::from(ncj_move_word);
    let dd_bound = i32::from(words.dd_bound);
    let xn = i32::from(words.base);
    let mut xb = xn + ncj_move;
    let mut xj = i32::from(i16::MIN);
    let mut xc = i32::from(i16::MIN);
    let floor_v = Simd::<i16, N>::splat(i16::MIN);

    for &residue in &dsq[..l] {
        let xi = usize::from(residue);
        let xi = if xi < words.kp { xi } else { 0 };
        let rsc_vectors = &emissions[xi * q..][..q];
        let xb_v = Simd::<i16, N>::splat(xb as i16);
        let mut dcv = floor_v;
        let mut xe_v = floor_v;
        let mut dmax_v = floor_v;

        // The previous row shifted by one lane (Farrar's shift); minus
        // infinity enters node 1.
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

            // D of the next vector, from M only for now.
            dcv = sv.saturating_add(t(TSC_MD));
            dmax_v = dmax_v.simd_max(dcv);

            // I(i,q) = max(M(i-1,q)+tMI, I(i-1,q)+tII)
            imx[qi] = mpv
                .saturating_add(t(TSC_MI))
                .simd_max(ipv.saturating_add(t(TSC_II)))
                .to_array();
        }

        let xe = i32::from(xe_v.reduce_max());
        if xe >= i32::from(i16::MAX) {
            return f32::INFINITY;
        }
        xc = i32::from(xc.max(xe + e_move) as i16);
        xj = i32::from(xj.max(xe + e_loop) as i16);
        xb = i32::from((xj + ncj_move).max(xn + ncj_move) as i16);

        // The delete-to-delete paths, lazily: when no delete state plus the
        // best such path can beat the begin state on the next row, none is
        // evaluated and the delete cells stay as the match states left them.
        let dmax = i32::from(dmax_v.reduce_max());
        dcv = K::shift_right_one(dcv);
        if dmax + dd_bound > xb {
            // One full pass, then passes that stop at the first vector the
            // carry does not improve; a lane can need as many passes as
            // there are lanes.
            for (dmx_q, dd) in dmx.iter_mut().zip(dd_transitions) {
                let cell = Simd::from_array(*dmx_q).simd_max(dcv);
                *dmx_q = cell.to_array();
                dcv = cell.saturating_add(Simd::from_array(*dd));
            }
            loop {
                dcv = K::shift_right_one(dcv);
                let mut completed = true;
                for (dmx_q, dd) in dmx.iter_mut().zip(dd_transitions) {
                    let cell = Simd::from_array(*dmx_q);
                    if !dcv.simd_gt(cell).any() {
                        completed = false;
                        break;
                    }
                    let cell = cell.simd_max(dcv);
                    *dmx_q = cell.to_array();
                    dcv = cell.saturating_add(Simd::from_array(*dd));
                }
                if !completed {
                    break;
                }
            }
        } else {
            dmx[0] = dcv.to_array();
        }
    }

    if xc > i32::from(i16::MIN) {
        (xc as f32 + f32::from(ncj_move_word) - f32::from(words.base)) / words.scale - 3.0
    } else {
        f32::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_round_to_the_nearest_unit_and_saturate() {
        let scale = (500.0 / std::f64::consts::LN_2) as f32;
        assert_eq!(wordify(500.0, 3.2), 1600);
        assert_eq!(wordify(scale, -std::f32::consts::LN_2), -500);
        assert_eq!(wordify(scale, f32::NEG_INFINITY), -32768);
        assert_eq!(wordify(scale, 100.0), 32767);
        assert_eq!(wordify(scale, 0.0), 0);
    }
}
