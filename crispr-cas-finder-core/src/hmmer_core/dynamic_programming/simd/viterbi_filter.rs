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
// The recurrence is Farrar's striped one. HMMER evaluates the
// delete-to-delete paths of a row lazily: it skips them when no delete state
// could beat the begin state on the next row, and then keeps the delete
// cells it has, reached from a match state only. `lazy_row_cells` and
// `lazy_delete_paths` are that recurrence as HMMER writes it. The search
// runs `eager_row` instead, which evaluates the paths on every row within
// its main loop: the match cells, and with them the score, are HMMER's
// either way (see `eager_row`), in less time.
//
// Every cell of a model node, and the test that decides which rows skip,
// are integers that do not depend on the striping, so the result is the
// same for any vector width; the kernels run with the widest vectors the
// processor has, as the bounds of int_msv.rs do.

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
    /// Lanes per vector.
    pub lanes: usize,
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
    /// Whether that bound fits the 16-bit field HMMER keeps it in, and no
    /// delete-to-delete transition scores above zero. Both hold for any real
    /// profile; when one does not, HMMER's lazy evaluation of the
    /// delete-to-delete paths may leave out paths that count, and only its
    /// own recurrence reproduces what it then computes.
    pub dd_bound_is_exact: bool,
    /// For every second vector, the sum of its delete-to-delete transition
    /// and the next vector's, saturated: two steps of the delete chain in
    /// one (see `eager_row`).
    pub dd_pairs: Vec<i16>,
    /// Whether no sum of `dd_pairs` saturated where the delete state it
    /// leads to has a match state after it. Two saturating additions are
    /// one addition of the sum only then.
    pub dd_pairs_are_exact: bool,
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

        // HMMER keeps the running maximum in a 16-bit field: a gain that
        // does not fit is truncated as it is stored.
        let mut dd_bound = i16::MIN;
        let mut dd_bound_is_exact = true;
        for k in 2..m.saturating_sub(1) {
            let gain = i32::from(wordify(scale, transition(k, PTsc::DeleteToDelete)))
                + i32::from(wordify(scale, transition(k + 1, PTsc::DeleteToMatch)))
                - i32::from(wordify(scale, transition(k + 1, PTsc::BeginToMatch)));
            let largest = i32::from(dd_bound).max(gain);
            dd_bound_is_exact &= i16::try_from(largest).is_ok();
            dd_bound = largest as i16;
        }

        // The other transitions are capped at zero above, as HMMER caps
        // them; a delete-to-delete transition is not, and one above zero (a
        // probability above one) would undo the argument for evaluating
        // every path, which has scores only fall along a run of deletes.
        let dd_start = NTSC_PER_Q * q * lanes;
        dd_bound_is_exact &= transitions[dd_start..].iter().all(|&word| word <= 0);

        // Vectors 2j and 2j+1 hold, in lane z, the transitions out of nodes
        // k and k+1, k = 2j + 1 + z·q; their sum takes a delete cell from
        // node k to node k+2, which leads on to a match state when
        // k + 2 < M. Past that, no cell that counts follows.
        let mut dd_pairs = vec![i16::MIN; (q / 2) * lanes];
        let mut dd_pairs_are_exact = true;
        for pair in 0..q / 2 {
            for z in 0..lanes {
                let first = i32::from(transitions[dd_start + 2 * pair * lanes + z]);
                let second = i32::from(transitions[dd_start + (2 * pair + 1) * lanes + z]);
                let sum = first + second;
                dd_pairs[pair * lanes + z] = sum.max(i32::from(i16::MIN)) as i16;
                let k = 2 * pair + 1 + z * q;
                if sum < i32::from(i16::MIN) && k + 2 < m {
                    dd_pairs_are_exact = false;
                }
            }
        }

        Self {
            scale,
            base: 12000,
            kp,
            q,
            lanes,
            emissions,
            transitions,
            e_loop: wordify(scale, gm.special_scores.e_loop),
            e_move: wordify(scale, gm.special_scores.e_move),
            dd_bound,
            dd_bound_is_exact,
            dd_pairs,
            dd_pairs_are_exact,
        }
    }

    /// The match emission score of residue `x` at node `k`, from 1: an
    /// entry of `rwv`.
    #[inline]
    pub fn emission(&self, x: usize, k: usize) -> i16 {
        let (qi, z) = ((k - 1) % self.q, (k - 1) / self.q);
        self.emissions[(x * self.q + qi) * self.lanes + z]
    }

    /// The transition score of kind `kind` at node `k`, from 1: an entry of
    /// `twv`. `kind` is one of the `TSC_*` constants of oprofile.rs, for
    /// which nodes hold what the recurrence adds when it computes node `k`
    /// (the transitions into a match state are those from node `k - 1`), or
    /// `NTSC_PER_Q` for the delete-to-delete transition out of node `k`.
    #[inline]
    pub fn transition(&self, kind: usize, k: usize) -> i16 {
        let (qi, z) = ((k - 1) % self.q, (k - 1) / self.q);
        let vector = if kind < NTSC_PER_Q {
            qi * NTSC_PER_Q + kind
        } else {
            NTSC_PER_Q * self.q + qi
        };
        self.transitions[vector * self.lanes + z]
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
    // Evaluating every delete-to-delete path gives HMMER's score as long as
    // its bound on those paths is the one its lazy evaluation relies on.
    let words = &om.viterbi_words;
    let recurrence = if !words.dd_bound_is_exact {
        LAZY
    } else if words.dd_pairs_are_exact {
        EAGER_BY_PAIRS
    } else {
        EAGER
    };
    dispatch(dsq, l, om, rows, recurrence)
}

/// [`viterbi_filter_with`] by `p7_ViterbiFilter()`'s own recurrence, which
/// evaluates the delete-to-delete paths of a row only when its bound says
/// they may matter: the reference the faster recurrence is tested against,
/// and the one that runs when that bound does not fit HMMER's 16-bit field.
pub fn viterbi_filter_lazy_with(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    rows: &mut Rows,
) -> f32 {
    dispatch(dsq, l, om, rows, LAZY)
}

/// [`viterbi_filter_with`] with every delete-to-delete path evaluated one
/// vector at a time, whatever the profile: what runs when the sums of
/// `ViterbiWords::dd_pairs` are not exact, exposed for the tests.
pub fn viterbi_filter_eager_with(
    dsq: &[u8],
    l: usize,
    om: &OptimizedProfile,
    rows: &mut Rows,
) -> f32 {
    dispatch(dsq, l, om, rows, EAGER)
}

/// The recurrences of the delete-to-delete paths (see `eager_row`).
const LAZY: u8 = 0;
const EAGER: u8 = 1;
const EAGER_BY_PAIRS: u8 = 2;

/// Runs the kernel compiled for the instruction set of `om`.
#[inline(always)]
fn dispatch(dsq: &[u8], l: usize, om: &OptimizedProfile, rows: &mut Rows, recurrence: u8) -> f32 {
    // The kernels of an instruction set run only where it is available,
    // which `OptimizedProfile` guarantees of the one it records (see
    // `int_msv::Kernels`).
    match om.isa() {
        Isa::Portable => portable::viterbi_filter::<{ Isa::Portable.lanes::<i16>() }>(
            dsq, l, om, rows, recurrence,
        ),
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => unsafe {
            sse41::viterbi_filter::<{ Isa::Sse41.lanes::<i16>() }>(dsq, l, om, rows, recurrence)
        },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe {
            avx2::viterbi_filter::<{ Isa::Avx2.lanes::<i16>() }>(dsq, l, om, rows, recurrence)
        },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe {
            avx512::viterbi_filter::<{ Isa::Avx512.lanes::<i16>() }>(dsq, l, om, rows, recurrence)
        },
    }
}

/// The kernels compiled for one instruction set.
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
                recurrence: u8,
            ) -> f32 {
                match recurrence {
                    LAZY => viterbi_filter_impl::<N, $kernels, LAZY>(dsq, l, om, rows),
                    EAGER => viterbi_filter_impl::<N, $kernels, EAGER>(dsq, l, om, rows),
                    _ => viterbi_filter_impl::<N, $kernels, EAGER_BY_PAIRS>(dsq, l, om, rows),
                }
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

/// The word tables and DP rows of one call, vector by vector.
struct Cells<'a, const N: usize> {
    mmx: &'a mut [[i16; N]],
    imx: &'a mut [[i16; N]],
    dmx: &'a mut [[i16; N]],
    /// Seven transitions per vector, in the order of the `TSC_*` constants.
    transitions: &'a [[[i16; N]; NTSC_PER_Q]],
    /// The delete-to-delete transition out of each vector.
    dd_transitions: &'a [[i16; N]],
    /// `ViterbiWords::dd_pairs`.
    dd_pairs: &'a [[i16; N]],
}

/// [`viterbi_filter_with`] on vectors of `N` lanes with the shift of `K`;
/// `om`'s word tables must be striped for `N` lanes. `RECURRENCE` is `LAZY`
/// for `p7_ViterbiFilter()`'s own evaluation of the delete-to-delete paths,
/// `EAGER` or `EAGER_BY_PAIRS` for `eager_row`.
#[inline(always)]
fn viterbi_filter_impl<const N: usize, K: Kernels, const RECURRENCE: u8>(
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
    let (transitions, _) = words.transitions.as_chunks::<N>();
    let (main_transitions, dd_transitions) = transitions.split_at(NTSC_PER_Q * q);
    let (main_transitions, _) = main_transitions.as_chunks::<NTSC_PER_Q>();
    let (emissions, _) = words.emissions.as_chunks::<N>();
    assert!(emissions.len() >= words.kp * q, "emission table too short");
    assert!(
        q >= 2 && main_transitions.len() == q && dd_transitions.len() >= q,
        "transition table too short"
    );
    let mut cells = Cells::<N> {
        mmx: rows.mmx.as_chunks_mut::<N>().0,
        imx: rows.imx.as_chunks_mut::<N>().0,
        dmx: rows.dmx.as_chunks_mut::<N>().0,
        transitions: main_transitions,
        dd_transitions: &dd_transitions[..q],
        dd_pairs: words.dd_pairs.as_chunks::<N>().0,
    };
    assert!(cells.dd_pairs.len() >= q / 2, "pair table too short");
    let emissions_of = |residue: u8| -> &[[i16; N]] {
        let xi = usize::from(residue);
        let xi = if xi < words.kp { xi } else { 0 };
        &emissions[xi * q..][..q]
    };

    // The special states are 16-bit integers in HMMER, computed as `int`
    // and stored back; none of the stores below can wrap, as B never falls
    // below N's move from the base and E stays under the top of the word.
    let (e_loop, e_move) = (i32::from(words.e_loop), i32::from(words.e_move));
    let ncj_move_word = words.ncj_move(l, om.nj);
    let ncj_move = i32::from(ncj_move_word);
    let dd_bound = i32::from(words.dd_bound);
    // N costs nothing to stay in, so it stays at the base, and B at N's
    // move from it until a hit lifts J above N.
    let xn = i32::from(words.base);
    let begin = xn + ncj_move;
    let mut xb = begin;
    let mut xj = i32::from(i16::MIN);
    let mut xc = i32::from(i16::MIN);

    // While J has not risen above N, B is the same for every row: a row
    // then depends on the one before through its cells only, and the
    // processor starts it before the special states of the previous one are
    // known. The first row that lifts J hands over to the loop below.
    let mut row = 0;
    let begin_v = Simd::<i16, N>::splat(begin as i16);
    while row < l {
        let (xe, dmax, dcv) =
            row_cells::<N, K, RECURRENCE>(&mut cells, emissions_of(dsq[row]), begin_v);
        if xe >= i32::from(i16::MAX) {
            return f32::INFINITY;
        }
        xc = i32::from(xc.max(xe + e_move) as i16);
        xj = i32::from(xj.max(xe + e_loop) as i16);
        row += 1;
        let lifted = xj > xn;
        if lifted {
            xb = i32::from((xj + ncj_move) as i16);
        }
        if RECURRENCE == LAZY {
            // HMMER tests its bound against the B that follows the row.
            lazy_delete_paths::<N, K>(&mut cells, dcv, dmax + dd_bound > xb);
        }
        if lifted {
            break;
        }
    }
    for &residue in &dsq[row..l] {
        let xb_v = Simd::<i16, N>::splat(xb as i16);
        let (xe, dmax, dcv) =
            row_cells::<N, K, RECURRENCE>(&mut cells, emissions_of(residue), xb_v);
        if xe >= i32::from(i16::MAX) {
            return f32::INFINITY;
        }
        xc = i32::from(xc.max(xe + e_move) as i16);
        xj = i32::from(xj.max(xe + e_loop) as i16);
        xb = i32::from((xj + ncj_move).max(xn + ncj_move) as i16);
        if RECURRENCE == LAZY {
            lazy_delete_paths::<N, K>(&mut cells, dcv, dmax + dd_bound > xb);
        }
    }

    if xc > i32::from(i16::MIN) {
        (xc as f32 + f32::from(ncj_move_word) - f32::from(words.base)) / words.scale - 3.0
    } else {
        f32::NEG_INFINITY
    }
}

/// One row of the recurrence, updated in place: returns its largest match
/// cell and, when `RECURRENCE` is `LAZY`, what `lazy_delete_paths` needs to
/// settle its delete cells once the B that follows the row is known.
#[inline(always)]
fn row_cells<const N: usize, K: Kernels, const RECURRENCE: u8>(
    cells: &mut Cells<'_, N>,
    rsc_vectors: &[[i16; N]],
    xb_v: Simd<i16, N>,
) -> (i32, i32, Simd<i16, N>) {
    match RECURRENCE {
        LAZY => lazy_row_cells::<N, K>(cells, rsc_vectors, xb_v),
        EAGER => (
            eager_row::<N, K, false>(cells, rsc_vectors, xb_v),
            0,
            Simd::splat(i16::MIN),
        ),
        _ => (
            eager_row::<N, K, true>(cells, rsc_vectors, xb_v),
            0,
            Simd::splat(i16::MIN),
        ),
    }
}

/// The match and insert cells of one vector of a row, updated in place:
/// returns the new match cell and the previous row's match and insert cells
/// of the vector, which the next vector continues from.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn match_and_insert<const N: usize>(
    xb_v: Simd<i16, N>,
    mpv: Simd<i16, N>,
    dpv: Simd<i16, N>,
    ipv: Simd<i16, N>,
    tsc: &[[i16; N]; NTSC_PER_Q],
    rsc: &[i16; N],
    m: &mut [i16; N],
    i: &mut [i16; N],
) -> (Simd<i16, N>, Simd<i16, N>, Simd<i16, N>) {
    let t = |kind: usize| Simd::<i16, N>::from_array(tsc[kind]);
    // M(i,q) = max(B+tBM, M(i-1,q-1)+tMM, I(i-1,q-1)+tIM, D(i-1,q-1)+tDM) + rsc
    let mut sv = xb_v
        .saturating_add(t(TSC_BM))
        .simd_max(mpv.saturating_add(t(TSC_MM)));
    sv = sv.simd_max(ipv.saturating_add(t(TSC_IM)));
    sv = sv.simd_max(dpv.saturating_add(t(TSC_DM)));
    sv = sv.saturating_add(Simd::from_array(*rsc));
    let old_m = Simd::from_array(*m);
    let old_i = Simd::from_array(*i);
    *m = sv.to_array();
    // I(i,q) = max(M(i-1,q)+tMI, I(i-1,q)+tII)
    *i = old_m
        .saturating_add(t(TSC_MI))
        .simd_max(old_i.saturating_add(t(TSC_II)))
        .to_array();
    (sv, old_m, old_i)
}

/// Vectors of a row whose delete cells are corrected for paths from the
/// previous lane without asking first whether they need it (see
/// `eager_row`).
const UNASKED_CORRECTIONS: usize = 4;

/// One row with every delete-to-delete path evaluated: updates the cells in
/// place and returns the largest match cell.
///
/// HMMER leaves these paths out of a row when its bound shows that none can
/// beat the begin state on the next row. A match cell of the next row is
/// then the same with or without them, and the match cells are all a row
/// hands on (the delete cells of a row come from its own match cells), so
/// evaluating the paths on every row gives HMMER's cells and HMMER's score.
/// It is also cheaper than deciding: the delete cells then follow the match
/// cells through the same loop, where a pass of their own, on the half of
/// the rows that need one, waits on a chain as long as the row.
///
/// A delete cell continues from the one before it, in the vector before,
/// and the first vector's from the last vector's, one lane down. So the
/// last vector goes first: its match cell gives the delete cell entering
/// the first vector, as far as a match state reaches it. What is then
/// missing is a run of deletes crossing from one lane into the next, which
/// most rows have in some lane and which dies out within a few vectors; the
/// first `UNASKED_CORRECTIONS` vectors are corrected for it unconditionally
/// (a correction that changes nothing costs less than a branch that cannot
/// be predicted) and the rest only while a lane still improves.
///
/// The chain from one delete cell to the next is a saturating addition and
/// a maximum, and it is what a row waits for. With `PAIRS` it goes two
/// vectors at a time, D(q+2) = max(M(q+1)+tMD, M(q)+tMD+tDD, D(q)+tDD+tDD),
/// the two transitions of the last term added beforehand (`dd_pairs`): half
/// the wait. All these additions saturate at the bottom only and add
/// nothing above zero, so their order does not matter as long as the sum of
/// the two transitions is itself exact, which the caller has checked.
#[inline(always)]
fn eager_row<const N: usize, K: Kernels, const PAIRS: bool>(
    cells: &mut Cells<'_, N>,
    rsc_vectors: &[[i16; N]],
    xb_v: Simd<i16, N>,
) -> i32 {
    let q = cells.mmx.len();
    let last = q - 1;
    let load = |cell: &[i16; N]| Simd::<i16, N>::from_array(*cell);
    // The previous row's last vector, one lane down: what the first vector
    // continues from. Minus infinity enters node 1.
    let mut mpv = K::shift_right_one(load(&cells.mmx[last]));
    let mut dpv = K::shift_right_one(load(&cells.dmx[last]));
    let mut ipv = K::shift_right_one(load(&cells.imx[last]));

    // The last vector, from the previous row's last but one.
    let (before_last_m, last_m) = cells.mmx.split_at_mut(last);
    let (before_last_i, last_i) = cells.imx.split_at_mut(last);
    let (before_last_d, last_d) = cells.dmx.split_at_mut(last);
    let (sv_last, _, _) = match_and_insert(
        xb_v,
        load(&before_last_m[last - 1]),
        load(&before_last_d[last - 1]),
        load(&before_last_i[last - 1]),
        &cells.transitions[last],
        &rsc_vectors[last],
        &mut last_m[0],
        &mut last_i[0],
    );
    let from_last_match = sv_last.saturating_add(load(&cells.transitions[last][TSC_MD]));
    // The delete cell of the vector at hand.
    let mut dcv = K::shift_right_one(from_last_match);
    // Two running maxima, one per vector of a pair.
    let mut xe_v = [sv_last, Simd::splat(i16::MIN)];

    // The other vectors, two at a time.
    let (m_pairs, m_rest) = before_last_m.as_chunks_mut::<2>();
    let (i_pairs, i_rest) = before_last_i.as_chunks_mut::<2>();
    let (d_pairs, d_rest) = before_last_d.as_chunks_mut::<2>();
    let (tsc_pairs, tsc_rest) = cells.transitions[..last].as_chunks::<2>();
    let (dd_pairs, dd_rest) = cells.dd_transitions[..last].as_chunks::<2>();
    let (rsc_pairs, rsc_rest) = rsc_vectors[..last].as_chunks::<2>();
    let vectors = m_pairs
        .iter_mut()
        .zip(i_pairs.iter_mut())
        .zip(d_pairs.iter_mut());
    let tables = tsc_pairs
        .iter()
        .zip(dd_pairs)
        .zip(cells.dd_pairs)
        .zip(rsc_pairs);
    for (((m, i), d), (((tsc, dd), dd_sum), rsc)) in vectors.zip(tables) {
        let [m0, m1] = m;
        let [i0, i1] = i;
        let [d0, d1] = d;
        let (sv0, old_m, old_i) = match_and_insert(xb_v, mpv, dpv, ipv, &tsc[0], &rsc[0], m0, i0);
        let old_d = load(d0);
        *d0 = dcv.to_array();
        // D(i,q+1) = max(M(i,q)+tMD, D(i,q)+tDD)
        let from_match0 = sv0.saturating_add(load(&tsc[0][TSC_MD]));
        let next = from_match0.simd_max(dcv.saturating_add(load(&dd[0])));
        let (sv1, old_m, old_i) =
            match_and_insert(xb_v, old_m, old_d, old_i, &tsc[1], &rsc[1], m1, i1);
        dpv = load(d1);
        *d1 = next.to_array();
        let from_match1 = sv1.saturating_add(load(&tsc[1][TSC_MD]));
        dcv = if PAIRS {
            from_match1
                .simd_max(from_match0.saturating_add(load(&dd[1])))
                .simd_max(dcv.saturating_add(load(dd_sum)))
        } else {
            from_match1.simd_max(next.saturating_add(load(&dd[1])))
        };
        xe_v = [xe_v[0].simd_max(sv0), xe_v[1].simd_max(sv1)];
        mpv = old_m;
        ipv = old_i;
    }
    // One vector is left when the row has an even number of them.
    let vectors = m_rest
        .iter_mut()
        .zip(i_rest.iter_mut())
        .zip(d_rest.iter_mut());
    for (((m, i), d), ((tsc, dd), rsc)) in vectors.zip(tsc_rest.iter().zip(dd_rest).zip(rsc_rest)) {
        let (sv, _, _) = match_and_insert(xb_v, mpv, dpv, ipv, tsc, rsc, m, i);
        xe_v[0] = xe_v[0].simd_max(sv);
        *d = dcv.to_array();
        dcv = sv
            .saturating_add(load(&tsc[TSC_MD]))
            .simd_max(dcv.saturating_add(load(dd)));
    }
    last_d[0] = dcv.to_array();

    // Delete runs that cross from the last vector into the next lane.
    dcv = K::shift_right_one(dcv.saturating_add(load(&cells.dd_transitions[last])));
    let unasked = UNASKED_CORRECTIONS.min(q);
    let (first_d, rest_d) = cells.dmx.split_at_mut(unasked);
    let (first_dd, rest_dd) = cells.dd_transitions.split_at(unasked);
    for (d, dd) in first_d.iter_mut().zip(first_dd) {
        let cell = load(d).simd_max(dcv);
        *d = cell.to_array();
        dcv = cell.saturating_add(load(dd));
    }
    let mut improving = true;
    for (d, dd) in rest_d.iter_mut().zip(rest_dd) {
        let cell = load(d);
        if !dcv.simd_gt(cell).any() {
            improving = false;
            break;
        }
        let cell = cell.simd_max(dcv);
        *d = cell.to_array();
        dcv = cell.saturating_add(load(dd));
    }
    // A run that reached the end of the row again goes on, one more lane
    // down; a lane can need as many passes as there are lanes.
    while improving {
        dcv = K::shift_right_one(dcv);
        for (d, dd) in cells.dmx.iter_mut().zip(cells.dd_transitions) {
            let cell = load(d);
            if !dcv.simd_gt(cell).any() {
                improving = false;
                break;
            }
            let cell = cell.simd_max(dcv);
            *d = cell.to_array();
            dcv = cell.saturating_add(load(dd));
        }
    }

    i32::from(xe_v[0].simd_max(xe_v[1]).reduce_max())
}

/// The match and insert cells of a row as `p7_ViterbiFilter()` computes
/// them, and its delete cells as far as a match state reaches them, the
/// first vector's excepted: returns the largest match cell, the largest such
/// delete cell and the delete cells that enter the first vector, not yet
/// shifted.
#[inline(always)]
fn lazy_row_cells<const N: usize, K: Kernels>(
    cells: &mut Cells<'_, N>,
    rsc_vectors: &[[i16; N]],
    xb_v: Simd<i16, N>,
) -> (i32, i32, Simd<i16, N>) {
    let q = cells.mmx.len();
    let floor_v = Simd::<i16, N>::splat(i16::MIN);
    let load = |cell: &[i16; N]| Simd::<i16, N>::from_array(*cell);
    let mut dcv = floor_v;
    let mut xe_v = floor_v;
    let mut dmax_v = floor_v;
    let mut mpv = K::shift_right_one(load(&cells.mmx[q - 1]));
    let mut dpv = K::shift_right_one(load(&cells.dmx[q - 1]));
    let mut ipv = K::shift_right_one(load(&cells.imx[q - 1]));
    let vectors = cells
        .mmx
        .iter_mut()
        .zip(cells.imx.iter_mut())
        .zip(cells.dmx.iter_mut());
    for (((m, i), d), (tsc, rsc)) in vectors.zip(cells.transitions.iter().zip(rsc_vectors)) {
        let (sv, old_m, old_i) = match_and_insert(xb_v, mpv, dpv, ipv, tsc, rsc, m, i);
        xe_v = xe_v.simd_max(sv);
        mpv = old_m;
        ipv = old_i;
        dpv = load(d);
        *d = dcv.to_array();
        // D of the next vector, from M only for now.
        dcv = sv.saturating_add(load(&tsc[TSC_MD]));
        dmax_v = dmax_v.simd_max(dcv);
    }
    (
        i32::from(xe_v.reduce_max()),
        i32::from(dmax_v.reduce_max()),
        dcv,
    )
}

/// The "lazy F" step of `p7_ViterbiFilter()`: with `evaluate`, the
/// delete-to-delete paths of the row, one full pass then passes that stop at
/// the first vector the carry does not improve; without, the delete cells
/// stay as the match states left them.
#[inline(always)]
fn lazy_delete_paths<const N: usize, K: Kernels>(
    cells: &mut Cells<'_, N>,
    dcv: Simd<i16, N>,
    evaluate: bool,
) {
    let load = |cell: &[i16; N]| Simd::<i16, N>::from_array(*cell);
    let mut dcv = K::shift_right_one(dcv);
    if !evaluate {
        cells.dmx[0] = dcv.to_array();
        return;
    }
    for (d, dd) in cells.dmx.iter_mut().zip(cells.dd_transitions) {
        let cell = load(d).simd_max(dcv);
        *d = cell.to_array();
        dcv = cell.saturating_add(load(dd));
    }
    loop {
        dcv = K::shift_right_one(dcv);
        let mut completed = true;
        for (d, dd) in cells.dmx.iter_mut().zip(cells.dd_transitions) {
            let cell = load(d);
            if !dcv.simd_gt(cell).any() {
                completed = false;
                break;
            }
            let cell = cell.simd_max(dcv);
            *d = cell.to_array();
            dcv = cell.saturating_add(load(dd));
        }
        if !completed {
            break;
        }
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
