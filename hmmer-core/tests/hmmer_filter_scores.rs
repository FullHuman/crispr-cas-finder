//! The scores of HMMER's own MSV and Viterbi filters against cell-by-cell
//! transcriptions of `p7_MSVFilter()` (with `p7_SSVFilter()`) and
//! `p7_ViterbiFilter()`, HMMER 3.4. The transcriptions work on the unstriped
//! tables in the arithmetic of the C code; the kernels must return the same
//! bits on every instruction set.
use hmmer_core::{
    alphabet::Alphabet,
    background::BackgroundModel,
    config::{HTransition, SearchMode},
    hmm::Hmm,
    modelconfig,
    profile::Profile,
    rng::XorShift64,
    simd::{
        int_msv::Isa,
        msv_filter::{MsvBytes, msv_filter, msv_filter_of},
        oprofile::{
            NTSC_PER_Q, OptimizedProfile, TSC_BM, TSC_DM, TSC_II, TSC_IM, TSC_MD, TSC_MI, TSC_MM,
        },
        viterbi_filter::{self, ViterbiWords, viterbi_filter},
    },
    test_helpers::{hmm_sample, random_digital_seq},
};
use std::collections::HashMap;

/// Which part of `p7_MSVFilter()` produced a score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MsvPath {
    /// `p7_SSVFilter()` returned the score.
    SingleSegment,
    /// `p7_SSVFilter()` overflowed where the full recurrence would too.
    SingleSegmentOverflow,
    /// The full recurrence ran: the shortcut is unproven for the profile
    /// and length, a segment paid for the J state, or the single-segment
    /// pass overflowed where the full recurrence might not.
    Full,
}

/// `p7_SSVFilter()` and `p7_MSVFilter()` cell by cell on the unstriped cost
/// table, in the arithmetic of the C code: signed bytes from -128 with a
/// saturating subtraction of `sbv` and an unsigned maximum for the
/// single-segment pass, unsigned saturating bytes for the full recurrence.
fn scalar_hmmer_msv(seq: &[u8], m: usize, bytes: &MsvBytes) -> (f32, MsvPath) {
    let (base, bias, tjb, tbm, tec) = (bytes.base, bytes.bias, bytes.tjb, bytes.tbm, bytes.tec);
    let score = |xj: i32| ((xj - i32::from(tjb)) as f32 - f32::from(base)) / bytes.scale - 3.0;
    let wide = |byte: u8| i32::from(byte);

    if wide(tjb) + wide(tbm) + wide(tec) + wide(bias) < 127 {
        // sf_conversion(): ((127 + bias) - rbv) ^ 127, the subtraction
        // unsigned and saturating, read as a signed byte.
        let sbv = |x: u8, k: usize| -> i8 {
            (bias
                .wrapping_add(127)
                .saturating_sub(bytes.cost(usize::from(x), k))
                ^ 127) as i8
        };
        // get_xE(): row[0] is the begin state, shifted in at -128.
        let mut row = vec![-128i8; m + 1];
        let mut xe = 128u8;
        for &x in seq {
            for k in (1..=m).rev() {
                row[k] = row[k - 1].saturating_sub(sbv(x, k));
                xe = xe.max(row[k] as u8);
            }
        }
        let begin = wide(base) - wide(tjb) - wide(tbm);
        let mut xe = i32::from(xe);
        if xe >= 255 - wide(bias) {
            if begin >= 128 {
                return (f32::INFINITY, MsvPath::SingleSegmentOverflow);
            }
        } else {
            xe += begin;
            xe -= 128;
            if xe >= 255 - wide(bias) {
                return (f32::INFINITY, MsvPath::SingleSegmentOverflow);
            }
            let xj = xe - wide(tec);
            if xj <= wide(base) {
                return (score(xj), MsvPath::SingleSegment);
            }
        }
    }

    let mut row = vec![0u8; m + 1];
    let tjbm = tjb.wrapping_add(tbm);
    let mut xj = 0u8;
    let mut xb = base.saturating_sub(tjbm);
    for &x in seq {
        let mut xe = 0u8;
        for k in (1..=m).rev() {
            row[k] = row[k - 1]
                .max(xb)
                .saturating_add(bias)
                .saturating_sub(bytes.cost(usize::from(x), k));
            xe = xe.max(row[k]);
        }
        if xe.saturating_add(bias) == u8::MAX {
            return (f32::INFINITY, MsvPath::Full);
        }
        xj = xj.max(xe.saturating_sub(tec));
        xb = base.max(xj).saturating_sub(tjbm);
    }
    (score(i32::from(xj)), MsvPath::Full)
}

/// `p7_ViterbiFilter()` cell by cell on the unstriped word tables, with its
/// lazy evaluation of the delete-to-delete paths. Returns the score and the
/// number of rows that evaluated them.
fn scalar_hmmer_viterbi(seq: &[u8], m: usize, nj: f32, words: &ViterbiWords) -> (f32, usize) {
    let floor = i16::MIN;
    let transition = |kind: usize, k: usize| words.transition(kind, k);
    let empty = || vec![floor; m + 2];
    let (mut mmx, mut imx, mut dmx) = (empty(), empty(), empty());
    let ncj_move = i32::from(words.ncj_move(seq.len(), nj));
    let xn = i32::from(words.base);
    let mut xb = xn + ncj_move;
    let (mut xj, mut xc) = (i32::from(floor), i32::from(floor));
    let mut evaluated = 0;
    for &x in seq {
        let (mut new_m, mut new_i, mut new_d) = (empty(), empty(), empty());
        let (mut xe, mut dmax) = (floor, floor);
        for k in 1..=m {
            let mut sv = (xb as i16).saturating_add(transition(TSC_BM, k));
            sv = sv.max(mmx[k - 1].saturating_add(transition(TSC_MM, k)));
            sv = sv.max(imx[k - 1].saturating_add(transition(TSC_IM, k)));
            sv = sv.max(dmx[k - 1].saturating_add(transition(TSC_DM, k)));
            sv = sv.saturating_add(words.emission(usize::from(x), k));
            xe = xe.max(sv);
            new_m[k] = sv;
            // The delete cell of the next node, from this match cell.
            let dcv = sv.saturating_add(transition(TSC_MD, k));
            dmax = dmax.max(dcv);
            new_d[k + 1] = dcv;
            new_i[k] = mmx[k]
                .saturating_add(transition(TSC_MI, k))
                .max(imx[k].saturating_add(transition(TSC_II, k)));
        }
        if xe == i16::MAX {
            return (f32::INFINITY, evaluated);
        }
        let xe = i32::from(xe);
        xc = xc.max(xe + i32::from(words.e_move));
        xj = xj.max(xe + i32::from(words.e_loop));
        xb = (xj + ncj_move).max(xn + ncj_move);
        if i32::from(dmax) + i32::from(words.dd_bound) > xb {
            evaluated += 1;
            for k in 1..=m {
                new_d[k + 1] = new_d[k + 1].max(new_d[k].saturating_add(transition(NTSC_PER_Q, k)));
            }
        }
        (mmx, imx, dmx) = (new_m, new_i, new_d);
    }
    let score = if xc > i32::from(floor) {
        (xc as f32 + ncj_move as f32 - f32::from(words.base)) / words.scale - 3.0
    } else {
        f32::NEG_INFINITY
    };
    (score, evaluated)
}

/// The most likely residue of node `k` of `profile`.
fn best_residue(profile: &Profile, k: usize) -> u8 {
    (0..profile.alphabet.canonical_size)
        .max_by(|&left, &right| {
            profile.residue_scores_for(left)[k * 2]
                .total_cmp(&profile.residue_scores_for(right)[k * 2])
        })
        .unwrap() as u8
}

/// `random` with a stretch of consensus in which nodes are skipped and
/// residues inserted: its best path runs through delete and insert states.
fn gapped_consensus(profile: &Profile, random: &[u8]) -> Vec<u8> {
    let (m, l) = (profile.num_nodes, random.len());
    let mut gapped = random.to_vec();
    let mut k = 1;
    for (position, residue) in gapped.iter_mut().enumerate().skip(l / 4) {
        if k > m {
            break;
        }
        if position % 11 == 5 {
            continue;
        }
        *residue = best_residue(profile, k);
        k += if position % 7 == 3 { 3 } else { 1 };
    }
    gapped
}

/// The three ways the Viterbi filter can be run on `om`, each with the
/// score it gives for `seq`.
fn viterbi_filters(seq: &[u8], om: &OptimizedProfile) -> [(&'static str, f32); 3] {
    let l = seq.len();
    let mut rows = viterbi_filter::Rows::default();
    [
        ("as the search runs it", viterbi_filter(seq, l, om)),
        (
            "with HMMER's lazy delete paths",
            viterbi_filter::viterbi_filter_lazy_with(seq, l, om, &mut rows),
        ),
        (
            "with every delete path, one vector at a time",
            viterbi_filter::viterbi_filter_eager_with(seq, l, om, &mut rows),
        ),
    ]
}

/// The filters the default search decides on return what the
/// transcriptions return, bit for bit, on every instruction set: random
/// sequences, runs of consensus that overflow, two separated segments that
/// need the J state, and consensus with residues dropped and inserted.
#[test]
fn hmmer_filter_scores_match_cell_by_cell_transcriptions() {
    let alphabet = Alphabet::amino();
    let background = BackgroundModel::new(&alphabet);
    let isas = Isa::available();
    eprintln!("instruction sets: {isas:?}");
    let mut msv_paths: HashMap<(MsvPath, bool), usize> = HashMap::new();
    let (mut viterbi_finite, mut viterbi_overflowed) = (0usize, 0usize);
    let (mut rows, mut rows_evaluated) = (0usize, 0usize);
    for seed in [0, 5, 42] {
        let mut rng = XorShift64::new(seed);
        for m in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 129, 257, 600] {
            let hmm = hmm_sample(&mut rng, m, &alphabet);
            for l in [1, 2, 8, 51, 300, 1500] {
                let mut profile = Profile::new(m, &alphabet);
                modelconfig::profile_config(&hmm, &background, &mut profile, l, SearchMode::Local);
                let optimized: Vec<OptimizedProfile> = isas
                    .iter()
                    .map(|&isa| OptimizedProfile::from_profile_with_isa(&profile, isa))
                    .collect();
                let random = random_digital_seq(
                    &mut rng,
                    &background.residue_frequencies,
                    alphabet.canonical_size,
                    l,
                );
                let consensus: Vec<u8> = (0..l)
                    .map(|position| best_residue(&profile, position % m + 1))
                    .collect();
                // Two short runs of consensus in a random sequence: each
                // can pay for the J state without overflowing.
                let mut segments = random.clone();
                for start in [l / 5, 3 * l / 5] {
                    let first = 1 + start % m;
                    for offset in 0..(l / 8).min(m - first + 1).min(12) {
                        segments[start + offset] = best_residue(&profile, first + offset);
                    }
                }
                let gapped = gapped_consensus(&profile, &random);
                for seq in [&random, &consensus, &segments, &gapped] {
                    let (msv, path) = scalar_hmmer_msv(seq, m, &optimized[0].msv_bytes);
                    *msv_paths.entry((path, msv.is_finite())).or_default() += 1;
                    let (viterbi, evaluated) =
                        scalar_hmmer_viterbi(seq, m, optimized[0].nj, &optimized[0].viterbi_words);
                    if viterbi.is_finite() {
                        viterbi_finite += 1;
                        rows += l;
                        rows_evaluated += evaluated;
                    } else {
                        viterbi_overflowed += 1;
                    }
                    for (isa, om) in isas.iter().zip(&optimized) {
                        let context = format!("{isa:?} seed={seed} M={m} L={l}");
                        assert_eq!(om.isa(), *isa);
                        let score = msv_filter(seq, l, om);
                        assert_eq!(
                            score.to_bits(),
                            msv.to_bits(),
                            "MSV filter, {context}: {score} != {msv} ({path:?})"
                        );
                        assert!(om.viterbi_words.dd_bound_is_exact);
                        for (name, score) in viterbi_filters(seq, om) {
                            assert_eq!(
                                score.to_bits(),
                                viterbi.to_bits(),
                                "Viterbi filter {name}, {context}: {score} != {viterbi}"
                            );
                        }
                    }
                }
            }
        }
    }
    // Every way through the MSV filter was taken, and the Viterbi filter
    // both skipped and evaluated its delete-to-delete paths.
    eprintln!("MSV filter: {msv_paths:?}");
    eprintln!(
        "Viterbi filter: {viterbi_finite} scores, {viterbi_overflowed} overflows, \
         {rows_evaluated} of {rows} rows with delete-to-delete paths"
    );
    for wanted in [
        (MsvPath::SingleSegment, true),
        (MsvPath::SingleSegmentOverflow, false),
        (MsvPath::Full, true),
        (MsvPath::Full, false),
    ] {
        assert!(
            msv_paths.get(&wanted).copied().unwrap_or(0) >= 5,
            "{wanted:?} in {msv_paths:?}"
        );
    }
    assert!(viterbi_finite > 300 && viterbi_overflowed > 50);
    assert!(rows_evaluated * 20 > rows && rows_evaluated * 20 < rows * 19);
}

/// The MSV filter on tables that reach the corners of `p7_SSVFilter()`: a
/// profile and length for which the shortcut is unproven, a begin state so
/// low that an overflow of the single-segment pass says nothing about the
/// full recurrence, costs that saturate, and transition costs whose sum
/// wraps around the byte.
#[test]
fn msv_filter_follows_hmmer_where_the_single_segment_shortcut_does_not_hold() {
    const KP: usize = 4;
    let mut rng = XorShift64::new(9);
    let mut paths: HashMap<(MsvPath, bool), usize> = HashMap::new();
    for isa in Isa::available() {
        for m in [1, 5, 16, 17, 40, 200, 450] {
            // (bias, cost of the begin transition, cost of the move from N)
            for (bias, tbm, tjb) in [
                (12u8, 30u8, 20u8),
                (19, 57, 21),
                (15, 70, 45),
                (15, 20, 50),
                (40, 60, 30),
                (30, 200, 200),
                (0, 10, 10),
            ] {
                let table: Vec<u8> = (0..KP * m)
                    .map(|_| match rng.roll(10) {
                        0 => 255,
                        1..=3 => bias - rng.roll(usize::from(bias) + 1) as u8,
                        _ => bias + 1 + rng.roll(30) as u8,
                    })
                    .collect();
                let mut bytes = MsvBytes::from_costs(m, KP, bias, isa, |x, k| table[x * m + k - 1]);
                assert_eq!(bytes.isa(), isa);
                bytes.tbm = tbm;
                bytes.tjb = tjb;
                for l in [1, 9, 64, 400] {
                    // The residue each node likes best, so that segments
                    // form, with random stretches between them.
                    let seq: Vec<u8> = (0..l)
                        .map(|position| {
                            let k = position % m + 1;
                            if (position / 13) % 3 == 0 {
                                rng.roll(KP) as u8
                            } else {
                                (0..KP).min_by_key(|&x| table[x * m + k - 1]).unwrap() as u8
                            }
                        })
                        .collect();
                    let (expected, path) = scalar_hmmer_msv(&seq, m, &bytes);
                    *paths.entry((path, expected.is_finite())).or_default() += 1;
                    let actual = msv_filter_of(&seq, &bytes, &mut Vec::new(), &mut Vec::new());
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{isa:?} M={m} L={l} bias={bias} tbm={tbm} tjb={tjb}: \
                         {actual} != {expected} ({path:?})"
                    );
                }
            }
        }
    }
    eprintln!("MSV filter: {paths:?}");
    for wanted in [
        (MsvPath::SingleSegment, true),
        (MsvPath::SingleSegmentOverflow, false),
        (MsvPath::Full, true),
        (MsvPath::Full, false),
    ] {
        assert!(
            paths.get(&wanted).copied().unwrap_or(0) >= 5,
            "{wanted:?} in {paths:?}"
        );
    }
}

/// A sampled model of `m` nodes and its profile for targets of `l` residues,
/// after `edit` has changed the model.
fn edited_profile(
    rng: &mut XorShift64,
    m: usize,
    l: usize,
    edit: impl Fn(&mut Hmm),
) -> (BackgroundModel, Profile) {
    let alphabet = Alphabet::amino();
    let background = BackgroundModel::new(&alphabet);
    let mut hmm = hmm_sample(rng, m, &alphabet);
    edit(&mut hmm);
    let mut profile = Profile::new(m, &alphabet);
    modelconfig::profile_config(&hmm, &background, &mut profile, l, SearchMode::Local);
    (background, profile)
}

/// What takes a shortcut of the Viterbi filter away from a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Oddity {
    /// Delete-to-delete transitions so unlikely that the sum of two of them
    /// does not fit a word.
    DeleteStateWithoutSuccessor,
    /// A bound for HMMER's lazy evaluation that does not fit the 16-bit
    /// field HMMER keeps it in, so that HMMER tests a truncated one.
    BeginTransitionNeverTaken,
    /// A delete-to-delete "probability" above one, whose score is above
    /// zero.
    DeleteRunThatGains,
}

/// The Viterbi filter evaluates every delete-to-delete path, two vectors at
/// a time, because that gives HMMER's score for any real profile. Some
/// profiles take that away, and the filter must notice: it then goes one
/// vector at a time, or runs HMMER's own recurrence, which may skip paths
/// that count.
#[test]
fn viterbi_filter_follows_hmmer_where_its_shortcuts_do_not_hold() {
    let (m, l) = (120, 400);
    let transition = |hmm: &mut Hmm, node: usize, kind: HTransition, probability: f32| {
        hmm.transitions_mut(node)[kind as usize] = probability;
    };
    let mut rng = XorShift64::new(31);
    for oddity in [
        Oddity::DeleteStateWithoutSuccessor,
        Oddity::BeginTransitionNeverTaken,
        Oddity::DeleteRunThatGains,
    ] {
        let (background, profile) = edited_profile(&mut rng, m, l, |hmm| match oddity {
            Oddity::DeleteStateWithoutSuccessor => {
                // Delete states that always return to a match state.
                for node in [7, 8, 40, 77] {
                    transition(hmm, node, HTransition::DeleteToDelete, 0.0);
                    transition(hmm, node, HTransition::DeleteToMatch, 1.0);
                }
            }
            Oddity::BeginTransitionNeverTaken => {
                // Nothing but match states up to node 21, so that each is
                // always visited; the match state of node 21 always deletes,
                // so that of node 22 is never entered from B, and the delete
                // states of nodes 20 and 21 cost nothing to leave: the gain
                // of the delete path into node 22 over a begin transition
                // that cannot be taken is more than a word holds.
                for node in 0..21 {
                    transition(hmm, node, HTransition::MatchToMatch, 1.0);
                    transition(hmm, node, HTransition::MatchToInsert, 0.0);
                    transition(hmm, node, HTransition::MatchToDelete, 0.0);
                }
                transition(hmm, 20, HTransition::DeleteToDelete, 1.0);
                transition(hmm, 20, HTransition::DeleteToMatch, 0.0);
                transition(hmm, 21, HTransition::MatchToMatch, 0.0);
                transition(hmm, 21, HTransition::MatchToInsert, 0.0);
                transition(hmm, 21, HTransition::MatchToDelete, 1.0);
                transition(hmm, 21, HTransition::DeleteToMatch, 0.9999);
                transition(hmm, 21, HTransition::DeleteToDelete, 0.0001);
            }
            Oddity::DeleteRunThatGains => {
                for node in [50, 51, 52] {
                    transition(hmm, node, HTransition::DeleteToDelete, 1.5);
                }
            }
        });
        let pairs_exact = oddity != Oddity::DeleteStateWithoutSuccessor;
        let every_path_is_hmmer = oddity == Oddity::DeleteStateWithoutSuccessor;
        for isa in Isa::available() {
            let om = OptimizedProfile::from_profile_with_isa(&profile, isa);
            let words = &om.viterbi_words;
            assert_eq!(words.dd_pairs_are_exact, pairs_exact, "{oddity:?}");
            assert_eq!(words.dd_bound_is_exact, every_path_is_hmmer, "{oddity:?}");
            // The bound as vf_conversion() leaves it: a running maximum
            // stored in 16 bits after every node.
            let (mut bound, mut largest_gain) = (i16::MIN, i32::MIN);
            for k in 2..m - 1 {
                let gain = i32::from(words.transition(NTSC_PER_Q, k))
                    + i32::from(words.transition(TSC_DM, k + 2))
                    - i32::from(words.transition(TSC_BM, k + 2));
                largest_gain = largest_gain.max(gain);
                bound = i32::from(bound).max(gain) as i16;
            }
            assert_eq!(words.dd_bound, bound, "{oddity:?}");
            assert_eq!(
                largest_gain <= i32::from(i16::MAX),
                oddity != Oddity::BeginTransitionNeverTaken,
                "{oddity:?}"
            );
            for round in 0..40 {
                let random = random_digital_seq(
                    &mut rng,
                    &background.residue_frequencies,
                    profile.alphabet.canonical_size,
                    l,
                );
                let seq = match round % 3 {
                    0 => random,
                    1 => gapped_consensus(&profile, &random),
                    _ => {
                        // The consensus of nodes 44 to 68 without five
                        // nodes in the middle: one run of delete states,
                        // between two stretches too short to overflow.
                        let mut skipping = random;
                        let nodes = (44..=53).chain(59..=68);
                        for (residue, k) in skipping[l / 4..].iter_mut().zip(nodes) {
                            *residue = best_residue(&profile, k);
                        }
                        skipping
                    }
                };
                let (expected, _) = scalar_hmmer_viterbi(&seq, m, om.nj, words);
                for (kernel, score) in viterbi_filters(&seq, &om) {
                    // Where nothing says that the paths HMMER skips do not
                    // count, evaluating all of them need not give its score;
                    // the search runs HMMER's own recurrence then.
                    if !every_path_is_hmmer && kernel.starts_with("with every") {
                        continue;
                    }
                    assert_eq!(
                        score.to_bits(),
                        expected.to_bits(),
                        "{oddity:?}, {isa:?}, {kernel}: {score} != {expected}"
                    );
                }
            }
        }
    }
}
