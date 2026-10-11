//! Independent, unstriped recurrences for the SIMD filter boundary cases.
use hmmer_core::{
    alphabet::Alphabet,
    background::BackgroundModel,
    config::{PTsc, SearchMode},
    modelconfig,
    profile::Profile,
    rng::XorShift64,
    simd::{
        f32_msv::msv_filter_f32, f32_viterbi::viterbi_filter_f32, forward_filter::forward_filter,
        i16_msv::msv_upper_bound, i16_viterbi::viterbi_upper_bound, oprofile::OptimizedProfile,
    },
    test_helpers::{hmm_sample, random_digital_seq},
};

fn combine(a: f32, b: f32, forward: bool) -> f32 {
    let max = a.max(b);
    if !forward || max == f32::NEG_INFINITY {
        return max;
    }
    max + ((-(a as f64 - b as f64).abs()).exp().ln_1p() as f32)
}

fn scalar_full(seq: &[u8], profile: &Profile, forward: bool) -> f32 {
    let m = profile.num_nodes;
    let neg = f32::NEG_INFINITY;
    let special = &profile.special_scores;
    let mut previous = vec![[neg; 3]; m + 1];
    let (mut n, mut b, mut j, mut c) = (0.0, special.n_move, neg, neg);
    for &residue in seq {
        let mut current = vec![[neg; 3]; m + 1];
        let mut e = neg;
        for k in 1..=m {
            let transition = |kind| profile.transition_score_at(k, kind);
            let emission = profile.residue_scores_for(residue as usize)[k * 2];
            current[k][0] = combine(
                combine(
                    previous[k - 1][0] + transition(PTsc::MatchToMatch),
                    previous[k - 1][1] + transition(PTsc::InsertToMatch),
                    forward,
                ),
                combine(
                    previous[k - 1][2] + transition(PTsc::DeleteToMatch),
                    b + transition(PTsc::BeginToMatch),
                    forward,
                ),
                forward,
            ) + emission;
            current[k][2] = combine(
                current[k - 1][0] + transition(PTsc::MatchToDelete),
                current[k - 1][2] + transition(PTsc::DeleteToDelete),
                forward,
            );
            if k < m {
                current[k][1] = combine(
                    previous[k][0] + profile.transition_score_at(k + 1, PTsc::MatchToInsert),
                    previous[k][1] + profile.transition_score_at(k + 1, PTsc::InsertToInsert),
                    forward,
                ) + profile.residue_scores_for(residue as usize)[k * 2 + 1];
            }
            e = combine(e, combine(current[k][0], current[k][2], forward), forward);
        }
        j = combine(j + special.j_loop, e + special.e_loop, forward);
        c = combine(c + special.c_loop, e + special.e_move, forward);
        n += special.n_loop;
        b = combine(n + special.n_move, j + special.j_move, forward);
        previous = current;
    }
    c + special.c_move
}

fn scalar_msv(seq: &[u8], profile: &Profile) -> f32 {
    let m = profile.num_nodes;
    let l = seq.len() as f32;
    let (tloop, tmove) = ((l / (l + 3.0)).ln(), (3.0 / (l + 3.0)).ln());
    let entry = (2.0 / (m * (m + 1)) as f32).ln();
    let exit = 0.5f32.ln();
    let neg = f32::NEG_INFINITY;
    let mut previous = vec![neg; m + 1];
    let (mut n, mut b, mut j, mut c) = (0.0, tmove, neg, neg);
    for &residue in seq {
        let mut current = vec![neg; m + 1];
        for k in 1..=m {
            current[k] = previous[k - 1].max(b + entry)
                + profile.residue_scores_for(residue as usize)[k * 2];
        }
        let e = current.iter().copied().fold(neg, f32::max);
        j = (j + tloop).max(e + exit);
        c = (c + tloop).max(e + exit);
        n += tloop;
        b = (n + tmove).max(j + tmove);
        previous = current;
    }
    c + tmove
}

#[test]
fn filters_match_scalar_recurrences_across_stripe_boundaries() {
    let alphabet = Alphabet::amino();
    let background = BackgroundModel::new(&alphabet);
    for seed in [0, 1, 17, 42] {
        let mut rng = XorShift64::new(seed);
        for m in [1, 2, 3, 4, 5, 8, 15, 16, 17, 99, 100, 101] {
            let hmm = hmm_sample(&mut rng, m, &alphabet);
            for l in [1, 8, 51] {
                let mut profile = Profile::new(m, &alphabet);
                modelconfig::profile_config(&hmm, &background, &mut profile, l, SearchMode::Local);
                let optimized = OptimizedProfile::from_profile(&profile);
                let seq = random_digital_seq(
                    &mut rng,
                    &background.residue_frequencies,
                    alphabet.canonical_size,
                    l,
                );
                for (name, actual, expected) in [
                    (
                        "MSV",
                        msv_filter_f32(&seq, l, &optimized),
                        scalar_msv(&seq, &profile),
                    ),
                    (
                        "Viterbi",
                        viterbi_filter_f32(&seq, l, &optimized),
                        scalar_full(&seq, &profile, false),
                    ),
                    (
                        "Forward",
                        forward_filter(&seq, l, &optimized).score,
                        scalar_full(&seq, &profile, true),
                    ),
                ] {
                    assert!(
                        (actual - expected).abs() < 0.002,
                        "{name}: seed={seed}, M={m}, L={l}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}

/// The integer pre-filters may only reject what the exact filters would
/// reject, so their scores must never fall below the exact ones. They must
/// also stay close to them, or they would let everything through.
#[test]
fn integer_bounds_are_never_below_the_exact_scores_and_stay_tight() {
    let alphabet = Alphabet::amino();
    let background = BackgroundModel::new(&alphabet);
    // (comparisons, summed excess in nats) for MSV and Viterbi.
    let mut tightness = [(0usize, 0.0f64); 2];
    for seed in [0, 1, 17, 42, 1234] {
        let mut rng = XorShift64::new(seed);
        for m in [1, 2, 7, 8, 9, 16, 17, 64, 100, 257, 600] {
            let hmm = hmm_sample(&mut rng, m, &alphabet);
            for l in [1, 2, 8, 51, 300, 1500] {
                let mut profile = Profile::new(m, &alphabet);
                modelconfig::profile_config(&hmm, &background, &mut profile, l, SearchMode::Local);
                let optimized = OptimizedProfile::from_profile(&profile);
                let random = random_digital_seq(
                    &mut rng,
                    &background.residue_frequencies,
                    alphabet.canonical_size,
                    l,
                );
                // Repeating the best residue of each model position gives
                // long, high-scoring diagonals and exercises saturation.
                let best_residue = |k: usize| {
                    (0..alphabet.canonical_size)
                        .max_by(|&left, &right| {
                            profile.residue_scores_for(left)[k * 2]
                                .total_cmp(&profile.residue_scores_for(right)[k * 2])
                        })
                        .unwrap() as u8
                };
                let consensus: Vec<u8> = (0..l)
                    .map(|position| best_residue(position % m + 1))
                    .collect();
                // A consensus stretch with residues dropped and inserted
                // sends the best path through delete and insert states.
                let mut gapped = random.clone();
                let mut k = 1;
                for (position, residue) in gapped.iter_mut().enumerate().skip(l / 4) {
                    if k > m {
                        break;
                    }
                    if position % 11 == 5 {
                        continue; // keep the random residue: an insertion
                    }
                    *residue = best_residue(k);
                    k += if position % 7 == 3 { 3 } else { 1 }; // skip nodes: deletions
                }
                for (kind, seq) in [
                    ("random", &random),
                    ("consensus", &consensus),
                    ("gapped", &gapped),
                ] {
                    let checks = [
                        (
                            "MSV",
                            msv_filter_f32(seq, l, &optimized),
                            msv_upper_bound(seq, l, &optimized),
                        ),
                        (
                            "Viterbi",
                            viterbi_filter_f32(seq, l, &optimized),
                            viterbi_upper_bound(seq, l, &optimized),
                        ),
                    ];
                    for (index, (name, exact, bound)) in checks.into_iter().enumerate() {
                        assert!(
                            bound >= exact,
                            "{name} {kind}: seed={seed}, M={m}, L={l}: bound {bound} < exact {exact}"
                        );
                        if bound.is_finite() && kind == "random" {
                            tightness[index].0 += 1;
                            tightness[index].1 += f64::from(bound - exact);
                            assert!(
                                bound - exact < 1.5,
                                "{name} {kind}: seed={seed}, M={m}, L={l}: bound {bound} far above {exact}"
                            );
                        }
                    }
                }
            }
        }
    }
    for (name, (compared, total_excess)) in ["MSV", "Viterbi"].into_iter().zip(tightness) {
        assert!(compared > 300, "{name}: only {compared} finite bounds");
        let mean_excess = total_excess / compared as f64;
        assert!(mean_excess < 0.3, "{name}: mean excess {mean_excess} nats");
    }
}
