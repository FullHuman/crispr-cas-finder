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
        oprofile::OptimizedProfile,
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
