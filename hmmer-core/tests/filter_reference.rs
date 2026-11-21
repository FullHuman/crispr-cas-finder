//! An independent, unstriped recurrence for the SIMD Forward filter at the
//! stripe boundaries.
use hmmer_core::{
    alphabet::Alphabet,
    background::BackgroundModel,
    config::{PTsc, SearchMode},
    modelconfig,
    profile::Profile,
    rng::XorShift64,
    simd::{forward_filter::forward_filter, oprofile::OptimizedProfile},
    test_helpers::{hmm_sample, random_digital_seq},
};

/// log(e^a + e^b), the exponential in double precision.
fn log_sum(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    if max == f32::NEG_INFINITY {
        return max;
    }
    max + ((-(a as f64 - b as f64).abs()).exp().ln_1p() as f32)
}

/// The Forward score of `seq`, cell by cell on the generic profile.
fn scalar_forward(seq: &[u8], profile: &Profile) -> f32 {
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
            current[k][0] = log_sum(
                log_sum(
                    previous[k - 1][0] + transition(PTsc::MatchToMatch),
                    previous[k - 1][1] + transition(PTsc::InsertToMatch),
                ),
                log_sum(
                    previous[k - 1][2] + transition(PTsc::DeleteToMatch),
                    b + transition(PTsc::BeginToMatch),
                ),
            ) + emission;
            current[k][2] = log_sum(
                current[k - 1][0] + transition(PTsc::MatchToDelete),
                current[k - 1][2] + transition(PTsc::DeleteToDelete),
            );
            if k < m {
                current[k][1] = log_sum(
                    previous[k][0] + profile.transition_score_at(k + 1, PTsc::MatchToInsert),
                    previous[k][1] + profile.transition_score_at(k + 1, PTsc::InsertToInsert),
                ) + profile.residue_scores_for(residue as usize)[k * 2 + 1];
            }
            e = log_sum(e, log_sum(current[k][0], current[k][2]));
        }
        j = log_sum(j + special.j_loop, e + special.e_loop);
        c = log_sum(c + special.c_loop, e + special.e_move);
        n += special.n_loop;
        b = log_sum(n + special.n_move, j + special.j_move);
        previous = current;
    }
    c + special.c_move
}

#[test]
fn forward_filter_matches_the_scalar_recurrence_across_stripe_boundaries() {
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
                let actual = forward_filter(&seq, l, &optimized);
                let expected = scalar_forward(&seq, &profile);
                assert!(
                    (actual - expected).abs() < 0.002,
                    "seed={seed}, M={m}, L={l}: {actual} != {expected}"
                );
            }
        }
    }
}
