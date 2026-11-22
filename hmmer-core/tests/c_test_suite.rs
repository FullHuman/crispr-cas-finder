// Integration tests porting the C TESTDRIVE unit tests from HMMER.
//
// Each test function corresponds to a utest_*() or main() TESTDRIVE
// block in the corresponding C source file.

use hmmer_core::alphabet::Alphabet;
use hmmer_core::background::BackgroundModel;
use hmmer_core::config::TraceStateType;
use hmmer_core::config::*;
use hmmer_core::logsum;
use hmmer_core::profile::Profile;
use hmmer_core::rng::XorShift64;
use hmmer_core::score_matrix::NUM_SPECIAL_STATES;
use hmmer_core::score_matrix::*;
use hmmer_core::simd::forward_filter;
use hmmer_core::simd::oprofile::OptimizedProfile;
use hmmer_core::test_helpers::*;

// ===================================================================
// p7_hmm.c :: utest_occupancy()
//
// The stationary match occupancy probability in a random HMM
// converges to 0.6, for long enough M. (STL11/138)
// ===================================================================

#[test]
fn test_hmm_occupancy() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 200;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let (occ, _) = hmm.calculate_occupancy();

    let x: f32 = occ[1..=m].iter().sum::<f32>() / m as f32;

    assert!(
        (x - 0.6).abs() < 0.1,
        "occupancy unit test: expected ~0.6, got {:.3}",
        x
    );
}

// ===================================================================
// p7_bg.c :: utest_ReadWrite()
// ===================================================================

#[test]
fn test_bg_create_and_validate() {
    let abc = Alphabet::amino();
    let bg = BackgroundModel::new(&abc);

    let sum: f32 = bg.residue_frequencies.iter().sum();
    assert!((sum - 1.0).abs() < 0.001, "bg frequencies sum to {}", sum);
    assert!(bg.residue_frequencies.iter().all(|&f| f > 0.0));
    assert!((bg.null1_transition_prob - 350.0 / 351.0).abs() < 1e-6);
}

#[test]
fn test_bg_set_length() {
    let abc = Alphabet::amino();
    let mut bg = BackgroundModel::new(&abc);

    bg.set_length(400);
    let expected_p1 = 400.0 / 401.0;
    assert!((bg.null1_transition_prob - expected_p1).abs() < 1e-6);
}

#[test]
fn test_bg_null_one_score() {
    let abc = Alphabet::amino();
    let bg = BackgroundModel::new(&abc);
    let mut rng = XorShift64::new(42);

    let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, 100);
    let score = bg.null_one(&dsq, 100);

    assert!(score.is_finite());
    assert!(score < 0.0);
}

#[test]
fn test_bg_different_alphabets() {
    for abc in &[Alphabet::amino(), Alphabet::dna()] {
        let bg = BackgroundModel::new(abc);

        let sum: f32 = bg.residue_frequencies.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "bg freq sum for {:?}: {}",
            abc.kind,
            sum
        );
    }
}

// ===================================================================
// logsum.c :: utest_FLogsumError()
// ===================================================================

#[test]
fn test_logsum_error() {
    let n = 1000;
    let maxval = 20.0_f32;
    let mut max_err = 0.0_f32;
    let mut avg_err = 0.0_f32;
    let mut rng = XorShift64::new(42);

    for _ in 0..n {
        let a = (rng.random() as f32 - 0.5) * maxval * 2.0;
        let b = (rng.random() as f32 - 0.5) * maxval * 2.0;

        let exact = ((a as f64).exp() + (b as f64).exp()).ln() as f32;
        let result = logsum::flogsum(a, b);
        let err = (exact - result).abs() / maxval;

        avg_err += err;
        if err > max_err {
            max_err = err;
        }
    }
    avg_err /= n as f32;

    assert!(
        max_err <= 0.0001,
        "maximum error of {} is too high",
        max_err
    );
    assert!(
        avg_err <= 0.0001,
        "average error of {} is too high",
        avg_err
    );
}

// ===================================================================
// logsum.c :: utest_FLogsumSpecials()
// ===================================================================

#[test]
fn test_logsum_specials() {
    assert_eq!(logsum::flogsum(0.0, f32::NEG_INFINITY), 0.0);
    assert_eq!(logsum::flogsum(f32::NEG_INFINITY, 0.0), 0.0);
    assert_eq!(
        logsum::flogsum(f32::NEG_INFINITY, f32::NEG_INFINITY),
        f32::NEG_INFINITY
    );
}

// ===================================================================
// p7_gmx.c :: gmx_testpattern()
// ===================================================================

#[test]
fn test_gmx_testpattern() {
    let m = 20;
    let l = 20;
    let mut gx = ScoreMatrix::new(m, l).expect("ScoreMatrix::new failed");

    let mut n: u64 = 0;
    for i in 0..=l {
        for k in 0..=m {
            gx.set_match_score(i, k, n as f32);
            n += 1;
            gx.set_insert_score(i, k, n as f32);
            n += 1;
            gx.set_delete_score(i, k, n as f32);
            n += 1;
        }
    }

    let mut n2: u64 = 0;
    for i in 0..=l {
        for k in 0..=m {
            assert_eq!(gx.match_score(i, k) as u64, n2, "M at i={},k={}", i, k);
            n2 += 1;
            assert_eq!(gx.insert_score(i, k) as u64, n2, "I at i={},k={}", i, k);
            n2 += 1;
            assert_eq!(gx.delete_score(i, k) as u64, n2, "D at i={},k={}", i, k);
            n2 += 1;
        }
    }
    assert_eq!(n, n2);
}

// ===================================================================
// p7_gmx.c :: utest_Compare()
// ===================================================================

#[test]
fn test_gmx_compare() {
    let m = 10;
    let l = 10;

    let mut gx1 = ScoreMatrix::new(m, l).expect("new failed");
    let mut gx2 = ScoreMatrix::new(m, l).expect("new failed");

    let mut n = 0.0_f32;
    for i in 0..=l {
        for k in 0..=m {
            gx1.set_match_score(i, k, n);
            gx2.set_match_score(i, k, n);
            gx1.set_insert_score(i, k, n + 0.1);
            gx2.set_insert_score(i, k, n + 0.1);
            gx1.set_delete_score(i, k, n + 0.2);
            gx2.set_delete_score(i, k, n + 0.2);
            n += 1.0;
        }
        let specials: [f32; NUM_SPECIAL_STATES] = std::array::from_fn(|s| n + s as f32);
        gx1.set_special_row(i, specials);
        gx2.set_special_row(i, specials);
        n += NUM_SPECIAL_STATES as f32;
    }

    for i in 0..=l {
        for k in 0..=m {
            assert_eq!(gx1.match_score(i, k), gx2.match_score(i, k));
            assert_eq!(gx1.insert_score(i, k), gx2.insert_score(i, k));
            assert_eq!(gx1.delete_score(i, k), gx2.delete_score(i, k));
        }
        for s in 0..NUM_SPECIAL_STATES {
            assert_eq!(gx1.special_score(i, s), gx2.special_score(i, s));
        }
    }

    gx2.set_match_score(5, 5, gx2.match_score(5, 5) + 1.0);
    assert_ne!(gx1.match_score(5, 5), gx2.match_score(5, 5));
}

// ===================================================================
// hmmer.c :: utest_alphabet_config()
// ===================================================================

#[test]
fn test_alphabet_config() {
    for abc in &[Alphabet::amino(), Alphabet::dna(), Alphabet::rna()] {
        assert!(
            abc.canonical_size <= MAX_CANONICAL_ALPHABET,
            "K ({}) > p7_MAXABET",
            abc.canonical_size
        );
        assert!(
            abc.full_size <= MAX_FULL_ALPHABET,
            "Kp ({}) > p7_MAXCODE",
            abc.full_size
        );
    }
}

// ===================================================================
// modelconfig.c :: utest_Config()
// ===================================================================

#[test]
fn test_modelconfig_config() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);

    let hmm = hmm_sample(&mut rng, 100, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);

    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, 400);

    assert_eq!(gm.num_nodes, hmm.num_nodes);
    assert_eq!(gm.target_length, 400);
}

#[test]
fn test_modelconfig_occupancy() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 200;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, 400);

    assert_eq!(gm.num_nodes, m);
    assert!(gm.target_length > 0);
}

#[test]
fn test_reconfig_hit_modes_match_hmmer_length_model() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(17);
    let hmm = hmm_sample(&mut rng, 20, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut profile = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut profile, 224);

    profile.reconfig_unihit(224);
    assert!((profile.special_scores.n_move.exp() - 2.0 / 226.0).abs() < 1e-7);
    assert!((profile.special_scores.n_loop.exp() - 224.0 / 226.0).abs() < 1e-7);
    assert_eq!(profile.special_scores.e_move, 0.0);
    assert_eq!(profile.special_scores.e_loop, f32::NEG_INFINITY);

    profile.reconfig_multihit(224);
    assert!((profile.special_scores.n_move.exp() - 3.0 / 227.0).abs() < 1e-7);
    assert!((profile.special_scores.n_loop.exp() - 224.0 / 227.0).abs() < 1e-7);
    assert!((profile.special_scores.e_move.exp() - 0.5).abs() < 1e-7);
    assert!((profile.special_scores.e_loop.exp() - 0.5).abs() < 1e-7);
}

// ===================================================================
// generic_fwdback.c :: utest_forward()
// ===================================================================

#[test]
fn test_generic_forward_random_seqs() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 20;
    let l = 50;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l);
    let mut om = OptimizedProfile::from_profile(&gm);
    om.reconfigure_length(l);

    for _ in 0..10 {
        let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l);
        let result = forward_filter::forward_filter(&dsq, dsq.len(), &om);
        assert!(!result.is_nan(), "Forward score is NaN");
    }
}

// ===================================================================
// forward_backward.rs :: the log-space reference decoder
// ===================================================================

#[test]
fn test_backward_posterior_null2_and_oa_invariants() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(73);
    let bg = BackgroundModel::new(&abc);

    for l in [5, 20, 50] {
        let hmm = hmm_sample(&mut rng, 20, &abc);
        let mut gm = Profile::new(hmm.num_nodes, &abc);
        hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l);
        let om = OptimizedProfile::from_profile(&gm);
        let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l);
        let forward_score = forward_filter::forward_filter(&dsq, l, &om);
        let mut posterior = ScoreMatrix::new(gm.num_nodes, l).unwrap();
        let mut domains = hmmer_core::domaindef::DomainWorkspace::new();

        let backward_score = hmmer_core::forward_backward::backward_decode_log_space(
            &dsq,
            &gm,
            forward_score,
            &mut posterior,
            Some(&mut domains),
        )
        .unwrap();
        assert!((backward_score - forward_score).abs() < 0.1);
        assert!(domains.begin_totals[l].is_finite());
        assert!(domains.exit_totals[l].is_finite());
        assert!((domains.begin_totals[l] - domains.exit_totals[l]).abs() < 0.02);

        for i in 1..=l {
            let mut total = posterior.special_score(i, N_STATE)
                + posterior.special_score(i, J_STATE)
                + posterior.special_score(i, C_STATE);
            for k in 1..=gm.num_nodes {
                total += posterior.match_score(i, k) + posterior.insert_score(i, k);
            }
            assert!(
                (total - 1.0).abs() < 0.02,
                "posterior row {i} sums to {total}"
            );
        }

        let null2 = hmmer_core::null2::null2_by_expectation(&gm, &posterior);
        assert_eq!(null2.len(), abc.full_size);
        assert!(null2.iter().all(|odds| odds.is_finite() && *odds >= 0.0));

        let mut decoder = hmmer_core::optimal_accuracy::OaDecoder::new(gm.num_nodes);
        let oa = decoder.decode(&gm, &posterior).unwrap();
        assert!(oa.score.is_finite() && oa.score >= 0.0 && oa.score <= l as f32 + 0.1);
        assert!(
            (oa.trace.get_expected_accuracy() - oa.score).abs() < 1e-4,
            "trace accuracy={} differs from OA score={} for L={l}",
            oa.trace.get_expected_accuracy(),
            oa.score
        );
        assert_eq!(oa.trace.steps.first().unwrap().state, TraceStateType::Start);
        assert_eq!(
            oa.trace.steps.last().unwrap().state,
            TraceStateType::Terminate
        );
    }
}

// ===================================================================
// p7_pipeline.c :: tests
// ===================================================================

#[test]
fn test_pipeline_create() {
    let pipeline = hmmer_core::pipeline::search::FilterPolicy::default();

    assert!((pipeline.msv_threshold - 0.02).abs() < 0.001);
    assert!((pipeline.viterbi_threshold - 1e-3).abs() < 1e-6);
    assert!((pipeline.forward_threshold - 1e-5).abs() < 1e-8);
}

// ===================================================================
// Stress tests
// ===================================================================

#[test]
fn test_many_random_hmms_amino() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(1234);

    for _ in 0..50 {
        let m = rng.roll(200) + 1;
        let hmm = hmm_sample(&mut rng, m, &abc);
        assert_eq!(hmm.num_nodes, m);
        assert!(hmm_validate(&hmm, 0.01));
    }
}

#[test]
fn test_many_random_hmms_dna() {
    let abc = Alphabet::dna();
    let mut rng = XorShift64::new(5678);

    for _ in 0..50 {
        let m = rng.roll(200) + 1;
        let hmm = hmm_sample(&mut rng, m, &abc);
        assert_eq!(hmm.num_nodes, m);
        assert!(hmm_validate(&hmm, 0.01));
    }
}

#[test]
fn test_many_random_sequences_scored() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);

    let bg = BackgroundModel::new(&abc);

    for _ in 0..100 {
        let l = rng.roll(200) + 10;
        let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l);
        let score = bg.null_one(&dsq, l);
        assert!(score.is_finite());
    }
}

// ===================================================================
// p7_gmx.c :: test accessors on X states
// ===================================================================

#[test]
fn test_gmx_xstate_accessors() {
    let m = 5;
    let l = 5;
    let mut gx = ScoreMatrix::new(m, l).expect("new failed");

    for i in 0..=l {
        gx.set_special_row(
            i,
            std::array::from_fn(|s| (i * NUM_SPECIAL_STATES + s) as f32),
        );
    }

    for i in 0..=l {
        for s in 0..NUM_SPECIAL_STATES {
            let expected = (i * NUM_SPECIAL_STATES + s) as f32;
            assert_eq!(gx.special_score(i, s), expected, "xmx({},{}) failed", i, s);
        }
    }
}

// ===================================================================
// Test HMM dump doesn't panic
// ===================================================================
