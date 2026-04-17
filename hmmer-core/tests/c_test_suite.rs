// Integration tests porting the C TESTDRIVE unit tests from HMMER.
//
// Each test function corresponds to a utest_*() or main() TESTDRIVE
// block in the corresponding C source file.

use hmmer_core::alphabet::Alphabet;
use hmmer_core::background::BackgroundModel;
use hmmer_core::config::TraceStateType;
use hmmer_core::config::*;
use hmmer_core::hmm::Hmm;
use hmmer_core::logsum;
use hmmer_core::profile::Profile;
use hmmer_core::rng::XorShift64;
use hmmer_core::score_matrix::NUM_SPECIAL_STATES;
use hmmer_core::score_matrix::*;
use hmmer_core::simd::f32_msv;
use hmmer_core::simd::f32_viterbi;
use hmmer_core::simd::forward_filter;
use hmmer_core::simd::fwd_bck;
use hmmer_core::simd::oprofile::OptimizedProfile;
use hmmer_core::test_helpers::*;
use hmmer_core::tophits::TopHits;
use hmmer_core::trace::Trace;

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
// p7_hmm.c :: utest_composition()
// ===================================================================

#[test]
fn test_hmm_composition() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 50;
    let k = abc.canonical_size;

    let mut hmm = hmm_sample(&mut rng, m, &abc);
    hmm_set_composition(&mut hmm);

    let compo = hmm.model_composition.as_ref().expect("compo should be set");
    let sum: f32 = compo[..k].iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "composition sum = {}, expected ~1.0",
        sum
    );

    for (a, value) in compo.iter().enumerate().take(k) {
        assert!(*value >= 0.0, "composition[{}] = {} (negative!)", a, value);
    }
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
// p7_trace.c :: creation and comparison
// ===================================================================

#[test]
fn test_trace_create_and_reuse() {
    let mut tr = Trace::new();
    assert_eq!(tr.len(), 0);

    tr.append(TraceStateType::Start, 0, 0);
    tr.append(TraceStateType::NTerminal, 0, 0);
    tr.append(TraceStateType::Begin, 0, 0);
    tr.append(TraceStateType::Match, 1, 1);
    tr.append(TraceStateType::Match, 2, 2);
    tr.append(TraceStateType::End, 0, 0);
    tr.append(TraceStateType::CTerminal, 0, 0);
    tr.append(TraceStateType::Terminate, 0, 0);
    assert_eq!(tr.len(), 8);

    tr.reuse();
    assert_eq!(tr.len(), 0);
}

#[test]
fn test_trace_compare() {
    let mut tr1 = Trace::new();
    let mut tr2 = Trace::new();

    for &st in &[
        TraceStateType::Start,
        TraceStateType::NTerminal,
        TraceStateType::Begin,
        TraceStateType::Match,
        TraceStateType::Match,
        TraceStateType::End,
        TraceStateType::CTerminal,
        TraceStateType::Terminate,
    ] {
        tr1.append(st, 0, 0);
        tr2.append(st, 0, 0);
    }

    assert!(tr1.compare(&tr2, 0.001));

    tr2.append(TraceStateType::Match, 3, 3);
    assert!(!tr1.compare(&tr2, 0.001));
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
// p7_gmx.c :: utest_GrowTo()
// ===================================================================

#[test]
fn test_gmx_resize() {
    let mut gx = ScoreMatrix::new(20, 20).expect("ScoreMatrix::new failed");

    gx.resize(40, 20).unwrap();
    assert_eq!(gx.num_nodes, 40);
    assert_eq!(gx.sequence_length, 20);
    assert!(gx.main.dim().1 >= 41);

    gx.resize(40, 40).unwrap();
    assert_eq!(gx.sequence_length, 40);
    assert!(gx.special.dim().0 >= 41);

    gx.resize(80, 80).unwrap();
    assert_eq!(gx.num_nodes, 80);
    assert!(gx.main.dim().1 >= 81);
    assert!(gx.special.dim().0 >= 81);

    gx.resize(100, 100).unwrap();
    let m = 100;
    let l = 100;
    let mut n: u64 = 0;
    for i in 0..=l {
        for k in 0..=m {
            gx.set_match_score(i, k, n as f32);
            n += 1;
        }
    }
    let mut n2: u64 = 0;
    for i in 0..=l {
        for k in 0..=m {
            assert_eq!(gx.match_score(i, k) as u64, n2);
            n2 += 1;
        }
    }
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
        for s in 0..NUM_SPECIAL_STATES {
            gx1.set_special_score(i, s, n);
            gx2.set_special_score(i, s, n);
            n += 1.0;
        }
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
// p7_profile.c :: utest_Compare()
// ===================================================================

#[test]
fn test_profile_compare() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 200;
    let l = 400;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let bg = BackgroundModel::new(&abc);

    let mut gm1 = Profile::new(hmm.num_nodes, &abc);
    let mut gm2 = Profile::new(hmm.num_nodes, &abc);

    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm1, l, SearchMode::Local);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm2, l, SearchMode::Local);

    assert!(gm1.compare(&gm2, 0.001));
}

// ===================================================================
// p7_tophits.c :: TESTDRIVE
// ===================================================================

#[test]
fn test_tophits_sort_and_merge() {
    let mut rng = XorShift64::new(42);
    let n = 100;

    let mut h1 = TopHits::new();
    let mut h2 = TopHits::new();
    let mut h3 = TopHits::new();

    for _ in 0..n {
        let key = rng.random();
        {
            let hit = h1.create_next_hit();
            hit.name = "not_unique_name".to_string();
            hit.sortkey = key;
            hit.score = key as f32;
        }
        let key2 = 10.0 * rng.random();
        {
            let hit = h2.create_next_hit();
            hit.name = "not_unique_name".to_string();
            hit.sortkey = key2;
            hit.score = key2 as f32;
        }
        let key3 = 0.1 * rng.random();
        {
            let hit = h3.create_next_hit();
            hit.name = "not_unique_name".to_string();
            hit.sortkey = key3;
            hit.score = key3 as f32;
        }
    }

    // Add named hits for sort verification
    {
        let hit = h1.create_next_hit();
        hit.name = "last".to_string();
        hit.sortkey = -1.0;
        hit.score = -1.0;
    }
    {
        let hit = h1.create_next_hit();
        hit.name = "first".to_string();
        hit.sortkey = 20.0;
        hit.score = 20.0;
    }

    h1.sort_by_sortkey();

    let first = h1.get_hit(0).expect("should have first hit");
    assert_eq!(first.name, "first");

    let nhits = h1.len();
    let last = h1.get_hit(nhits - 1).expect("should have last hit");
    assert_eq!(last.name, "last");
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

    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, 400, SearchMode::Local);

    assert_eq!(gm.num_nodes, hmm.num_nodes);
    assert_eq!(gm.target_length, 400);
    assert_eq!(gm.mode, SearchMode::Local);
}

#[test]
fn test_modelconfig_occupancy() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 200;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, 400, SearchMode::Local);

    assert_eq!(gm.num_nodes, m);
    assert!(gm.target_length > 0);
}

#[test]
fn test_modelconfig_all_modes() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);

    let hmm = hmm_sample(&mut rng, 50, &abc);
    let bg = BackgroundModel::new(&abc);

    for &mode in &[
        SearchMode::Local,
        SearchMode::Glocal,
        SearchMode::UniLocal,
        SearchMode::UniGlocal,
    ] {
        let mut gm = Profile::new(hmm.num_nodes, &abc);
        hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, 200, mode);
        assert_eq!(gm.mode, mode);
    }
}

// ===================================================================
// p7_hmm.c :: HMM clone and compare
// ===================================================================

#[test]
fn test_hmm_clone_and_compare() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 50;

    let hmm1 = hmm_sample(&mut rng, m, &abc);
    let hmm2 = hmm1.clone();

    assert!(hmm1.compare(&hmm2, 1e-6));
}

// ===================================================================
// p7_hmm.c :: scale and renormalize
// ===================================================================

#[test]
fn test_hmm_scale_renormalize() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 20;

    let mut hmm = hmm_sample(&mut rng, m, &abc);
    hmm.scale(2.0);
    hmm.renormalize();

    assert!(hmm_validate(&hmm, 0.01));
}

// ===================================================================
// p7_hmm.c :: zero
// ===================================================================

#[test]
fn test_hmm_zero() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);

    let mut hmm = hmm_sample(&mut rng, 10, &abc);
    hmm.zero();

    let k = abc.canonical_size;
    for node in 0..=hmm.num_nodes {
        for j in 0..HMM_NUM_TRANSITIONS {
            assert_eq!(hmm.transitions(node)[j], 0.0);
        }
        for a in 0..k {
            assert_eq!(hmm.match_emissions(node)[a], 0.0);
            assert_eq!(hmm.insert_emissions(node)[a], 0.0);
        }
    }
}

// ===================================================================
// p7_hmm.c :: encode/decode statetype
// ===================================================================

#[test]
fn test_hmm_encode_decode_statetype() {
    let states = vec![
        ("M", TraceStateType::Match),
        ("D", TraceStateType::Delete),
        ("I", TraceStateType::Insert),
        ("S", TraceStateType::Start),
        ("N", TraceStateType::NTerminal),
        ("B", TraceStateType::Begin),
        ("E", TraceStateType::End),
        ("C", TraceStateType::CTerminal),
        ("T", TraceStateType::Terminate),
        ("J", TraceStateType::Jump),
        ("X", TraceStateType::Missing),
    ];

    for (name, st) in &states {
        let encoded = Hmm::encode_statetype(name);
        assert_eq!(encoded, *st, "encode '{}' failed", name);
        let decoded = Hmm::decode_statetype(*st);
        assert_eq!(decoded, *name, "decode {:?} failed", st);
    }

    assert_eq!(Hmm::encode_statetype("m"), TraceStateType::Match);
    assert_eq!(Hmm::encode_statetype("d"), TraceStateType::Delete);
    assert_eq!(Hmm::encode_statetype("Q"), TraceStateType::Bogus);
}

// ===================================================================
// seqmodel.c :: utest_normalization()
// ===================================================================

// #[test]
// fn test_seqmodel_normalization() {
//     let abc = Alphabet::amino();
//     let mut rng = XorShift64::new(42);
//     let l = 20;

//     let bg = BackgroundModel::new(&abc);
//     let dsq = random_digital_seq(&mut rng, &bg.f, abc.k, l);

//     let hmm = hmmer_core::seqmodel::modelsample_single_pathed(&abc, "test_seq", &dsq, l);

//     assert_eq!(hmm.m, l);
//     assert_eq!(hmm.name, "test_seq");

//     for k in 1..=l {
//         let sum: f32 = hmm.match_emissions(k).iter().sum();
//         assert!((sum - 1.0).abs() < 0.01, "node {} mat sum = {}", k, sum);
//     }

//     let uniform = 1.0 / abc.k as f32;
//     for k in 0..=l {
//         for a in 0..abc.k {
//             assert!(
//                 (hmm.insert_emissions(k)[a] - uniform).abs() < 0.001,
//                 "node {} ins[{}] = {}", k, a, hmm.insert_emissions(k)[a]
//             );
//         }
//     }
// }

// ===================================================================
// generic_viterbi.c :: utest_viterbi()
// ===================================================================

#[test]
fn test_generic_viterbi_random_seqs() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 20;
    let l = 50;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
    let mut om = OptimizedProfile::from_profile(&gm);
    om.reconfigure_length(l);

    for _ in 0..10 {
        let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l);
        let score = f32_viterbi::viterbi_filter_f32(&dsq, dsq.len(), &om);
        assert!(!score.is_nan(), "Viterbi score is NaN");
    }
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
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
    let mut om = OptimizedProfile::from_profile(&gm);
    om.reconfigure_length(l);

    for _ in 0..10 {
        let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l);
        let result = forward_filter::forward_filter(&dsq, dsq.len(), &om);
        assert!(!result.score.is_nan(), "Forward score is NaN");
    }
}

// ===================================================================
// ===================================================================
// forward_backward.rs :: forward_score_only / forward_checkpointed
// Verify that both new functions agree with the reference forward().
// ===================================================================

#[test]
fn test_forward_score_only_agrees_with_forward() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 20;
    let bg = BackgroundModel::new(&abc);

    for l in [1, 5, 20, 50, 200] {
        let hmm = hmm_sample(&mut rng, m, &abc);
        let mut gm = Profile::new(hmm.num_nodes, &abc);
        hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
        let mut om = OptimizedProfile::from_profile(&gm);
        om.reconfigure_length(l);

        for _ in 0..10 {
            let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l);
            let ref_score = forward_filter::forward_filter(&dsq, dsq.len(), &om).score;
            let (checkpoints, _) = fwd_bck::forward_checkpointed_simd(&dsq, dsq.len(), &om);
            let score_only = checkpoints.overall_score;

            let diff = (ref_score - score_only).abs();
            assert!(
                diff < 1e-4,
                "checkpointed forward mismatch: ref={ref_score}, got={score_only}, diff={diff}, L={l}"
            );
        }
    }
}

#[test]
fn test_forward_checkpointed_agrees_with_forward() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 20;
    let bg = BackgroundModel::new(&abc);

    for l in [1, 5, 20, 50, 200] {
        let hmm = hmm_sample(&mut rng, m, &abc);
        let mut gm = Profile::new(hmm.num_nodes, &abc);
        hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
        let mut om = OptimizedProfile::from_profile(&gm);
        om.reconfigure_length(l);

        for _ in 0..10 {
            let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l);
            let ref_result = forward_filter::forward_filter(&dsq, dsq.len(), &om);
            let (checkpoints, simd_data) = fwd_bck::forward_checkpointed_simd(&dsq, dsq.len(), &om);

            let diff = (ref_result.score - checkpoints.overall_score).abs();
            assert!(
                diff < 1e-4,
                "forward_checkpointed score mismatch: ref={}, got={}, diff={diff}, L={l}",
                ref_result.score,
                checkpoints.overall_score
            );

            let mut cumulative_scale = 0.0f32;
            for i in 0..=l {
                cumulative_scale += ref_result.scale_factors[i].ln();
                for s in 0..5 {
                    let raw = ref_result.xmx[i][s];
                    let ref_sp = if raw > 0.0 {
                        raw.ln() + cumulative_scale
                    } else {
                        f32::NEG_INFINITY
                    };
                    let chk_sp = checkpoints.special_score(i, s);
                    let sp_diff = (ref_sp - chk_sp).abs();
                    assert!(
                        sp_diff < 1e-4 || (ref_sp.is_infinite() && chk_sp.is_infinite()),
                        "special state mismatch at row={i}, state={s}: ref={ref_sp}, got={chk_sp}"
                    );
                }
            }

            for seg in 0..checkpoints.checkpoint_indices.len() - 1 {
                let start = checkpoints.checkpoint_indices[seg];
                let end = checkpoints.checkpoint_indices[seg + 1];
                let rows = fwd_bck::recompute_forward_segment_simd(
                    &dsq,
                    &om,
                    &checkpoints,
                    &simd_data,
                    seg,
                    start,
                    end,
                );

                let (_, recomputed_end) = rows.last().expect("segment should include its endpoint");
                let expected_end = &checkpoints.checkpoint_main[seg + 1];
                for k in 0..=gm.num_nodes {
                    for s in 0..3 {
                        let ref_val = expected_end[[k, s]];
                        let rec_val = recomputed_end[[k, s]];
                        let val_diff = (ref_val - rec_val).abs();
                        assert!(
                            val_diff < 1e-4 || (ref_val.is_infinite() && rec_val.is_infinite()),
                            "recomputed checkpoint mismatch at row={end}, k={k}, s={s}: ref={ref_val}, got={rec_val}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_recompute_forward_segment() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 20;
    let l = 100;
    let bg = BackgroundModel::new(&abc);

    let hmm = hmm_sample(&mut rng, m, &abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
    let mut om = OptimizedProfile::from_profile(&gm);
    om.reconfigure_length(l);

    let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l);
    let (checkpoints, simd_data) = fwd_bck::forward_checkpointed_simd(&dsq, dsq.len(), &om);

    for seg in 0..checkpoints.checkpoint_indices.len() - 1 {
        let start = checkpoints.checkpoint_indices[seg];
        let end = checkpoints.checkpoint_indices[seg + 1];
        let rows = fwd_bck::recompute_forward_segment_simd(
            &dsq,
            &om,
            &checkpoints,
            &simd_data,
            seg,
            start,
            end,
        );

        let (_, start_row) = rows.first().expect("segment should include start row");
        for k in 0..=gm.num_nodes {
            for s in 0..3 {
                let ref_val = checkpoints.checkpoint_main[seg][[k, s]];
                let rec_val = start_row[[k, s]];
                let diff = (ref_val - rec_val).abs();
                assert!(
                    diff < 1e-4 || (ref_val.is_infinite() && rec_val.is_infinite()),
                    "recompute mismatch at start row={start}, k={k}, s={s}: ref={ref_val}, got={rec_val}"
                );
            }
        }

        let (_, end_row) = rows.last().expect("segment should include end row");
        for k in 0..=gm.num_nodes {
            for s in 0..3 {
                let ref_val = checkpoints.checkpoint_main[seg + 1][[k, s]];
                let rec_val = end_row[[k, s]];
                let diff = (ref_val - rec_val).abs();
                assert!(
                    diff < 1e-4 || (ref_val.is_infinite() && rec_val.is_infinite()),
                    "recompute mismatch at end row={end}, k={k}, s={s}: ref={ref_val}, got={rec_val}"
                );
            }
        }
    }
}

// ===================================================================
// generic_msv.c :: utest_msv()
// ===================================================================

#[test]
fn test_generic_msv_random_seqs() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 20;
    let l = 50;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
    let mut om = OptimizedProfile::from_profile(&gm);
    om.reconfigure_length(l);

    for _ in 0..10 {
        let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l);
        let score = f32_msv::msv_filter_f32(&dsq, dsq.len(), &om);
        assert!(score.is_finite(), "MSV score not finite: {}", score);
    }
}

// ===================================================================
// p7_alidisplay.c :: tests
// ===================================================================

#[test]
fn test_alidisplay_create_and_clone() {
    let ad = hmmer_core::alidisplay::AliDisplay::new();
    assert_eq!(ad.alignment_length, 0);
    assert!(ad.model.is_empty());

    let ad2 = ad.clone_display();
    assert_eq!(ad.alignment_length, ad2.alignment_length);
}

// ===================================================================
// p7_domain.c :: tests
// ===================================================================

#[test]
fn test_domain_create() {
    let dom = hmmer_core::domain::Domain::default();
    assert_eq!(dom.envelope_start, 0);
    assert_eq!(dom.envelope_end, 0);
    assert!(dom.scores_per_pos.is_none());
}

// ===================================================================
// p7_pipeline.c :: tests
// ===================================================================

#[test]
fn test_pipeline_create() {
    let pipeline = hmmer_core::pipeline::FilterPolicy::default();

    assert!((pipeline.msv_threshold - 0.02).abs() < 0.001);
    assert!((pipeline.viterbi_threshold - 1e-3).abs() < 1e-6);
    assert!((pipeline.forward_threshold - 1e-5).abs() < 1e-8);
}

// ===================================================================
// p7_domaindef.c :: creation tests
// ===================================================================

#[test]
fn test_domaindef_create() {
    let ws = hmmer_core::domaindef::DomainWorkspace::new();
    assert!(ws.model_occupancy.is_empty());
    let config = hmmer_core::domaindef::DomainConfig::default();
    assert_eq!(config.nsamples, 200);
}

#[test]
fn test_domaindef_grow_to() {
    let mut ws = hmmer_core::domaindef::DomainWorkspace::new();
    ws.grow_to(500);
}

// ===================================================================
// p7_hit.c :: tests
// ===================================================================

#[test]
fn test_hit_create_and_flags() {
    let hit = hmmer_core::hit::Hit::default();
    assert_eq!(hit.num_domains, 0);
    assert!(hit.name.is_empty());
    assert!(hit.sortkey.is_finite());
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
        for s in 0..NUM_SPECIAL_STATES {
            let v = (i * NUM_SPECIAL_STATES + s) as f32;
            gx.set_special_score(i, s, v);
        }
    }

    for i in 0..=l {
        for s in 0..NUM_SPECIAL_STATES {
            let expected = (i * NUM_SPECIAL_STATES + s) as f32;
            assert_eq!(gx.special_score(i, s), expected, "xmx({},{}) failed", i, s);
        }
    }
}

// ===================================================================
// Random HMM -> clone -> compare
// ===================================================================

#[test]
fn test_random_hmm_clone_compare() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);

    for _ in 0..10 {
        let m = rng.roll(100) + 5;
        let hmm = hmm_sample(&mut rng, m, &abc);
        let cloned = hmm.clone();
        assert!(hmm.compare(&cloned, 1e-6));
    }
}

// ===================================================================
// Test HMM dump doesn't panic
// ===================================================================

#[test]
fn test_hmm_dump() {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let hmm = hmm_sample(&mut rng, 5, &abc);

    let mut buf = Vec::new();
    hmm.dump(&mut buf).expect("dump should succeed");
    let output = String::from_utf8(buf).expect("should be valid UTF-8");
    assert!(output.contains("HMM:"));
}

// ===================================================================
// Test bg dump doesn't panic
// ===================================================================

#[test]
fn test_bg_dump() {
    let abc = Alphabet::amino();
    let bg = BackgroundModel::new(&abc);

    let mut buf = Vec::new();
    bg.dump(&mut buf).expect("dump should succeed");
    let output = String::from_utf8(buf).expect("should be valid UTF-8");
    assert!(output.contains("Background model"));
}
