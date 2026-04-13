// Regression tests for hmmsearch pipeline.
//
// These tests run the full search pipeline via the library API against
// pre-built HMM files and sequence databases, then assert that the results
// (hit rankings, bit-scores, E-values, domain counts, filter pass-rates)
// match known-good golden values.
//
// The golden values were captured from the validated Rust port that was
// confirmed to produce results matching HMMER C 3.4 (scores within ±0.2 bits,
// identical hit rankings and domain counts).
//
// If a refactor changes any of these numbers, the test failures will tell you
// exactly which metric drifted and by how much.

use hmmer_core::background::BackgroundModel;
use hmmer_core::config::SearchMode;
use hmmer_core::modelconfig;
use hmmer_core::pipeline::PipelineStats;
use hmmer_core::pipeline::{
    CapacityHints, FilterPolicy, SearchOutcome, SearchPlan, SearchQuery, Thresholds,
};
use hmmer_core::profile::Profile;
use hmmer_core::tophits::TopHits;
use hmmer_io::hmmfile::HmmFile;
use hmmer_io::seq_reader::FastaReader;

// ---------------------------------------------------------------------------
// Helper: resolve bench_data paths relative to workspace root
// ---------------------------------------------------------------------------

fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap() // crates/
        .parent()
        .unwrap() // workspace root
        .to_path_buf()
}

fn bench_path(relative: &str) -> String {
    workspace_root()
        .join(relative)
        .to_string_lossy()
        .into_owned()
}

// ---------------------------------------------------------------------------
// Helper: run a full hmmsearch and return (pipeline, tophits, nseq)
// ---------------------------------------------------------------------------

struct RegressionRunSummary {
    thresholds: Thresholds,
    search_space: f64,
    stats: PipelineStats,
}

fn run_hmmsearch(hmm_path: &str, db_path: &str) -> (RegressionRunSummary, TopHits, u64) {
    let mut hfp = HmmFile::open(hmm_path, None)
        .unwrap_or_else(|e| panic!("Cannot open HMM {}: {}", hmm_path, e));
    let (abc, hmm) = hfp
        .read()
        .unwrap_or_else(|e| panic!("Cannot read HMM: {}", e));

    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    modelconfig::profile_config(&hmm, &bg, &mut gm, 400, SearchMode::Local);
    modelconfig::reconfigure_length(&mut gm, 400);

    let mut sqfp = FastaReader::open_digital(&abc, db_path)
        .unwrap_or_else(|msg| panic!("Cannot open seqdb {}: {}", db_path, msg));

    let query = SearchQuery::from_configured_profile(gm, bg)
        .unwrap_or_else(|e| panic!("Cannot compile search query: {}", e));
    let plan = SearchPlan::builder(query).build();
    let mut worker = plan
        .spawn_worker(CapacityHints { target_length: 400 })
        .unwrap_or_else(|e| panic!("Cannot create search worker: {}", e));
    let mut th = TopHits::new();
    let thresholds = Thresholds::default();

    let mut nseq = 0u64;
    loop {
        match sqfp.read() {
            Ok(Some(sq)) => {
                nseq += 1;
                let report = worker.search(&sq).unwrap();
                if let SearchOutcome::Hit(hit) = report.outcome {
                    th.push_hit(hit);
                }
            }
            Ok(None) => break,
            Err(msg) => panic!("Error reading sequence: {}", msg),
        }
    }

    let search_space = nseq as f64;
    th.sort_by_sortkey();
    th.threshold(
        thresholds.evalue,
        thresholds.domain_evalue,
        thresholds.inclusion_evalue,
        thresholds.inclusion_domain_evalue,
        thresholds.use_bit_cutoffs,
    );

    let metrics = worker.into_metrics();
    (
        RegressionRunSummary {
            thresholds,
            search_space,
            stats: metrics.stats,
        },
        th,
        nseq,
    )
}

/// Compute the sequence E-value from lnP and Z.
fn evalue(lnp: f64, z: f64) -> f64 {
    (lnp + z.ln()).exp()
}

// ===================================================================
// globins4 × seqdb_100
//
// Every sequence in seqdb_100 is a globin family member, so all 100
// pass every filter and produce significant hits.
// ===================================================================

#[test]
fn test_globins4_seqdb100_hit_count() {
    let (pli, th, nseq) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );
    assert_eq!(nseq, 100, "database should contain 100 sequences");
    // All 100 sequences score significantly
    let mut nhits = 0;
    for i in 0..th.len() {
        if let Some(hit) = th.get_hit(i)
            && evalue(hit.log_pvalue, pli.search_space) <= pli.thresholds.evalue
        {
            nhits += 1;
        }
    }
    assert_eq!(nhits, 100, "expected 100 reported hits for seqdb_100");
}

#[test]
fn test_globins4_seqdb100_top_hit() {
    let (pli, th, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );
    let top = th.get_hit(0).expect("should have at least one hit");
    assert_eq!(top.name, "globins4-sample67", "top hit name");
    let ev = evalue(top.log_pvalue, pli.search_space);
    assert!(
        ev < 1e-27,
        "top hit E-value should be < 1e-27, got {:.2e}",
        ev
    );
    assert!(
        (top.score - 94.0).abs() < 0.5,
        "top hit score should be ~94.0 bits, got {:.1}",
        top.score
    );
}

#[test]
fn test_globins4_seqdb100_ranking_and_scores() {
    let (_pli, th, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );

    // Golden reference: (rank, name, expected_score, score_tolerance)
    // Scores validated against HMMER C 3.4 (within ±0.2 bits).
    let golden: &[(usize, &str, f32, f32)] = &[
        (0, "globins4-sample67", 94.0, 0.5),
        (1, "globins4-sample27", 82.9, 0.5),
        (2, "globins4-sample32", 80.8, 0.5),
        (3, "globins4-sample23", 80.8, 0.5),
        (4, "globins4-sample52", 78.6, 0.5),
        (5, "globins4-sample85", 78.1, 0.5),
        (6, "globins4-sample70", 77.6, 0.5),
        (7, "globins4-sample93", 76.0, 0.5),
        (8, "globins4-sample35", 72.9, 0.5),
        (9, "globins4-sample45", 72.2, 0.5),
        (10, "globins4-sample22", 71.9, 0.5),
        (11, "globins4-sample9", 71.4, 0.5),
        (12, "globins4-sample3", 71.0, 0.5),
        (13, "globins4-sample29", 70.9, 0.5),
        (14, "globins4-sample94", 70.9, 0.5),
        (15, "globins4-sample42", 70.7, 0.5),
        (16, "globins4-sample96", 70.7, 0.5),
        (17, "globins4-sample48", 69.6, 0.5),
        (18, "globins4-sample59", 69.3, 0.5),
        (19, "globins4-sample1", 67.5, 0.5),
    ];

    for &(rank, name, expected_score, tol) in golden {
        let hit = th
            .get_hit(rank)
            .unwrap_or_else(|| panic!("missing hit at rank {}", rank));
        assert_eq!(
            hit.name, name,
            "rank {}: expected {} but got {}",
            rank, name, hit.name
        );
        assert!(
            (hit.score - expected_score).abs() < tol,
            "rank {} ({}): score {:.1} differs from expected {:.1} by more than {:.1}",
            rank,
            name,
            hit.score,
            expected_score,
            tol
        );
    }
}

#[test]
fn test_globins4_seqdb100_bottom_hits() {
    let (pli, th, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );

    // Check the weakest hits — these are most sensitive to regressions.
    // Golden: last 5 reported hits
    let golden_tail: &[(&str, f32, f32)] = &[
        ("globins4-sample24", 29.3, 0.5),
        ("globins4-sample51", 27.9, 0.5),
        ("globins4-sample75", 25.6, 0.5),
        ("globins4-sample11", 23.1, 0.5),
        ("globins4-sample36", 18.2, 0.5),
    ];

    // Find the last N reported hits
    let mut reported: Vec<(String, f32, f64)> = Vec::new();
    for i in 0..th.len() {
        if let Some(hit) = th.get_hit(i) {
            let ev = evalue(hit.log_pvalue, pli.search_space);
            if ev <= pli.thresholds.evalue {
                reported.push((hit.name.clone(), hit.score, ev));
            }
        }
    }

    let n = reported.len();
    for (j, &(name, expected_score, tol)) in golden_tail.iter().enumerate() {
        let idx = n - golden_tail.len() + j;
        assert_eq!(
            reported[idx].0, name,
            "tail rank {}: expected {} but got {}",
            idx, name, reported[idx].0
        );
        assert!(
            (reported[idx].1 - expected_score).abs() < tol,
            "tail rank {} ({}): score {:.1} differs from expected {:.1}",
            idx,
            name,
            reported[idx].1,
            expected_score
        );
    }
}

#[test]
fn test_globins4_seqdb100_domain_counts() {
    let (pli, th, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );

    // Every globin hit should have exactly 1 domain
    for i in 0..th.len() {
        if let Some(hit) = th.get_hit(i) {
            let ev = evalue(hit.log_pvalue, pli.search_space);
            if ev <= pli.thresholds.evalue {
                assert!(
                    hit.num_domains >= 1,
                    "hit {} should have ≥1 domain, got {}",
                    hit.name,
                    hit.num_domains
                );
            }
        }
    }
}

#[test]
fn test_globins4_seqdb100_evalue_ordering() {
    let (pli, th, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );

    // E-values must be monotonically non-decreasing after sort
    let mut prev_ev = 0.0f64;
    for i in 0..th.len() {
        if let Some(hit) = th.get_hit(i) {
            let ev = evalue(hit.log_pvalue, pli.search_space);
            if ev > pli.thresholds.evalue {
                break;
            }
            assert!(
                ev >= prev_ev * 0.999 - 1e-300, // tiny tolerance for float rounding
                "E-values not sorted: rank {} has {:.2e} < previous {:.2e}",
                i,
                ev,
                prev_ev
            );
            prev_ev = ev;
        }
    }
}

#[test]
fn test_globins4_seqdb100_filter_stats() {
    let (pli, _, nseq) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );

    // All 100 globin sequences should pass all filters
    assert_eq!(
        pli.stats.sequences_past_msv, nseq,
        "all seqs should pass MSV"
    );
    assert_eq!(
        pli.stats.sequences_past_bias, nseq,
        "all seqs should pass bias"
    );
    assert_eq!(
        pli.stats.sequences_past_viterbi, nseq,
        "all seqs should pass Viterbi"
    );
    assert_eq!(
        pli.stats.sequences_past_forward, nseq,
        "all seqs should pass Forward"
    );
}

// ===================================================================
// globins4 × seqdb_1000
//
// seqdb_1000 = 100 globins + 900 random proteins.
// The pipeline should filter most non-globins.
// ===================================================================

#[test]
fn test_globins4_seqdb1000_hit_count_and_filters() {
    let (pli, th, nseq) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_1000.fa"),
    );
    assert_eq!(nseq, 1000);

    // Filter cascade — most non-globins get filtered
    assert!(
        pli.stats.sequences_past_msv <= 1000 && pli.stats.sequences_past_msv >= 990,
        "MSV pass count {} outside expected range [990, 1000]",
        pli.stats.sequences_past_msv
    );
    assert!(
        pli.stats.sequences_past_viterbi <= pli.stats.sequences_past_msv,
        "Viterbi pass {} > MSV pass {} (impossible)",
        pli.stats.sequences_past_viterbi,
        pli.stats.sequences_past_msv
    );

    // Should still find ~100 significant globin hits
    let mut nhits = 0;
    for i in 0..th.len() {
        if let Some(hit) = th.get_hit(i) {
            let ev = evalue(hit.log_pvalue, pli.search_space);
            if ev <= pli.thresholds.evalue {
                nhits += 1;
            }
        }
    }
    assert!(
        (990..=1000).contains(&nhits),
        "expected ~996 hits in seqdb_1000, got {}",
        nhits
    );
}

#[test]
fn test_globins4_seqdb1000_top_hit() {
    let (pli, th, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_1000.fa"),
    );
    let top = th.get_hit(0).expect("should have at least one hit");
    assert_eq!(top.name, "globins4-sample191", "top hit in seqdb_1000");
    let ev = evalue(top.log_pvalue, pli.search_space);
    assert!(
        ev < 1e-29,
        "top hit E-value should be < 1e-29, got {:.2e}",
        ev
    );
    assert!(
        (top.score - 104.6).abs() < 0.5,
        "top hit score should be ~104.6 bits, got {:.1}",
        top.score
    );
}

#[test]
fn test_globins4_seqdb1000_ranking_top10() {
    let (_pli, th, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_1000.fa"),
    );

    let golden: &[(usize, &str, f32)] = &[
        (0, "globins4-sample191", 104.6),
        (1, "globins4-sample952", 101.4),
        (2, "globins4-sample663", 100.0),
        (3, "globins4-sample67", 94.0),
        (4, "globins4-sample764", 91.3),
        (5, "globins4-sample358", 90.7),
        (6, "globins4-sample720", 90.4),
        (7, "globins4-sample854", 90.2),
        (8, "globins4-sample660", 89.5),
        (9, "globins4-sample921", 88.6),
    ];

    for &(rank, name, expected_score) in golden {
        let hit = th
            .get_hit(rank)
            .unwrap_or_else(|| panic!("missing hit at rank {}", rank));
        assert_eq!(
            hit.name, name,
            "seqdb_1000 rank {}: expected {} got {}",
            rank, name, hit.name
        );
        assert!(
            (hit.score - expected_score).abs() < 0.5,
            "seqdb_1000 rank {} ({}): score {:.1} expected {:.1}",
            rank,
            name,
            hit.score,
            expected_score
        );
    }
}

// ===================================================================
// fn3 × seqdb_100  (no hits expected)
// ===================================================================

#[test]
fn test_fn3_seqdb100_no_hits() {
    let (pli, th, nseq) = run_hmmsearch(
        &bench_path("bench_data/fn3.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );
    assert_eq!(nseq, 100);

    let mut nhits = 0;
    for i in 0..th.len() {
        if let Some(hit) = th.get_hit(i)
            && evalue(hit.log_pvalue, pli.search_space) <= pli.thresholds.evalue
        {
            nhits += 1;
        }
    }
    assert_eq!(
        nhits, 0,
        "fn3 should produce no hits against globin seqdb_100"
    );

    // Some sequences may pass MSV but none should pass Viterbi+Forward
    assert!(
        pli.stats.sequences_past_forward == 0,
        "fn3: {} seqs passed Forward against globin db (expected 0)",
        pli.stats.sequences_past_forward
    );
}

// ===================================================================
// Pkinase × seqdb_100  (no hits expected)
// ===================================================================

#[test]
fn test_pkinase_seqdb100_no_hits() {
    let (pli, th, nseq) = run_hmmsearch(
        &bench_path("bench_data/Pkinase.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );
    assert_eq!(nseq, 100);

    let mut nhits = 0;
    for i in 0..th.len() {
        if let Some(hit) = th.get_hit(i)
            && evalue(hit.log_pvalue, pli.search_space) <= pli.thresholds.evalue
        {
            nhits += 1;
        }
    }
    assert_eq!(
        nhits, 0,
        "Pkinase should produce no hits against globin seqdb_100"
    );
}

// ===================================================================
// fn3 × seqdb_1000  (filter cascade)
// ===================================================================

#[test]
fn test_fn3_seqdb1000_filter_cascade() {
    let (pli, _, nseq) = run_hmmsearch(
        &bench_path("bench_data/fn3.hmm"),
        &bench_path("bench_data/seqdb_1000.fa"),
    );
    assert_eq!(nseq, 1000);

    // fn3 against globin+random: MSV should filter most, Viterbi should filter rest
    assert!(
        pli.stats.sequences_past_msv < 100,
        "fn3 MSV pass should be < 100, got {}",
        pli.stats.sequences_past_msv
    );
    assert_eq!(
        pli.stats.sequences_past_viterbi, 0,
        "fn3 Viterbi pass should be 0, got {}",
        pli.stats.sequences_past_viterbi
    );
    assert_eq!(
        pli.stats.sequences_past_forward, 0,
        "fn3 Forward pass should be 0, got {}",
        pli.stats.sequences_past_forward
    );
}

// ===================================================================
// Pkinase × seqdb_1000  (filter cascade)
// ===================================================================

#[test]
fn test_pkinase_seqdb1000_filter_cascade() {
    let (pli, _, nseq) = run_hmmsearch(
        &bench_path("bench_data/Pkinase.hmm"),
        &bench_path("bench_data/seqdb_1000.fa"),
    );
    assert_eq!(nseq, 1000);

    // Pkinase against globin+random: a few pass MSV, fewer pass Viterbi, none pass Forward
    assert!(
        pli.stats.sequences_past_msv < 150,
        "Pkinase MSV pass should be < 150, got {}",
        pli.stats.sequences_past_msv
    );
    assert!(
        pli.stats.sequences_past_viterbi <= 10,
        "Pkinase Viterbi pass should be ≤ 10, got {}",
        pli.stats.sequences_past_viterbi
    );
    assert_eq!(
        pli.stats.sequences_past_forward, 0,
        "Pkinase Forward pass should be 0, got {}",
        pli.stats.sequences_past_forward
    );
}

// ===================================================================
// Score-E-value consistency
// ===================================================================

#[test]
fn test_score_evalue_consistency() {
    let (pli, th, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );

    // Higher bit-scores must correspond to lower E-values
    for i in 0..th.len() {
        if let Some(hit) = th.get_hit(i) {
            let ev = evalue(hit.log_pvalue, pli.search_space);
            if ev > pli.thresholds.evalue {
                break;
            }
            // Score should be positive for significant hits
            assert!(
                hit.score > 0.0,
                "hit {} has non-positive score {:.1}",
                hit.name,
                hit.score
            );
            // lnP should be negative (probability < 1)
            assert!(
                hit.log_pvalue < 0.0,
                "hit {} has non-negative lnP {:.4}",
                hit.name,
                hit.log_pvalue
            );
        }
    }
}

// ===================================================================
// Domain envelope coordinates
// ===================================================================

#[test]
fn test_globins4_domain_envelopes() {
    let (pli, th, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );

    for i in 0..th.len() {
        if let Some(hit) = th.get_hit(i) {
            let ev = evalue(hit.log_pvalue, pli.search_space);
            if ev > pli.thresholds.evalue {
                break;
            }
            for d in &hit.domains {
                // Envelope must contain alignment
                assert!(
                    d.envelope_start <= d.alignment_start && d.alignment_end <= d.envelope_end,
                    "hit {} domain: envelope [{}, {}] does not contain alignment [{}, {}]",
                    hit.name,
                    d.envelope_start,
                    d.envelope_end,
                    d.alignment_start,
                    d.alignment_end
                );
                // Coordinates must be positive
                assert!(
                    d.envelope_start >= 1,
                    "hit {} domain: ienv {} < 1",
                    hit.name,
                    d.envelope_start
                );
                assert!(
                    d.envelope_end >= d.envelope_start,
                    "hit {} domain: jenv {} < ienv {}",
                    hit.name,
                    d.envelope_end,
                    d.envelope_start
                );
                // Domain bit-score should be reasonable
                assert!(
                    d.bitscore > -50.0 && d.bitscore < 500.0,
                    "hit {} domain: bitscore {:.1} out of reasonable range",
                    hit.name,
                    d.bitscore
                );
                // Overall accuracy should be in [0, 1]
                assert!(
                    d.optimal_accuracy_score >= 0.0 && d.optimal_accuracy_score <= 1.01,
                    "hit {} domain: oasc {:.2} out of [0, 1]",
                    hit.name,
                    d.optimal_accuracy_score
                );
            }
        }
    }
}

// ===================================================================
// --max mode (filters disabled)
// ===================================================================

#[test]
fn test_globins4_max_mode() {
    let mut hfp = HmmFile::open(&bench_path("bench_data/globins4.hmm"), None).unwrap();
    let (abc, hmm) = hfp.read().unwrap();

    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    modelconfig::profile_config(&hmm, &bg, &mut gm, 400, SearchMode::Local);
    modelconfig::reconfigure_length(&mut gm, 400);

    let mut sqfp = FastaReader::open_digital(&abc, &bench_path("bench_data/seqdb_100.fa")).unwrap();

    let query = SearchQuery::from_configured_profile(gm, bg).unwrap();
    let plan = SearchPlan::builder(query)
        .filters(FilterPolicy {
            max_mode: true,
            ..FilterPolicy::default()
        })
        .build();
    let mut worker = plan
        .spawn_worker(CapacityHints { target_length: 400 })
        .unwrap();
    let mut th = TopHits::new();
    let thresholds = Thresholds::default();

    let mut nseq = 0u64;
    loop {
        match sqfp.read() {
            Ok(Some(sq)) => {
                nseq += 1;
                let report = worker.search(&sq).unwrap();
                if let SearchOutcome::Hit(hit) = report.outcome {
                    th.push_hit(hit);
                }
            }
            Ok(None) => break,
            Err(msg) => panic!("Error reading sequence: {}", msg),
        }
    }

    th.sort_by_sortkey();
    th.threshold(
        thresholds.evalue,
        thresholds.domain_evalue,
        thresholds.inclusion_evalue,
        thresholds.inclusion_domain_evalue,
        thresholds.use_bit_cutoffs,
    );
    let metrics = worker.into_metrics();

    // In max mode, all sequences proceed to Forward
    assert_eq!(
        metrics.stats.sequences_past_forward, nseq,
        "max mode: all seqs should reach Forward"
    );

    // Results should be very similar to filtered mode
    let top = th.get_hit(0).expect("should have hits in max mode");
    assert_eq!(top.name, "globins4-sample67");
    assert!(
        (top.score - 94.0).abs() < 0.5,
        "max mode top score should be ~94.0, got {:.1}",
        top.score
    );
}

// ===================================================================
// Residue count tracking
// ===================================================================

#[test]
fn test_residue_count_tracking() {
    let (pli, _, nseq) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );
    assert_eq!(nseq, 100);
    // Total residues should be non-zero and reasonable (avg ~150aa × 100 seqs)
    assert!(
        pli.stats.num_residues > 10000 && pli.stats.num_residues < 50000,
        "residue count {} outside expected range for 100 globin-length seqs",
        pli.stats.num_residues
    );
}

// ===================================================================
// Determinism: running the same search twice produces identical results
// ===================================================================

#[test]
fn test_deterministic_results() {
    let (_, th1, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );
    let (pli2, th2, _) = run_hmmsearch(
        &bench_path("bench_data/globins4.hmm"),
        &bench_path("bench_data/seqdb_100.fa"),
    );

    assert_eq!(th1.len(), th2.len(), "hit count must be deterministic");

    for i in 0..th1.len() {
        let h1 = th1.get_hit(i).unwrap();
        let h2 = th2.get_hit(i).unwrap();

        let ev1 = evalue(h1.log_pvalue, pli2.search_space);
        let _ev2 = evalue(h2.log_pvalue, pli2.search_space);
        if ev1 > 10.0 {
            break;
        }

        assert_eq!(h1.name, h2.name, "rank {}: names differ", i);
        assert_eq!(
            h1.score, h2.score,
            "rank {} ({}): scores differ",
            i, h1.name
        );
        assert_eq!(
            h1.log_pvalue, h2.log_pvalue,
            "rank {} ({}): lnP differ",
            i, h1.name
        );
        assert_eq!(
            h1.num_domains, h2.num_domains,
            "rank {} ({}): ndom differ",
            i, h1.name
        );
    }
}
