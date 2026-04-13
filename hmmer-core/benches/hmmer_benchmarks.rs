// Criterion benchmarks for the HMMER Rust port.
//
// Benchmarks are structured to match the C test-speed/ benchmark pattern:
// - Core DP algorithms (Viterbi, Forward, Backward, MSV)
// - Data structure operations (logsum, HMM sampling, profile config)
// - Background model scoring
//
// To run: cargo bench -p hmmer-core

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use hmmer_core::alphabet::{Alphabet, Dsq};
use hmmer_core::background::BackgroundModel;
use hmmer_core::config::*;
use hmmer_core::logsum;
use hmmer_core::profile::Profile;
use hmmer_core::rng::XorShift64;
use hmmer_core::score_matrix::*;
use hmmer_core::simd::f32_msv;
use hmmer_core::simd::f32_viterbi;
use hmmer_core::simd::forward_filter;
use hmmer_core::simd::fwd_bck;
use hmmer_core::simd::oprofile::OptimizedProfile;
use hmmer_core::test_helpers::*;
use hmmer_core::tophits::TopHits;

// ===================================================================
// Logsum benchmark
// ===================================================================

fn bench_logsum(c: &mut Criterion) {
    let mut rng = XorShift64::new(42);
    let n = 10000;
    let pairs: Vec<(f32, f32)> = (0..n)
        .map(|_| {
            let a = (rng.random() as f32 - 0.5) * 40.0;
            let b = (rng.random() as f32 - 0.5) * 40.0;
            (a, b)
        })
        .collect();

    c.bench_function("logsum/10k_pairs", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for &(a, val_b) in &pairs {
                sum += logsum::flogsum(black_box(a), black_box(val_b));
            }
            black_box(sum)
        })
    });
}

// ===================================================================
// HMM sampling benchmark
// ===================================================================

fn bench_hmm_sample(c: &mut Criterion) {
    let abc = Alphabet::amino();

    let mut group = c.benchmark_group("hmm_sample");
    for m in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(m), m, |b, &m| {
            let mut rng = XorShift64::new(42);
            b.iter(|| black_box(hmm_sample(&mut rng, m, &abc)))
        });
    }
    group.finish();
}

// ===================================================================
// Profile configuration benchmark
// ===================================================================

fn bench_profile_config(c: &mut Criterion) {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let l = 400;

    let mut group = c.benchmark_group("profile_config");
    for m in [50, 100, 200].iter() {
        let hmm = hmm_sample(&mut rng, *m, &abc);
        let bg = BackgroundModel::new(&abc);

        group.bench_with_input(BenchmarkId::from_parameter(m), m, |b, &m| {
            b.iter(|| {
                let mut gm = Profile::new(m, &abc);
                hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
                black_box(&gm);
            })
        });
    }
    group.finish();
}

// ===================================================================
// Null model scoring benchmark
// ===================================================================

fn bench_null_one(c: &mut Criterion) {
    let abc = Alphabet::amino();
    let bg = BackgroundModel::new(&abc);
    let mut rng = XorShift64::new(42);

    let mut group = c.benchmark_group("null_one");
    for l in [100, 400, 1000].iter() {
        let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, *l);

        group.bench_with_input(BenchmarkId::from_parameter(l), l, |b, &l| {
            b.iter(|| black_box(bg.null_one(&dsq, l)))
        });
    }
    group.finish();
}

// ===================================================================
// Generic Viterbi benchmark
// ===================================================================

fn bench_generic_viterbi(c: &mut Criterion) {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 100;
    let l = 400;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
    let mut om = OptimizedProfile::from_profile(&gm);
    om.reconfigure_length(l);

    let nseqs = 100;
    let seqs: Vec<Vec<Dsq>> = (0..nseqs)
        .map(|_| random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l))
        .collect();

    c.bench_function(&format!("generic_viterbi/M{}_L{}", m, l), |b| {
        b.iter(|| {
            for dsq in &seqs {
                let _ = f32_viterbi::viterbi_filter_f32(black_box(dsq), dsq.len(), &om);
            }
        })
    });
}

// ===================================================================
// Generic Forward benchmark
// ===================================================================

fn bench_generic_forward(c: &mut Criterion) {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 100;
    let l = 400;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
    let mut om = OptimizedProfile::from_profile(&gm);
    om.reconfigure_length(l);

    let nseqs = 100;
    let seqs: Vec<Vec<Dsq>> = (0..nseqs)
        .map(|_| random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l))
        .collect();

    c.bench_function(&format!("generic_forward/M{}_L{}", m, l), |b| {
        b.iter(|| {
            for dsq in &seqs {
                let _ = forward_filter::forward_filter(black_box(dsq), dsq.len(), &om);
            }
        })
    });
}

// ===================================================================
// Generic Backward benchmark
// ===================================================================

fn bench_checkpointed_forward(c: &mut Criterion) {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 100;
    let l = 400;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
    let mut om = OptimizedProfile::from_profile(&gm);
    om.reconfigure_length(l);

    let nseqs = 100;
    let seqs: Vec<Vec<Dsq>> = (0..nseqs)
        .map(|_| random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l))
        .collect();

    c.bench_function(&format!("checkpointed_forward/M{}_L{}", m, l), |b| {
        b.iter(|| {
            for dsq in &seqs {
                let _ = fwd_bck::forward_checkpointed_simd(black_box(dsq), dsq.len(), &om);
            }
        })
    });
}

// ===================================================================
// Generic MSV filter benchmark
// ===================================================================

fn bench_generic_msv(c: &mut Criterion) {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);
    let m = 100;
    let l = 400;

    let hmm = hmm_sample(&mut rng, m, &abc);
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    hmmer_core::modelconfig::profile_config(&hmm, &bg, &mut gm, l, SearchMode::Local);
    let mut om = OptimizedProfile::from_profile(&gm);
    om.reconfigure_length(l);

    let nseqs = 100;
    let seqs: Vec<Vec<Dsq>> = (0..nseqs)
        .map(|_| random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l))
        .collect();

    c.bench_function(&format!("generic_msv/M{}_L{}", m, l), |b| {
        b.iter(|| {
            for dsq in &seqs {
                let _ = f32_msv::msv_filter_f32(black_box(dsq), dsq.len(), &om);
            }
        })
    });
}

// ===================================================================
// DP matrix allocation/growth benchmark
// ===================================================================

fn bench_gmx_operations(c: &mut Criterion) {
    c.bench_function("gmx_create/M100_L400", |b| {
        b.iter(|| black_box(ScoreMatrix::new(100, 400).unwrap()))
    });

    c.bench_function("gmx_resize/20x20_to_100x400", |b| {
        b.iter_batched(
            || ScoreMatrix::new(20, 20).unwrap(),
            |mut gx| {
                gx.resize(100, 400).unwrap();
                black_box(&gx);
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

// ===================================================================
// HMM occupancy calculation benchmark
// ===================================================================

fn bench_occupancy(c: &mut Criterion) {
    let abc = Alphabet::amino();
    let mut rng = XorShift64::new(42);

    let mut group = c.benchmark_group("occupancy");
    for m in [50, 200, 500].iter() {
        let hmm = hmm_sample(&mut rng, *m, &abc);

        group.bench_with_input(BenchmarkId::from_parameter(m), m, |b, _m| {
            b.iter(|| black_box(hmm.calculate_occupancy()))
        });
    }
    group.finish();
}

// ===================================================================
// TopHits sort benchmark
// ===================================================================

fn bench_tophits_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("tophits_sort");
    for n in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter_batched(
                || {
                    let mut rng = XorShift64::new(42);
                    let mut th = TopHits::new();
                    for _ in 0..n {
                        let hit = th.create_next_hit();
                        hit.sortkey = rng.random();
                        hit.score = hit.sortkey as f32;
                    }
                    th
                },
                |mut th: TopHits| {
                    th.sort_by_sortkey();
                    black_box(&th);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_logsum,
    bench_hmm_sample,
    bench_profile_config,
    bench_null_one,
    bench_generic_viterbi,
    bench_generic_forward,
    bench_checkpointed_forward,
    bench_generic_msv,
    bench_gmx_operations,
    bench_occupancy,
    bench_tophits_sort,
);
criterion_main!(benches);
