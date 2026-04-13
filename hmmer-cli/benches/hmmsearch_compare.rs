// Comparative benchmark: Rust hmmsearch vs C hmmsearch
//
// Benchmarks both implementations across different:
// - HMM profiles (globins4, fn3, Pkinase)
// - Database sizes (100, 1000, 10000 sequences)
// - Thread counts (1, 2, 4)
//
// Prerequisites:
// - C HMMER installed at /opt/homebrew/bin/hmmsearch (or adjust C_HMMSEARCH below)
// - Rust hmmsearch built in release mode: cargo build --release -p hmmer-cli
// - Benchmark data present in bench_data/ directory
//
// To run:
//   cargo bench -p hmmer-cli --bench hmmsearch_compare
//
// To compare against a saved baseline:
//   cargo bench -p hmmer-cli --bench hmmsearch_compare -- --save-baseline <name>
//   # ... make changes ...
//   cargo bench -p hmmer-cli --bench hmmsearch_compare -- --baseline <name>

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// Path to the C HMMER hmmsearch binary (Homebrew default on macOS).
/// Override by setting the C_HMMSEARCH environment variable.
fn c_hmmsearch() -> PathBuf {
    std::env::var("C_HMMSEARCH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/opt/homebrew/bin/hmmsearch"))
}

/// Path to the Rust hmmsearch binary (release build).
/// Override by setting the RUST_HMMSEARCH environment variable.
fn rust_hmmsearch() -> PathBuf {
    std::env::var("RUST_HMMSEARCH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Walk up from the bench binary's location to find workspace root
            let manifest = env!("CARGO_MANIFEST_DIR");
            Path::new(manifest)
                .parent() // crates/
                .and_then(|p| p.parent()) // workspace root
                .map(|p| p.join("target/release/hmmsearch"))
                .unwrap_or_else(|| PathBuf::from("target/release/hmmsearch"))
        })
}

/// Workspace root (for bench_data/ paths).
fn workspace_root() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn run_hmmsearch(binary: &Path, hmm: &Path, db: &Path, cpu: usize) {
    let status = Command::new(binary)
        .arg("--cpu")
        .arg(cpu.to_string())
        .arg(hmm)
        .arg(db)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", binary.display(), e));
    assert!(
        status.success(),
        "{} exited with {}",
        binary.display(),
        status
    );
}

/// Main benchmark group: varies HMM, database size, and thread count.
fn bench_hmmsearch(c: &mut Criterion) {
    let c_bin = c_hmmsearch();
    let rust_bin = rust_hmmsearch();
    let root = workspace_root();

    // Verify binaries exist
    assert!(
        c_bin.exists(),
        "C hmmsearch not found at {}. Set C_HMMSEARCH env var.",
        c_bin.display()
    );
    assert!(
        rust_bin.exists(),
        "Rust hmmsearch not found at {}. Run `cargo build --release -p hmmer-cli` first, or set RUST_HMMSEARCH env var.",
        rust_bin.display()
    );

    let hmms = ["globins4.hmm", "fn3.hmm", "Pkinase.hmm"];
    let dbs = [
        ("100", "seqdb_100.fa"),
        ("1k", "seqdb_1000.fa"),
        ("10k", "seqdb_10000.fa"),
    ];
    let thread_counts: &[usize] = &[1, 2, 4];

    // One group per HMM profile.
    for hmm_name in &hmms {
        let hmm_path = root.join("bench_data").join(hmm_name);
        let profile_stem = hmm_name.strip_suffix(".hmm").unwrap_or(hmm_name);

        let mut group = c.benchmark_group(format!("hmmsearch/{}", profile_stem));
        // These are whole-process benchmarks, so increase measurement time
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(30));
        group.warm_up_time(Duration::from_secs(5));

        for &(db_label, db_file) in &dbs {
            let db_path = root.join("bench_data").join(db_file);

            for &cpus in thread_counts {
                let param = format!("{}_{}cpu", db_label, cpus);

                // C HMMER
                group.bench_with_input(
                    BenchmarkId::new("C", &param),
                    &(&c_bin, &hmm_path, &db_path, cpus),
                    |b, &(bin, hmm, db, cpu)| {
                        b.iter(|| run_hmmsearch(bin, hmm, db, cpu));
                    },
                );

                // Rust HMMER
                group.bench_with_input(
                    BenchmarkId::new("Rust", &param),
                    &(&rust_bin, &hmm_path, &db_path, cpus),
                    |b, &(bin, hmm, db, cpu)| {
                        b.iter(|| run_hmmsearch(bin, hmm, db, cpu));
                    },
                );
            }
        }

        group.finish();
    }
}

criterion_group!(benches, bench_hmmsearch);
criterion_main!(benches);
