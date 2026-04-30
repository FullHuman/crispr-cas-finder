use crispr_cas_finder_cli::run_from_args;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn ecoli_fasta() -> PathBuf {
    workspace_root()
        .join("crispr-cas-finder-cli")
        .join("data")
        .join("ecoli.fasta")
}

fn casfinder_data_root() -> PathBuf {
    workspace_root()
        .join("crispr-cas-finder-cli")
        .join("data")
        .join("CasFinder-2.0.3")
}

fn subtyping_models_dir() -> PathBuf {
    casfinder_data_root().join("DEF-SubTyping-2.0.3")
}

fn profiles_dir() -> PathBuf {
    casfinder_data_root().join("CASprofiles-2.0.3")
}

fn assert_benchmark_inputs() {
    let fasta = ecoli_fasta();
    let models = subtyping_models_dir();
    let profiles = profiles_dir();

    assert!(
        fasta.is_file(),
        "Benchmark FASTA not found at {}",
        fasta.display()
    );
    assert!(
        models.is_dir(),
        "CAS model directory not found at {}",
        models.display()
    );
    assert!(
        profiles.is_dir(),
        "CAS profile directory not found at {}",
        profiles.display()
    );
}

fn build_cli_args(outdir: &Path, include_cas: bool) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("crispr-cas-finder"),
        OsString::from("-i"),
        ecoli_fasta().into_os_string(),
        OsString::from("--outdir"),
        outdir.as_os_str().to_os_string(),
    ];

    if include_cas {
        args.extend([
            OsString::from("--cas"),
            OsString::from("--cas-models-dir"),
            subtyping_models_dir().into_os_string(),
            OsString::from("--cas-profiles-dir"),
            profiles_dir().into_os_string(),
        ]);
    }

    args
}

fn run_cli_benchmark(outdir: &Path, include_cas: bool) {
    let args = build_cli_args(outdir, include_cas);
    run_from_args(args).unwrap_or_else(|err| panic!("CLI benchmark run failed: {err}"));
}

fn bench_crispr_only(c: &mut Criterion) {
    assert_benchmark_inputs();

    let mut group = c.benchmark_group("crispr-cas-finder-cli/ecoli/crispr-only");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(20));
    group.bench_function("default", |b| {
        b.iter_batched(
            || TempDir::new().expect("failed to create temporary output directory"),
            |outdir| run_cli_benchmark(outdir.path(), false),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    assert_benchmark_inputs();

    let mut group = c.benchmark_group("crispr-cas-finder-cli/ecoli/with-cas-subtyping");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(5));
    // Each full pipeline sample is multi-second, so give Criterion enough
    // wall time to collect the configured sample size without warning.
    group.measurement_time(Duration::from_secs(90));
    group.bench_function("default", |b| {
        b.iter_batched(
            || TempDir::new().expect("failed to create temporary output directory"),
            |outdir| run_cli_benchmark(outdir.path(), true),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_crispr_only, bench_full_pipeline);
criterion_main!(benches);
