use crisprcas_cli::run_from_args;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Duration;
use tempfile::TempDir;

/// Rayon's global thread pool can only be initialised once per process.
/// All benchmarks in this file share the same 4-thread pool so that both the
/// warm-up and measurement phases use the same degree of parallelism.
static RAYON_INIT: Once = Once::new();
const BENCH_THREADS: usize = 4;

fn init_thread_pool() {
    RAYON_INIT.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(BENCH_THREADS)
            .build_global()
            .expect("failed to initialise rayon global thread pool");
    });
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn ecoli_fasta() -> PathBuf {
    workspace_root()
        .join("crisprcas-cli")
        .join("data")
        .join("ecoli.fasta")
}

fn casfinder_data_root() -> PathBuf {
    workspace_root()
        .join("crisprcas-cli")
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

// ---------------------------------------------------------------------------
// Rust (this implementation) helpers
// ---------------------------------------------------------------------------

fn build_cli_args(outdir: &Path, include_cas: bool) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("crisprcas"),
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
    init_thread_pool();
    assert_benchmark_inputs();

    let mut group = c.benchmark_group("ecoli/crispr-only");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(60));
    group.bench_function("rust", |b| {
        b.iter_batched(
            || TempDir::new().expect("failed to create temporary output directory"),
            |outdir| run_cli_benchmark(outdir.path(), false),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    init_thread_pool();
    assert_benchmark_inputs();

    let mut group = c.benchmark_group("ecoli/with-cas-subtyping");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(5));
    // Each full pipeline sample is multi-second, so give Criterion enough
    // wall time to collect the configured sample size without warning.
    group.measurement_time(Duration::from_secs(120));
    group.bench_function("rust", |b| {
        b.iter_batched(
            || TempDir::new().expect("failed to create temporary output directory"),
            |outdir| run_cli_benchmark(outdir.path(), true),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Perl CRISPRCasFinder (dcouvin/CRISPRCasFinder) helpers
//
// Set the environment variable CRISPRCASFINDER_PERL to the path of
// CRISPRCasFinder.pl before running the benchmarks, e.g.:
//
//   CRISPRCASFINDER_PERL=/path/to/CRISPRCasFinder.pl \
//       cargo bench --bench crisprcas_cli_ecoli
//
// The Perl benchmarks are automatically skipped when the variable is not set.
// ---------------------------------------------------------------------------

const PERL_PATH_ENV: &str = "CRISPRCASFINDER_PERL";

/// Returns the path to `CRISPRCasFinder.pl`, or `None` if the env var is not
/// set.  The benchmark functions call this to decide whether to skip.
fn perl_script_path() -> Option<PathBuf> {
    std::env::var_os(PERL_PATH_ENV).map(PathBuf::from)
}

/// Run the Perl CRISPRCasFinder tool for a single benchmark sample.
///
/// `script` — path to `CRISPRCasFinder.pl`
/// `outdir`  — per-sample temporary output directory
/// `include_cas` — whether to pass the `-cas` flag
fn run_perl_benchmark(script: &Path, outdir: &Path, include_cas: bool) {
    let mut cmd = std::process::Command::new("perl");
    cmd.arg(script)
        .arg("-in")
        .arg(ecoli_fasta())
        .arg("-out")
        .arg(outdir);

    if include_cas {
        cmd.arg("-cas");
    }

    let status = cmd
        .status()
        .unwrap_or_else(|err| panic!("failed to launch Perl CRISPRCasFinder: {err}"));

    assert!(
        status.success(),
        "Perl CRISPRCasFinder exited with non-zero status: {}",
        status
    );
}

fn bench_perl_crispr_only(c: &mut Criterion) {
    let Some(script) = perl_script_path() else {
        eprintln!(
            "Skipping Perl crispr-only benchmark: set {PERL_PATH_ENV} to the path of \
             CRISPRCasFinder.pl to enable it."
        );
        return;
    };

    let mut group = c.benchmark_group("ecoli/crispr-only");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(60));
    group.bench_function("perl", |b| {
        b.iter_batched(
            || TempDir::new().expect("failed to create temporary output directory"),
            |outdir| run_perl_benchmark(&script, outdir.path(), false),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_perl_full_pipeline(c: &mut Criterion) {
    let Some(script) = perl_script_path() else {
        eprintln!(
            "Skipping Perl full-pipeline benchmark: set {PERL_PATH_ENV} to the path of \
             CRISPRCasFinder.pl to enable it."
        );
        return;
    };

    let mut group = c.benchmark_group("ecoli/with-cas-subtyping");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(120));
    group.bench_function("perl", |b| {
        b.iter_batched(
            || TempDir::new().expect("failed to create temporary output directory"),
            |outdir| run_perl_benchmark(&script, outdir.path(), true),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_crispr_only,
    bench_full_pipeline,
    bench_perl_crispr_only,
    bench_perl_full_pipeline
);
criterion_main!(benches);
