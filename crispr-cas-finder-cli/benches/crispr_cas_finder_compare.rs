/// Comparison benchmark: original CRISPRCasFinder (Perl) vs our Rust implementation.
///
/// The original tool is cloned into `vendor/CRISPRCasFinder/` (gitignored).
/// If the vendor directory is absent the benchmark is skipped gracefully so
/// that CI passes on machines where only the Rust implementation is available.
use crispr_cas_finder_cli::run_from_args;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use tempfile::TempDir;

// ── path helpers ─────────────────────────────────────────────────────────────

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

fn vendor_dir() -> PathBuf {
    workspace_root().join("vendor").join("CRISPRCasFinder")
}

fn original_script() -> PathBuf {
    vendor_dir().join("CRISPRCasFinder.pl")
}

fn original_so() -> PathBuf {
    vendor_dir().join("sel392v2.so")
}

// ── availability checks ───────────────────────────────────────────────────────

fn original_available() -> bool {
    let script = original_script();
    let so = original_so();
    let fasta = ecoli_fasta();
    if !script.is_file() {
        eprintln!(
            "[compare bench] skipping original: script not found at {}",
            script.display()
        );
        return false;
    }
    if !so.is_file() {
        eprintln!(
            "[compare bench] skipping original: sel392v2.so not found at {}",
            so.display()
        );
        return false;
    }
    if !fasta.is_file() {
        eprintln!(
            "[compare bench] skipping original: FASTA not found at {}",
            fasta.display()
        );
        return false;
    }
    // Quick check: confirm that `conda run -n ccf_orig perl -MBio::SeqIO -e 1` succeeds
    let bioperl_check = Command::new("conda")
        .args(["run", "-n", "ccf_orig", "perl", "-MBio::SeqIO", "-e", "1"])
        .output();
    match bioperl_check {
        Ok(out) if out.status.success() => true,
        _ => {
            eprintln!(
                "[compare bench] skipping original: ccf_orig conda env with BioPerl not found. \
                 Set up via: conda create -n ccf_orig -c bioconda -c conda-forge perl-bioperl perl=5.32"
            );
            false
        }
    }
}

// ── original CRISPRCasFinder runner ──────────────────────────────────────────

/// Run original CRISPRCasFinder.pl with `workdir` as CWD and `outdir` as the output.
///
/// The Perl script writes `NC_000913.fna` into CWD, then after processing does
/// `chdir ".."` twice (from inside `outdir/NC_000913`). Both steps land back in
/// `workdir` only when `outdir` is a **direct child** of `workdir`.
/// The caller must guarantee this relationship.
fn run_original_crispr_only(workdir: &Path, outdir: &Path) {
    let status = Command::new("conda")
        .args(["run", "-n", "ccf_orig", "perl"])
        .arg(original_script())
        .arg("-in")
        .arg(ecoli_fasta())
        .arg("-out")
        .arg(outdir)
        .arg("-so")
        .arg(original_so())
        .arg("-q")
        .current_dir(workdir)
        .status()
        .expect("failed to spawn perl CRISPRCasFinder.pl via conda");

    assert!(
        status.success(),
        "original CRISPRCasFinder.pl exited with: {}",
        status
    );
}

// ── Rust CLI runner ───────────────────────────────────────────────────────────

fn run_rust_crispr_only(outdir: &Path) {
    let args: Vec<OsString> = vec![
        OsString::from("crispr-cas-finder"),
        OsString::from("-i"),
        ecoli_fasta().into_os_string(),
        OsString::from("--outdir"),
        outdir.as_os_str().to_os_string(),
    ];
    run_from_args(args).expect("Rust CLI benchmark run failed");
}

// ── benchmarks ───────────────────────────────────────────────────────────────

fn bench_crispr_only_comparison(c: &mut Criterion) {
    let fasta = ecoli_fasta();
    assert!(fasta.is_file(), "FASTA not found at {}", fasta.display());

    let mut group = c.benchmark_group("crispr-only/ecoli");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(30));

    // Rust implementation (always present)
    group.bench_function(BenchmarkId::new("rust", "default"), |b| {
        b.iter_batched(
            || TempDir::new().expect("tmp dir"),
            |outdir| run_rust_crispr_only(outdir.path()),
            BatchSize::SmallInput,
        );
    });

    // Original Perl implementation (only if vendor dir is present)
    if original_available() {
        group.bench_function(BenchmarkId::new("original-perl", "default"), |b| {
            b.iter_batched(
                // The Perl script:
                //   1. Refuses to run if the output dir already exists (so we
                //      pass a non-existent direct child of workdir as output).
                //   2. Writes NC_000913.fna in CWD and does `chdir ".."` twice
                //      from inside `outdir/NC_000913` to get back to workdir.
                //   3. On macOS, std TempDir lives under /var/folders/… — a
                //      very deep path.  After those two chdir ".." calls the
                //      script may land several levels above workdir and end up
                //      in /var (unwritable).  We therefore create the workdir
                //      directly under /tmp so the path depth is small and
                //      predictable.
                || {
                    let workdir = tempfile::Builder::new()
                        .prefix("ccf_bench_")
                        .tempdir_in("/tmp")
                        .expect("tmp dir under /tmp");
                    let out = workdir.path().join("ccf_out");
                    (workdir, out)
                },
                |(workdir, out)| run_original_crispr_only(workdir.path(), &out),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_crispr_only_comparison);
criterion_main!(benches);
