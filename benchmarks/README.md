# Comparison with the original CRISPRCasFinder

Scripts that run this project and the original Perl pipeline on the same
genomes, on the same machine, and compare run time, memory and results.

| File | Purpose |
| --- | --- |
| `setup_upstream_macos_arm64.sh` | Builds what the original needs to run natively on Apple silicon |
| `bench.py` | Runs both tools, records wall time, CPU time and peak memory, keeps their outputs |
| `compare.py` | Pairs the arrays and Cas systems of the two outputs |
| `summarize.py` | Writes `summary.md` from the CSV files |
| `browser_timing.cjs` | Times the WebAssembly build in a headless browser |
| `vmatch_oracle.py` | Checks the repeats this project finds against those Vmatch reports |

## What is compared

- **Original**: CRISPRCasFinder 4.3.2 (`dcouvin/CRISPRCasFinder` at `9a7f07e`),
  run as `perl CRISPRCasFinder.pl -in genome.fasta -cas -so sel392v2.so -rpts
  Repeat_List.csv -drpt repeatDirection.tsv -out res`. It calls Vmatch 2.3.1,
  Prodigal 2.6.3, MacSyFinder 2.1.6 with HMMER 3.4, MUSCLE 3.8.1551 and
  EMBOSS 6.6.0 under Perl 5.32.1 with BioPerl.
- **Rust**: `crispr-cas-finder -i genome.fasta --cas --cas-models-dir CasFinder
  --outdir res -q`, release build of this repository.
- Both read the same CasFinder 3.1.0 models. The Rust CLI embeds CasFinder
  2.0.3 by default; that is not what is measured here.

Configurations in `bench.py`:

| Name | Meaning |
| --- | --- |
| `upstream` | Original with its defaults: one MacSyFinder worker |
| `upstream-cpuM` | Original with `-cpuM <cores>` |
| `upstream-arrays` | Original without `-cas` |
| `rust-all` | Rust with its defaults: every core |
| `rust-tN` | Rust with `--workers N` and `RAYON_NUM_THREADS=N` |
| `rust-arrays`, `rust-arrays-tN` | Rust without `--cas` |

## How it is measured

Each run is one process tree started by `bench.py` and reaped with `wait4`.
Wall time is the elapsed time of that tree; CPU time is user plus system time
of every process in it; memory is the largest resident set size reached by any
single process. Runs are sequential (`--jobs 1`) and written to `timings.csv`.
`--jobs N` exists to collect outputs faster; those rows go to
`runs_parallel.csv` and are not used as timings.

`compare.py` pairs two arrays, or two Cas systems, when they are on the same
sequence and overlap by at least half of the shorter one. A spacer counts as
reported by both when the two intervals overlap by 80% or more, because the
tools can place a repeat boundary a base or two apart.

## Running the original on Apple silicon

The conda environment documented upstream does not resolve for `osx-arm64` or
`linux-aarch64`: Vmatch has no ARM package and the pinned MacSyFinder requires
HMMER 3.3.2 or older, which predates HMMER's ARM support. The file
`sel392v2.so` shipped upstream is an x86_64 binary.

`setup_upstream_macos_arm64.sh` therefore compiles Vmatch 2.3.1 and `sel392.c`
from the `genometools/vstree` sources and wraps MacSyFinder so that it finds the
models. Three things differ from an upstream installation and are worth
stating next to any number taken from here:

- MacSyFinder 2.1.6 and HMMER 3.4 instead of 2.1.2 and 3.3.2.
- MUSCLE 3.8 behind a wrapper that translates the MUSCLE 5 options upstream
  passes.
- Vmatch built with `-O3` by Apple clang from source instead of the vendor
  binary.

## Results of 2026-10-13

The runs used a MacBook Pro (M1 Max, 8 performance and 2 efficiency cores,
32 GB, macOS 27.0.1) and `nightly-2026-07-29`. Their outputs are kept under
`results/`, which is not tracked; the tables are `results/summary.md` (local
set) and `results/panel/summary.md` (panel).

- `results/outputs/`: both tools on the local set of 51 inputs: E. coli K-12
  MG1655, the Acetobacter plasmid of the upstream install test, and 49
  assemblies of Phytobacter and related enterobacteria with 1 to 376 contigs
  (3,564 sequences in all). Collected several runs at a time; `arrays.csv`,
  `cas_systems.csv` and `genomes.csv` come from `compare.py`.
- `results/timing/`: sequential runs on the local set. Arrays and Cas systems
  on 19 inputs, three repetitions for four of them and one for the rest;
  arrays only on all 51; thread counts 1 to 10 on E. coli; the browser build
  on E. coli.
- `results/panel/`: sequential runs of both tools on the 21 reference genomes
  of `panel.tsv` (`fetch_panel.py` downloads them from NCBI): bacteria of
  seven phyla and five archaea, twelve Cas subtypes, four genomes without a
  CRISPR-Cas system.
- `results/before-fix/` and `results/at-b7fb24d/`: the Rust results at
  `26d60c7`, before the array detection fixes that this comparison led to, and
  at `b7fb24d`, before the repeat search became a port of Vmatch's. The
  measurements of the original are the same throughout.

Things to state next to any number taken from here:

- Agreement with CRISPRCasFinder is not accuracy; no independent truth set was
  used.
- The local set is one family of bacteria with type I-E, I-F and unassigned
  type I systems only.
- The original was run with MacSyFinder 2.1.6 and HMMER 3.4, not the 2.1.2 and
  3.3.2 it documents. The Cas systems it loses (see `compare.py`) were checked
  against MacSyFinder's own output for two genomes with these versions only.
- Most timings are single runs on one machine.

## Checking the repeat search against Vmatch

`vmatch_oracle.py genomes/*.fasta` builds a Vmatch index of every sequence,
runs Vmatch with the arguments CRISPRCasFinder passes and compares its repeats
with those of `cargo run --release -p crispr-cas-finder-core --example
repeat_matches`. It prints, for each input, how many repeats both report and
how many only one of them does.

## Reproducing

```sh
cargo build --locked --release -p crispr-cas-finder-cli
VSTREE_SRC=/path/to/vstree benchmarks/setup_upstream_macos_arm64.sh
python3 benchmarks/bench.py genomes/*.fasta \
    --configs upstream,upstream-cpuM,rust-t1,rust-all \
    --reps 3 --out benchmarks/results/timing --work /tmp/ccf-bench
python3 benchmarks/compare.py benchmarks/results/timing
python3 benchmarks/summarize.py benchmarks/results/timing
```

`summarize.py --outputs DIR` reads the comparison from another results
directory, as done here.
