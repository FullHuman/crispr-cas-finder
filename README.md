# crispr-cas-finder

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

An experimental, parallel Rust implementation for exploring CRISPR arrays and candidate Cas systems in prokaryotic genomes.

## What is crispr-cas-finder?

crispr-cas-finder is an independent implementation inspired by [CRISPRCasFinder](https://github.com/dcouvin/CRISPRCasFinder). This first public release is intended for experimental use and feedback. Detection behavior and APIs may change. See [limitations and validation](LIMITATIONS.md) for implemented behavior, initial comparisons, and known differences. Broad accuracy and performance comparisons are ongoing; [Benchmarks](#benchmarks) below compares run time and results with CRISPRCasFinder on 72 genomes.

**[Try the browser application](https://crisprcasfinder.full-human.com).** Your genome sequence is processed locally and is not uploaded.

## Components

crispr-cas-finder is available in multiple forms:

- **`crispr-cas-finder-cli`**: Command-line interface for CRISPR-Cas detection
- **`crispr-cas-finder-core`**: Rust library for integrating into your own projects
- **`crispr-cas-finder-python`**: Python bindings (via PyO3)
- **`crispr-cas-finder-wasm`**: WebAssembly module and browser application

HMMER code is bundled inside `crispr-cas-finder-core/src/hmmer_core/` and
`src/hmmer_io/`. The workspace crates `hmmer-core` and `hmmer-io` are unpublished
compatibility wrappers for internal tests and benchmarks (`publish = false`).
The crates.io workflow publishes only the core and CLI packages. CI verifies
both package archives in a fresh Cargo staging registry before release
(`python3 scripts/package_public_crates.py`).

## 📦 Installation

### Using Cargo

```bash
rustup toolchain install nightly-2026-07-29 --profile minimal
cargo +nightly-2026-07-29 install --locked crispr-cas-finder-cli
```

The CLI embeds its CasFinder 2.0.3 models, including for `cargo install` and
downloaded executables. No separate data download or source checkout is needed.
For Cas analysis, the embedded files are extracted to a private temporary directory
and removed when the run finishes. Custom data can be selected with
`--cas-models-dir` and `--cas-profiles-dir`. Other genetic codes are selected
with `--genetic-code` (for example 4 for Mycoplasma).

### From Source

Install Rust with rustup. The repository pins a nightly compiler because HMM
scoring uses `std::simd`; a stable compiler alone cannot build these crates.

```bash
git clone https://github.com/FullHuman/crispr-cas-finder.git
cd crispr-cas-finder
cargo install --path crispr-cas-finder-cli
```

### Python Bindings

```bash
pip install crispr-cas-finder
```

Prebuilt Python wheels do not require Rust. If pip builds from source, install
`nightly-2026-07-29` and set `RUSTUP_TOOLCHAIN=nightly-2026-07-29`. See the
[Python instructions](crispr-cas-finder-python/README.md).

### Rust Library

The core currently requires nightly Rust for `portable_simd`. Add a
`rust-toolchain.toml` containing the following to your consuming project:

```toml
[toolchain]
channel = "nightly-2026-07-29"
profile = "minimal"
```

Then add to your `Cargo.toml`:

```toml
[dependencies]
crispr-cas-finder-core = "0.1.0"
```

## 🏃 Quick Start

### Command Line (crispr-cas-finder-cli)

```bash
# Detect CRISPR arrays
crispr-cas-finder -i genome.fasta

# Detect CRISPR arrays and Cas systems
crispr-cas-finder -i genome.fasta --cas --outdir results/

# Use the CasFinder 3 models (download them first, see below)
crispr-cas-finder -i genome.fasta --cas --cas-models-dir models/CasFinder

# Use an existing annotation instead of predicting genes
crispr-cas-finder -i genome.fasta --cas --gff genes.gff --faa proteins.faa

# Group arrays and Cas systems lying within 20 kb of each other
crispr-cas-finder -i genome.fasta --cas --locus-distance 20000

# Metagenomic mode for every sequence
crispr-cas-finder -i metagenome.fasta --metagenome --cas

# Decide the HMM filters with exact scores (slower; see LIMITATIONS.md)
crispr-cas-finder -i genome.fasta --cas --exact-filters
```

Run `crispr-cas-finder --help` for every option. Most CRISPRCasFinder option
names are accepted as aliases (for example `--minDR`, `--rcfowce`, `--cSA`,
`--Mrpts`, `--dbc`).

#### Cas models

The CLI embeds CasFinder 2.0.3 and uses it by default. Current CRISPRCasFinder
releases use CasFinder 3.1 with MacSyFinder 2; those models are licensed
CC BY-NC-SA 4.0 and are therefore not distributed with this GPL project. To use
them, download the package and point `--cas-models-dir` at it:

```bash
pip install macsyfinder
msf_data install --target models CasFinder==3.1.0
crispr-cas-finder -i genome.fasta --cas --cas-models-dir models/CasFinder
```

MacSyFinder is needed only for the download. Definitions in the MacSyFinder 2
format are evaluated with MacSyFinder 2 rules implemented in Rust.

#### Output files

Without `--outdir`, the files go to `Result_<name>` in the current directory,
`<name>` being the input file's name without its extension, or to
`Result_<name>_2` and so on when that directory exists.

| File | Content |
| --- | --- |
| `report.json` | Everything below in one document: parameters, per-sequence summaries, arrays, Cas systems, arrays near a Cas system, loci, totals |
| `sequences.tsv` | One row per input sequence |
| `crisprs.tsv` | One row per array: coordinates, evidence level, consensus repeat, conservation scores, spacer statistics |
| `features.gff` | GFF3 with arrays, repeats, spacers, flanking regions and Cas genes |
| `arrays.fna`, `repeats.fna`, `spacers.fna` | Array sequences, consensus repeats and spacers |
| `cas_systems.tsv`, `cas_proteins.faa` | With `--cas`: one row per Cas gene, and the Cas proteins |
| `associations.tsv` | With `--cas`: arrays within `--vicinity` bp (default 600) of a Cas system |
| `loci.tsv` | With `--locus-distance`: arrays and Cas systems chained by proximity |
| `annotation/` | With `--cas`: `proteins.faa` with all predicted proteins, `genes.gff` with their coordinates, and `cas_hits.tsv` with every HMM hit that passed the thresholds |

### Python

```python
import crispr_cas_finder

# Find CRISPR repeats in a FASTA string
with open("genome.fasta") as f:
    fasta_content = f.read()

arrays = crispr_cas_finder.find_crispr_arrays(fasta_content)

# Customize detection parameters
arrays = crispr_cas_finder.find_crispr_arrays(
    fasta_content,
    min_repeat_length=23,
    max_repeat_length=55,
    min_spacer_length=25,
    max_spacer_length=60,
    min_evidence_level=2,
)

# Full report as a dictionary (same structure as report.json), with Cas systems
report = crispr_cas_finder.analyze(
    fasta_content,
    cas=True,
    cas_models_dir="CasFinder-2.0.3/DEF-SubTyping-2.0.3",
    cas_profiles_dir="CasFinder-2.0.3/CASprofiles-2.0.3",
)
print(report["summary"])
```

### Rust Library (crispr-cas-finder-core)

```rust
use crispr_cas_finder_core::{AnalysisOptions, analyze, parse_fasta};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sequences = parse_fasta(std::fs::File::open("genome.fasta")?)?;
    let analysis = analyze(&sequences, &AnalysisOptions::default())?;

    for array in &analysis.report.crisprs {
        println!(
            "{} ({}..{}): {} spacers, evidence level {}",
            array.id, array.start, array.end,
            array.spacers.len(), array.evidence_level
        );
    }

    Ok(())
}
```

Set `AnalysisOptions::cas` to a `CasFinderConfig` with model and profile
directories to detect Cas systems as well. `detect_crisprs` remains available
for array detection on in-memory sequences.

## Browser application

Build and run the threaded WebAssembly app using the instructions in
[crispr-cas-finder-wasm/README.md](crispr-cas-finder-wasm/README.md). Both CRISPR
and Cas detection run locally in the browser.

## Benchmarks

Both tools were run on the same machine, a MacBook Pro with an Apple M1 Max
(8 performance and 2 efficiency cores, 32 GB), on 40 genomes with Cas
detection and 72 without: the 21 reference genomes of bacteria and archaea
listed in [benchmarks/panel.tsv](benchmarks/panel.tsv), E. coli K-12, the
Acetobacter plasmid of the original's install test, and enterobacterial
assemblies of 1 to 376 contigs. The original is CRISPRCasFinder 4.3.2 run
natively on Apple silicon with Vmatch 2.3.1, Prodigal 2.6.3, MacSyFinder 2.1.6
and HMMER 3.4; both tools read the same CasFinder 3.1.0 models. "Original" is
the original with its defaults, one MacSyFinder worker; "original, all cores"
adds `-cpuM 10`. The speed-up compares the two tools at their defaults, the
original on one worker and this project on every core. Each figure is one run,
or the median of three for E. coli K-12 and three other inputs. The scripts,
every row and the caveats are in [benchmarks/README.md](benchmarks/README.md);
the Rust rows are of the build at `45c346f` (2026-10-30).

**Arrays and Cas systems** (`--cas`), a selection of the 40 genomes:

| genome | note | Mb | original | original, all cores | Rust, 1 thread | Rust, all cores | speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mycoplasmoides genitalium G37 | no CRISPR-Cas, genetic code 4 | 0.6 | 9.64 s | 3.72 s | 1.45 s | 0.27 s | 35x |
| Campylobacter jejuni NCTC 11168 | type II-C | 1.6 | 14.9 s | 6.02 s | 3.03 s | 0.46 s | 32x |
| Methanocaldococcus jannaschii DSM 2661 | archaeon, about twenty arrays | 1.7 | 37.0 s | 18.8 s | 4.12 s | 0.60 s | 62x |
| Saccharolobus solfataricus P2 | archaeon, types I-A and III-B | 3.0 | 27.5 s | 16.4 s | 8.08 s | 1.58 s | 17x |
| Haloferax volcanii DS2 | archaeon, type I-B, high GC | 3.5 | 33.0 s | 16.9 s | 6.85 s | 1.57 s | 21x |
| Bacillus subtilis 168 | no CRISPR-Cas | 4.2 | 30.5 s | 15.3 s | 7.17 s | 1.19 s | 26x |
| Clostridioides difficile 630 | type I-B, many arrays | 4.3 | 43.9 s | 29.5 s | 7.42 s | 1.10 s | 40x |
| Mycobacterium tuberculosis H37Rv | type III-A, high GC | 4.4 | 39.3 s | 26.2 s | 7.15 s | 1.39 s | 28x |
| Escherichia coli K-12 MG1655 | type I-E | 4.6 | 23.8 s | 12.3 s | 7.45 s | 1.29 s | 18x |
| Pseudomonas aeruginosa PA14 | type I-F | 6.5 | 32.3 s | 18.9 s | 9.88 s | 1.74 s | 19x |
| Streptomyces coelicolor A3(2) | 8.7 Mb linear chromosome | 8.7 | 53.7 s | 37.3 s | 12.6 s | 2.30 s | 23x |
| Phytobacter ursingii ATCC 27989 | draft, 19 contigs | 5.8 | 110.9 s | 35.9 s | 9.44 s | 1.68 s | 66x |
| Phytobacter diazotrophicus Pd1 | draft, 79 contigs | 5.6 | 7.6 min | 2.2 min | 10.4 s | 1.36 s | 335x |
| isolate 13020RM | enterobacterial draft, 234 contigs | 5.7 | 13.5 min | 3.9 min | 12.2 s | 1.64 s | 496x |

Over all 40 genomes, geometric means of the original's wall time over this
project's:

| | geometric mean | range |
| --- | --- | --- |
| both at their defaults | 31x | 15x to 496x |
| this project held to one thread | 5.3x | 2.8x to 67x |
| both using every core | 15x | 8x to 142x |

The gain grows with the number of contigs because the original runs the
whole Cas search once per sequence. With Cas detection the largest process
of this project peaks at about 1.4 times the original's memory; for arrays
only, at a third to a sixth of it. E. coli K-12 with Cas detection against
the number of threads:

| threads | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| wall time | 7.45 s | 3.88 s | 2.23 s | 1.35 s |

**Arrays only**, 72 genomes: the geometric mean is 82x at the defaults (26x to
278x) and 39x with this project held to one thread. The 51 inputs of the local
set take 6.1 minutes one after another with the original and 4.6 s here; the
21 of the panel take 2.3 minutes and 2.1 s.

**Agreement with the original.** Of the original's 448 arrays on these 72
inputs, 439 are reported here with the same start and the same number of
spacers: all of evidence levels 2, 3 and 4 and 263 of the 272 of level 1. The
nine arrays of the original not reported and the three reported only here are
loci of three spacers or more on the other side of the 60% spacer similarity
limit. Of the original's 48 Cas systems, 47 are reported with the same
subtype; 13 more are reported here, 9 of them found by MacSyFinder inside the
original and then lost by its Perl script, and 4 on sequences under 100 kb,
where the original reports none. [LIMITATIONS.md](LIMITATIONS.md) goes through
every difference.

Things to keep in mind:

- Agreement with the original is not accuracy; no independent truth set was
  used.
- The original ran with MacSyFinder 2.1.6 and HMMER 3.4 rather than the 2.1.2
  and 3.3.2 it documents, and with Vmatch compiled from source for ARM.
- Most figures are single runs on one machine. On Apple silicon the HMM
  kernels run their portable code path; the SSE4.1, AVX2 and AVX-512 paths
  used on x86-64 are not measured here.

## Contributing

We welcome contributions! Please open an issue or submit a pull request.

## References

If you use this software, please cite the original CRISPRCasFinder publications:

- Couvin D, Bernheim A, Toffano-Nioche C, Touchon M, Michalik J, Néron B, Rocha EPC, Vergnaud G, Gautheret D, Pourcel C. **CRISPRCasFinder, an update of CRISRFinder, includes a portable version, enhanced performance and integrates search for Cas proteins.** *Nucleic Acids Res.* 2018 Jul 2;46(W1):W246–W251. DOI: [10.1093/nar/gky425](https://doi.org/10.1093/nar/gky425) PMID:29790974

- Grissa I, Vergnaud G, Pourcel C. **CRISPRFinder: a web tool to identify clustered regularly interspaced short palindromic repeats.** *Nucleic Acids Res.* 2007 Jul;35(Web Server issue):W52–7. DOI: [10.1093/nar/gkm360](https://doi.org/10.1093/nar/gkm360) PMID:17537822

- Abby SS, Néron B, Ménager H, Touchon M, Rocha EP. **MacSyFinder: a program to mine genomes for molecular systems with an application to CRISPR-Cas systems.** *PLoS One.* 2014 Oct 17;9(10):e110726. DOI: [10.1371/journal.pone.0110726](https://doi.org/10.1371/journal.pone.0110726) PMID:25330359

- Néron B, Denise R, Coluzzi C, Touchon M, Rocha EPC, Abby SS. **MacSyFinder v2: Improved modelling and search engine to identify molecular systems in genomes.** *bioRxiv.* DOI: [10.1101/2022.09.02.506364](https://doi.org/10.1101/2022.09.02.506364)

Further information: [https://crisprcas.i2bc.paris-saclay.fr](https://crisprcas.i2bc.paris-saclay.fr)

## 📄 License

This project is licensed under GPL-3.0-or-later; see [LICENSE](LICENSE).
Bundled code and data retain their upstream notices in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This is an independent project;
the upstream authors have not endorsed it.
