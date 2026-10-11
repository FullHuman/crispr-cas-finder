# crispr-cas-finder

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

An experimental, parallel Rust implementation for exploring CRISPR arrays and candidate Cas systems in prokaryotic genomes.

## What is crispr-cas-finder?

crispr-cas-finder is an independent implementation inspired by [CRISPRCasFinder](https://github.com/dcouvin/CRISPRCasFinder). This first public release is intended for experimental use and feedback. Detection behavior and APIs may change. See [limitations and validation](LIMITATIONS.md) for implemented behavior, initial comparisons, and known differences. Broad accuracy and performance comparisons are ongoing.

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
| `annotation/` | With `--cas`: all predicted proteins, their coordinates, and every HMM hit that passed the thresholds |

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
