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
`--cas-models-dir` and `--cas-profiles-dir`. Only genetic code 11 is supported.

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

# Detect CRISPR arrays and run Cas protein subtyping
crispr-cas-finder -i genome.fasta --cas

# Specify output directory
crispr-cas-finder -i genome.fasta --outdir results/

# Metagenomic mode
crispr-cas-finder -i metagenome.fasta --metagenome --cas
```

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
```

### Rust Library (crispr-cas-finder-core)

```rust
use crispr_cas_finder_core::{DetectionParams, detect_crisprs_in_fasta_str};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fasta = std::fs::read_to_string("genome.fasta")?;
    let params = DetectionParams::default();

    let arrays = detect_crisprs_in_fasta_str(&fasta, &params)?;
    println!("Found {} CRISPR arrays", arrays.len());

    for array in &arrays {
        println!(
            "  {} ({}..{}) — {} spacers, evidence level {}",
            array.seq_id, array.start, array.end,
            array.spacers.len(), array.evidence_level
        );
    }

    Ok(())
}
```

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
