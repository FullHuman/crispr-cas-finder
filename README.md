# crispr-cas-finder

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A fast, parallel Rust implementation of CRISPRCasFinder, a tool for identifying CRISPR arrays and Cas proteins in genomic sequences.

## What is crispr-cas-finder?

crispr-cas-finder is a high-performance reimplementation of [CRISPRCasFinder](https://github.com/dcouvin/CRISPRCasFinder), the widely-used tool for detecting CRISPR-Cas systems in prokaryotic genomes. Written in Rust, it delivers the same accurate detection algorithm with improved performance and modern language features.

## Components

crispr-cas-finder is available in multiple forms:

- **`crispr-cas-finder-cli`**: Command-line interface for CRISPR-Cas detection
- **`crispr-cas-finder-core`**: Rust library for integrating into your own projects
- **`crispr-cas-finder-python`**: Python bindings (via PyO3)
- **`crispr-cas-finder-wasm`**: WebAssembly module for browser/Node.js usage

## 📦 Installation

### Using Cargo

```bash
cargo install crispr-cas-finder-cli
```

### From Source

```bash
git clone https://github.com/FullHuman/crispr-cas-finder.git
cd crispr-cas-finder
cargo install --path crispr-cas-finder-cli
```

### Python Bindings

```bash
pip install crispr-cas-finder
```

### Rust Library

Add to your `Cargo.toml`:

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

results_json = crispr_cas_finder.find_repeats(fasta_content)

# Customize detection parameters
results_json = crispr_cas_finder.find_repeats(
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

This project is licensed under the GPL-3.0 License — see the [LICENSE](LICENSE) file for details.
