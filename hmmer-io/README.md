# hmmer-io

Internal compatibility crate (`publish = false`) for workspace tests and
benchmarks. The implementation is bundled in
`crispr-cas-finder-core/src/hmmer_io/`; this crate re-exports the same types.
Publishing the core/CLI packages does not publish a separate HMMER crate.

File I/O for the HMMER Rust port.

## Features

- **HMM file reader** — Reads HMMER3/a-f ASCII models, including the bundled HMMER3/b profiles
- **HMM file writer** — Writes complete HMMER3/f ASCII models
- **Sequence reader** — Reads FASTA sequence databases, with digital encoding

## Usage

```rust
use hmmer_io::HmmFile;

let mut hfp = HmmFile::open("model.hmm", None).unwrap();
let (abc, hmm) = hfp.read().unwrap();
println!("Model: {} (M={})", hmm.name, hmm.num_nodes);
```
