# hmmer-io

File I/O for the HMMER Rust port.

## Features

- **HMM file reader** — Reads HMMER3 format HMM files (`.hmm`)
- **Sequence reader** — Reads FASTA sequence databases, with digital encoding
- **H2 I/O** — Legacy HMMER2 format support (stub)

## Usage

```rust
use hmmer_io::HmmFile;

let mut hfp = HmmFile::open("model.hmm", None).unwrap();
let (abc, hmm) = hfp.read().unwrap();
println!("Model: {} (M={})", hmm.name, hmm.m);
```
