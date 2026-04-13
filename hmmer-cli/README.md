# hmmer-cli

Command-line tools for the HMMER Rust port.

## Binaries

- **`hmmsearch`** — Search a profile HMM against a sequence database
- **`hmmbuild`** — Build a profile HMM from a multiple sequence alignment

## Usage

```bash
cargo run --bin hmmsearch -- model.hmm sequences.fa
cargo run --bin hmmsearch -- --tblout results.tbl model.hmm sequences.fa
```
