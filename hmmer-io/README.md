# hmmer-io

Internal compatibility crate (`publish = false`) for workspace tests and
benchmarks. The implementation is bundled in
`crispr-cas-finder-core/src/hmmer_io/`; this crate re-exports the same types.
Publishing the core/CLI packages does not publish a separate HMMER crate.

Reads HMMER3/a-f ASCII models, including the bundled HMMER3/b profiles
(`HmmFile`, `read_hmm`).

## Usage

```rust
use hmmer_io::HmmFile;

let mut hfp = HmmFile::open("model.hmm", None).unwrap();
let (abc, hmm) = hfp.read().unwrap();
println!("Model: {} (M={})", hmm.name, hmm.num_nodes);
```
