# hmmer-py

Python bindings for HMMER, built with [PyO3](https://pyo3.rs/) and
[maturin](https://maturin.rs/).

## Installation

```bash
cd crates/hmmer-py
maturin develop
```

## Usage

```python
import hmmer

# Search an HMM against a sequence database
results = hmmer.hmmsearch("globins4.hmm", "sequences.fa")
print(f"Found {results.nhits} hits in {results.nseq} sequences")

for hit in results:
    print(f"  {hit.name}: score={hit.score:.1f}, E={hit.evalue:.2e}")

# Alphabet info
abc = hmmer.Alphabet.amino()
print(f"Alphabet size: {abc.k}")
```

## API

### `hmmer.hmmsearch(hmm_path, seq_path, e_threshold=10.0)`

Search an HMM file against a FASTA sequence database. Returns a
`SearchResults` object.

### `SearchResults`

- `.nhits` — Number of hits
- `.nseq` — Number of sequences searched
- `len(results)` — Same as `.nhits`
- `results[i]` — Access hit by index
- Iterable: `for hit in results: ...`

### `HitResult`

- `.name` — Target sequence name
- `.score` — Bit score
- `.evalue` — E-value
- `.ndom` — Number of domains
- `.is_reported` / `.is_included` — Threshold flags

### `Alphabet`

- `Alphabet.amino()` — Amino acid alphabet (K=20)
- `Alphabet.dna()` — DNA alphabet (K=4)
