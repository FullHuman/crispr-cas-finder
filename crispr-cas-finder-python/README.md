# crispr-cas-finder (Python)

Python bindings for the CRISPR-Cas detection engine written in Rust.

## Installation

Build and install in-place (development):

```shell
cd crispr-cas-finder-python
maturin develop --release
```

Build a distributable wheel:

```shell
maturin build --release
```

## Quick start

```python
from crispr_cas_finder import find_crispr_arrays, find_crispr_arrays_in_file

# ── From a file ──────────────────────────────────────────────────────────────
arrays = find_crispr_arrays_in_file("genome.fasta")

# ── From a FASTA string ───────────────────────────────────────────────────────
with open("genome.fasta") as fh:
    arrays = find_crispr_arrays(fh.read())

# ── Inspect results ───────────────────────────────────────────────────────────
for array in arrays:
    print(
        f"{array.seq_id}  {array.start}–{array.end}  "
        f"({array.spacer_count} spacers, evidence level {array.evidence_level})"
    )
    for spacer in array.spacers:
        print(f"  spacer: {spacer.sequence}")
```

## API

### `find_crispr_arrays(fasta_content, *, ...)` → `list[CrisprArray]`

Scan a FASTA string for CRISPR arrays.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fasta_content` | `str` | — | Multi-sequence FASTA text |
| `min_repeat_length` | `int` | `23` | Minimum direct repeat length (bp) |
| `max_repeat_length` | `int` | `55` | Maximum direct repeat length (bp) |
| `min_spacer_length` | `int` | `25` | Minimum spacer length (bp) |
| `max_spacer_length` | `int` | `60` | Maximum spacer length (bp) |
| `no_mismatch` | `bool` | `False` | Require identical repeats |
| `min_spacer_count` | `int` | `1` | Minimum spacers per array |
| `min_evidence_level` | `int` | `1` | Minimum evidence level (1–4) |

All keyword arguments after `fasta_content` are keyword-only.

### `find_crispr_arrays_in_file(path, *, ...)` → `list[CrisprArray]`

Convenience wrapper that reads a FASTA file and calls `find_crispr_arrays`.
Raises `FileNotFoundError` if `path` does not exist.

### `CrisprArray`

| Attribute | Type | Description |
|---|---|---|
| `seq_id` | `str` | Source sequence ID |
| `start` | `int` | 1-based array start |
| `end` | `int` | 1-based array end (inclusive) |
| `consensus_repeat` | `str` | Consensus direct repeat sequence |
| `repeats` | `list[Repeat]` | All direct repeats |
| `spacers` | `list[Spacer]` | All spacers |
| `evidence_level` | `int` | 1–4 confidence score |
| `orientation` | `str` | `"+"`, `"-"`, or `"."` |
| `repeat_id` | `str` | CRISPRdb canonical repeat ID |
| `crispr_direction` | `str` | CRISPRDirection database direction |
| `spacer_count` *(property)* | `int` | `len(spacers)` |
| `repeat_count` *(property)* | `int` | `len(repeats)` |

`len(array)` returns the array length in base pairs.

### `Repeat` / `Spacer`

Both share the same three fields: `start` (int), `end` (int), `sequence` (str).
`len(repeat)` / `len(spacer)` returns the sequence length.
