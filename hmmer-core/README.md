# hmmer-core

Core HMMER algorithms implemented in pure Rust.

## Modules

- **`dynamic_programming`** — MSV filter, Viterbi, Forward/Backward, posterior
  decoding, optimal accuracy alignment, stochastic traceback
- **`pipeline`** — Full search pipeline with filter cascade and domain definition
- **`stats`** — E-value calibration, model statistics, entropy weighting
- **`results`** — Hit/domain data structures, alignment display, top hits list
- **`align`** — Trace-based sequence alignment
- **`easel`** — Alphabet, sequence, random number generator, and utility types
  (port of Easel library)

## SIMD Backends

The MSV filter supports multiple SIMD backends with runtime dispatch:
- SSE2 (x86_64)
- AVX2 (x86_64)
- NEON (aarch64)
- Scalar fallback (all platforms)
