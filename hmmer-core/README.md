# hmmer-core

Internal compatibility crate (`publish = false`) for workspace tests and
benchmarks. The implementation is bundled in
`crispr-cas-finder-core/src/hmmer_core/`; this crate re-exports the same types.
Publishing the core/CLI packages does not publish a separate HMMER crate.

Profile HMM algorithms implemented in pure Rust. Builds require the nightly
compiler pinned in the repository's `rust-toolchain.toml` for `std::simd`.

## Modules

- `alphabet`, `sequence`, `rng`, `background`: sequence and background models.
- `hmm`, `profile`, `modelconfig`, `trace`: model configuration and alignments.
- `dynamic_programming`: SIMD filters, Forward/Backward, null2 correction,
  posterior decoding, and optimal accuracy alignment.
- `pipeline`: search plans/workers, filter cascade, and domain definition.
- `stats`: score-to-P-value conversions.
- `results`: hits, alignment displays, and thresholding.

## SIMD

The kernels use four-lane portable `Simd<f32, 4>`. Rust lowers these operations
for the compilation target (including WASM SIMD); there is no runtime CPU
dispatch or separate AVX2 backend. Native builds use portable target defaults.
For local performance measurements, use `RUSTFLAGS="-C target-cpu=native" cargo bench`.

This crate is licensed under GPL-3.0-or-later.
