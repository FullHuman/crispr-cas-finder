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

The floating-point kernels use four-lane portable `Simd<f32, 4>`. Rust lowers
these operations for the compilation target (including WASM SIMD).

The integer MSV and Viterbi bounds (`simd/int_msv.rs`, `simd/i16_viterbi.rs`),
which decide the filters for nearly every sequence, are compiled once per
instruction set and chosen at run time (`int_msv::Isa`): 128-bit vectors on
the portable baseline and with SSE4.1, 256-bit with AVX2 and 512-bit with
AVX-512BW on x86-64 (256-bit for models of fewer than about 190 nodes, whose
short rows run faster that way). The bounds do not depend on the vector
width, so every instruction set gives the same results; `cargo bench -p
hmmer-core -- msv_lower_bound_i8` compares them. For local measurements of
the other kernels, use `RUSTFLAGS="-C target-cpu=native" cargo bench`.

This crate is licensed under GPL-3.0-or-later.
