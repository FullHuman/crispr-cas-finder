# Experimental scope and validation

This is an independent experimental Rust implementation inspired by
CRISPRCasFinder. APIs and detection behavior may change. Initial tests establish
specific regression cases and consistency across runtimes; broad sensitivity,
specificity, and performance comparisons remain ongoing.

## Supported behavior and known differences

- Cas detection supports **genetic code 11 only**. Other codes return an error
  in the CLI, native Cas API, and WASM API. Do not use this release to analyze
  Cas proteins in organisms requiring another code.
- Evidence levels are experimental: arrays with at most three spacers receive
  level 1 and longer arrays receive level 4 after filtering. The original
  CRISPRCasFinder level-2/3 conservation classifications are not implemented.
  Selecting minimum evidence 2 or 3 therefore has the same effect as selecting 4.
- Gene prediction uses Orphos 0.3.0. Cas detection uses the bundled CasFinder
  2.0.3 definitions/profiles and an internal Rust HMMER implementation.
- Cas hits are clustered per replicon and evaluated per locus, with linear or
  circular topology. The evaluator does not implement upstream `loner`,
  `multi_system`, `multi_model`, `multi_loci`, or gene-specific
  `inter_gene_max_space` semantics. Parsing these XML attributes does not imply
  that they affect evaluation. Some bundled models use `loner`, so even default
  model results can differ from MacSyFinder/CRISPRCasFinder.
- Python exposes CRISPR-array detection. The CLI and browser expose both array
  and Cas-system detection. Native Cas library callers must provide model and
  profile directories; the CLI embeds its defaults.
- Rust source builds require `nightly-2026-07-29` for `portable_simd`.
  Browser builds require WebAssembly SIMD, SharedArrayBuffer, and cross-origin
  isolation. Chromium and WebKit are exercised by the browser regression test.
  Memory and runtime limits depend on genome size and the device; arbitrary
  large assemblies have not been validated.

## Current regression cases

| Case | Source and options | What is checked | Boundary of the claim |
| --- | --- | --- | --- |
| E. coli NC_000913.3 array comparison | Tracked `crispr-cas-finder-cli/data/ecoli.fasta`; default detection options; reference positions attributed to CRISPRCasFinder 4.2.30 in the existing regression | Five reference regions; repeat positions within 10 bp, cluster overlap within 200 bp; high-evidence spacer count within 2 | Uses the existing frozen expectations, not a fresh upstream run or a false-positive benchmark |
| E. coli native/browser comparison | Same FASTA; genetic code 11; circular topology; SubTyping definitions; bundled CasFinder 2.0.3 | Full JSON equality across CLI, Chromium, WebKit, threaded and sequential search; baseline five arrays and one type I-E system | Agreement among builds of this implementation on one genome |
| HMMER Cas5 oracle | HMMER 3.4 `hmmsearch` frozen output; TIGR01868/Cas5_0_IE versus CP014688.1_298 | Sequence/domain scores, coordinates, bias, and alignment accuracy | One independently generated HMM/sequence oracle |
| HMM filter references | Synthetic profiles and sequences in `hmmer-core/tests/filter_reference.rs` | SIMD filters compared with independent scalar calculations | Algorithmic regression cases, not biological prevalence or accuracy |
| Python API | Synthetic repeat array and invalid inputs | Installed-wheel file/string API agreement and errors | Binding behavior, not additional biological validation |

The provenance and local correction to the bundled models are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dependency versions are pinned
in the Cargo lockfiles. Record the program version, model source, options,
input accession/checksum, and runtime when sharing results.

## Reproducing the checks

```sh
cargo test --locked --workspace --all-features
cargo build --locked --release -p crispr-cas-finder-cli
python3 scripts/test_cli_distribution.py target/release/crispr-cas-finder --report /tmp/native-report.json
python3 crispr-cas-finder-wasm/bundle_cas_models.py
./build_wasm.sh
npm ci --prefix crispr-cas-finder-wasm/tests
npx --prefix crispr-cas-finder-wasm/tests playwright install chromium webkit
SMOKE_NATIVE_REPORT=/tmp/native-report.json node crispr-cas-finder-wasm/tests/browser-smoke.cjs
SMOKE_BROWSER=webkit SMOKE_NATIVE_REPORT=/tmp/native-report.json node crispr-cas-finder-wasm/tests/browser-smoke.cjs
python3 scripts/test_python_distribution.py
```

On Windows use `target/release/crispr-cas-finder.exe` and PowerShell environment
variable syntax. Python distribution checks need maturin and a Python virtual
environment capable of installing the local wheel. Browser installation on
Linux may need Playwright's `--with-deps` option.

Future validation should include additional bacterial and archaeal genomes,
negative controls, fragmented assemblies, and multiple Cas subtypes. Publish
accuracy metrics and hardware/options alongside runtime comparisons before
making general equivalence or speedup claims.
