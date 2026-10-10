# Experimental scope and validation

This is an independent experimental Rust implementation inspired by
CRISPRCasFinder. APIs and detection behavior may change. Initial tests establish
specific regression cases and consistency across runtimes; broad sensitivity,
specificity, and performance comparisons remain ongoing.

## Supported behavior and known differences

- Gene prediction supports **genetic code 11 only**. Other codes return an
  error in the CLI, native Cas API, and WASM API, because Orphos 0.3.0 ignores a
  requested translation table. For organisms using another code (for example
  Mycoplasma, code 4), supply an existing annotation with `--gff` and `--faa`.
  With code 11, predicted genes and proteins are identical to Prodigal 2.6.3 on
  the tracked E. coli genome (4,318 of 4,318).
- Evidence levels follow the CRISPRCasFinder rules: level 1 for at most three
  spacers, level 2 when repeat conservation is below 70, level 3 when more than
  8% of aligned spacer columns are identical, and level 4 otherwise. Repeats and
  spacers are aligned by an internal center-star aligner, not MUSCLE, so scores
  near a threshold can differ from upstream. An array is rejected when its
  aligned spacers are on average at least `--spacer-similarity-threshold`
  percent identical; a residue facing a gap counts as a mismatch.
- `--force-detection` and `--better-detect-truncated` are reimplemented rather
  than copied: fostered start/end patterns are IUPAC codes (`.` = any base), and
  truncated detection adds at most one terminal repeat per side.
- Gene prediction uses Orphos 0.3.0, run once per sequence. Sequences shorter
  than 100 kb are predicted in metagenomic mode, as in CRISPRCasFinder;
  `--auto-metagenome-below 0` disables this. A GFF3 annotation and protein FASTA
  can be supplied instead (`--gff`, `--faa`). Prokka is not supported.
- Cas detection uses an internal Rust HMMER implementation. Bit scores can
  differ from HMMER 3.4 by a few tenths of a bit and, rarely, one domain is
  reported where `hmmsearch` reports two, which can matter for a hit sitting
  exactly on a threshold.
- Two generations of Cas models are supported. The bundled default is CasFinder
  2.0.3. CasFinder 3 packages (MacSyFinder 2 format) are loaded from a directory
  given with `--cas-models-dir`; they are licensed CC BY-NC-SA 4.0 and are not
  distributed with this project. The browser application uses CasFinder 2.0.3.
- CasFinder 3 definitions are evaluated with MacSyFinder 2 rules reimplemented in
  Rust: gathering cutoffs, one best profile per protein, clustering with
  gene-specific spacing, quorum, scoring, and the best combination of
  non-conflicting systems. `loner`, `multi_system` and `multi_loci` are not
  implemented and a package using them is rejected; CasFinder 3.1.0 uses none.
  `--definition` and `--min-cas-score` do not apply to these packages.
- CasFinder 2.0.3 definitions are evaluated by the original evaluator of this
  project, which clusters hits per replicon and does not implement `loner`,
  `multi_system`, `multi_model`, `multi_loci`, or gene-specific
  `inter_gene_max_space`. Some of these models use `loner`, so results can
  differ from MacSyFinder 1.
- Every sequence is searched as an ordered replicon. CRISPRCasFinder runs
  MacSyFinder in unordered mode for metagenomes and for sequences shorter than
  100 kb, which does not produce the file it reads Cas systems from; this
  project reports Cas systems on those sequences.
- An array is reported near a Cas system when the distance between them is at
  most `--vicinity` bp. Upstream instead tests whether the array boundary
  shifted by the vicinity falls inside the system, so the two lists can differ.
  Loci (`--locus-distance`) do not chain across the origin of a circular
  sequence.
- Output files are not those of CRISPRCasFinder. See the README for the files
  written and `report.json` for the complete result.
- The CLI, Python and browser expose array and Cas-system detection and return
  the same report structure. Bit scores and E-values can differ in their last
  digits between native and WebAssembly builds. Native library and Python
  callers must provide Cas model and profile directories; the CLI embeds its
  defaults.
- Rust source builds require `nightly-2026-07-29` for `portable_simd`.
  Browser builds require WebAssembly SIMD, SharedArrayBuffer, and cross-origin
  isolation. Chromium and WebKit are exercised by the browser regression test.
  Memory and runtime limits depend on genome size and the device; arbitrary
  large assemblies have not been validated.

## Current regression cases

| Case | Source and options | What is checked | Boundary of the claim |
| --- | --- | --- | --- |
| E. coli NC_000913.3 CRISPRdb coordinates | Tracked `crispr-cas-finder-cli/data/ecoli.fasta`; default detection options; the four arrays listed for NC_000913 in CRISPRCasFinder's `CRISPR_crisprdb.csv` | Exact start, end and spacer count of each of the four arrays | One genome; a fifth, level-1 array reported here is not in CRISPRdb |
| E. coli NC_000913.3 array comparison | Tracked `crispr-cas-finder-cli/data/ecoli.fasta`; default detection options; reference positions attributed to CRISPRCasFinder 4.2.30 in the existing regression | Five reference regions; repeat positions within 10 bp, cluster overlap within 200 bp; high-evidence spacer count within 2 | Uses the existing frozen expectations, not a fresh upstream run or a false-positive benchmark |
| E. coli native/browser comparison | Same FASTA; genetic code 11; circular topology; SubTyping definitions; bundled CasFinder 2.0.3 | Report equality across CLI, Chromium, WebKit, threaded and sequential search, with a 1e-4 relative tolerance on floating-point values between native and browser; baseline five arrays and one type I-E system | Agreement among builds of this implementation on one genome |
| CasFinder 3.1.0 versus MacSyFinder 2.1.6 | Eight bacterial genomes (E. coli K-12, three Erwinia amylovora, four other Enterobacterales, up to six sequences each); proteins predicted by this project; MacSyFinder run with the options CRISPRCasFinder passes | Identical systems: model, proteins, matched profiles, score and wholeness. HMM hits identical except one domain boundary | Run locally, not in CI, because the models cannot be redistributed. One genus-level clade; six systems of type I-E or I and two genomes without a system. An opt-in test (`CASFINDER3_DIR`) repeats the E. coli case |
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
