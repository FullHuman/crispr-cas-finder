# Experimental scope and validation

This is an independent experimental Rust implementation inspired by
CRISPRCasFinder. APIs and detection behavior may change. Tests establish
specific regression cases and consistency across runtimes, and a first
comparison with CRISPRCasFinder on 72 inputs is summarised below. Sensitivity
and specificity against an independent truth set have not been measured.

## Supported behavior and known differences

- Gene prediction and translation accept the NCBI genetic codes Prodigal
  supports (1-6, 9-16, 21-25) through `--genetic-code`; the default is 11. On the
  tracked E. coli genome, predicted genes and proteins are identical to Prodigal
  2.6.3 for codes 11, 4 and 25. Codes other than 11 have not been checked on an
  organism that actually uses them.
- Repeats are discovered with the settings CRISPRCasFinder gives Vmatch (23 to
  55 bp, at most one mismatch, 25 to 60 bp apart) by an internal seed search.
  It does not reproduce every Vmatch match. Arrays of evidence level 3 and 4
  are found either way; small and poorly conserved ones (levels 1 and 2) differ
  more, and repeat-rich genomes get more of them here than from
  CRISPRCasFinder. See the comparison below.
- A locus is divided into arrays wherever two neighbouring repeats are farther
  apart than the longest spacer the repeat allows, as CRISPRCasFinder does.
  Repeats within 1.5 kb of each other form one locus and are searched with one
  consensus repeat, so of two unrelated arrays that close only one is reported.
- Only A, C, G and T match between copies of a repeat. N and the other
  ambiguity codes match nothing, so a run of N is never a repeat.
- Evidence levels follow the CRISPRCasFinder rules: level 1 for at most three
  spacers, level 2 when repeat conservation is below 70, level 3 when more than
  8% of aligned spacer columns are identical, and level 4 otherwise. Spacers are
  aligned by an internal center-star aligner, not MUSCLE, and the repeats of an
  array, which all have the same length, are compared without gaps. Scores near
  a threshold can therefore differ from upstream: unrelated spacers that MUSCLE
  lines up by chance leave an array at level 4 here and level 3 there, and a
  repeat carrying an insertion or deletion lowers repeat conservation more here.
  An array is rejected when its aligned spacers are on average at least
  `--spacer-similarity-threshold` percent identical; a residue facing a gap
  counts as a mismatch.
- `--force-detection` and `--better-detect-truncated` are reimplemented rather
  than copied: fostered start/end patterns are IUPAC codes (`.` = any base), and
  truncated detection adds at most one terminal repeat per side.
- Gene prediction uses Orphos 0.4.0, run once per sequence. Sequences shorter
  than 100 kb are predicted in metagenomic mode, as in CRISPRCasFinder;
  `--auto-metagenome-below 0` disables this. A GFF3 annotation and protein FASTA
  can be supplied instead (`--gff`, `--faa`). Prokka is not supported.
- Cas detection uses an internal Rust HMMER implementation. Bit scores can
  differ from HMMER 3.4 by a few tenths of a bit and, rarely, one domain is
  reported where `hmmsearch` reports two. Either can matter for a hit sitting
  exactly on a threshold, and the second can change which of two related
  profiles is a protein's best match, which in turn can change the systems
  found around it.
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
| Comparison with CRISPRCasFinder 4.3.2 | 51 enterobacterial inputs of 1 to 376 contigs and 21 reference genomes of bacteria and archaea (`benchmarks/panel.tsv`); both tools with CasFinder 3.1.0; scripts and method in `benchmarks/` | Arrays and Cas systems of the two outputs, paired by overlap | Run locally on 2026-10-13, not in CI, the original with MacSyFinder 2.1.6 and HMMER 3.4. All 161 level-4 arrays of the original are reported, 159 of them within 5 bp, as are its 8 level-3 arrays (at level 4 here), 4 of its 7 level-2 and 222 of its 272 level-1 arrays. Reported here only: 91 arrays of level 1, 32 of level 2 and 3 of level 3, most of them in M. tuberculosis, S. coelicolor and C. difficile. Of the 48 Cas systems in the original's report, 47 are reported with the same subtype, 46 with the same genes; a type I-B system of T. kodakarensis is missed because of the domain difference described above |
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

The comparison covers twelve Cas subtypes and four genomes without a
CRISPR-Cas system. It is still small: type IV, most type V and type VI
subtypes, metagenomes and genomes above 9 Mb are not in it, and it measures
agreement with CRISPRCasFinder, not accuracy. State the hardware and the
options next to any run time taken from it.
