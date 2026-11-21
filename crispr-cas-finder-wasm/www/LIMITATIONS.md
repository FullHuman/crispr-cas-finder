# Experimental scope and validation

This is an independent experimental Rust implementation inspired by
CRISPRCasFinder. APIs and detection behavior may change. Tests establish
specific regression cases and consistency across runtimes, and a comparison
with CRISPRCasFinder on 72 inputs is summarised below: on those inputs the two
report the same arrays and the same Cas systems, apart from the differences
listed here as deliberate. Sensitivity and specificity against an independent
truth set have not been measured.

## Supported behavior and known differences

- Gene prediction and translation accept the NCBI genetic codes Prodigal
  supports (1-6, 9-16, 21-25) through `--genetic-code`; the default is 11. On the
  tracked E. coli genome, predicted genes and proteins are identical to Prodigal
  2.6.3 for codes 11, 4 and 25. Codes other than 11 have not been checked on an
  organism that actually uses them.
- Repeats are discovered by a Rust port of the Vmatch search CRISPRCasFinder
  runs (23 to 55 bp, at most one difference, 25 to 60 bp apart). On the 72
  inputs of the comparison below it reports the same 5,922 repeats as Vmatch
  2.3.1, with none missing and none added; `benchmarks/vmatch_oracle.py` runs
  that check. Vmatch's limit of a million repeats per sequence is not applied.
- The steps from those repeats to arrays follow CRISPRCasFinder: the consensus
  repeat of a locus is one of the repeats found there, chosen the same way and
  reported as it is; its copies within 500 bp of the locus give the spacers;
  the locus is dropped when its spacers are too alike or its only spacer is too
  much like the repeat; and it is divided into arrays at spacers longer than
  the repeat allows. Three things are deliberately not followed:
  CRISPRCasFinder divides a locus only once the sequence has given an array of
  three spacers or more, it loses arrays when one side of a division has no
  spacer, and it reports the first part of a divided locus one base too long.
- Repeats within 1.5 kb of each other form one locus and are searched with one
  consensus repeat, so of two unrelated arrays that close only one is reported.
  This is also what CRISPRCasFinder does.
- CRISPRCasFinder aligns the spacers and the repeats of a locus with MUSCLE
  and reads off those alignments whether the spacers are too alike for an
  array (60% identity, counting only bases that face each other, as BioPerl
  does) and the scores behind the evidence levels. The alignments here are
  those of MUSCLE 5.3, the version CRISPRCasFinder 4.3.2 asks for, computed by
  a Rust port of its `-align` command: the same rows as the binary
  `muscle 5.3.osxarm64` writes for all 975 alignments the comparison below
  triggers and for about 6,000 generated ones. The version matters: with
  MUSCLE 3.8 on its path the original reports 448 arrays on those inputs, with
  MUSCLE 5.3 it reports 485, because MUSCLE 5 finds the spacers of borderline
  loci less alike. So does the C library: MUSCLE draws the partitions of its
  refinement from `rand()` without seeding it, and the port follows the
  sequence of the macOS library. A MUSCLE built for Linux draws another
  sequence and may align a few inputs in a hundred differently; no Linux
  binary was compared. The test for a single spacer replaces EMBOSS `needle`
  and gave its verdict on all 301 repeat and spacer pairs of the comparison.
- Aligning as MUSCLE does costs what MUSCLE's method costs. The time grows
  with the square of the number of distinct spacers of a locus for their
  pairwise alignments, and with its cube for the consistency step. On an M1
  Max an array of a hundred spacers takes about 1.3 s on one thread and 0.2 s
  on ten: Saccharolobus solfataricus P2, with six such arrays, is searched for
  arrays in 0.8 s where version 0.1.0 took 0.06 s, and E. coli K-12 in 0.08 s
  as before. Three hundred spacers take about 23 s on one thread and 3.5 s on
  ten. Memory grows with the square of the number of spacers: a generated
  array of 400 spacers takes 6 s on ten cores and 0.7 GB, one of 800 spacers
  49 s and 2.8 GB, where version 0.1.0 took 0.02 s and 12 MB for either. No
  locus is refused for its size, so an array of many hundreds of spacers can
  exhaust the memory of a browser tab. The sums of the method are fused multiply-adds; where the processor
  instruction is not available to the build, in WebAssembly and in x86-64
  builds made without `+fma`, they are computed exactly another way and the
  alignment takes about three times as long.
- Only A, C, G and T match between copies of a repeat. N and the other
  ambiguity codes match nothing, so a run of N is never a repeat.
- Case is ignored: a sequence gives the same arrays in uppercase, in lowercase
  and soft-masked. Repeats and spacers are reported in the case the input has
  them in, as CRISPRCasFinder does; the consensus repeat is in uppercase.
- Evidence levels follow the CRISPRCasFinder rules: level 1 for at most three
  spacers, level 2 when repeat conservation is below 70, level 3 when more than
  8% of aligned spacer columns are identical, and level 4 otherwise. Repeats
  and spacers are aligned as above, the repeats too although they all have one
  length, so the two scores are those of the original: on the 485 arrays of
  the comparison, the same level and the same two scores to the two decimals
  it prints.
- The orientation of an array follows CRISPRCasFinder: an orientation recorded
  for the repeat in `Repeat_List.csv` goes first; otherwise the leader is on
  the side whose 100 bp flank is richer in A and T, when that flank is also
  richer than the sequence as a whole. A flank counts only when all its bases
  exist, and a flank without an A or a T, a run of N for instance, leaves the
  orientation unknown. One thing is not followed: when the identifier of a
  sequence ends in a range, as in `NC_000913.3:1-4641652`, BioPerl takes the
  range for coordinates, the original finds no right flank and reports every
  array of that sequence as of unknown orientation.
- `--force-detection` and `--better-detect-truncated` are reimplemented rather
  than copied: fostered start/end patterns are IUPAC codes (`.` = any base), and
  truncated detection adds at most one terminal repeat per side.
- Gene prediction uses Orphos 0.5.0, run once per sequence. Sequences shorter
  than 100 kb are predicted in metagenomic mode, as in CRISPRCasFinder;
  `--auto-metagenome-below 0` disables this. A GFF3 annotation and protein FASTA
  can be supplied instead (`--gff`, `--faa`). Prokka is not supported.
- Cas detection uses an internal Rust HMMER implementation, which follows
  `hmmsearch` 3.4 stage by stage. It decides the MSV, bias and Viterbi filters
  on the scores HMMER's own filters return, in bytes and 16-bit words with
  their rounding and overflow rules, computes the profile, the null score and
  the bias filter score in HMMER's arithmetic, and resolves a stretch of a
  protein that may hold several domains the way HMMER does, by clustering 200
  tracebacks sampled with HMMER's random number generator and seed. On the
  proteins of E. coli K-12 and of the 21 genomes of `benchmarks/panel.tsv`,
  searched with the 535 CasFinder 3.1 profiles, as many sequences pass each of
  the four filters as in `hmmsearch` for every profile and every proteome
  (11,770 searches), and the two report the same 7727 protein and profile
  pairs, with the same number of domains, the same alignment coordinates, and
  domain scores within 0.06 bit of the ones `hmmsearch` prints to a tenth of a
  bit. One pair differs beyond that, a hit of -36.7 bits that no threshold
  keeps: its envelope ends two residues later and its domain scores 0.6 bit
  more. With seed 7 in both tools, the 439 pairs of Thermococcus kodakarensis
  agree in the same way.
- That comparison is with the arm64 build of HMMER on macOS, and it holds on
  that platform. Where a score hangs on the last bit of a single-precision
  number, HMMER itself depends on how it was built: its arm64 build fuses some
  multiplications with the addition that follows, which this implementation
  does on every platform, and it reads profiles and scores the bias filter
  with the C library's `expf` and `logf`, which macOS does not round as other
  systems do. On another system a profile's probabilities can differ in their
  last bit from those `hmmsearch` computes on macOS, and a hit sitting exactly
  on a threshold can fall on the other side of it.
- The HMM search decodes posterior probabilities in probability space, as
  HMMER itself does. On E. coli K-12 (4318 proteins) with the 535 CasFinder
  3.1 profiles, a run of the tool on one thread of an M1 Max takes 6.6 s, gene
  prediction and array detection included, against 9.1 s for `hmmsearch` 3.4
  on the proteins alone; with every core, 1.0 s of wall time against 4.6 s for
  `hmmsearch --cpu 10`.
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
- CRISPRCasFinder reads the systems of MacSyFinder's best solution by number,
  from 1 to their count, while MacSyFinder numbers them among all its
  candidates: a solution holding systems 1, 3 and 4 is reported as system 1
  and an empty entry. This project reports every system of the solution.
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
| E. coli NC_000913.3 arrays | Tracked `crispr-cas-finder-cli/data/ecoli.fasta`; default detection options; the five arrays CRISPRCasFinder 4.2.30 reports, four of which CRISPRCasFinder's `CRISPR_crisprdb.csv` lists for NC_000913 with the same coordinates | Exact consensus repeat, evidence level, end and start of every repeat of each array, and that nothing else is reported | One genome; frozen expectations, not a fresh upstream run |
| Repeat search versus Vmatch 2.3.1 | The 72 inputs of the comparison below; `benchmarks/vmatch_oracle.py` | Every repeat of every sequence: same two copies, none missing, none added (5,922 repeats) | Run locally, not in CI, where Vmatch is not available. Unit tests freeze Vmatch's output for small synthetic cases |
| Alignments versus MUSCLE 5.3 | 52 sets of repeats or spacers frozen in `crispr-cas-finder-core/src/muscle5/frozen_cases.txt`; with `MUSCLE5_BIN` set, generated inputs against the binary | Every row of every alignment, with and without hardware fused multiply-add | The frozen cases run in CI. The wider agreement was measured locally against `muscle 5.3.osxarm64 [6d49a2f]`: all 975 alignments of the comparison below and about 6,000 generated inputs of 2 to 80 sequences, 500 of them also in a WebAssembly build. Not compared with a Linux or x86-64 build of MUSCLE |
| Single-spacer test versus EMBOSS | Repeat and spacer pairs taken from the genomes of the comparison | Share of gaps as EMBOSS `needle` 6.6 gives it for four pairs | Frozen values in unit tests. The wider agreement was measured locally: the same verdict for 301 of 301 pairs |
| E. coli native/browser comparison | Same FASTA; genetic code 11; circular topology; SubTyping definitions; bundled CasFinder 2.0.3 | Report equality across CLI, Chromium, WebKit, threaded and sequential search, with a 1e-4 relative tolerance on floating-point values between native and browser; baseline five arrays and one type I-E system | Agreement among builds of this implementation on one genome |
| CasFinder 3.1.0 versus MacSyFinder 2.1.6 | Eight bacterial genomes (E. coli K-12, three Erwinia amylovora, four other Enterobacterales, up to six sequences each); proteins predicted by this project; MacSyFinder run with the options CRISPRCasFinder passes | Identical systems: model, proteins, matched profiles, score and wholeness. HMM hits identical except one domain boundary | Run locally, not in CI, because the models cannot be redistributed. One genus-level clade; six systems of type I-E or I and two genomes without a system. An opt-in test (`CASFINDER3_DIR`) repeats the E. coli case |
| Comparison with CRISPRCasFinder 4.3.2 | 51 inputs of 1 to 376 contigs (E. coli K-12, an Acetobacter plasmid and 49 enterobacterial assemblies) and 21 reference genomes of bacteria and archaea (`benchmarks/panel.tsv`); both tools with CasFinder 3.1.0; scripts and method in `benchmarks/` | Arrays and Cas systems of the two outputs, paired by overlap | Run locally, not in CI: the arrays of the original on 2026-11-14 with MUSCLE 5.3, its Cas systems on 2026-10-13 with MacSyFinder 2.1.6 and HMMER 3.4. The original reports 485 arrays, 303 of level 1, 14 of level 2, one of level 3 and 167 of level 4. All 485 are reported here and no other, each with the same start, consensus repeat, spacers, evidence level, conservation scores, repeat identifier and CRISPRDirection. For 455 the end is the same too; the other 30 are first parts of a divided locus, which the original reports one base past its last repeat. The orientation is the same for 482; the other three are the arrays of E. coli K-12, whose identifier in the tracked file ends in a range (see above). The original reports 48 Cas systems; all 48 are reported here with the same subtype, the same genes at the same coordinates and every gene matched to the same profile. Thirteen more are reported here: nine that MacSyFinder finds inside the original and its Perl script then loses, and four on sequences under 100 kb |
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
subtypes, metagenomes and genomes above 9 Mb are not in it, it was run with
the default options of both tools only, and it measures agreement with
CRISPRCasFinder, not accuracy. State the hardware and the
options next to any run time taken from it.
