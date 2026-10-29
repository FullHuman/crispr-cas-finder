# Changelog

Nothing has been released yet; every manifest is at version 0.1.0. The entries
below cover the work since the port began, newest first within each section,
with the pull request that merged it.

## [Unreleased]

### Array detection

- The repeat search compares the hashes of the windows a repeat's partner can
  start in eight at a time, with the processor's widest vectors, instead of
  one by one: about half the time of the array detection on E. coli K-12 on
  one thread, with the same repeats.
- Repeats are found by a port of the Vmatch search CRISPRCasFinder runs, which
  reports the same repeats as Vmatch 2.3.1 on the 72 inputs of the comparison
  in `benchmarks/`. Arrays are then built in the original's order: the left
  copy of a repeat is the hit, a locus is searched 500 bp around its hits, the
  consensus is the hit sequence found most often, spacers follow
  `definespacers` and `check_cris_div`, and a locus is dropped on a
  MUSCLE-scored spacer alignment before it is divided. Of the original's 448
  arrays on those inputs, 439 are reported with the same start and the same
  number of spacers. (#6)
- Case is ignored, so soft-masked input gives the same arrays as uppercase.
  Repeats and spacers keep the input's case; the consensus repeat is
  uppercase. (#6)
- Each spacer is read once, at the positions reported for it; a copy right
  after the previous one becomes part of the next spacer instead of giving an
  empty one. (#6)
- Every array of a locus is reported where before a cluster of seed repeats
  gave at most one; a degenerate copy before an array no longer drops it;
  tandem repeats no longer seed arrays; N matches nothing; spacer length
  limits are rounded inwards as upstream rounds them. (#6)
- Arrays carry evidence levels 1 to 4 with conservation statistics from a Rust
  aligner. User-supplied repeats, small-array reclassification and CRISPRdb
  lookups work; every option on the upstream gap list works or is rejected
  explicitly, and most upstream option names are accepted as aliases. (#4)

### Cas detection

- The integer MSV and Viterbi bounds of the HMM search, which decide the
  filters for nearly every sequence, run with the widest vectors the
  processor has: the kernels are compiled once per instruction set and chosen
  at run time (SSE4.1, AVX2 and AVX-512BW on x86-64, the portable baseline
  elsewhere), and the bounds come out identical on every one. The HMM reader
  parses a model without a string per line or a vector per row. One thread
  searches the E. coli K-12 proteome against the embedded CasFinder 2.0.3
  profiles in about half the time, with identical output. The model
  definitions and the HMM profiles are read while the genes are predicted,
  which is one long serial task on a single genome.
- The HMM search decides its MSV and bias filters with an 8-bit lower bound on
  the MSV score, its Viterbi filter with a 16-bit upper bound, and decodes
  posteriors in probability space, as HMMER does. One thread of an M1 Max
  searches the E. coli K-12 proteome against the 535 CasFinder 3.1 profiles in
  7.9 s, against 9.8 s for `hmmsearch` 3.4 and 26.7 s before; `--exact-filters`
  keeps the exact filter scores and the log-space decoder. (#6)
- CasFinder 3 packages, in the MacSyFinder 2 format, load from a directory
  given with `--cas-models-dir` and are evaluated with MacSyFinder 2 rules
  reimplemented in Rust: identical systems to MacSyFinder 2.1.6 on eight
  genomes. The bundled default remains CasFinder 2.0.3. (#4)
- Gene prediction and translation accept the genetic codes Prodigal supports,
  1-6, 9-16 and 21-25, through `--genetic-code`; other codes are rejected
  before the analysis starts. Predictions are identical to Prodigal 2.6.3 on
  E. coli for codes 11, 4 and 25. (#4)
- Arrays near a Cas system are reported (`--vicinity`), arrays and Cas systems
  can be chained into loci (`--locus-distance`), and an existing annotation can
  replace gene prediction (`--gff`, `--faa`). (#4)

### Interfaces and outputs

- `benchmarks/` holds the scripts that run this project and CRISPRCasFinder on
  the same genomes and compare results, run time and memory, and the check of
  the repeat search against Vmatch. (#6)
- The website is restyled with the Lab Report design system and stays a plain
  front for the library: a drop zone for the FASTA file, spelled-out option
  labels, a progress bar during the HMM search, one light theme. (#5)
- The CLI, the Python bindings and the browser produce one `report.json`
  structure, alongside TSV, GFF3 and FASTA outputs. (#4)

### Release preparation

- This first public release is experimental; `LIMITATIONS.md` documents the
  supported behaviour, the known differences from CRISPRCasFinder and the
  regression cases.
- The CLI embeds the CasFinder 2.0.3 models, so installed and downloaded
  binaries run Cas analysis without a source checkout.
- Project and upstream license notices ship with the Rust, Python, browser and
  CLI distributions; `THIRD_PARTY_NOTICES.md` records the provenance of the
  bundled data and the local correction to one XML definition.
- Python source distributions include the pinned nightly toolchain file, and
  the nightly requirement is documented for Rust consumers.
- Release checks cover the E. coli reference comparison, the installed
  packages, the browser against the native report, the archive contents and
  the release wheels.
