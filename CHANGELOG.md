# Changelog

## [Unreleased]

On the 72 genomes of the comparison in `benchmarks/`, this version and
CRISPRCasFinder 4.3.2 now report the same 485 arrays and the same 48 Cas
systems, down to the spacers, the evidence levels and the profile each gene is
matched to. What still differs is listed in `LIMITATIONS.md` and is deliberate:
the end of the first part of a divided locus, which the original reports one
base too far, and thirteen Cas systems the original loses.

### Array detection

- The repeats and the spacers of a locus are aligned as MUSCLE 5.3 aligns
  them, by a Rust port of its `-align` command (`muscle5`), in place of the
  center-star alignment used until now. CRISPRCasFinder reads off those
  alignments whether the spacers of a locus are too alike for an array and the
  two scores behind evidence levels 2 to 4, so a locus near a threshold now
  falls on the side the original puts it. The port returns the rows the binary
  `muscle 5.3.osxarm64` writes for all 975 alignments the 72 genomes trigger
  and for about 6,000 generated ones.
- The comparison of 2026-10-13 ran the original with MUSCLE 3.8, although
  version 4.3.2 asks for MUSCLE 5.1 or later. With MUSCLE 5.3 the original
  reports 485 arrays on the 72 genomes, not 448, and 8 other evidence levels.
  Against that, version 0.1.0 reported 442 of the 485, with one level
  different; this version reports all 485 and no other, each with the same
  repeats, spacers, level and conservation scores.
- Results change for loci near a threshold: 43 arrays are reported on the 72
  genomes that 0.1.0 dropped as tandem repeats, and the evidence level of one
  changes.
- The price is the time MUSCLE's method takes on large arrays, which grows
  with the square and the cube of the number of distinct spacers. The loci of
  a sequence are processed in parallel and the alignment made to test the
  spacers of a locus is reused for the scores of its array.
- The orientation of an array is decided as in CRISPRCasFinder: an orientation
  recorded for the repeat in `Repeat_List.csv` goes first, a flank counts only
  when all its 100 bases exist, and a flank without an A or a T leaves the
  orientation unknown. Six arrays of the 72 genomes change, to what the
  original reports.

### Cas detection

- The HMM search lets the same sequences through its filters as `hmmsearch`
  3.4. HMMER decides its MSV and bias filters on the score its byte filter
  returns and its Viterbi filter on the score of its 16-bit filter, each with
  its own rounding, its own overflow rule and a constant in place of the N, C
  and J loops; the search decided them on an 8-bit lower bound of the exact
  MSV score and a 16-bit upper bound of the exact Viterbi score, up to a nat
  away. It now computes HMMER's two filter scores and applies HMMER's tests:
  an overflowed filter passes, and a sequence whose MSV score already meets
  the Viterbi threshold skips the Viterbi filter. Three smaller differences
  are gone with it: the bias filter kept one transition of its two-state
  model fixed where HMMER sets it from the target length, a model's
  composition was normalised where HMMER uses it as written, and the
  profile's scores were computed in single precision where HMMER takes its
  logarithms in double precision. On E. coli K-12 and the 21 genomes of the
  panel, with the 535 CasFinder 3.1 profiles, as many sequences pass each of
  the four filters as in `hmmsearch` in all 11,770 searches, where none
  agreed, and the two tools report the same 7727 protein and profile pairs,
  where 540 were reported by `hmmsearch` only, 45 of them of 20 bits or
  more, and 1153 by this tool only, 18 of 20 bits or more. The two envelopes
  that started a residue or two away from HMMER's are HMMER's now. The 7187
  pairs reported before and still reported keep their coordinates otherwise;
  709 scores move, 17 by more than a thousandth of a bit and none by more
  than 0.014 bit.
- On the 72 genomes the Cas systems do not change: 48 reported by both tools
  with the same genes and profiles, 13 by this tool only. Seven weak hits
  enter `annotation/cas_hits.tsv`, one of 35.9 bits that `hmmsearch` drops
  leaves it, none of them in a system, and the third digit of 14 E-values
  moves by one. E. coli K-12 takes 6.57 s on one thread of an M1 Max where it
  took 6.38 s, and 1.01 s on all its cores where it took 0.99 s: HMMER's
  filters let more sequences through to the Viterbi filter than the bounds
  did.
- `--exact-filters` still decides the filters on exact scores, and its bias
  filter follows HMMER's too: on the 22 proteomes it reports 7757 pairs where
  it reported 8666, 7592 of them among the 7727 of `hmmsearch` where 7327
  were.
- A stretch of a protein that may hold several domains is resolved as HMMER
  3.4 resolves it. The Forward matrix of the region is filled with the
  multihit model, 200 tracebacks are sampled from it with Easel's generator
  and HMMER's seed, the domains they contain are clustered into envelopes, the
  composition correction of the region comes from the same tracebacks, and
  each envelope is scored on its own; before, such a region was scored as one
  domain. On E. coli K-12 and the 21 genomes of the panel, with the 535
  CasFinder 3.1 profiles, 1759 of the 7187 protein and profile pairs reported
  by both tools differed from `hmmsearch` 3.4 in their number of domains (850)
  or in a domain score by more than 0.05 bit; none differs in its number of
  domains now, and the 4 that still differ in a score do so by less than 0.06
  bit, against scores `hmmsearch` prints to a tenth of a bit. With
  `--exact-filters`, 2217 of 7327 pairs differed and 505 do. The 35 Cas
  systems the original pipeline reports on the panel are all found with every
  gene matched to the same profile, where 29 were, five had a gene matched to
  another profile or one gene more, and the type I-B system of Thermococcus
  kodakarensis was missed. Hits without such a region are unchanged. E. coli
  K-12 takes 6.54 s on one thread of an M1 Max where it took 6.50 s.
- With that, the 48 Cas systems the original reports on the 72 genomes are all
  reported with the same subtype, the same genes at the same coordinates and
  every gene matched to the same profile; version 0.1.0 reported 47 with the
  same subtype and 42 with every gene on the same profile.

## [0.1.0] - 2026-11-14

The first release: the crates `crispr-cas-finder-core` and
`crispr-cas-finder-cli` on crates.io, the Python package `crispr-cas-finder` on
PyPI, and command-line binaries for Linux, macOS and Windows on the GitHub
release. The entries below cover the work since the port began, newest first
within each section, with the pull request that merged it.

### Array detection

- The AT content of a sequence, which the orientation of each array is judged
  against, is counted once per sequence instead of once per array: detecting
  the arrays of E. coli K-12 takes 0.08 s where it took 0.15 s, with the same
  arrays. (#9)
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

- Gene prediction uses Orphos 0.5.0, which predicts the same genes faster
  (0.2 s less on E. coli K-12). The 8-bit MSV bound that decides the first filter of the HMM
  search has kernels of its own for the common case: a row of up to 24 vectors
  is updated in its registers from the last vector down, a longer one four
  rows per pass along chains of cells that are loaded and stored once for four
  rows of work, and the emissions of a row are found in four instructions. The
  forward filter adds the delete states to its exit sum during its last carry
  pass, and the order the proteins are searched in is worked out once for all
  profiles. With the 535 CasFinder 3.1 profiles, E. coli K-12 takes 6.5 s on
  one thread of an M1 Max where it took 7.9 s, and 1.0 s on all cores where it
  took 1.4 s; every output file is identical on 29 genomes, with and without
  `--exact-filters`. (#9)
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

- The command-line tool ends a run with one line saying how many arrays and Cas
  systems it found and where the files are; `--quiet`, which did nothing
  before, leaves that line out. Every option has a description in `--help`.
- `benchmarks/` holds the scripts that run this project and CRISPRCasFinder on
  the same genomes and compare results, run time and memory, and the check of
  the repeat search against Vmatch. (#6)
- The website is restyled with the Lab Report design system and stays a plain
  front for the library: a drop zone for the FASTA file, spelled-out option
  labels, a progress bar during the HMM search, one light theme. (#5)
- The CLI, the Python bindings and the browser produce one `report.json`
  structure, alongside TSV, GFF3 and FASTA outputs. (#4)

### Release preparation

- The crates and the Python package carry keywords, categories and links for
  crates.io and PyPI; the Python package has a `py.typed` marker, so type
  checkers read its stubs; the Python and WebAssembly crates are marked as not
  for crates.io. `rustix` is updated to 1.1.6 for RUSTSEC-2026-0486.
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
