//! Tests of the MUSCLE 5.3 port.
//!
//! `frozen_cases.txt` holds inputs and the alignment the binary
//! `muscle 5.3.osxarm64 [6d49a2f]` wrote for each of them, so the port is
//! checked against MUSCLE without the program. To check it against the
//! program on freshly generated inputs, name the binary in `MUSCLE5_BIN`:
//!
//! ```text
//! MUSCLE5_BIN=/path/to/muscle cargo test -p crispr-cas-finder-core muscle5
//! ```
//!
//! (`MUSCLE5_CASES` sets how many inputs, 300 by default, `MUSCLE5_SEED`
//! where the generator starts.)

use super::*;

/// One case of `frozen_cases.txt`: its name and, for each sequence, the
/// input and the row MUSCLE gave it.
struct Frozen {
    name: &'static str,
    rows: Vec<(&'static [u8], &'static [u8])>,
}

fn frozen_cases() -> Vec<Frozen> {
    let mut cases: Vec<Frozen> = Vec::new();
    for line in include_str!("frozen_cases.txt").lines() {
        if let Some(name) = line.strip_prefix("# ") {
            cases.push(Frozen {
                name,
                rows: Vec::new(),
            });
        } else if let Some((input, row)) = line.split_once('\t') {
            let case = cases.last_mut().expect("a name before the rows");
            case.rows.push((input.as_bytes(), row.as_bytes()));
        }
    }
    cases
}

fn check_frozen(soft_fma: bool) {
    let cases = frozen_cases();
    assert!(cases.len() >= 50, "{} cases", cases.len());
    for case in cases {
        let input: Vec<&[u8]> = case.rows.iter().map(|&(input, _)| input).collect();
        let rows = align_with(&input, soft_fma);
        assert_eq!(rows.len(), input.len(), "{}", case.name);
        for (index, (row, &(_, expected))) in rows.iter().zip(&case.rows).enumerate() {
            assert_eq!(
                String::from_utf8_lossy(row),
                String::from_utf8_lossy(expected),
                "{} row {index}",
                case.name
            );
        }
    }
}

#[test]
fn frozen_alignments_are_those_of_muscle() {
    check_frozen(!HARDWARE_FMA);
}

/// The two ways of computing the consistency sums, with the fused
/// multiply-add instruction and without, give the same alignments.
#[test]
fn frozen_alignments_hold_with_either_fma() {
    check_frozen(HARDWARE_FMA);
}

#[test]
fn zero_or_one_sequence_comes_back_unaligned() {
    assert!(align(&[]).is_empty());
    assert_eq!(align(&[b"acgtn-ACG T."]), vec![b"ACGTNACGT".to_vec()]);
    assert_eq!(align(&[b""]), vec![Vec::<u8>::new()]);
}

#[test]
fn identical_sequences_come_back_unaligned() {
    let rows = align(&[b"ACGTACGT", b"acgtacgt", b"ACGT-ACGT"]);
    assert_eq!(rows, vec![b"ACGTACGT".to_vec(); 3]);
}

#[test]
fn copies_of_a_sequence_share_its_row() {
    let rows = align(&[
        b"ACGTACGTACGTAAGCTAGCT",
        b"ACGTACGACGTAAGCTAGCTT",
        b"acgtacgtacgtaagctagct",
        b"ACGTACGACGTAAGCTAGCTT",
    ]);
    assert_eq!(rows[0], b"ACGTACGTACGTAAGCTAGC-T");
    assert_eq!(rows[1], b"ACGTACG-ACGTAAGCTAGCTT");
    assert_eq!(rows[2], rows[0]);
    assert_eq!(rows[3], rows[1]);
}

/// MUSCLE crashes on an empty sequence; here it gets a row of gaps and the
/// other rows are those of the alignment without it.
#[test]
fn an_empty_sequence_gets_a_row_of_gaps() {
    let filled: [&[u8]; 3] = [
        b"ACGTACGTACGTAAGCTAGCT",
        b"ACGTACGTACTTAAGCTAGCT",
        b"ACGTACGACGTAAGCTAGCTT",
    ];
    let expected = align(&filled);
    let rows = align(&[filled[0], b"", filled[1], b"--", filled[2]]);
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0], expected[0]);
    assert_eq!(rows[2], expected[1]);
    assert_eq!(rows[4], expected[2]);
    assert_eq!(rows[1], vec![b'-'; expected[0].len()]);
    assert_eq!(rows[3], rows[1]);
    assert_eq!(
        align(&[b"", b"ACGT"]),
        vec![b"----".to_vec(), b"ACGT".to_vec()]
    );
    assert_eq!(align(&[b"", b""]), vec![Vec::<u8>::new(); 2]);
}

/// Values of the C library's `expf` on macOS, among them arguments for
/// which it does not return the nearest `float`.
#[test]
fn expf_is_that_of_the_c_library() {
    let samples: [(u32, u32); 24] = [
        (0x8000_0001, 0x3f80_0000),
        (0xb2ff_fff1, 0x3f7f_ffff),
        (0xb8d1_b717, 0x3f7f_f972),
        (0xbdcc_cccd, 0x3f67_a36d),
        (0xbf00_0000, 0x3f1b_4598),
        (0xbf31_7218, 0x3f00_0000),
        (0xbf80_0000, 0x3ebc_5ab2),
        (0xc000_0000, 0x3e0a_9555),
        (0xc080_0000, 0x3c96_0aae),
        (0xc093_5d8e, 0x3c23_d70a),
        (0xb300_0000, 0x3f7f_ffff),
        (0xb41f_fff8, 0x3f7f_fffd),
        (0xb517_fffa, 0x3f7f_fff6),
        (0xb60e_0002, 0x3f7f_ffdc),
        (0xb702_801a, 0x3f7f_ff7d),
        (0xb7f6_c0e0, 0x3f7f_fe12),
        (0xb8ea_f351, 0x3f7f_f8a8),
        (0xb9df_181a, 0x3f7f_e41e),
        (0xbad3_3c87, 0x3f7f_9677),
        (0xbbc7_60b6, 0x3f7e_7274),
        (0xbcbb_84d2, 0x3f7a_34e3),
        (0xbdaf_a8f1, 0x3f6a_f534),
        (0xbea3_cd42, 0x3f39_e874),
        (0xbf97_f179, 0x3e9c_386d),
    ];
    for (x, expected) in samples {
        assert_eq!(
            expf(f32::from_bits(x)).to_bits(),
            expected,
            "expf({x:#010x})"
        );
    }
}

/// Small generator for the tests (splitmix64).
struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn between(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn pick(&mut self, letters: &[u8]) -> u8 {
        letters[self.below(letters.len())]
    }

    fn sequence(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.pick(b"ACGT")).collect()
    }
}

#[test]
fn soft_fma_rounds_once() {
    let mut rng = TestRng(1);
    // How often rounding the double-precision sum to `f32` would have been
    // wrong: the cases `soft_fma` exists for.
    let mut double_rounding_errors = 0;
    let mut check = |a: f32, b: f32, c: f32| {
        let fused = a.mul_add(b, c);
        assert_eq!(
            soft_fma(a, b, c).to_bits(),
            fused.to_bits(),
            "{a:e} * {b:e} + {c:e}"
        );
        if (f64::from(a) * f64::from(b) + f64::from(c)) as f32 != fused {
            double_rounding_errors += 1;
        }
    };
    for _ in 0..200_000 {
        // Probabilities and sums of the sizes the consistency step sees.
        let a = (rng.next() >> 40) as f32 / (1u64 << 24) as f32;
        let b = (rng.next() >> 40) as f32 / (1u64 << 24) as f32;
        let c = (rng.next() >> 40) as f32 / (1u64 << 20) as f32;
        check(a, b, c);
    }
    for _ in 0..100_000 {
        // A product exactly half way between two floats (two odd integers
        // whose product has 25 bits), and a third operand too small to
        // show in the double-precision sum but that decides the rounding.
        let a = (rng.between(1 << 11, (1 << 12) - 1) | 1) as f32;
        let b = (((1usize << 24) / a as usize + rng.between(1, 4000)) | 1) as f32;
        let c = f32::from_bits(0x2000_0000 | (rng.next() as u32 & 0x007f_ffff));
        let scale = [1.0f32, 0.5, 0.001_953_125, 64.0][rng.below(4)];
        for (a, b, c) in [(a, b, c), (a, b, -c), (a, -b, c), (a, -b, -c)] {
            check(a * scale, b, c);
        }
    }
    assert!(double_rounding_errors > 50_000, "{double_rounding_errors}");
}

#[test]
fn libc_rand_is_the_minimal_standard_generator() {
    let mut rng = LibcRand::new();
    let first: Vec<u32> = (0..5).map(|_| rng.next()).collect();
    assert_eq!(
        first,
        [
            16_807,
            282_475_249,
            1_622_650_073,
            984_943_658,
            1_144_108_930
        ]
    );
}

/// The polynomials of scoretype.h `LOGEXP1`, at the ends of their ranges.
#[test]
fn log_add_follows_scoretype_h() {
    assert_eq!(log_add(LOG_ZERO, -1.5), -1.5);
    assert_eq!(log_add(-1.5, LOG_ZERO), -1.5);
    assert_eq!(log_add(-9.0, -1.5), -1.5);
    assert_eq!(log_add(-1.5, -9.0), -1.5);
    let ln2 = f32::from_bits(0x3f31_75c2);
    assert_eq!(log_add(0.0, 0.0), ln2);
    for (x, exact) in [
        (0.5f32, 0.974_077),
        (2.0, 2.126_928),
        (4.0, 4.018_15),
        (7.0, 7.000_911),
    ] {
        let sum = log_add(x, 0.0);
        assert!((sum - exact).abs() < 2e-3, "log(e^{x} + 1) = {sum}");
        assert_eq!(log_add(0.0, x), sum);
    }
}

/// The rows of an alignment spell the sequences and have one length.
#[test]
fn rows_spell_the_sequences() {
    let mut rng = TestRng(7);
    for _ in 0..40 {
        let input = random_input(&mut rng, 12, 40);
        let refs: Vec<&[u8]> = input.iter().map(Vec::as_slice).collect();
        let rows = align(&refs);
        assert_eq!(rows.len(), input.len());
        for (seq, row) in input.iter().zip(&rows) {
            assert_eq!(row.len(), rows[0].len());
            let letters: Vec<u8> = row.iter().copied().filter(|&c| c != b'-').collect();
            assert_eq!(letters, seq.to_ascii_uppercase());
        }
        assert_eq!(align_with(&refs, true), align_with(&refs, false));
    }
}

/// Any bytes are taken, short and empty sequences too: what has no score of
/// its own counts as a wildcard.
#[test]
fn arbitrary_bytes_are_aligned() {
    let mut rng = TestRng(3);
    for _ in 0..300 {
        let input: Vec<Vec<u8>> = (0..rng.below(8))
            .map(|_| {
                let alphabet = rng.between(1, 256);
                (0..rng.below(30))
                    .map(|_| rng.below(alphabet) as u8)
                    .collect()
            })
            .collect();
        let refs: Vec<&[u8]> = input.iter().map(Vec::as_slice).collect();
        let rows = align(&refs);
        assert_eq!(rows.len(), input.len());
        let aligned = input.iter().filter(|seq| !read_sequence(seq).is_empty());
        let width = if aligned.count() > 1 {
            rows[0].len()
        } else {
            0
        };
        for (seq, row) in input.iter().zip(&rows) {
            let letters: Vec<u8> = row.iter().copied().filter(|&c| c != b'-').collect();
            assert_eq!(letters, read_sequence(seq));
            if width > 0 {
                assert_eq!(row.len(), width);
            }
        }
    }
}

/// Sequences of the kinds an array yields: unrelated ones, near-copies of a
/// parent (substitutions, insertions, deletions, shifts), exact copies, a
/// few letters other than ACGT, lowercase.
fn random_input(rng: &mut TestRng, max_seqs: usize, max_len: usize) -> Vec<Vec<u8>> {
    let n = if rng.chance(25) {
        2
    } else {
        rng.between(3, max_seqs)
    };
    let equal = rng.chance(50);
    let base_len = rng.between(15, max_len);
    let related = rng.chance(65);
    let parents: Vec<Vec<u8>> = (0..rng.between(1, 3))
        .map(|_| {
            let len = if equal {
                base_len
            } else {
                rng.between(15, max_len)
            };
            rng.sequence(len)
        })
        .collect();
    let divergence = [2, 5, 10, 20, 40][rng.below(5)];
    let mut seqs: Vec<Vec<u8>> = Vec::with_capacity(n);
    for _ in 0..n {
        if !related || rng.chance(15) {
            let len = if equal {
                base_len
            } else {
                rng.between(15, max_len)
            };
            seqs.push(rng.sequence(len));
            continue;
        }
        let mut seq = parents[rng.below(parents.len())].clone();
        for position in 0..seq.len() {
            if rng.chance(divergence) {
                seq[position] = rng.pick(b"ACGT");
            }
        }
        if !equal {
            for _ in 0..rng.below(4) {
                let at = rng.below(seq.len());
                if rng.chance(50) && seq.len() > 18 {
                    let end = (at + rng.between(1, 3)).min(seq.len());
                    seq.drain(at..end);
                } else {
                    for _ in 0..rng.between(1, 3) {
                        seq.insert(at, rng.pick(b"ACGT"));
                    }
                }
            }
        }
        if rng.chance(15) {
            let shift = rng.between(1, 6);
            seq.rotate_left(shift);
            let len = seq.len();
            for letter in &mut seq[len - shift..] {
                *letter = rng.pick(b"ACGT");
            }
        }
        seqs.push(seq);
    }
    if rng.chance(40) {
        for _ in 0..rng.between(1, (n / 3).max(1)) {
            let copy = seqs[rng.below(n)].clone();
            seqs[rng.below(n)] = copy;
        }
    }
    if rng.chance(8) {
        for _ in 0..rng.between(1, 6) {
            let seq = rng.below(n);
            let at = rng.below(seqs[seq].len());
            seqs[seq][at] = rng.pick(b"NNNNRYKMSWBDHVX");
        }
    } else if rng.chance(3) {
        // Enough of them to tip MUSCLE's guess of the alphabet.
        let percent = [10, 20, 25, 30, 50][rng.below(5)];
        for seq in &mut seqs {
            for letter in seq.iter_mut() {
                if rng.chance(percent) {
                    *letter = b'N';
                }
            }
        }
    }
    if rng.chance(8) {
        for seq in &mut seqs {
            for letter in seq.iter_mut() {
                if rng.chance(40) {
                    *letter = letter.to_ascii_lowercase();
                }
            }
        }
    }
    seqs
}

/// Compares with the MUSCLE 5.3 binary named in `MUSCLE5_BIN` on generated
/// inputs; does nothing without it.
#[test]
fn agrees_with_the_muscle_binary() {
    let Some(binary) = std::env::var_os("MUSCLE5_BIN") else {
        eprintln!("MUSCLE5_BIN is not set: not comparing with the MUSCLE binary");
        return;
    };
    let number = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let cases = number("MUSCLE5_CASES", 300);
    let mut rng = TestRng(number("MUSCLE5_SEED", 1));
    let dir = tempfile::tempdir().expect("temporary directory");
    let fasta = dir.path().join("in.fa");
    let aligned = dir.path().join("out.afa");
    let mut mismatches = 0;
    for case in 0..cases {
        let mut input = match case % 20 {
            0 => random_input(&mut rng, 80, 80),
            1..=4 => random_input(&mut rng, 40, 80),
            _ => random_input(&mut rng, 12, 80),
        };
        if case % 16 == 7 {
            // One sequence much longer than the others.
            let long = rng.between(200, 1200);
            let slot = rng.below(input.len());
            input[slot] = rng.sequence(long);
            input.truncate(12);
        }
        let mut text = String::new();
        for (index, seq) in input.iter().enumerate() {
            text.push_str(&format!(">{index}\n{}\n", String::from_utf8_lossy(seq)));
        }
        std::fs::write(&fasta, text).expect("write the input");
        let _ = std::fs::remove_file(&aligned);
        let status = std::process::Command::new(&binary)
            .arg("-align")
            .arg(&fasta)
            .arg("-output")
            .arg(&aligned)
            .arg("-quiet")
            .status()
            .expect("run the MUSCLE binary");
        assert!(status.success(), "MUSCLE failed on case {case}");
        let output = std::fs::read_to_string(&aligned).expect("read the alignment");
        let mut expected = vec![Vec::new(); input.len()];
        let mut current = usize::MAX;
        for line in output.lines() {
            if let Some(label) = line.strip_prefix('>') {
                current = label.trim().parse().expect("records are named by index");
            } else {
                expected[current].extend_from_slice(line.trim().as_bytes());
            }
        }
        let refs: Vec<&[u8]> = input.iter().map(Vec::as_slice).collect();
        if align(&refs) != expected {
            mismatches += 1;
            eprintln!("case {case} differs from MUSCLE:");
            for seq in &input {
                eprintln!("  {}", String::from_utf8_lossy(seq));
            }
        }
    }
    assert_eq!(mismatches, 0, "of {cases} inputs");
}
