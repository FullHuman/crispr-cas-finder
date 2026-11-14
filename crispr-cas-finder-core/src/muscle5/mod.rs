//! Multiple sequence alignment as MUSCLE 5.3 `-align` computes it.
//!
//! This is a port to Rust of the `-align` command of MUSCLE 5.3 (commit
//! 6d49a2f) by Robert C. Edgar, <https://github.com/rcedgar/muscle>,
//! released under the GNU General Public License version 3. The original
//! CRISPRCasFinder pipeline shells out to that program to align the spacers
//! and the repeats of each locus; [`align`] returns the same alignment
//! without the program.
//!
//! "The same" is meant literally: for every input tried, the rows are
//! those the released binary `muscle 5.3.osxarm64 [6d49a2f]` writes for
//! `muscle -align in.fa -output out.afa`. MUSCLE's alignment is the outcome
//! of single-precision sums whose last bit can decide a tie, so the port
//! performs the binary's floating-point operations one for one: the
//! operations of the C++ source, with the changes the compiler made to them
//! (GCC 11 with `-O3 -ffast-math` fused multiplications into additions,
//! turned a division into a multiplication by a reciprocal and reordered
//! one sum; these were read off the binary's machine code), the pair-HMM
//! scores the binary computes at start-up ([`tables`]), and the `expf` and
//! `rand` of the C library it runs against ([`expf`], [`LibcRand`]).
//!
//! The stages, and the MUSCLE source files each follows:
//!
//! - reading: sequence.cpp, multisequence.cpp (`GuessIsNucleo`, which draws
//!   on the generator of myutils.cpp), alpha.cpp, alpha3.cpp;
//! - dereplication: derep.cpp;
//! - pair HMM: hmmparams.cpp, defaulthmmparams.cpp, pairhmm.cpp,
//!   scoretype.h;
//! - posterior of every pair: fwdflat3.cpp, bwdflat3.cpp, totalprobflat.cpp,
//!   calcposteriorflat.cpp, mysparsemx.cpp, calcalnscoreflat.cpp;
//! - guide tree: upgma5.cpp, tree.cpp, guidetreejoinorder.cpp;
//! - consistency: consflat.cpp, conspairflat.cpp, relaxflat.cpp;
//! - progressive alignment: progalnflat.cpp, alnalnsflat.cpp,
//!   buildpostflat.cpp, calcalnflat.cpp, tracebackflat.cpp;
//! - refinement: refineflat.cpp, project.cpp;
//! - the whole: align.cpp, mpcflat.cpp.
//!
//! MUSCLE runs the same method whatever the number of sequences (it only
//! warns above 1000). The time grows with the square of the number of
//! distinct sequences times the product of their lengths (posteriors) plus
//! the cube of that number times their length (consistency).
//!
//! With enough sequences the work is spread over the rayon thread pool, and
//! the result is the same on any number of threads, because every sum is
//! still made of the same terms in the same order: the posterior of a pair
//! depends on nothing else; a consistency round reads only the matrices of
//! the round before (MUSCLE spreads these two stages over its threads the
//! same way); and the sum of posteriors that aligns two alignments, in the
//! progressive stage and in each round of refinement, is split by rows of
//! the sum. Small inputs stay on the calling thread.

#![allow(clippy::needless_range_loop)]

mod expf;
mod tables;
#[cfg(test)]
mod tests;

use expf::expf;
use rayon::prelude::*;
use std::collections::HashMap;
use std::simd::prelude::*;
use std::simd::{StdFloat, f64x4, u64x4};

const M: usize = 0;
const IX: usize = 1;
const IY: usize = 2;
const JX: usize = 3;
const JY: usize = 4;
const STATES: usize = 5;

const LOG_ZERO: f32 = -2e20;
const LOG_UNDERFLOW_THRESHOLD: f32 = 7.5;
/// `MIN_SPARSE_PROB` of mysparsemx.h.
const MIN_SPARSE_PROB: f32 = 0.01;
/// `logf(MIN_SPARSE_PROB)`.
const MIN_SPARSE_SCORE: f32 = f32::from_bits(0xc093_5d8e);
const CONSISTENCY_ITERS: usize = 2;
const REFINE_ITERS: usize = 100;

/// On the thread pool, a task of the posterior stage takes pairs of sequences
/// whose forward and backward matrices have about this many cells in all, and
/// a consistency task takes one Y and `CONSISTENCY_BLOCK` X: small enough
/// that the cores share the work evenly whatever their speeds (and, for the
/// consistency, that the sums of a task fit in the first-level cache).
const POSTERIOR_TASK_CELLS: usize = 10_000;
const CONSISTENCY_BLOCK: usize = 24;

/// Whether the target multiplies and adds with a single rounding in one
/// instruction; otherwise the same result is computed in double precision.
const HARDWARE_FMA: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// The choices that change how the alignment is computed and not what it is.
#[derive(Clone, Copy)]
struct Tuning {
    /// Compute the consistency sums in double precision, as on a target
    /// without the fused multiply-add.
    soft_fma: bool,
    /// The posteriors go to the thread pool from this many cells of forward
    /// and backward matrix in all (the sum over the pairs of the product of
    /// the two lengths); below, the pool would not shorten the few
    /// milliseconds they take.
    parallel_posterior_cells: usize,
    /// The consistency rounds go to the pool from this number of sequences
    /// cubed times their mean length.
    parallel_consistency_work: usize,
    /// The sum of posteriors that aligns two alignments goes to the pool
    /// from this many pairs of sequences times their mean length.
    parallel_sum_work: usize,
}

impl Tuning {
    const DEFAULT: Self = Self {
        soft_fma: !HARDWARE_FMA,
        parallel_posterior_cells: 40_000,
        parallel_consistency_work: 60_000,
        parallel_sum_work: 200_000,
    };
}

/// Rows of the multiple alignment MUSCLE 5.3 `-align` gives for these sequences,
/// in the order the sequences were given, uppercase, padded with b'-'.
///
/// As MUSCLE does when it reads a FASTA file, letters are upper-cased and
/// white space, `-` and `.` are dropped. MUSCLE decides by itself whether
/// the sequences are nucleotides (it samples 100 letters and wants more than
/// 75 of them to be A, C, G, T or U); sequences with many other letters are
/// therefore aligned with its amino-acid scores, here as there. Zero or one
/// sequence comes back as it is, and so do sequences that are all the same.
///
/// The order of the sequences matters, as it does to MUSCLE: it decides
/// ties, the order of sums and the random splits of the refinement, and the
/// same sequences in another order now and then align differently. So
/// does the C library: the refinement draws on `rand()`, and this follows
/// the macOS one, like the binary it was checked against.
///
/// MUSCLE crashes on an empty sequence. Here empty sequences are set aside,
/// the others are aligned, and each empty one gets a row of gaps. MUSCLE
/// refuses a pair of sequences whose lengths multiply to more than about
/// 400 million (21,000 by 21,000); nothing is refused here, the memory
/// taken grows with that product.
pub fn align(sequences: &[&[u8]]) -> Vec<Vec<u8>> {
    align_with(sequences, Tuning::DEFAULT)
}

/// [`align`], computed as `tuning` says (the result is the same whatever it
/// says).
fn align_with(sequences: &[&[u8]], tuning: Tuning) -> Vec<Vec<u8>> {
    let input: Vec<Vec<u8>> = sequences.iter().map(|s| read_sequence(s)).collect();
    let filled: Vec<&[u8]> = input
        .iter()
        .filter(|seq| !seq.is_empty())
        .map(Vec::as_slice)
        .collect();
    let mut rows = align_filled(&filled, tuning).into_iter();
    let width = rows.as_slice().first().map_or(0, Vec::len);
    input
        .iter()
        .map(|seq| {
            if seq.is_empty() {
                vec![b'-'; width]
            } else {
                rows.next().expect("one row per sequence")
            }
        })
        .collect()
}

/// mpcflat.cpp `MPCFlat::Run` for sequences that are upper-case and not
/// empty.
fn align_filled(input: &[&[u8]], tuning: Tuning) -> Vec<Vec<u8>> {
    if input.len() < 2 {
        return input.iter().map(|seq| seq.to_vec()).collect();
    }

    // derep.cpp: the first copy of each distinct sequence stands for the
    // others, which get its row.
    let mut slot_of: HashMap<&[u8], usize> = HashMap::new();
    let mut unique: Vec<&[u8]> = Vec::new();
    let slots: Vec<usize> = input
        .iter()
        .map(|&seq| {
            *slot_of.entry(seq).or_insert_with(|| {
                unique.push(seq);
                unique.len() - 1
            })
        })
        .collect();
    if unique.len() == 1 {
        return input.iter().map(|seq| seq.to_vec()).collect();
    }

    let hmm = PairHmm::new(guess_is_nucleo(input));
    let rows = Mpc::new(&hmm, &unique, tuning).run();
    slots.iter().map(|&slot| rows[slot].clone()).collect()
}

/// sequence.cpp `Sequence::FromFileBuffer` with `stripGaps`: white space and
/// the gap characters are dropped, letters are upper-cased.
fn read_sequence(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .filter(|&&c| !(c == b'-' || c == b'.' || c == b' ' || (9..=13).contains(&c)))
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// myutils.cpp `randu32`: Marsaglia multiply-with-carry seeded through a
/// linear congruential generator, default seed 1.
struct Mwc {
    x: [u32; 5],
}

impl Mwc {
    fn new(seed: u32) -> Self {
        let mut state = seed;
        let mut slcg = || {
            state = state.wrapping_mul(214_013).wrapping_add(2_531_011);
            state
        };
        for _ in 0..10 {
            slcg();
        }
        let mut x = [0u32; 5];
        for value in &mut x {
            *value = slcg();
        }
        let mut rng = Self { x };
        for _ in 0..100 {
            rng.increment();
        }
        rng
    }

    fn increment(&mut self) {
        let sum = 2_111_111_111u64 * u64::from(self.x[3])
            + 1492 * u64::from(self.x[2])
            + 1776 * u64::from(self.x[1])
            + 5115 * u64::from(self.x[0])
            + u64::from(self.x[4]);
        self.x[3] = self.x[2];
        self.x[2] = self.x[1];
        self.x[1] = self.x[0];
        self.x[4] = (sum >> 32) as u32;
        self.x[0] = sum as u32;
    }

    fn next(&mut self) -> u32 {
        self.increment();
        self.x[0]
    }
}

/// The C library `rand()` the macOS binary links to (Park and Miller's
/// minimal standard generator, never seeded, so it starts from 1). MUSCLE
/// uses it to split the alignment in two during refinement (refineflat.cpp).
/// A MUSCLE built against another C library draws other numbers (glibc's
/// `rand()` is a different generator) and refines differently.
struct LibcRand {
    state: u64,
}

impl LibcRand {
    fn new() -> Self {
        Self { state: 1 }
    }

    fn next(&mut self) -> u32 {
        let hi = (self.state / 127_773) as i64;
        let lo = (self.state % 127_773) as i64;
        let mut x = 16_807 * lo - 2_836 * hi;
        if x < 0 {
            x += 0x7fff_ffff;
        }
        self.state = x as u64;
        (x as u64 % 0x8000_0000) as u32
    }
}

/// multisequence.cpp `MultiSequence::GuessIsNucleo`: 100 letters drawn at
/// random, nucleotide when more than 75 of them are A, C, G, T or U.
fn guess_is_nucleo(seqs: &[&[u8]]) -> bool {
    let mut rng = Mwc::new(1);
    let mut nucleo = 0;
    for _ in 0..100 {
        let seq = seqs[rng.next() as usize % seqs.len()];
        let letter = seq[rng.next() as usize % seq.len()];
        if matches!(letter, b'A' | b'C' | b'G' | b'T' | b'U') {
            nucleo += 1;
        }
    }
    nucleo > 75
}

/// pairhmm.cpp / hmmparams.cpp: the pair HMM scores (natural logarithms).
struct PairHmm {
    t_sm: f32,
    t_si: f32,
    t_sj: f32,
    t_mm: f32,
    t_mi: f32,
    t_mj: f32,
    t_ii: f32,
    t_im: f32,
    t_jj: f32,
    t_jm: f32,
    /// Letter class of each byte; the last class is the wildcard.
    class: [u8; 256],
    classes: usize,
    ins: Vec<f32>,
    emit: Vec<f32>,
}

impl PairHmm {
    fn new(nucleo: bool) -> Self {
        let f = f32::from_bits;
        // The letters with scores of their own; every other byte is scored
        // as a wildcard, the last class. `PairHMM::FixUT` gives U the
        // nucleotide scores of T.
        let letters: &[u8] = if nucleo {
            b"ACGT"
        } else {
            tables::AMINO_LETTERS
        };
        let wild = letters.len();
        let classes = wild + 1;
        let mut class = [wild as u8; 256];
        for (code, &letter) in letters.iter().enumerate() {
            class[letter as usize] = code as u8;
            class[letter.to_ascii_lowercase() as usize] = code as u8;
        }
        let mut ins;
        let mut emit;
        if nucleo {
            class[b'U' as usize] = class[b'T' as usize];
            class[b'u' as usize] = class[b'T' as usize];
            ins = vec![f(tables::NUCLEO_INS); classes];
            ins[wild] = f(tables::NUCLEO_INS_WILDCARD);
            emit = vec![f(tables::NUCLEO_EMIT_WILDCARD); classes * classes];
            for a in 0..wild {
                for b in 0..wild {
                    emit[a * classes + b] = f(if a == b {
                        tables::NUCLEO_MATCH
                    } else {
                        tables::NUCLEO_MISMATCH
                    });
                }
            }
        } else {
            ins = vec![f(tables::AMINO_INS_WILDCARD); classes];
            emit = vec![f(tables::AMINO_EMIT_WILDCARD); classes * classes];
            let mut next = tables::AMINO_EMIT.iter();
            for a in 0..wild {
                ins[a] = f(tables::AMINO_INS[a]);
                for b in 0..=a {
                    let score = f(*next.next().expect("210 scores"));
                    emit[a * classes + b] = score;
                    emit[b * classes + a] = score;
                }
            }
        }
        Self {
            t_sm: f(tables::START[0]),
            t_si: f(tables::START[1]),
            t_sj: f(tables::START[2]),
            t_mm: f(tables::FROM_MATCH[0]),
            t_mi: f(tables::FROM_MATCH[1]),
            t_mj: f(tables::FROM_MATCH[2]),
            t_ii: f(tables::SHORT_GAP[0]),
            t_im: f(tables::SHORT_GAP[1]),
            t_jj: f(tables::LONG_GAP[0]),
            t_jm: f(tables::LONG_GAP[1]),
            class,
            classes,
            ins,
            emit,
        }
    }

    fn encode(&self, seq: &[u8]) -> Vec<u8> {
        seq.iter().map(|&c| self.class[c as usize]).collect()
    }

    #[inline(always)]
    fn emit(&self, a: u8, b: u8) -> f32 {
        self.emit[a as usize * self.classes + b as usize]
    }
}

/// The cubics of scoretype.h `LOGEXP1`, log(exp(x) + 1) for x up to 1,
/// 2.5, 4.5 and 7.5: coefficients from the highest power down.
const LOGEXP1: [[u32; 4]; 4] = [
    [0xbc19_343d, 0x3e05_cb9c, 0x3eff_62b0, 0x3f31_75c2],
    [0xbc6e_18fa, 0x3e0f_4d0a, 0x3efd_c3f0, 0x3f31_3020],
    [0xbb96_e5ce, 0x3d81_e63c, 0x3f32_2a34, 0x3f03_a75f],
    [0xb9f0_7885, 0x3c1e_dbbf, 0x3f6e_44a1, 0x3e2c_11ef],
];

/// scoretype.h `LOG_ADD`: log(exp(x) + exp(y)), the larger of the two when
/// the other is `LOG_ZERO` or 7.5 below it, otherwise the smaller plus
/// `LOGEXP1` of their difference. The binary evaluates the cubic with three
/// fused multiply-adds. Written without branches (the cubic is evaluated
/// whether or not its value is used), which changes no result.
#[inline(always)]
fn log_add(x: f32, y: f32) -> f32 {
    let (small, large) = if x < y { (x, y) } else { (y, x) };
    let diff = large - small;
    let range = usize::from(diff > 1.0) + usize::from(diff > 2.5) + usize::from(diff > 4.5);
    let [a, b, c, d] = LOGEXP1[range].map(f32::from_bits);
    let sum = fma(diff, fma(diff, fma(diff, a, b), c), d) + small;
    if small == LOG_ZERO || diff >= LOG_UNDERFLOW_THRESHOLD {
        large
    } else {
        sum
    }
}

/// fwdflat3.cpp `CalcFwdFlat`.
fn calc_fwd(hmm: &PairHmm, x: &[u8], y: &[u8], flat: &mut [f32]) {
    let lx = x.len();
    let ly = y.len();
    let ly1 = ly + 1;
    let inc_i = STATES * ly1;
    let inc_j = STATES;
    let base_1_1 = inc_i + inc_j;
    let base_1_0 = inc_i;
    let base_0_1 = inc_j;

    let ins_x0 = hmm.ins[x[0] as usize];
    let ins_y0 = hmm.ins[y[0] as usize];
    let start_m = hmm.t_sm + hmm.emit(x[0], y[0]);

    flat[..STATES].fill(LOG_ZERO);
    flat[base_1_1 + M] = start_m;
    flat[base_1_0 + IX] = hmm.t_si + ins_x0;
    flat[base_1_0 + JX] = hmm.t_sj + ins_x0;
    flat[base_0_1 + IY] = hmm.t_si + ins_y0;
    flat[base_0_1 + JY] = hmm.t_sj + ins_y0;

    for i in 1..=lx {
        let base = i * inc_i;
        flat[base + M] = LOG_ZERO;
        flat[base + IY] = LOG_ZERO;
        flat[base + JY] = LOG_ZERO;
    }
    for j in 1..=ly {
        let base = j * inc_j;
        flat[base + M] = LOG_ZERO;
        flat[base + IX] = LOG_ZERO;
        flat[base + JX] = LOG_ZERO;
    }

    // The binary adds the emission to the previous cell first and the
    // transition second (the source has them the other way round).
    for i in 1..lx {
        let base = i * inc_i;
        let next = base + inc_i;
        let emit = hmm.ins[x[i] as usize];
        flat[next + IX] = (emit + flat[base + IX]) + hmm.t_ii;
        flat[next + JX] = (emit + flat[base + JX]) + hmm.t_jj;
    }
    for j in 1..ly {
        let base = j * inc_j;
        let next = base + inc_j;
        let emit = hmm.ins[y[j] as usize];
        flat[next + IY] = (emit + flat[base + IY]) + hmm.t_ii;
        flat[next + JY] = (emit + flat[base + JY]) + hmm.t_jj;
    }

    for i in 1..=lx {
        let xc = x[i - 1];
        let emit_x = hmm.ins[xc as usize];
        for j in 1..=ly {
            let yc = y[j - 1];
            let emit_y = hmm.ins[yc as usize];
            let ij = STATES * (i * ly1 + j);
            let i1j = ij - inc_i;
            let ij1 = ij - inc_j;
            let i1j1 = i1j - inc_j;
            if i == 1 && j == 1 {
                flat[ij + M] = start_m;
            } else {
                let m_m = hmm.t_mm + flat[i1j1 + M];
                let ix_m = hmm.t_im + flat[i1j1 + IX];
                let jx_m = hmm.t_jm + flat[i1j1 + JX];
                let iy_m = hmm.t_im + flat[i1j1 + IY];
                let jy_m = hmm.t_jm + flat[i1j1 + JY];
                let sum = log_add(m_m, log_add(ix_m, log_add(jx_m, log_add(iy_m, jy_m))));
                flat[ij + M] = hmm.emit(xc, yc) + sum;
            }
            let prev_m_i1_j = flat[i1j + M];
            let prev_m_i_j1 = flat[ij1 + M];
            flat[ij + IX] = emit_x + log_add(hmm.t_ii + flat[i1j + IX], hmm.t_mi + prev_m_i1_j);
            flat[ij + JX] = emit_x + log_add(hmm.t_jj + flat[i1j + JX], hmm.t_mj + prev_m_i1_j);
            flat[ij + IY] = emit_y + log_add(hmm.t_ii + flat[ij1 + IY], hmm.t_mi + prev_m_i_j1);
            flat[ij + JY] = emit_y + log_add(hmm.t_jj + flat[ij1 + JY], hmm.t_mj + prev_m_i_j1);
        }
    }
}

/// bwdflat3.cpp `CalcBwdFlat`.
fn calc_bwd(hmm: &PairHmm, x: &[u8], y: &[u8], flat: &mut [f32]) {
    let lx = x.len();
    let ly = y.len();
    let ly1 = ly + 1;
    let inc_i = STATES * ly1;
    let inc_j = STATES;

    for i in 0..lx {
        let base = STATES * (i * ly1 + ly);
        flat[base + IY] = LOG_ZERO;
        flat[base + JY] = LOG_ZERO;
    }
    for j in 0..ly {
        let base = STATES * (lx * ly1 + j);
        flat[base + IX] = LOG_ZERO;
        flat[base + JX] = LOG_ZERO;
    }

    for i in (0..=lx).rev() {
        let emit_x = if i == lx { 0.0 } else { hmm.ins[x[i] as usize] };
        for j in (0..=ly).rev() {
            let ij = STATES * (i * ly1 + j);
            let i1j = ij + inc_i;
            let ij1 = ij + inc_j;
            let i1j1 = i1j + inc_j;
            if i == lx && j == ly {
                flat[ij + M] = hmm.t_sm;
                flat[ij + IX] = hmm.t_si;
                flat[ij + IY] = hmm.t_si;
                flat[ij + JX] = hmm.t_sj;
                flat[ij + JY] = hmm.t_sj;
                continue;
            }
            let emit_y = if j == ly { 0.0 } else { hmm.ins[y[j] as usize] };

            if i < lx && j < ly {
                let next_m = flat[i1j1 + M] + hmm.emit(x[i], y[j]);
                let next_ix = flat[i1j + IX] + emit_x;
                let next_jx = flat[i1j + JX] + emit_x;
                let next_iy = flat[ij1 + IY] + emit_y;
                let next_jy = flat[ij1 + JY] + emit_y;

                flat[ij + M] = if i > 0 && j > 0 {
                    let m_m = hmm.t_mm + next_m;
                    let m_ix = hmm.t_mi + next_ix;
                    let m_jx = hmm.t_mj + next_jx;
                    let m_iy = hmm.t_mi + next_iy;
                    let m_jy = hmm.t_mj + next_jy;
                    log_add(m_m, log_add(m_ix, log_add(m_jx, log_add(m_iy, m_jy))))
                } else {
                    LOG_ZERO
                };
                if i > 0 {
                    flat[ij + IX] = log_add(hmm.t_ii + next_ix, hmm.t_im + next_m);
                    flat[ij + JX] = log_add(hmm.t_jj + next_jx, hmm.t_jm + next_m);
                } else {
                    flat[ij + IX] = LOG_ZERO;
                    flat[ij + JX] = LOG_ZERO;
                }
                if j > 0 {
                    flat[ij + IY] = log_add(hmm.t_ii + next_iy, hmm.t_im + next_m);
                    flat[ij + JY] = log_add(hmm.t_jj + next_jy, hmm.t_jm + next_m);
                } else {
                    flat[ij + IY] = LOG_ZERO;
                    flat[ij + JY] = LOG_ZERO;
                }
                continue;
            }

            if i < lx {
                if i > 0 {
                    let next_ix = flat[i1j + IX] + emit_x;
                    let next_jx = flat[i1j + JX] + emit_x;
                    flat[ij + M] = log_add(hmm.t_mi + next_ix, hmm.t_mj + next_jx);
                    flat[ij + IX] = hmm.t_ii + next_ix;
                    flat[ij + JX] = hmm.t_jj + next_jx;
                } else {
                    flat[ij + M] = LOG_ZERO;
                    flat[ij + IX] = LOG_ZERO;
                    flat[ij + JX] = LOG_ZERO;
                }
            }
            if j < ly {
                if j > 0 {
                    let next_iy = flat[ij1 + IY] + emit_y;
                    let next_jy = flat[ij1 + JY] + emit_y;
                    flat[ij + M] = log_add(hmm.t_mi + next_iy, hmm.t_mj + next_jy);
                    flat[ij + IY] = hmm.t_ii + next_iy;
                    flat[ij + JY] = hmm.t_jj + next_jy;
                } else {
                    flat[ij + M] = LOG_ZERO;
                    flat[ij + IY] = LOG_ZERO;
                    flat[ij + JY] = LOG_ZERO;
                }
            }
        }
    }
}

/// totalprobflat.cpp `CalcTotalProbFlat`.
fn calc_total_prob(fwd: &[f32], bwd: &[f32], lx: usize, ly: usize) -> f32 {
    let base = STATES * (lx * (ly + 1) + ly);
    let mut sum = LOG_ZERO;
    for s in 0..STATES {
        sum = log_add(sum, fwd[base + s] + bwd[base + s]);
    }
    sum
}

/// calcposteriorflat.cpp `CalcPostFlat`.
fn calc_post(fwd: &[f32], bwd: &[f32], lx: usize, ly: usize, post: &mut [f32]) {
    let total = calc_total_prob(fwd, bwd, lx, ly);
    let ly1 = ly + 1;
    for i in 0..lx {
        for j in 0..ly {
            let cell = STATES * ((i + 1) * ly1 + j + 1);
            let score = (fwd[cell] + bwd[cell]) - total;
            post[i * ly + j] = if score < MIN_SPARSE_SCORE {
                0.0
            } else if score >= 0.0 {
                1.0
            } else {
                expf(score)
            };
        }
    }
}

/// Working memory for the posterior of one pair, kept from one pair to the
/// next.
#[derive(Default)]
struct PairBuffers {
    fwd: Vec<f32>,
    bwd: Vec<f32>,
    post: Vec<f32>,
    dp_row: Vec<f32>,
}

impl PairBuffers {
    /// calcposteriorflat.cpp `MPCFlat::CalcPosterior`: the sparse posterior
    /// of two encoded sequences and the expected accuracy of their
    /// alignment.
    fn posterior(&mut self, hmm: &PairHmm, x: &[u8], y: &[u8]) -> (SparseMx, f32) {
        let (lx, ly) = (x.len(), y.len());
        let grow = |buffer: &mut Vec<f32>, len: usize| {
            if buffer.len() < len {
                buffer.resize(len, 0.0);
            }
        };
        grow(&mut self.fwd, STATES * (lx + 1) * (ly + 1));
        grow(&mut self.bwd, STATES * (lx + 1) * (ly + 1));
        grow(&mut self.post, lx * ly);
        grow(&mut self.dp_row, ly + 1);
        calc_fwd(hmm, x, y, &mut self.fwd);
        calc_bwd(hmm, x, y, &mut self.bwd);
        calc_post(&self.fwd, &self.bwd, lx, ly, &mut self.post);
        let score = calc_aln_score(&self.post, lx, ly, &mut self.dp_row);
        let ea = score / lx.min(ly) as f32;
        (SparseMx::from_post(&self.post, lx, ly), ea)
    }
}

/// mysparsemx.h `MySparseMx`: the cells of a posterior matrix that reached
/// `MIN_SPARSE_PROB`, row by row.
#[derive(Clone, Default)]
struct SparseMx {
    offsets: Vec<u32>,
    cols: Vec<u32>,
    probs: Vec<f32>,
}

impl SparseMx {
    /// `MySparseMx::FromPost`.
    fn from_post(post: &[f32], lx: usize, ly: usize) -> Self {
        let cells = post[..lx * ly]
            .iter()
            .filter(|&&p| p >= MIN_SPARSE_PROB)
            .count();
        let mut mx = Self {
            offsets: Vec::with_capacity(lx + 1),
            cols: Vec::with_capacity(cells),
            probs: Vec::with_capacity(cells),
        };
        for i in 0..lx {
            mx.offsets.push(mx.cols.len() as u32);
            for (j, &p) in post[i * ly..(i + 1) * ly].iter().enumerate() {
                if p >= MIN_SPARSE_PROB {
                    mx.cols.push(j as u32);
                    mx.probs.push(p);
                }
            }
        }
        mx.offsets.push(mx.cols.len() as u32);
        mx
    }

    fn rows(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline(always)]
    fn row(&self, i: usize) -> std::ops::Range<usize> {
        self.offsets[i] as usize..self.offsets[i + 1] as usize
    }

    /// First column and one past the last column of row `i` (0, 0 if empty).
    #[inline(always)]
    fn span(&self, i: usize) -> (usize, usize) {
        let row = self.row(i);
        if row.is_empty() {
            (0, 0)
        } else {
            (
                self.cols[row.start] as usize,
                self.cols[row.end - 1] as usize + 1,
            )
        }
    }
}

/// a * b + c with a single rounding, as the `fmadd` instruction computes it.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    if HARDWARE_FMA {
        a.mul_add(b, c)
    } else {
        soft_fma(a, b, c)
    }
}

/// [`fma`] without the instruction. The product of two `f32` is exact in an
/// `f64`; adding the third and converting rounds twice, which gives the
/// wrong `f32` only when the `f64` sum is inexact and lies exactly half way
/// between two `f32`, and then the error of the sum says which of the two
/// is right. For finite operands and a result that neither overflows nor is
/// subnormal, which is all this module produces.
#[inline(always)]
fn soft_fma(a: f32, b: f32, c: f32) -> f32 {
    let xy = f64::from(a) * f64::from(b);
    let z = f64::from(c);
    let result = xy + z;
    let bits = result.to_bits();
    if bits & 0x1fff_ffff != 0x1000_0000 || (result - xy == z && result - z == xy) {
        return result as f32;
    }
    let negative = bits >> 63 != 0;
    let err = if negative == (z > xy) {
        xy - result + z
    } else {
        z - result + xy
    };
    let adjusted = if negative == (err < 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(adjusted) as f32
}

/// Cells of a consistency sum handled together.
const WINDOW: usize = 16;

/// P(z~y) for one Z and one Y, dense: row z holds columns 0..len(Y) and
/// then `WINDOW` zeros, so that a window starting at any column can be read.
#[derive(Default)]
struct DenseRows {
    stride: usize,
    values: Vec<f32>,
    /// First column and one past the last column of each row; the first is
    /// not below the last for a row without cells.
    span: Vec<(u32, u32)>,
}

impl DenseRows {
    /// From the posterior of (Z, Y).
    fn fill(&mut self, mx: &SparseMx, ly: usize) {
        let rows = mx.rows();
        self.stride = ly + WINDOW;
        self.values.clear();
        self.values.resize(rows * self.stride, 0.0);
        self.span.clear();
        for i in 0..rows {
            let (lo, hi) = mx.span(i);
            self.span.push((lo as u32, hi as u32));
            for k in mx.row(i) {
                self.values[i * self.stride + mx.cols[k] as usize] = mx.probs[k];
            }
        }
    }

    /// From the posterior of (Y, Z), where Z has `lz` letters.
    fn fill_transposed(&mut self, mx: &SparseMx, lz: usize) {
        let ly = mx.rows();
        self.stride = ly + WINDOW;
        self.values.clear();
        self.values.resize(lz * self.stride, 0.0);
        self.span.clear();
        self.span.resize(lz, (u32::MAX, 0));
        for i in 0..ly {
            for k in mx.row(i) {
                let col = mx.cols[k] as usize;
                self.values[col * self.stride + i] = mx.probs[k];
                let span = &mut self.span[col];
                span.0 = span.0.min(i as u32);
                span.1 = i as u32 + 1;
            }
        }
    }

    fn row(&self, i: usize) -> &[f32] {
        &self.values[i * self.stride..(i + 1) * self.stride]
    }
}

/// The sums of one row of a pair (X, Y): `cells` holds the columns
/// `lo..hi` (the row's first to last cell) and what is needed to fill whole
/// windows.
struct SumRow<'a> {
    cells: &'a mut [f32],
    lo: usize,
    hi: usize,
}

impl SumRow<'_> {
    /// The row takes the terms of one Z: for each cell z of the row of
    /// P(x~z) (`cols`, `probs`), probability * row z of P(z~y), fused
    /// (relaxflat.cpp `RelaxFlat_XZ_ZY`, `RelaxFlat_XZ_YZ`). A term whose
    /// P(z~y) is zero leaves the sum as it is, and columns from `hi` on are
    /// never read, so whole windows are taken.
    #[inline(always)]
    fn add_terms(&mut self, cols: &[u32], probs: &[f32], zy: &DenseRows, soft: bool) {
        if soft {
            for (&pos_z, &p) in cols.iter().zip(probs) {
                self.add_term_soft(p, zy.row(pos_z as usize), zy.span[pos_z as usize]);
            }
            return;
        }
        let windows = self.cells.as_chunks_mut::<WINDOW>().0;
        for (window, cells) in windows.iter_mut().enumerate() {
            let column = self.lo + window * WINDOW;
            let mut sums = f32x16::from_array(*cells);
            for (&pos_z, &p) in cols.iter().zip(probs) {
                let source = &zy.row(pos_z as usize)[column..column + WINDOW];
                sums = f32x16::from_slice(source).mul_add(f32x16::splat(p), sums);
            }
            *cells = sums.to_array();
        }
    }

    /// The row takes one term, p * `source`, a row of P(z~y) whose cells lie
    /// in the columns `span` (relaxflat.cpp `RelaxFlat_ZX_ZY`).
    #[inline(always)]
    fn add_term(&mut self, p: f32, source: &[f32], span: (u32, u32), soft: bool) {
        if soft {
            self.add_term_soft(p, source, span);
            return;
        }
        let windows = self.cells.as_chunks_mut::<WINDOW>().0;
        for (window, cells) in windows.iter_mut().enumerate() {
            let column = self.lo + window * WINDOW;
            *cells = f32x16::from_slice(&source[column..column + WINDOW])
                .mul_add(f32x16::splat(p), f32x16::from_array(*cells))
                .to_array();
        }
    }

    /// [`Self::add_term`] without the instruction, over the columns where
    /// the term is not zero: [`soft_fma`], four cells at a time when none of
    /// the four sums is one of those the double rounding gets wrong.
    #[inline(always)]
    fn add_term_soft(&mut self, p: f32, source: &[f32], span: (u32, u32)) {
        let from = (span.0 as usize).max(self.lo);
        let to = (span.1 as usize).min(self.hi);
        if from >= to {
            return;
        }
        let (cells, last_cells) = self.cells[from - self.lo..to - self.lo].as_chunks_mut::<4>();
        let (source, last_source) = source[from..to].as_chunks::<4>();
        let p_wide = f64x4::splat(f64::from(p));
        for (cells, source) in cells.iter_mut().zip(source) {
            let sum = f32x4::from_array(*source).cast::<f64>() * p_wide
                + f32x4::from_array(*cells).cast();
            let half_way =
                (sum.to_bits() & u64x4::splat(0x1fff_ffff)).simd_eq(u64x4::splat(0x1000_0000));
            if half_way.any() {
                for (cell, &s) in cells.iter_mut().zip(source) {
                    *cell = soft_fma(p, s, *cell);
                }
            } else {
                *cells = sum.cast::<f32>().to_array();
            }
        }
        for (cell, &s) in last_cells.iter_mut().zip(last_source) {
            *cell = soft_fma(p, s, *cell);
        }
    }
}

/// The sums of the pairs (X, Y) for the current Y and a block of X.
#[derive(Default)]
struct Sums {
    values: Vec<f32>,
    /// Where each row starts in `values`; one more entry than rows.
    start: Vec<usize>,
    /// First column and one past the last column of each row.
    span: Vec<(u32, u32)>,
    /// The first row of each X of the block.
    first_row: Vec<usize>,
}

impl Sums {
    /// Starts the sums of the pairs (X, y), X in `xs`, at twice their
    /// posterior (the terms Z = X and Z = Y).
    fn start(&mut self, mpc: &Mpc, y: usize, xs: std::ops::Range<usize>) {
        self.values.clear();
        self.start.clear();
        self.span.clear();
        self.first_row.clear();
        for x in xs {
            self.first_row.push(self.span.len());
            let xy = &mpc.posts[mpc.pair_index(x, y)];
            for i in 0..xy.rows() {
                let (lo, hi) = xy.span(i);
                let start = self.values.len();
                self.start.push(start);
                self.span.push((lo as u32, hi as u32));
                self.values
                    .resize(start + (hi - lo).next_multiple_of(WINDOW), 0.0);
                for k in xy.row(i) {
                    self.values[start + xy.cols[k] as usize - lo] = xy.probs[k] * 2.0;
                }
            }
        }
        self.start.push(self.values.len());
    }

    #[inline(always)]
    fn row(&mut self, row: usize) -> SumRow<'_> {
        let (lo, hi) = self.span[row];
        SumRow {
            cells: &mut self.values[self.start[row]..self.start[row + 1]],
            lo: lo as usize,
            hi: hi as usize,
        }
    }
}

/// One cell of a posterior matrix.
#[derive(Clone, Copy)]
struct Cell {
    row: u32,
    col: u32,
    p: f32,
}

/// The final posteriors of every pair, cell after cell in one block of
/// memory: what the sums that align two alignments walk through, pair after
/// pair (`Mpc::build_post`).
#[derive(Default)]
struct Cells {
    cells: Vec<Cell>,
    /// Where the cells of each pair start; one more entry than pairs.
    start: Vec<usize>,
}

impl Cells {
    /// Takes the matrices, in the order of `Mpc::pair_index`, freeing each
    /// as it goes.
    fn from_posts(posts: Vec<SparseMx>) -> Self {
        let total = posts.iter().map(|mx| mx.cols.len()).sum();
        let mut all = Self {
            cells: Vec::with_capacity(total),
            start: Vec::with_capacity(posts.len() + 1),
        };
        for mx in posts {
            all.start.push(all.cells.len());
            for row in 0..mx.rows() {
                for k in mx.row(row) {
                    all.cells.push(Cell {
                        row: row as u32,
                        col: mx.cols[k],
                        p: mx.probs[k],
                    });
                }
            }
        }
        all.start.push(all.cells.len());
        all
    }

    fn pair(&self, pair: usize) -> &[Cell] {
        &self.cells[self.start[pair]..self.start[pair + 1]]
    }
}

/// One row of an alignment under construction: which input sequence it is
/// and its letters and gaps.
#[derive(Clone)]
struct Row {
    seq: usize,
    chars: Vec<u8>,
}

type Msa = Vec<Row>;

/// mpcflat.h `MPCFlat`: ProbCons-style alignment of the distinct sequences.
struct Mpc<'a> {
    hmm: &'a PairHmm,
    seqs: &'a [&'a [u8]],
    /// Posterior of pair (i, j), i < j, at `pair_index(i, j)`, until the
    /// consistency rounds are over.
    posts: Vec<SparseMx>,
    /// The same afterwards.
    cells: Cells,
    /// Expected accuracy of the alignment of each pair, n by n.
    dist: Vec<f32>,
    tuning: Tuning,
}

impl<'a> Mpc<'a> {
    fn new(hmm: &'a PairHmm, seqs: &'a [&'a [u8]], tuning: Tuning) -> Self {
        Self {
            hmm,
            seqs,
            posts: Vec::new(),
            cells: Cells::default(),
            dist: Vec::new(),
            tuning,
        }
    }

    fn pair_index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < j);
        let n = self.seqs.len();
        i * n - i * (i + 1) / 2 + (j - i - 1)
    }

    /// `MPCFlat::Run` after dereplication.
    fn run(mut self) -> Vec<Vec<u8>> {
        let n = self.seqs.len();
        self.calc_posteriors();
        let (left, right) = self.calc_guide_tree();
        if n >= 3 {
            for _ in 0..CONSISTENCY_ITERS {
                self.cons_iter();
            }
        }
        self.cells = Cells::from_posts(std::mem::take(&mut self.posts));
        let (joins1, joins2) = join_order(n, &left, &right);
        let mut msa = self.progressive_align(&joins1, &joins2);
        if n >= 3 {
            let mut rng = LibcRand::new();
            for _ in 0..REFINE_ITERS {
                self.refine_iter(&mut msa, &mut rng);
            }
        }
        let mut rows = vec![Vec::new(); n];
        for row in msa {
            rows[row.seq] = row.chars;
        }
        rows
    }

    /// `MPCFlat::CalcPosteriors`: the posterior and the expected accuracy of
    /// every pair, each independent of the others.
    fn calc_posteriors(&mut self) {
        let n = self.seqs.len();
        let codes: Vec<Vec<u8>> = self.seqs.iter().map(|s| self.hmm.encode(s)).collect();
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect();
        let cells: usize = pairs
            .iter()
            .map(|&(i, j)| codes[i].len() * codes[j].len())
            .sum();
        let hmm = self.hmm;
        let posterior = |buffers: &mut PairBuffers, &(i, j): &(usize, usize)| {
            buffers.posterior(hmm, &codes[i], &codes[j])
        };
        let results: Vec<(SparseMx, f32)> = if pairs.len() > 1
            && cells >= self.tuning.parallel_posterior_cells
            && rayon::current_num_threads() > 1
        {
            let pairs_per_task = (POSTERIOR_TASK_CELLS * pairs.len() / cells.max(1)).max(1);
            pairs
                .par_iter()
                .with_min_len(pairs_per_task)
                .with_max_len(2 * pairs_per_task)
                .map_init(PairBuffers::default, posterior)
                .collect()
        } else {
            let mut buffers = PairBuffers::default();
            pairs
                .iter()
                .map(|pair| posterior(&mut buffers, pair))
                .collect()
        };

        self.dist = vec![0f32; n * n];
        self.posts = Vec::with_capacity(pairs.len());
        for (&(i, j), (post, ea)) in pairs.iter().zip(results) {
            self.posts.push(post);
            self.dist[i * n + j] = ea;
            self.dist[j * n + i] = ea;
        }
    }

    /// `MPCFlat::CalcGuideTree`: `UPGMA5::FixEADistMx` then `UPGMA5::Run`
    /// with biased linkage. Returns the children of each internal node, in
    /// the order the nodes were made; leaves are 0..n, internal node k is
    /// n + k, and the last one is the root.
    fn calc_guide_tree(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.seqs.len();
        const NONE: usize = usize::MAX;
        let tri = |a: usize, b: usize| {
            if a >= b {
                b + a * (a.wrapping_sub(1)) / 2
            } else {
                a + b * (b - 1) / 2
            }
        };
        let mut dist = vec![0f32; n * (n - 1) / 2];
        let mut min_dist = vec![f32::MAX; n];
        let mut nearest = vec![NONE; n];
        let mut node_index: Vec<usize> = (0..n).collect();
        for i in 1..n {
            for j in 0..i {
                let mut d = 1.0 - self.dist[i * n + j];
                if d < 0.0 {
                    d = 0.0;
                }
                dist[tri(i, j)] = d;
                if d < min_dist[i] {
                    min_dist[i] = d;
                    nearest[i] = j;
                }
                if d < min_dist[j] {
                    min_dist[j] = d;
                    nearest[j] = i;
                }
            }
        }

        let mut left = Vec::with_capacity(n - 1);
        let mut right = Vec::with_capacity(n - 1);
        for internal in 0..n - 1 {
            let mut lmin = NONE;
            let mut rmin = NONE;
            let mut best = f32::MAX;
            for j in 0..n {
                if node_index[j] == NONE {
                    continue;
                }
                let d = min_dist[j];
                if d < best {
                    best = d;
                    lmin = j;
                    rmin = nearest[j];
                }
            }

            let mut new_min_dist = f32::MAX;
            let mut new_nearest = NONE;
            for j in 0..n {
                if j == lmin || j == rmin || node_index[j] == NONE {
                    continue;
                }
                let v_l = tri(lmin, j);
                let v_r = tri(rmin, j);
                let d_l = dist[v_l];
                let d_r = dist[v_r];
                // LINKAGE_Biased, 0.1*AVG(dL, dR) + 0.9*min(dL, dR), as the
                // binary evaluates it.
                let new_dist = fma(
                    d_l + d_r,
                    f32::from_bits(0x3d4c_cccd),
                    d_l.min(d_r) * f32::from_bits(0x3f66_6666),
                );
                if nearest[j] == rmin {
                    nearest[j] = lmin;
                }
                dist[v_l] = new_dist;
                if new_dist < new_min_dist {
                    new_min_dist = new_dist;
                    new_nearest = j;
                }
            }

            left.push(node_index[lmin]);
            right.push(node_index[rmin]);
            node_index[lmin] = n + internal;
            nearest[lmin] = new_nearest;
            min_dist[lmin] = new_min_dist;
            node_index[rmin] = NONE;
        }
        (left, right)
    }

    /// `MPCFlat::ConsIter` / `ConsPair`: one round of the consistency
    /// transformation over every pair.
    ///
    /// For a pair (X, Y) the new posterior of a cell is
    /// (2 P(x~y) + sum over the other sequences Z and their positions z of
    /// P(x~z) P(z~y)) / N, the terms added one by one, Z in input order and z
    /// in sequence order, each product fused with its addition
    /// (relaxflat.cpp). Only the cells the sparse matrix of (X, Y) already
    /// holds are kept (`MySparseMx::UpdateFromPost`), so the sums are
    /// accumulated only over the columns each row spans. The loops run over
    /// Y, then Z, then X, so that P(z~y) can be laid out densely once for
    /// all X; every cell still receives the same terms in the same order as
    /// in MUSCLE, and a term whose P(z~y) is zero leaves the sum as it is.
    ///
    /// A round reads the posteriors of the round before and nothing of its
    /// own (MUSCLE writes into a second set of matrices and swaps the two
    /// sets afterwards, consflat.cpp), so the pairs (X, Y) are computed
    /// independently of one another, on the thread pool when there are
    /// enough sequences.
    fn cons_iter(&mut self) {
        let n = self.seqs.len();
        let mean_len = self.seqs.iter().map(|seq| seq.len()).sum::<usize>() / n;
        let parallel = n * n * n * mean_len >= self.tuning.parallel_consistency_work
            && rayon::current_num_threads() > 1;
        // One task per Y on the calling thread; per Y and block of X on the
        // pool.
        let block = if parallel { CONSISTENCY_BLOCK } else { n };
        let tasks: Vec<(usize, usize)> = (1..n)
            .flat_map(|y| (0..y).step_by(block).map(move |x| (y, x)))
            .collect();
        let this = &*self;
        let relax = |(zy, sums): &mut (DenseRows, Sums), &(y, x): &(usize, usize)| {
            this.cons_block(y, x..(x + block).min(y), zy, sums)
        };
        let updated: Vec<Vec<Vec<f32>>> = if parallel {
            tasks
                .par_iter()
                .with_max_len(1)
                .map_init(Default::default, relax)
                .collect()
        } else {
            let mut buffers = Default::default();
            tasks.iter().map(|task| relax(&mut buffers, task)).collect()
        };
        for (&(y, first), block) in tasks.iter().zip(updated) {
            for (x, probs) in (first..).zip(block) {
                let pair = self.pair_index(x, y);
                self.posts[pair].probs = probs;
            }
        }
    }

    /// The new posteriors of the pairs (X, y) for the X of `xs`, all before
    /// `y`, in the order of X: the values of the cells of each, in the order
    /// of `SparseMx::probs`.
    fn cons_block(
        &self,
        y: usize,
        xs: std::ops::Range<usize>,
        zy: &mut DenseRows,
        sums: &mut Sums,
    ) -> Vec<Vec<f32>> {
        let n = self.seqs.len();
        let soft = self.tuning.soft_fma;
        // The binary multiplies by the reciprocal of the sequence count.
        let scale = 1.0 / n as f32;
        sums.start(self, y, xs.clone());
        for z in 0..n {
            if z == y {
                continue;
            }
            if z < y {
                zy.fill(&self.posts[self.pair_index(z, y)], self.seqs[y].len());
            } else {
                zy.fill_transposed(&self.posts[self.pair_index(y, z)], self.seqs[z].len());
            }
            for x in xs.clone() {
                if x == z {
                    continue;
                }
                let first_row = sums.first_row[x - xs.start];
                if x < z {
                    let xz = &self.posts[self.pair_index(x, z)];
                    for pos_x in 0..xz.rows() {
                        let entries = xz.row(pos_x);
                        let mut row = sums.row(first_row + pos_x);
                        if row.lo < row.hi && !entries.is_empty() {
                            row.add_terms(&xz.cols[entries.clone()], &xz.probs[entries], zy, soft);
                        }
                    }
                } else {
                    let zx = &self.posts[self.pair_index(z, x)];
                    for pos_z in 0..zx.rows() {
                        let span = zy.span[pos_z];
                        if span.0 >= span.1 {
                            continue;
                        }
                        let source = zy.row(pos_z);
                        let entries = zx.row(pos_z);
                        for (&pos_x, &p) in zx.cols[entries.clone()].iter().zip(&zx.probs[entries])
                        {
                            sums.row(first_row + pos_x as usize)
                                .add_term(p, source, span, soft);
                        }
                    }
                }
            }
        }

        // `MySparseMx::UpdateFromPost`: same cells, new values.
        xs.clone()
            .map(|x| {
                let xy = &self.posts[self.pair_index(x, y)];
                let first_row = sums.first_row[x - xs.start];
                let mut new = Vec::with_capacity(xy.probs.len());
                for i in 0..xy.rows() {
                    let row = sums.row(first_row + i);
                    for k in xy.row(i) {
                        new.push(row.cells[xy.cols[k] as usize - row.lo] * scale);
                    }
                }
                new
            })
            .collect()
    }

    /// `MPCFlat::ProgressiveAlign`.
    fn progressive_align(&self, joins1: &[usize], joins2: &[usize]) -> Msa {
        let n = self.seqs.len();
        let mut msas: Vec<Option<Msa>> = (0..n)
            .map(|i| {
                Some(vec![Row {
                    seq: i,
                    chars: self.seqs[i].to_vec(),
                }])
            })
            .collect();
        for (&index1, &index2) in joins1.iter().zip(joins2) {
            let msa1 = msas[index1].take().expect("join order uses each node once");
            let msa2 = msas[index2].take().expect("join order uses each node once");
            msas.push(Some(self.align_alns(&msa1, &msa2)));
        }
        msas.pop().flatten().expect("at least one join")
    }

    /// refineflat.cpp `MPCFlat::RefineIter`.
    fn refine_iter(&self, msa: &mut Msa, rng: &mut LibcRand) {
        let mut group1 = Vec::new();
        let mut group2 = Vec::new();
        for index in 0..msa.len() {
            if rng.next().is_multiple_of(2) {
                group1.push(index);
            } else {
                group2.push(index);
            }
        }
        if group1.is_empty() || group2.is_empty() {
            return;
        }
        let msa1 = project(msa, &group1);
        let msa2 = project(msa, &group2);
        *msa = self.align_alns(&msa1, &msa2);
    }

    /// alnalnsflat.cpp `MPCFlat::AlignAlns`.
    fn align_alns(&self, msa1: &Msa, msa2: &Msa) -> Msa {
        let cols1 = msa1[0].chars.len();
        let cols2 = msa2[0].chars.len();
        let mut post = vec![0f32; cols1 * cols2];
        self.build_post(msa1, msa2, cols2, &mut post);
        let path = calc_aln(&post, cols1, cols2);
        let mut result = Vec::with_capacity(msa1.len() + msa2.len());
        for row in msa1 {
            result.push(add_gaps_path(row, &path, b'X'));
        }
        for row in msa2 {
            result.push(add_gaps_path(row, &path, b'Y'));
        }
        result
    }

    /// buildpostflat.cpp `MPCFlat::BuildPost`: sum of the pairwise posteriors
    /// over every sequence of one alignment against every sequence of the
    /// other, in alignment columns.
    ///
    /// A cell of the sum takes at most one term from each pair of sequences,
    /// and takes them in MUSCLE's order: the sequences of the first
    /// alignment in turn, and for each the sequences of the second. With
    /// enough pairs the work goes to the thread pool in two steps that keep
    /// that order: the terms of each sequence of the first alignment are
    /// collected (which cell, how much), sorted into bands of rows of the
    /// sum; then each band adds its terms, sequence after sequence.
    fn build_post(&self, msa1: &Msa, msa2: &Msa, cols2: usize, post: &mut [f32]) {
        let cols_of_2: Vec<Vec<u32>> = msa2.iter().map(|row| letter_columns(row, 1)).collect();
        let mean_len = self.seqs.iter().map(|seq| seq.len()).sum::<usize>() / self.seqs.len();
        let threads = if msa1.len() * msa2.len() * mean_len >= self.tuning.parallel_sum_work {
            rayon::current_num_threads()
        } else {
            1
        };
        if threads == 1 {
            for row1 in msa1 {
                self.post_terms(row1, msa2, &cols_of_2, cols2, |cell, p| {
                    post[cell as usize] += p;
                });
            }
            return;
        }

        // More bands than threads, so that the cores share them evenly
        // whatever their speeds.
        let band = (post.len() / cols2).div_ceil(4 * threads).max(1) * cols2;
        let bands = post.len().div_ceil(band);
        let collected: Vec<Vec<Vec<(u32, f32)>>> = msa1
            .par_iter()
            .with_max_len(1)
            .map(|row1| {
                let terms: usize = msa2
                    .iter()
                    .map(|row2| {
                        let (first, second) = (row1.seq.min(row2.seq), row1.seq.max(row2.seq));
                        self.cells.pair(self.pair_index(first, second)).len()
                    })
                    .sum();
                let mut by_band: Vec<Vec<(u32, f32)>> = (0..bands)
                    .map(|_| Vec::with_capacity(terms / bands * 5 / 4 + 16))
                    .collect();
                self.post_terms(row1, msa2, &cols_of_2, cols2, |cell, p| {
                    by_band[cell as usize / band].push((cell, p));
                });
                by_band
            })
            .collect();
        post.par_chunks_mut(band)
            .with_max_len(1)
            .enumerate()
            .for_each(|(index, cells)| {
                let first = index * band;
                for by_band in &collected {
                    for &(cell, p) in &by_band[index] {
                        cells[cell as usize - first] += p;
                    }
                }
            });
    }

    /// The terms of `build_post` for one sequence of the first alignment
    /// against every sequence of the second, in order: the cell of the sum
    /// and the posterior to add to it.
    #[inline(always)]
    fn post_terms(
        &self,
        row1: &Row,
        msa2: &Msa,
        cols_of_2: &[Vec<u32>],
        cols2: usize,
        mut add: impl FnMut(u32, f32),
    ) {
        let starts1 = letter_columns(row1, cols2);
        for (row2, cols2) in msa2.iter().zip(cols_of_2) {
            // The posterior has the earlier sequence in its rows. (Slices,
            // not vectors: the loop below must not look their place up again
            // after every addition.)
            let (pair, in_rows, in_cols): (usize, &[u32], &[u32]) = if row1.seq < row2.seq {
                (self.pair_index(row1.seq, row2.seq), &starts1, cols2)
            } else {
                (self.pair_index(row2.seq, row1.seq), cols2, &starts1)
            };
            for cell in self.cells.pair(pair) {
                add(
                    in_rows[cell.row as usize] + in_cols[cell.col as usize],
                    cell.p,
                );
            }
        }
    }
}

/// The column of each letter of a row, in order, times `scale`.
fn letter_columns(row: &Row, scale: usize) -> Vec<u32> {
    row.chars
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c != b'-')
        .map(|(col, _)| (col * scale) as u32)
        .collect()
}

/// calcalnscoreflat.cpp `CalcAlnScoreFlat`: the score of the best path
/// through a posterior matrix.
fn calc_aln_score(post: &[f32], lx: usize, ly: usize, row: &mut [f32]) -> f32 {
    row[..=ly].fill(0.0);
    for i in 0..lx {
        let mut curr_j1 = 0f32;
        let mut prev_j1 = row[0];
        row[0] = 0.0;
        for j in 1..=ly {
            let prev_j = row[j];
            let b = prev_j1 + post[i * ly + j - 1];
            prev_j1 = prev_j;
            curr_j1 = if b >= prev_j {
                if b >= curr_j1 { b } else { curr_j1 }
            } else if prev_j >= curr_j1 {
                prev_j
            } else {
                curr_j1
            };
            row[j] = curr_j1;
        }
    }
    row[ly]
}

/// calcalnflat.cpp `CalcAlnFlat` and tracebackflat.cpp `TraceBackFlat`: the
/// best path, as a string of B (both), X (first only) and Y (second only).
fn calc_aln(post: &[f32], lx: usize, ly: usize) -> Vec<u8> {
    let ly1 = ly + 1;
    let mut tb = vec![b'Y'; (lx + 1) * ly1];
    let mut old_row = vec![0f32; ly1];
    let mut new_row = vec![0f32; ly1];
    for i in 1..=lx {
        tb[i * ly1] = b'X';
        new_row[0] = 0.0;
        let post_row = &post[(i - 1) * ly..i * ly];
        let tb_row = &mut tb[i * ly1..(i + 1) * ly1];
        for j in 1..=ly {
            let b = old_row[j - 1] + post_row[j - 1];
            let x = old_row[j];
            let y = new_row[j - 1];
            let (best, code) = if b >= x {
                if b >= y { (b, b'B') } else { (y, b'Y') }
            } else if x >= y {
                (x, b'X')
            } else {
                (y, b'Y')
            };
            new_row[j] = best;
            tb_row[j] = code;
        }
        std::mem::swap(&mut old_row, &mut new_row);
    }

    let mut path = Vec::with_capacity(lx + ly);
    let (mut i, mut j) = (lx, ly);
    while i != 0 || j != 0 {
        let code = tb[i * ly1 + j];
        path.push(code);
        match code {
            b'B' => {
                i -= 1;
                j -= 1;
            }
            b'X' => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path
}

/// sequence.cpp `Sequence::AddGapsPath`.
fn add_gaps_path(row: &Row, path: &[u8], id: u8) -> Row {
    let mut chars = Vec::with_capacity(path.len());
    let mut next = row.chars.iter();
    for &step in path {
        if step == b'B' || step == id {
            chars.push(*next.next().expect("path consumes the row"));
        } else {
            chars.push(b'-');
        }
    }
    Row {
        seq: row.seq,
        chars,
    }
}

/// project.cpp `MultiSequence::Project`: the chosen rows without the
/// columns that hold only gaps in them.
fn project(msa: &Msa, indexes: &[usize]) -> Msa {
    let cols = msa[0].chars.len();
    let keep: Vec<usize> = (0..cols)
        .filter(|&col| indexes.iter().any(|&index| msa[index].chars[col] != b'-'))
        .collect();
    indexes
        .iter()
        .map(|&index| Row {
            seq: msa[index].seq,
            chars: keep.iter().map(|&col| msa[index].chars[col]).collect(),
        })
        .collect()
}

/// guidetreejoinorder.cpp `GetGuideTreeJoinOrder`: the joins in depth-first
/// order (left subtree, right subtree, node). Join k makes node n + k.
fn join_order(n: usize, left: &[usize], right: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut joins1 = Vec::with_capacity(n - 1);
    let mut joins2 = Vec::with_capacity(n - 1);
    let mut results: Vec<usize> = Vec::new();
    // (node, children already visited)
    let mut pending: Vec<(usize, bool)> = vec![(2 * n - 2, false)];
    let mut next_join = n;
    while let Some((node, visited)) = pending.pop() {
        if node < n {
            results.push(node);
        } else if visited {
            let second = results.pop().expect("two results per join");
            let first = results.pop().expect("two results per join");
            joins1.push(first);
            joins2.push(second);
            results.push(next_join);
            next_join += 1;
        } else {
            pending.push((node, true));
            pending.push((right[node - n], false));
            pending.push((left[node - n], false));
        }
    }
    (joins1, joins2)
}
