//! Degenerate repeats as Vmatch reports them.
//!
//! CRISPRCasFinder finds the repeats that seed its arrays with
//! `vmatch -l <length> <min gap> <max gap> -e 1 -evalue 1 -selfun sel392v2.so
//! <max length>`. This module follows the steps of that search so that the
//! same repeats come out:
//!
//! 1. Seeds are maximal exact repeats of at least `length / (errors + 1)`
//!    bases.
//! 2. Each seed is extended to the left and to the right by greedy unit edit
//!    distance alignment, within the allowed number of differences. An
//!    extension to the left may not run through another seed, which would
//!    report the same repeat again.
//! 3. Of all extensions of a seed, the one with the smallest E-value is kept.
//! 4. It is reported if both copies are long enough, the left one is not too
//!    long, and the copies lie `min gap` to `max gap` apart.
//! 5. A reported repeat whose two copies lie within the copies of another
//!    one is dropped. Vmatch does this whenever `-sort` and `-best` are both
//!    given, as they are here. In a tandem repeat it keeps the longest pair
//!    of copies and drops the shorter pairs inside it.
//!
//! `-best 1000000` itself is not followed: no sequence comes near a million
//! repeats.
//!
//! `-evalue 1` has no effect in that command: CRISPRCasFinder also passes
//! `-noevalue`, which makes Vmatch skip the E-value of the final repeat, so
//! no repeat is ever above the limit. E-values still decide step 3.
//!
//! The E-values are those of Kurtz et al., "Computation and visualization of
//! degenerate repeats in complete genomes", ISMB 2000. See
//! THIRD_PARTY_NOTICES.md for the Vmatch sources this is derived from.

use crate::hmmer_core::simd::int_msv::Isa;
use rayon::prelude::*;
use std::simd::prelude::*;

/// Rayon setup and merge overhead outweighs parallel scanning below this many
/// candidate starts.
const PARALLEL_SCAN_MIN_POSITIONS: usize = 8_192;
const ROLLING_HASH_BASE: u64 = 257;

/// Probability that two bases match.
const MATCH_PROBABILITY: f64 = 0.25;
/// E-values at or below this are zero for Vmatch.
const SMALLEST_EVALUE: f64 = 1.0e-300;
/// Average quotient of edit distance and Hamming distance E-values for one
/// to twenty differences.
const EDIT_DISTANCE_QUOTIENT: [f64; 21] = [
    0.0, 3.97e+00, 1.28e+01, 3.26e+01, 7.60e+01, 1.71e+02, 3.77e+02, 8.22e+02, 1.78e+03, 3.91e+03,
    8.50e+03, 1.76e+04, 3.78e+04, 7.98e+04, 1.66e+05, 3.58e+05, 7.44e+05, 1.52e+06, 3.20e+06,
    6.40e+06, 1.31e+07,
];

/// A front value that stands for "cannot be reached".
const UNREACHABLE: i64 = i64::MIN / 4;

/// What to look for; the arguments CRISPRCasFinder passes to Vmatch.
#[derive(Debug, Clone, Copy)]
pub struct RepeatSearch {
    /// Shortest copy of a repeat (`-l`, first argument).
    pub min_length: usize,
    /// Fewest and most bases between the two copies (`-l`, second and third
    /// arguments).
    pub min_gap: usize,
    pub max_gap: usize,
    /// Longest left copy (the argument of the `sel392v2.so` selection).
    pub max_length: usize,
    /// Differences allowed between the copies (`-e`); zero for exact repeats.
    pub max_differences: usize,
}

/// A repeat: two copies of a sequence, 0-based starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RepeatMatch {
    pub left_start: usize,
    pub left_length: usize,
    pub right_start: usize,
    pub right_length: usize,
    /// Edit distance between the two copies.
    pub differences: usize,
}

/// Whether two aligned bases count as a match.
///
/// Only A, C, G and T match. N and the other ambiguity codes are wildcards
/// for Vmatch and match nothing, themselves included. Case is not looked at
/// here: [`find_repeats`] searches a sequence in one case.
#[inline]
fn bases_match(left: u8, right: u8) -> bool {
    left == right && matches!(left.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T')
}

/// Hash every fixed-length window in linear time using a wrapping polynomial.
/// Candidate repeats are still compared base-for-base, so a hash collision can
/// add work but cannot create a false repeat hit.
fn rolling_window_hashes(seq: &[u8], window_length: usize) -> Vec<u64> {
    if window_length == 0 || seq.len() < window_length {
        return Vec::new();
    }

    let highest_place =
        (1..window_length).fold(1u64, |power, _| power.wrapping_mul(ROLLING_HASH_BASE));
    let mut hash = seq[..window_length].iter().fold(0u64, |value, byte| {
        value
            .wrapping_mul(ROLLING_HASH_BASE)
            .wrapping_add(u64::from(*byte) + 1)
    });
    let mut hashes = Vec::with_capacity(seq.len() - window_length + 1);
    hashes.push(hash);

    for incoming_index in window_length..seq.len() {
        let outgoing = u64::from(seq[incoming_index - window_length]) + 1;
        let incoming = u64::from(seq[incoming_index]) + 1;
        hash = hash
            .wrapping_sub(outgoing.wrapping_mul(highest_place))
            .wrapping_mul(ROLLING_HASH_BASE)
            .wrapping_add(incoming);
        hashes.push(hash);
    }
    hashes
}

/// Hash of every seed-length window, indexed by window start.
///
/// A repeat's partner lies at most one repeat and one spacer ahead, so pairs
/// are found by comparing a window's hash with the few dozen hashes in that
/// range. No genome-wide position table is needed, which keeps memory at
/// eight bytes per base.
fn seed_hashes(seq: &[u8], seed_length: usize) -> Vec<u64> {
    let mut hashes = rolling_window_hashes(seq, seed_length);
    // A window holding an ambiguous base gets a hash of its own. Runs of N
    // would otherwise all be seeds for one another.
    let is_ambiguous = |base: u8| !bases_match(base, base);
    let mut ambiguous = seq
        .iter()
        .take(seed_length)
        .filter(|&&base| is_ambiguous(base))
        .count();
    for start in 0..hashes.len() {
        if start > 0 {
            ambiguous -= usize::from(is_ambiguous(seq[start - 1]));
            ambiguous += usize::from(is_ambiguous(seq[start + seed_length - 1]));
        }
        if ambiguous > 0 {
            hashes[start] = u64::MAX - start as u64;
        }
    }
    hashes
}

/// A maximal exact repeat: `length` bases at `left` and at `right` that agree
/// and cannot be extended on either side.
#[derive(Debug, Clone, Copy)]
struct Seed {
    left: usize,
    right: usize,
    length: usize,
}

/// Lanes of the vectors that compare window hashes.
const HASH_LANES: usize = 8;

/// Call `found` with the offset of every element of `window` equal to
/// `hash`, in increasing order. Eight hashes are compared at a time.
#[inline(always)]
fn for_each_match_impl(window: &[u64], hash: u64, mut found: impl FnMut(usize)) {
    let needle = Simd::<u64, HASH_LANES>::splat(hash);
    let (chunks, rest) = window.as_chunks::<HASH_LANES>();
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let mut bits = Simd::from_array(*chunk).simd_eq(needle).to_bitmask();
        while bits != 0 {
            found(chunk_index * HASH_LANES + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
    for (offset, &candidate) in rest.iter().enumerate() {
        if candidate == hash {
            found(chunks.len() * HASH_LANES + offset);
        }
    }
}

/// [`for_each_match_impl`] compiled for the instruction set `isa`, so that
/// the comparisons use the processor's wide vectors.
#[inline(always)]
fn for_each_match(isa: Isa, window: &[u64], hash: u64, found: impl FnMut(usize)) {
    // The kernels of an instruction set run only where it is available,
    // which `Isa::detect` guarantees of the set it returns.
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { for_each_match_avx512(window, hash, found) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { for_each_match_avx2(window, hash, found) },
        _ => for_each_match_impl(window, hash, found),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn for_each_match_avx2(window: &[u64], hash: u64, found: impl FnMut(usize)) {
    for_each_match_impl(window, hash, found);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn for_each_match_avx512(window: &[u64], hash: u64, found: impl FnMut(usize)) {
    for_each_match_impl(window, hash, found);
}

/// Seeds whose left copy starts at `left`.
fn seeds_at(
    isa: Isa,
    seq: &[u8],
    hashes: &[u64],
    left: usize,
    seed_length: usize,
    search: &RepeatSearch,
    seeds: &mut Vec<Seed>,
) {
    let hash = hashes[left];
    // The copies of a reported repeat start between `min_length + min_gap`
    // and `max_length + max_gap` apart, and an insertion or deletion to the
    // left of the seed shifts one of them by a base.
    let first = left + (search.min_length + search.min_gap).saturating_sub(search.max_differences);
    let last =
        (left + search.max_length + search.max_gap + search.max_differences).min(hashes.len() - 1);
    if first > last {
        return;
    }
    for_each_match(isa, &hashes[first..=last], hash, |offset| {
        let right = first + offset;
        if left > 0 && bases_match(seq[left - 1], seq[right - 1]) {
            return;
        }
        let length = seq[left..]
            .iter()
            .zip(&seq[right..])
            .take_while(|&(&a, &b)| bases_match(a, b))
            .count();
        // A repeat is at least as long as its seed.
        if length < seed_length || length > search.max_length {
            return;
        }
        seeds.push(Seed {
            left,
            right,
            length,
        });
    });
}

/// One front of the greedy alignment: for each diagonal from `left` on, the
/// number of bases of the first sequence consumed, or `UNREACHABLE`.
struct Front {
    left: i64,
    rows: Vec<i64>,
}

impl Front {
    fn get(&self, diagonal: i64) -> i64 {
        usize::try_from(diagonal - self.left)
            .ok()
            .and_then(|index| self.rows.get(index).copied())
            .unwrap_or(UNREACHABLE)
    }

    fn entries(&self) -> impl Iterator<Item = (i64, i64)> + '_ {
        self.rows
            .iter()
            .enumerate()
            .map(|(index, &row)| (self.left + index as i64, row))
    }
}

/// Fronts for zero to `max_differences` differences when extending two
/// sequences from their starts, and the last front holding a reachable entry.
///
/// `u` and `v` give the base at an offset from the seed, or `None` past the
/// end. `reach_length` makes the extension give up on an entry when it runs
/// through that many matching bases, as Vmatch does to the left of a seed:
/// those bases are a seed themselves.
fn extension_fronts(
    u: &dyn Fn(usize) -> Option<u8>,
    u_length: usize,
    v: &dyn Fn(usize) -> Option<u8>,
    v_length: usize,
    max_differences: usize,
    reach_length: Option<usize>,
) -> (Vec<Front>, usize) {
    let mut fronts = vec![Front {
        left: 0,
        rows: vec![0],
    }];
    if u_length == 0 && v_length == 0 {
        return (fronts, 0);
    }
    let shortest = u_length.min(v_length) as i64;
    let mut found_seed = false;
    for differences in 1..=max_differences as i64 {
        let remaining = differences - shortest;
        let (left, width) = if remaining <= 0 {
            (-differences, 2 * differences + 1)
        } else {
            let left = (-(u_length as i64)).max(-differences);
            (left, (v_length as i64).min(differences) - left + 1)
        };
        let previous = fronts.last().expect("the first front is always there");
        let mut rows = Vec::with_capacity(width.max(0) as usize);
        let mut reachable = false;
        for diagonal in left..left + width {
            let mut row = UNREACHABLE;
            if remaining <= 0 || diagonal <= -remaining || diagonal >= remaining {
                row = (previous.get(diagonal) + 1)
                    .max(previous.get(diagonal - 1))
                    .max(previous.get(diagonal + 1) + 1);
                if row < 0 || row + diagonal < 0 {
                    row = UNREACHABLE;
                } else {
                    if u_length != 0 && v_length != 0 {
                        let mut matched = 0usize;
                        while let (Some(a), Some(b)) = (
                            u(row as usize + matched),
                            v((row + diagonal) as usize + matched),
                        ) {
                            if !bases_match(a, b) {
                                break;
                            }
                            matched += 1;
                        }
                        if reach_length.is_some_and(|reach| matched >= reach) {
                            found_seed = true;
                            row = UNREACHABLE;
                        } else {
                            row += matched as i64;
                        }
                    }
                    if row > u_length as i64 || row + diagonal > v_length as i64 {
                        row = UNREACHABLE;
                    }
                }
            }
            reachable |= row >= 0;
            rows.push(row);
        }
        fronts.push(Front { left, rows });
        if !reachable {
            return (fronts, differences as usize - 1);
        }
        if found_seed {
            return (fronts, differences as usize);
        }
    }
    (fronts, max_differences)
}

/// Longest extension, in bases of the second sequence, among the fronts
/// Vmatch looks at: the last one and, as long as a front has an unreachable
/// entry, the one before it.
fn longest_extension(fronts: &[Front], last: usize) -> i64 {
    let mut longest = 0;
    for front in fronts[..=last].iter().rev() {
        let mut look_further = false;
        for (diagonal, row) in front.entries() {
            if row < 0 {
                look_further = true;
            } else {
                longest = longest.max(row + diagonal);
            }
        }
        if !look_further {
            break;
        }
    }
    longest
}

/// E-values of repeats, computed as Vmatch computes them so that comparisons
/// between them come out the same.
struct Evalues {
    /// `tables[k][l - k - 1]`: probability of a match of length `l` with `k`
    /// mismatches, down to `SMALLEST_EVALUE`.
    tables: Vec<Vec<f64>>,
}

impl Evalues {
    fn new(max_differences: usize) -> Self {
        let no_match = 1.0 - MATCH_PROBABILITY;
        let mut first = MATCH_PROBABILITY * (no_match * no_match);
        let mut tables = Vec::with_capacity(max_differences + 1);
        for k in 0..=max_differences {
            let mut table = Vec::new();
            let mut probability = first;
            first *= ((k + 2) as f64 / (k + 1) as f64) * no_match;
            let mut length = k + 1;
            while probability > SMALLEST_EVALUE {
                table.push(probability);
                probability *= (length + 1) as f64 * MATCH_PROBABILITY / (length + 1 - k) as f64;
                length += 1;
            }
            tables.push(table);
        }
        Self { tables }
    }

    /// E-value of a repeat of `length` with `differences` edit operations,
    /// per pair of positions.
    fn evalue(&self, differences: usize, length: usize) -> f64 {
        let probability = length
            .checked_sub(differences + 1)
            .and_then(|index| self.tables[differences].get(index))
            .copied()
            .unwrap_or(0.0);
        if differences == 0 {
            probability
        } else {
            EDIT_DISTANCE_QUOTIENT[differences] * probability
        }
    }

    /// Whether a repeat with `new` (differences, length) replaces one with
    /// `old` as the best extension of a seed: smaller E-value, then higher
    /// identity, then longer, then the later one.
    fn replaces(&self, old: (usize, usize), new: (usize, usize)) -> bool {
        let old_evalue = self.evalue(old.0, old.1);
        let new_evalue = self.evalue(new.0, new.1);
        if old_evalue != new_evalue {
            return old_evalue > new_evalue;
        }
        let identity = |(differences, length): (usize, usize)| {
            100.0 * (1.0 - differences as f64 / length as f64)
        };
        if identity(old) != identity(new) {
            return identity(old) < identity(new);
        }
        old.1 <= new.1
    }
}

/// Whether Vmatch accepts two copies as a repeat of a sequence with itself:
/// the left one first, and not embedded in or mostly overlapping the other.
fn accepted_as_self_match(repeat: &RepeatMatch) -> bool {
    let (pos1, length1) = (repeat.left_start, repeat.left_length);
    let (pos2, length2) = (repeat.right_start, repeat.right_length);
    if pos1 >= pos2 {
        return false;
    }
    if pos1 + length1 - 1 < pos2 {
        return true;
    }
    if pos1 + length1 >= pos2 + length2 {
        return false;
    }
    (pos2 - pos1) + (pos2 + length2) - (pos1 + length1) > repeat.differences
}

/// The best extension of a seed, if it has one long enough.
fn extend_seed(
    seq: &[u8],
    seed: &Seed,
    seed_length: usize,
    search: &RepeatSearch,
    evalues: &Evalues,
) -> Option<RepeatMatch> {
    let (left_fronts, left_last) = extension_fronts(
        &|offset| seed.left.checked_sub(offset + 1).map(|index| seq[index]),
        seed.left,
        &|offset| seed.right.checked_sub(offset + 1).map(|index| seq[index]),
        seed.right,
        search.max_differences,
        Some(seed_length),
    );
    let after_left = seed.left + seed.length;
    let after_right = seed.right + seed.length;
    let (right_fronts, right_last) = extension_fronts(
        &|offset| seq.get(after_left + offset).copied(),
        seq.len() - after_left,
        &|offset| seq.get(after_right + offset).copied(),
        seq.len() - after_right,
        search.max_differences,
        None,
    );

    let missing = search.min_length.saturating_sub(seed.length) as i64;
    if longest_extension(&left_fronts, left_last) + longest_extension(&right_fronts, right_last)
        < missing
    {
        return None;
    }

    let mut best: Option<RepeatMatch> = None;
    for differences in 0..=search.max_differences.min(left_last + right_last) {
        let fewest_left = differences.saturating_sub(right_last);
        for left_differences in fewest_left..=differences.min(left_last) {
            let left_front = &left_fronts[left_differences];
            let right_front = &right_fronts[differences - left_differences];
            for (left_diagonal, left_rows) in left_front.entries().filter(|entry| entry.1 >= 0) {
                for (right_diagonal, right_rows) in
                    right_front.entries().filter(|entry| entry.1 >= 0)
                {
                    let first_extension = left_rows + right_rows;
                    let second_extension = first_extension + left_diagonal + right_diagonal;
                    if first_extension < missing || second_extension < missing {
                        continue;
                    }
                    let candidate = RepeatMatch {
                        left_start: seed.left - left_rows as usize,
                        left_length: seed.length + first_extension as usize,
                        right_start: seed.right - (left_rows + left_diagonal) as usize,
                        right_length: seed.length + second_extension as usize,
                        differences,
                    };
                    if !accepted_as_self_match(&candidate) {
                        continue;
                    }
                    let length = |repeat: &RepeatMatch| repeat.left_length.max(repeat.right_length);
                    if best.as_ref().is_none_or(|best| {
                        evalues.replaces(
                            (best.differences, length(best)),
                            (candidate.differences, length(&candidate)),
                        )
                    }) {
                        best = Some(candidate);
                    }
                }
            }
        }
    }
    best
}

/// Whether Vmatch reports a repeat.
fn reported(repeat: &RepeatMatch, search: &RepeatSearch) -> bool {
    if repeat.left_length > search.max_length
        || repeat.left_length < search.min_length
        || repeat.right_length < search.min_length
    {
        return false;
    }
    let left_end = repeat.left_start + repeat.left_length;
    left_end + search.min_gap <= repeat.right_start
        && repeat.right_start <= left_end + search.max_gap
}

/// Whether both copies of `inner` lie within the copies of `outer`.
fn contains(outer: &RepeatMatch, inner: &RepeatMatch) -> bool {
    outer.left_start <= inner.left_start
        && inner.left_start + inner.left_length <= outer.left_start + outer.left_length
        && outer.right_start <= inner.right_start
        && inner.right_start + inner.right_length <= outer.right_start + outer.right_length
}

/// Drop every repeat contained in another one. `repeats` must be sorted and
/// hold no two repeats with the same copies.
fn remove_contained(repeats: &mut Vec<RepeatMatch>) {
    let mut contained = vec![false; repeats.len()];
    // First repeat with the same left start as the current one.
    let mut group_start = 0;
    for (index, outer) in repeats.iter().enumerate() {
        if repeats[group_start].left_start != outer.left_start {
            group_start = index;
        }
        let left_end = outer.left_start + outer.left_length;
        let candidates = repeats[group_start..]
            .iter()
            .take_while(|inner| inner.left_start <= left_end);
        for (other, inner) in (group_start..).zip(candidates) {
            if other != index && contains(outer, inner) {
                contained[other] = true;
            }
        }
    }
    let mut contained = contained.into_iter();
    repeats.retain(|_| !contained.next().unwrap_or(false));
}

/// Find the repeats of `seq` that Vmatch reports for `search`, sorted by
/// position. Case is ignored, as Vmatch ignores it. A search allowing more
/// than 20 differences, beyond Vmatch's E-value tables, reports nothing.
pub fn find_repeats(seq: &[u8], search: &RepeatSearch) -> Vec<RepeatMatch> {
    let seq = &*crate::dna::uppercase(seq);
    let seed_length = (search.min_length / (search.max_differences + 1)).max(1);
    let hashes = seed_hashes(seq, seed_length);
    if hashes.is_empty()
        || search.min_gap > search.max_gap
        || search.max_length < seed_length
        || search.max_differences >= EDIT_DISTANCE_QUOTIENT.len()
    {
        return Vec::new();
    }
    let evalues = Evalues::new(search.max_differences);
    let isa = Isa::detect();

    let repeats_at = |left: usize, seeds: &mut Vec<Seed>, repeats: &mut Vec<RepeatMatch>| {
        seeds.clear();
        seeds_at(isa, seq, &hashes, left, seed_length, search, seeds);
        for seed in seeds.iter() {
            if let Some(repeat) = extend_seed(seq, seed, seed_length, search, &evalues)
                && reported(&repeat, search)
            {
                repeats.push(repeat);
            }
        }
    };

    let mut repeats = if hashes.len() >= PARALLEL_SCAN_MIN_POSITIONS {
        (0..hashes.len())
            .into_par_iter()
            .fold(
                || (Vec::new(), Vec::new()),
                |(mut seeds, mut repeats), left| {
                    repeats_at(left, &mut seeds, &mut repeats);
                    (seeds, repeats)
                },
            )
            .map(|(_, repeats)| repeats)
            .reduce(Vec::new, |mut merged, mut repeats| {
                merged.append(&mut repeats);
                merged
            })
    } else {
        let (mut seeds, mut repeats) = (Vec::new(), Vec::new());
        for left in 0..hashes.len() {
            repeats_at(left, &mut seeds, &mut repeats);
        }
        repeats
    };
    repeats.sort_unstable();
    // Two seeds can extend to the same copies with different alignments.
    repeats.dedup_by_key(|repeat| {
        (
            repeat.left_start,
            repeat.left_length,
            repeat.right_start,
            repeat.right_length,
        )
    });
    remove_contained(&mut repeats);
    repeats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::filler;

    /// The search CRISPRCasFinder runs by default.
    const SEARCH: RepeatSearch = RepeatSearch {
        min_length: 23,
        min_gap: 25,
        max_gap: 60,
        max_length: 55,
        max_differences: 1,
    };

    /// Two copies `gap` bases apart, the left one at `lead`. The bases around
    /// them differ, so that the copies do not go on as an exact repeat.
    fn pair(left: &str, right: &str, gap: usize, lead: usize) -> Vec<u8> {
        [
            &filler(lead - 1, 1),
            "A",
            left,
            "C",
            &filler(gap - 2, 3),
            "G",
            right,
            "T",
            &filler(100, 4),
        ]
        .concat()
        .into_bytes()
    }

    /// `sequence` with the base at `index` replaced by another one.
    fn with_mismatch(sequence: &str, index: usize) -> String {
        let other = if sequence.as_bytes()[index] == b'A' {
            "C"
        } else {
            "A"
        };
        [&sequence[..index], other, &sequence[index + 1..]].concat()
    }

    fn without_base(sequence: &str, index: usize) -> String {
        [&sequence[..index], &sequence[index + 1..]].concat()
    }

    fn repeat(
        left_start: usize,
        left_length: usize,
        right_start: usize,
        right_length: usize,
        differences: usize,
    ) -> RepeatMatch {
        RepeatMatch {
            left_start,
            left_length,
            right_start,
            right_length,
            differences,
        }
    }

    // The expected repeats of the tests below are those Vmatch 2.3.1 reports
    // for the same sequences; benchmarks/vmatch_oracle.py runs the comparison.

    #[test]
    fn exact_repeat_is_reported_once() {
        let copy = filler(30, 2);
        assert_eq!(
            find_repeats(&pair(&copy, &copy, 35, 100), &SEARCH),
            [repeat(100, 30, 165, 30, 0)]
        );
    }

    #[test]
    fn long_sequences_take_the_parallel_scan_with_the_same_result() {
        let copy = filler(30, 2);
        assert_eq!(
            find_repeats(&pair(&copy, &copy, 35, 10_000), &SEARCH),
            [repeat(10_000, 30, 10_065, 30, 0)]
        );
    }

    #[test]
    fn mismatch_between_two_seeds_gives_one_repeat() {
        // Both halves are seeds. Only the left one is extended over the
        // mismatch; the right one stops at the other seed.
        let copy = filler(30, 2);
        assert_eq!(
            find_repeats(&pair(&copy, &with_mismatch(&copy, 15), 35, 100), &SEARCH),
            [repeat(100, 30, 165, 30, 1)]
        );
    }

    #[test]
    fn copies_may_differ_by_an_insertion_or_a_deletion() {
        let copy = filler(30, 2);
        let shorter = without_base(&copy, 15);
        assert_eq!(
            find_repeats(&pair(&copy, &shorter, 35, 100), &SEARCH),
            [repeat(100, 30, 165, 29, 1)]
        );
        assert_eq!(
            find_repeats(&pair(&shorter, &copy, 35, 100), &SEARCH),
            [repeat(100, 29, 164, 30, 1)]
        );
    }

    #[test]
    fn seed_is_half_the_shortest_repeat_when_a_difference_is_allowed() {
        // Eleven bases, a mismatch, eleven bases.
        let copy = filler(23, 11);
        let sequence = pair(&copy, &with_mismatch(&copy, 11), 35, 100);
        assert_eq!(
            find_repeats(&sequence, &SEARCH),
            [repeat(100, 23, 158, 23, 1)]
        );

        let exact = RepeatSearch {
            max_differences: 0,
            ..SEARCH
        };
        assert!(find_repeats(&sequence, &exact).is_empty());
        assert!(find_repeats(&pair(&copy[..22], &copy[..22], 35, 100), &exact).is_empty());
        assert_eq!(
            find_repeats(&pair(&copy, &copy, 35, 100), &exact),
            [repeat(100, 23, 158, 23, 0)]
        );
    }

    #[test]
    fn ambiguous_bases_match_nothing() {
        let copy = filler(30, 2);
        let one = [&copy[..15], "N", &copy[16..]].concat();
        assert_eq!(
            find_repeats(&pair(&one, &one, 35, 100), &SEARCH),
            [repeat(100, 30, 165, 30, 1)]
        );

        let two = [&copy[..10], "N", &copy[11..20], "N", &copy[21..]].concat();
        assert!(find_repeats(&pair(&two, &two, 35, 100), &SEARCH).is_empty());

        let runs = [
            filler(100, 7),
            "N".repeat(40),
            filler(40, 8),
            "N".repeat(40),
            filler(100, 9),
        ]
        .concat();
        assert!(find_repeats(runs.as_bytes(), &SEARCH).is_empty());
    }

    #[test]
    fn case_is_ignored() {
        let copy = filler(30, 2);
        let different = with_mismatch(&copy, 15);
        for (left, right) in [(&copy, &copy), (&copy, &different)] {
            let sequence = pair(left, right, 35, 100);
            let expected = find_repeats(&sequence, &SEARCH);
            assert_eq!(expected.len(), 1);

            assert_eq!(
                find_repeats(&sequence.to_ascii_lowercase(), &SEARCH),
                expected
            );
            // One copy in lowercase, as when a masked stretch ends between
            // the two.
            let mut one_copy_masked = sequence.clone();
            one_copy_masked[..140].make_ascii_lowercase();
            assert_eq!(find_repeats(&one_copy_masked, &SEARCH), expected);
            // The case changes within the seeds.
            let mut alternating = sequence.clone();
            for base in alternating.iter_mut().step_by(2) {
                base.make_ascii_lowercase();
            }
            assert_eq!(find_repeats(&alternating, &SEARCH), expected);
        }

        // An ambiguous base matches nothing in either case.
        let ambiguous = [&copy[..10], "n", &copy[11..20], "N", &copy[21..]].concat();
        assert!(find_repeats(&pair(&ambiguous, &ambiguous, 35, 100), &SEARCH).is_empty());
    }

    #[test]
    fn copies_lie_within_the_gap_limits() {
        let copy = &filler(30, 2)[..23];
        assert!(find_repeats(&pair(copy, copy, 24, 100), &SEARCH).is_empty());
        assert_eq!(
            find_repeats(&pair(copy, copy, 25, 100), &SEARCH),
            [repeat(100, 23, 148, 23, 0)]
        );
        assert_eq!(
            find_repeats(&pair(copy, copy, 60, 100), &SEARCH),
            [repeat(100, 23, 183, 23, 0)]
        );
        assert!(find_repeats(&pair(copy, copy, 61, 100), &SEARCH).is_empty());
    }

    #[test]
    fn repeat_longer_than_the_limit_is_not_reported_cut_short() {
        let longest = filler(55, 10);
        assert_eq!(
            find_repeats(&pair(&longest, &longest, 30, 100), &SEARCH),
            [repeat(100, 55, 185, 55, 0)]
        );
        let too_long = filler(56, 10);
        assert!(find_repeats(&pair(&too_long, &too_long, 30, 100), &SEARCH).is_empty());
    }

    #[test]
    fn tandem_repeat_gives_its_longest_pair_of_copies_only() {
        // 120 bases of period seven. Copies 77 and 84 bases apart both fit
        // the limits; the second pair lies within the first.
        let sequence = [
            &filler(199, 5),
            "G",
            &"ATCTTCT".repeat(17),
            "A",
            "G",
            &filler(100, 6),
        ]
        .concat();
        assert_eq!(
            find_repeats(sequence.as_bytes(), &SEARCH),
            [repeat(200, 43, 277, 43, 0)]
        );
    }

    #[test]
    fn contained_repeats_are_removed() {
        let mut repeats = vec![
            repeat(10, 30, 100, 30, 0),
            // Left copy within, right copy not.
            repeat(12, 25, 95, 25, 0),
            // Both copies within, on another diagonal.
            repeat(12, 25, 104, 25, 0),
            // Starts where the first one ends.
            repeat(40, 23, 100, 23, 0),
        ];
        remove_contained(&mut repeats);
        assert_eq!(
            repeats,
            [
                repeat(10, 30, 100, 30, 0),
                repeat(12, 25, 95, 25, 0),
                repeat(40, 23, 100, 23, 0),
            ]
        );
    }

    #[test]
    fn evalues_keep_the_proportions_vmatch_prints() {
        // Vmatch on a 500 base sequence: 9.09e-22 for 43 exact bases,
        // 1.49e-17 for 36, 2.44e-13 for 29, 2.74e-07 for 23 with one
        // difference, 6.55e-05 for 15 exact, 5.99e-01 for 12 with one.
        let evalues = Evalues::new(1);
        let close = |ratio: f64, expected: f64| (ratio / expected - 1.0).abs() < 0.01;
        assert!(close(
            evalues.evalue(0, 36) / evalues.evalue(0, 43),
            1.49e-17 / 9.09e-22
        ));
        assert!(close(
            evalues.evalue(1, 23) / evalues.evalue(0, 29),
            2.74e-07 / 2.44e-13
        ));
        assert!(close(
            evalues.evalue(1, 12) / evalues.evalue(0, 15),
            5.99e-01 / 6.55e-05
        ));
        // Too short to hold that many differences.
        assert_eq!(evalues.evalue(1, 1), 0.0);
    }

    #[test]
    fn better_extension_replaces_the_current_one() {
        let evalues = Evalues::new(1);
        // Smaller E-value: four more matching bases pay for a difference.
        assert!(evalues.replaces((0, 30), (1, 35)));
        assert!(!evalues.replaces((0, 30), (1, 33)));
        // Same E-value: the later candidate wins.
        assert!(evalues.replaces((0, 30), (0, 30)));
    }

    #[test]
    fn rolling_hashes_match_direct_polynomial_hashes() {
        let sequence = b"ACGTNACGT";
        let window_length = 4;
        let expected: Vec<u64> = sequence
            .windows(window_length)
            .map(|window| {
                window.iter().fold(0u64, |value, byte| {
                    value
                        .wrapping_mul(ROLLING_HASH_BASE)
                        .wrapping_add(u64::from(*byte) + 1)
                })
            })
            .collect();
        assert_eq!(rolling_window_hashes(sequence, window_length), expected);
        assert!(rolling_window_hashes(sequence, 0).is_empty());
    }

    #[test]
    fn windows_with_ambiguous_bases_hash_apart() {
        let hashes = seed_hashes(b"ACGTNACGTACGTN", 4);
        // ACGT at 0, 5 and 9.
        assert_eq!(hashes[0], hashes[5]);
        assert_eq!(hashes[5], hashes[9]);
        // CGTN at 1 and 10.
        assert_ne!(hashes[1], hashes[10]);
    }

    #[test]
    fn impossible_searches_find_nothing() {
        let copy = filler(30, 2);
        let sequence = pair(&copy, &copy, 35, 100);
        let reversed_gap = RepeatSearch {
            min_gap: 60,
            max_gap: 25,
            ..SEARCH
        };
        assert!(find_repeats(&sequence, &reversed_gap).is_empty());
        assert!(find_repeats(b"ACGT", &SEARCH).is_empty());
        assert!(find_repeats(b"", &SEARCH).is_empty());
    }
}
