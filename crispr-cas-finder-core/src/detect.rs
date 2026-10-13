use crate::dna::reverse_complement;
use crate::types::{DetectionParams, Orientation};
use crate::vmatch::{RepeatSearch, find_repeats};
use rayon::prelude::*;
use std::collections::HashMap;

/// Compute AT% of a DNA sequence (fraction of A+T bases, as a percentage).
///
/// Returns 0.0 if the sequence is empty.
pub fn at_percent(seq: &[u8]) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    let at_count = seq
        .iter()
        .filter(|&&b| matches!(b, b'A' | b'a' | b'T' | b't'))
        .count();
    (at_count as f64 / seq.len() as f64) * 100.0
}

/// Infer the orientation of a CRISPR array from flanking AT% content.
///
/// Uses the same heuristic as the original Perl CRISPRCasFinder:
/// - If left_AT > right_AT AND left_AT > global_AT → Forward (+)
/// - If right_AT > left_AT AND right_AT > global_AT → Reverse (-)
/// - Otherwise → Unknown
pub fn infer_orientation(
    seq: &[u8],
    array_start: usize,
    array_end: usize,
    flank_len: usize,
) -> Orientation {
    let global_at = at_percent(seq);

    // Left flanking region (before the array)
    let left_start = if array_start > flank_len + 1 {
        array_start - flank_len - 1
    } else {
        0
    };
    let left_end = array_start.saturating_sub(1);
    let left_flank = if left_start < left_end {
        &seq[left_start..left_end]
    } else {
        &[]
    };

    // Right flanking region (after the array)
    let right_start = if array_end < seq.len() {
        array_end
    } else {
        seq.len()
    };
    let right_end = if right_start + flank_len <= seq.len() {
        right_start + flank_len
    } else {
        seq.len()
    };
    let right_flank = if right_start < right_end {
        &seq[right_start..right_end]
    } else {
        &[]
    };

    let left_at = at_percent(left_flank);
    let right_at = at_percent(right_flank);

    if left_flank.is_empty() || right_flank.is_empty() || (left_at - right_at).abs() < f64::EPSILON
    {
        return Orientation::Unknown;
    }

    if left_at > right_at && left_at > global_at {
        Orientation::Forward
    } else if right_at > left_at && right_at > global_at {
        Orientation::Reverse
    } else {
        Orientation::Unknown
    }
}

/// Represents a direct repeat hit from vmatch output
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RepeatHit {
    pub repeat_length: usize,
    pub pos1: usize,
    pub repeat_sequence: String,
}
//------------------------------------------------------------------------------
/// Cluster repeats into candidate CRISPR arrays.
/// Repeats with positions within `max_gap` bp are merged into the same cluster.
pub fn cluster_repeats(hits: &[RepeatHit], max_gap: usize) -> Vec<(usize, usize)> {
    let mut clusters = Vec::new();
    if hits.is_empty() {
        return clusters;
    }
    // This function is public and intentionally accepts unsorted input. Sort
    // references so repeat sequences are not cloned merely for clustering.
    let mut sorted: Vec<&RepeatHit> = hits.iter().collect();
    sorted.sort_unstable_by_key(|hit| hit.pos1);
    // Initialize first cluster
    let mut start = sorted[0].pos1;
    let mut end = sorted[0].pos1;
    for hit in &sorted[1..] {
        if hit.pos1 <= end.saturating_add(max_gap) {
            // extend current cluster
            end = hit.pos1;
        } else {
            // finalize previous cluster
            clusters.push((start, end));
            // start new cluster
            start = hit.pos1;
            end = hit.pos1;
        }
    }
    // push last cluster
    clusters.push((start, end));
    clusters
}

// ─────────────────────────────────────────────────────────────────────────────
// Hamming helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` when `a` and `b` differ in at most `max_mismatches` positions.
/// Bails early as soon as `max_mismatches` is exceeded.
#[inline]
pub fn is_hamming_distance_within(a: &[u8], b: &[u8], max_mismatches: usize) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut mismatch_count = 0usize;
    for (&left_base, &right_base) in a.iter().zip(b.iter()) {
        if left_base != right_base {
            mismatch_count += 1;
            if mismatch_count > max_mismatches {
                return false;
            }
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Sliding-window approximate pattern search  (replaces EMBOSS fuzznuc)
// ─────────────────────────────────────────────────────────────────────────────

/// Find all positions in `text` where `pattern` occurs with at most
/// `max_mism` mismatches. Returns `(position, mismatch_count)` pairs, with
/// overlapping occurrences included.
///
/// Stands in for EMBOSS `fuzznuc`, which CRISPRCasFinder runs to find the
/// copies of a candidate repeat around a locus. Like `fuzznuc`, it ignores
/// case.
pub fn slide_search(text: &[u8], pattern: &[u8], max_mism: usize) -> Vec<(usize, usize)> {
    let pattern_len = pattern.len();
    let mut results = Vec::new();
    if pattern_len == 0 || text.len() < pattern_len {
        return results;
    }
    for start_index in 0..=(text.len() - pattern_len) {
        let mut mismatch_count = 0usize;
        for (&text_base, &pattern_base) in text[start_index..start_index + pattern_len]
            .iter()
            .zip(pattern.iter())
        {
            if !text_base.eq_ignore_ascii_case(&pattern_base) {
                mismatch_count += 1;
            }
            if mismatch_count > max_mism {
                break;
            }
        }
        if mismatch_count <= max_mism {
            results.push((start_index, mismatch_count));
        }
    }
    results
}

/// Find the repeats that seed CRISPR arrays in `seq`.
///
/// One hit per repeat Vmatch reports to CRISPRCasFinder (see
/// [`crate::vmatch`]): its left copy, which is all CRISPRCasFinder reads from
/// Vmatch's output. Hits are sorted by position; several can share a start.
pub fn find_direct_repeats(seq: &[u8], params: &DetectionParams) -> Vec<RepeatHit> {
    let min_repeat_length = params.min_repeat_length;
    let max_repeat_length = params.max_repeat_length;
    let min_spacer_length = params.min_spacer_length;
    let max_spacer_length = params.max_spacer_length;
    let max_mismatches: usize = if params.no_mismatch { 0 } else { 1 };
    if min_repeat_length == 0
        || min_repeat_length > max_repeat_length
        || min_spacer_length > max_spacer_length
    {
        return Vec::new();
    }
    if !params.query_repeats.is_empty() {
        return find_query_repeats(seq, &params.query_repeats, max_mismatches);
    }

    find_repeats(
        seq,
        &RepeatSearch {
            min_length: min_repeat_length,
            min_gap: min_spacer_length,
            max_gap: max_spacer_length,
            max_length: max_repeat_length,
            max_differences: max_mismatches,
        },
    )
    .into_iter()
    .map(|repeat| RepeatHit {
        repeat_length: repeat.left_length,
        pos1: repeat.left_start + 1,
        repeat_sequence: String::from_utf8_lossy(
            &seq[repeat.left_start..repeat.left_start + repeat.left_length],
        )
        .to_string(),
    })
    .collect()
}

/// Find occurrences of user-supplied repeats on either strand.
///
/// Replaces repeat discovery, like `vmatch -q` upstream. Overlapping
/// occurrences keep the leftmost, then the longest.
pub fn find_query_repeats(
    seq: &[u8],
    query_repeats: &[String],
    max_mismatches: usize,
) -> Vec<RepeatHit> {
    let mut patterns: Vec<Vec<u8>> = Vec::new();
    for query in query_repeats {
        let forward = query.trim().to_ascii_uppercase().into_bytes();
        if forward.is_empty() {
            continue;
        }
        let reverse = reverse_complement(&forward);
        for pattern in [forward, reverse] {
            if !patterns.contains(&pattern) {
                patterns.push(pattern);
            }
        }
    }

    let mut occurrences: Vec<(usize, usize)> = patterns
        .par_iter()
        .flat_map_iter(|pattern| {
            slide_search(seq, pattern, max_mismatches)
                .into_iter()
                .map(|(start, _)| (start, pattern.len()))
        })
        .collect();
    occurrences.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(right.1.cmp(&left.1)));
    occurrences.dedup();

    let mut hits: Vec<RepeatHit> = Vec::new();
    let mut next_free = 0usize;
    for (start, repeat_length) in occurrences {
        if !hits.is_empty() && start < next_free {
            continue;
        }
        next_free = start + repeat_length;
        hits.push(RepeatHit {
            repeat_length,
            pos1: start + 1,
            repeat_sequence: String::from_utf8_lossy(&seq[start..start + repeat_length])
                .to_string(),
        });
    }
    hits
}

/// Shortest and longest spacer allowed beside a repeat of `repeat_length`.
///
/// CRISPRCasFinder compares the spacer length with the unrounded products of
/// the repeat length and the two ratios, so the bounds are rounded inwards.
fn spacer_length_bounds(repeat_length: usize, params: &DetectionParams) -> (usize, usize) {
    let repeat_length = repeat_length as f64;
    (
        (repeat_length * params.min_spacer_to_repeat_ratio).ceil() as usize,
        (repeat_length * params.max_spacer_to_repeat_ratio).floor() as usize,
    )
}

/// Extract spacer sequences from a cluster of DR hits.
pub fn extract_spacers(
    hits: &[RepeatHit],
    cluster: (usize, usize),
    seq: &[u8],
    params: &DetectionParams,
) -> Vec<String> {
    let mut repeat_hits_in_cluster: Vec<&RepeatHit> = hits
        .iter()
        .filter(|h| h.pos1 >= cluster.0 && h.pos1 <= cluster.1)
        .collect();
    if repeat_hits_in_cluster.len() < 2 {
        return Vec::new();
    }
    repeat_hits_in_cluster.sort_by_key(|h| h.pos1);

    let consensus_repeat_length = {
        let mut length_frequency: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for h in &repeat_hits_in_cluster {
            *length_frequency.entry(h.repeat_length).or_default() += 1;
        }
        length_frequency
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(l, _)| l)
            .unwrap_or(params.min_repeat_length)
    };
    let (min_spacer_length, max_spacer_length) =
        spacer_length_bounds(consensus_repeat_length, params);

    let mut spacers = Vec::new();
    for repeat_pair in repeat_hits_in_cluster.windows(2) {
        let first_repeat = repeat_pair[0];
        let second_repeat = repeat_pair[1];
        let first_repeat_end = first_repeat.pos1 + first_repeat.repeat_length - 1;
        let spacer_start = first_repeat_end + 1;
        let spacer_end = second_repeat.pos1.saturating_sub(1);
        if spacer_start > spacer_end || spacer_end > seq.len() {
            continue;
        }
        let spacer_length = spacer_end - spacer_start + 1;
        if spacer_length < min_spacer_length || spacer_length > max_spacer_length {
            continue;
        }
        spacers.push(String::from_utf8_lossy(&seq[(spacer_start - 1)..spacer_end]).to_string());
    }
    spacers
}

/// The part of the sequence searched for copies of a cluster's repeat, as a
/// 0-based range: the hits of the cluster and 500 bases on each side. A
/// cluster whose hits all start at one position reaches 60 bases further.
pub fn cluster_region(cluster: (usize, usize), sequence_length: usize) -> std::ops::Range<usize> {
    let first = cluster.0.saturating_sub(1);
    let mut last = cluster.1.saturating_sub(1);
    if last == first {
        last += 60;
    }
    first.saturating_sub(500).min(sequence_length)..(last + 501).min(sequence_length)
}

/// How often each hit sequence occurs among the hits of a sequence.
///
/// Hit sequences are compared as they are written, here and in
/// [`repeat_candidates`]: the hits must come from a sequence in one case.
pub fn repeat_occurrences(hits: &[RepeatHit]) -> HashMap<&str, usize> {
    let mut occurrences = HashMap::new();
    for hit in hits {
        *occurrences.entry(hit.repeat_sequence.as_str()).or_default() += 1;
    }
    occurrences
}

/// The repeats tried as the consensus of a cluster: of the distinct sequences
/// of its hits, those occurring most often among all hits of the sequence,
/// on either strand, in order of appearance.
pub fn repeat_candidates<'a>(
    cluster_hits: &'a [RepeatHit],
    occurrences: &HashMap<&str, usize>,
) -> Vec<&'a str> {
    let count = |sequence: &str| {
        let reverse = reverse_complement(sequence.as_bytes());
        let forward_count = occurrences.get(sequence).copied().unwrap_or(0);
        match std::str::from_utf8(&reverse) {
            Ok(reverse) if reverse != sequence => {
                forward_count + occurrences.get(reverse).copied().unwrap_or(0)
            }
            _ => forward_count,
        }
    };

    let mut seen: Vec<&str> = Vec::new();
    let mut candidates: Vec<&str> = Vec::new();
    let mut highest = 0usize;
    for hit in cluster_hits {
        let sequence = hit.repeat_sequence.as_str();
        if seen.contains(&sequence) {
            continue;
        }
        seen.push(sequence);
        let occurrences = count(sequence);
        if candidates.is_empty() || occurrences > highest {
            highest = occurrences;
            candidates = vec![sequence];
        } else if occurrences == highest {
            candidates.push(sequence);
        }
    }
    candidates
}

/// Mismatches allowed in a copy of a repeat of `repeat_length`.
pub fn allowed_copy_mismatches(repeat_length: usize, params: &DetectionParams) -> usize {
    // Computed as CRISPRCasFinder does, which gives 9 and not 10 for a
    // repeat of 30 with the default of 33.3 percent.
    (repeat_length as f64 / (100.0 / params.truncated_repeat_mismatch_percent)).floor() as usize
}

/// Penalty of a candidate repeat given its copies, as `(start, mismatches)`
/// in `region`, and the share of mismatching bases among them.
///
/// The candidate with the lowest penalty becomes the consensus: each copy
/// that is not exact costs one plus its share of mismatches. With
/// `force_detection`, each fostered property of a copy takes two off.
///
/// The share of mismatching bases is the one CRISPRCasFinder compares with
/// the allowed repeat mismatches. It divides by the number of lines read
/// from the report listing the copies, 23 more than there are copies.
pub fn candidate_score(
    copies: &[(usize, usize)],
    region: &[u8],
    repeat_length: usize,
    params: &DetectionParams,
) -> (f64, f64) {
    let mut score = 0.0f64;
    let mut total_mismatches = 0usize;
    for &(start, mismatches) in copies {
        if params.force_detection {
            let copy = &region[start..start + repeat_length];
            for fostered in [
                crate::dna::starts_with_iupac(copy, params.foster_repeat_begin.as_bytes()),
                copy.len() >= params.foster_repeat_length,
                crate::dna::ends_with_iupac(copy, params.foster_repeat_end.as_bytes()),
            ] {
                if fostered {
                    score -= 2.0;
                }
            }
        }
        if mismatches > 0 {
            total_mismatches += mismatches;
            score += 1.0 + mismatches as f64 / repeat_length as f64;
        }
    }
    let lines_read = copies.len() + 23;
    (
        score,
        total_mismatches as f64 / (lines_read * repeat_length) as f64,
    )
}

/// A stretch between two copies of a repeat; `start` is 1-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpacerSpan {
    pub start: usize,
    pub length: usize,
}

/// The spacers between the copies of a repeat, and which of them are too
/// long to be spacers and divide the locus into several arrays.
///
/// `copies` are `(start, mismatches)` with 0-based starts, in order. A copy
/// closer to the previous one than the shortest spacer is skipped. Spacers
/// longer than the longest one are dropped from both ends, together with the
/// copies beyond them, and flagged in between. Nothing is returned when the
/// copies differ from the repeat by more than `repeat_mismatch_percent` on
/// average.
///
/// This follows `definespacers` and `check_cris_div` of CRISPRCasFinder,
/// including how mismatches are averaged: those of the first copy are left
/// out, and those of a skipped copy replace those of the copy before it.
pub fn define_spacers(
    copies: &[(usize, usize)],
    repeat_length: usize,
    params: &DetectionParams,
) -> Vec<(SpacerSpan, bool)> {
    let shortest = repeat_length as f64 * params.min_spacer_to_repeat_ratio;
    let longest = repeat_length as f64 * params.max_spacer_to_repeat_ratio;

    let mut spacers: Vec<SpacerSpan> = Vec::new();
    // 1-based end of the last copy kept.
    let mut previous_end: Option<usize> = None;
    let mut mismatch_share = 0.0f64;
    let mut total_share = 0.0f64;
    for (index, &(start, mismatches)) in copies.iter().enumerate() {
        let end = start + repeat_length;
        let Some(kept_end) = previous_end else {
            previous_end = Some(end);
            continue;
        };
        let length = start as f64 - kept_end as f64;
        let share = mismatches as f64 / repeat_length as f64;
        if length < shortest {
            mismatch_share = share;
            continue;
        }
        let is_last = index + 1 == copies.len();
        if !(is_last && length > longest) {
            spacers.push(SpacerSpan {
                start: kept_end + 1,
                length: start - kept_end,
            });
            total_share += mismatch_share;
        }
        previous_end = Some(end);
        mismatch_share = share;
    }
    let average_share = (total_share + mismatch_share) / (spacers.len() + 1) as f64;
    if average_share > params.repeat_mismatch_percent / 100.0 {
        return Vec::new();
    }

    // Over-long spacers at the ends go; the others divide the locus.
    while spacers
        .last()
        .is_some_and(|spacer| spacer.length as f64 > longest)
    {
        spacers.pop();
    }
    let leading = spacers
        .iter()
        .take_while(|spacer| spacer.length as f64 > longest)
        .count();
    spacers
        .into_iter()
        .skip(leading)
        .map(|spacer| (spacer, spacer.length as f64 > longest))
        .collect()
}

/// Add at most one truncated terminal repeat on each side of an array.
///
/// The repeat farthest from the leader is often degenerate in its distal
/// half. A candidate is accepted when the half of the window nearest the
/// array matches the same half of the consensus within
/// `truncated_mismatch_percent` (at least one mismatch) and the spacer it
/// creates has a valid length. Returns the number of repeats added.
pub fn extend_with_truncated_repeats(
    consensus: &str,
    repeats: &mut Vec<RepeatHit>,
    seq: &[u8],
    params: &DetectionParams,
) -> usize {
    let consensus = consensus.as_bytes();
    let repeat_length = consensus.len();
    let half_length = repeat_length / 2;
    if half_length == 0 || repeats.len() < 2 {
        return 0;
    }
    let allowed_mismatches =
        ((half_length as f64 * params.truncated_mismatch_percent / 100.0).floor() as usize).max(1);
    let (min_spacer_length, max_spacer_length) = spacer_length_bounds(repeat_length, params);
    let min_spacer_length = min_spacer_length.max(1);
    let spacer_lengths: Vec<usize> = repeats
        .windows(2)
        .map(|pair| {
            pair[1]
                .pos1
                .saturating_sub(pair[0].pos1 + pair[0].repeat_length)
        })
        .collect();
    let typical_spacer_length = spacer_lengths.iter().sum::<usize>() / spacer_lengths.len();

    // `half_offset` selects which half of the window must match.
    let best_start = |starts: std::ops::RangeInclusive<usize>,
                      half_offset: usize,
                      spacer_length: &dyn Fn(usize) -> usize| {
        starts
            .filter(|&start| start >= 1 && start - 1 + repeat_length <= seq.len())
            .filter_map(|start| {
                let window = &seq[start - 1 + half_offset..start - 1 + half_offset + half_length];
                let expected = &consensus[half_offset..half_offset + half_length];
                let mismatches = window
                    .iter()
                    .zip(expected)
                    .filter(|(base, expected)| !base.eq_ignore_ascii_case(expected))
                    .count();
                (mismatches <= allowed_mismatches).then(|| {
                    (
                        mismatches,
                        spacer_length(start).abs_diff(typical_spacer_length),
                        start,
                    )
                })
            })
            .min()
            .map(|(_, _, start)| start)
    };
    let hit_at = |start: usize| RepeatHit {
        repeat_length,
        pos1: start,
        repeat_sequence: String::from_utf8_lossy(&seq[start - 1..start - 1 + repeat_length])
            .to_string(),
    };

    let mut added = 0;
    let last_end = repeats.last().map_or(0, |hit| hit.pos1 + hit.repeat_length);
    if let Some(start) = best_start(
        last_end + min_spacer_length..=last_end + max_spacer_length,
        0,
        &|start| start - last_end,
    ) {
        repeats.push(hit_at(start));
        added += 1;
    }

    let first_start = repeats[0].pos1;
    if first_start > min_spacer_length + repeat_length {
        let highest = first_start - min_spacer_length - repeat_length;
        let lowest = first_start
            .saturating_sub(max_spacer_length + repeat_length)
            .max(1);
        if let Some(start) = best_start(lowest..=highest, repeat_length - half_length, &|start| {
            first_start - (start + repeat_length)
        }) {
            repeats.insert(0, hit_at(start));
            added += 1;
        }
    }
    added
}

/// Perl `check_short_crispr`: for 1-spacer arrays, align the DR consensus
/// against the single spacer using EMBOSS needle-equivalent scoring.
/// Accept only when the alignment has more than 50 % gaps (DR and spacer are
/// structurally different).  Reject if they align too well (likely a tandem
/// repeat, not a real CRISPR).
///
/// Uses EDNAFULL scoring: match 5, mismatch −4, gap_open 10, gap_extend 0.5
/// with free end-gap penalties (`-endweight N`), matching EMBOSS `needle` defaults.
/// Implements Needleman-Wunsch with affine gaps and end-gap-free boundary
/// conditions directly, because bio-rs clip mode has different semantics.
pub fn check_short_crispr(dr: &str, spacer: &str) -> bool {
    needle_gap_share(dr, spacer).is_some_and(|share| share > 0.5)
}

/// Share of gap columns in the alignment of a repeat with a spacer, as EMBOSS
/// `needle` reports it; `None` when either is empty.
pub fn needle_gap_share(dr: &str, spacer: &str) -> Option<f64> {
    let a = dr.as_bytes();
    let b = spacer.as_bytes();
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return None;
    }

    let match_sc: f64 = 5.0;
    let mismatch_sc: f64 = -4.0;
    let gap_open: f64 = 10.0; // positive = penalty
    let gap_extend: f64 = 0.5;

    let inf = f64::NEG_INFINITY;

    // Three DP matrices for affine gaps:
    // mm[i][j]: best score ending with a[i-1] aligned to b[j-1]
    // ga[i][j]: best score ending with gap in a (b[j-1] is unmatched)
    // gb[i][j]: best score ending with gap in b (a[i-1] is unmatched)
    let mut mm = vec![vec![inf; n + 1]; m + 1];
    let mut ga = vec![vec![inf; n + 1]; m + 1];
    let mut gb = vec![vec![inf; n + 1]; m + 1];

    mm[0][0] = 0.0;

    // End-gap-free boundary: leading gaps cost zero
    for cell in ga[0].iter_mut().skip(1) {
        *cell = 0.0; // j bases of b unmatched at start = free
    }
    for row in gb.iter_mut().skip(1) {
        row[0] = 0.0; // i bases of a unmatched at start = free
    }

    for i in 1..=m {
        for j in 1..=n {
            let s = if a[i - 1] == b[j - 1] {
                match_sc
            } else {
                mismatch_sc
            };
            mm[i][j] = s + mm[i - 1][j - 1].max(ga[i - 1][j - 1]).max(gb[i - 1][j - 1]);

            ga[i][j] = (mm[i][j - 1] - gap_open)
                .max(ga[i][j - 1] - gap_extend)
                .max(gb[i][j - 1] - gap_open);

            gb[i][j] = (mm[i - 1][j] - gap_open)
                .max(ga[i - 1][j] - gap_open)
                .max(gb[i - 1][j] - gap_extend);
        }
    }

    // End-gap-free: trailing gaps are free.
    // Best final score: max over last row (trailing gaps in b are free)
    // and last column (trailing gaps in a are free), plus (m,n) itself.
    let mut best_i = m;
    let mut best_j = n;
    let mut best_score = mm[m][n].max(ga[m][n]).max(gb[m][n]);

    for j in 1..n {
        let sc = mm[m][j].max(ga[m][j]).max(gb[m][j]);
        if sc > best_score {
            best_score = sc;
            best_i = m;
            best_j = j;
        }
    }
    for i in 1..m {
        let sc = mm[i][n].max(ga[i][n]).max(gb[i][n]);
        if sc > best_score {
            best_score = sc;
            best_i = i;
            best_j = n;
        }
    }

    // Trailing end gaps
    let trailing_a_gaps = n - best_j;
    let trailing_b_gaps = m - best_i;

    // Traceback from (best_i, best_j) to (0,0)
    #[derive(Clone, Copy)]
    enum TracebackState {
        /// A repeat base is aligned to a spacer base.
        Match,
        /// A spacer base is aligned to a gap in the repeat.
        GapInRepeat,
        /// A repeat base is aligned to a gap in the spacer.
        GapInSpacer,
    }

    let mut state = TracebackState::Match;
    {
        let mut bv = mm[best_i][best_j];
        if ga[best_i][best_j] > bv {
            bv = ga[best_i][best_j];
            state = TracebackState::GapInRepeat;
        }
        if gb[best_i][best_j] > bv {
            state = TracebackState::GapInSpacer;
        }
    }

    let mut i = best_i;
    let mut j = best_j;
    let mut gaps = trailing_a_gaps + trailing_b_gaps;
    let mut total = gaps;
    let ep = 1e-6;

    while i > 0 || j > 0 {
        match state {
            TracebackState::Match => {
                if i == 0 || j == 0 {
                    break;
                }
                total += 1;
                let s = if a[i - 1] == b[j - 1] {
                    match_sc
                } else {
                    mismatch_sc
                };
                let prev = mm[i][j] - s;
                if (prev - mm[i - 1][j - 1]).abs() < ep {
                    state = TracebackState::Match;
                } else if (prev - ga[i - 1][j - 1]).abs() < ep {
                    state = TracebackState::GapInRepeat;
                } else {
                    state = TracebackState::GapInSpacer;
                }
                i -= 1;
                j -= 1;
            }
            TracebackState::GapInRepeat => {
                if j == 0 {
                    break;
                }
                total += 1;
                gaps += 1;
                let cur = ga[i][j];
                if i == 0 {
                    // Free leading gap
                    j -= 1;
                    continue;
                }
                if (cur - (ga[i][j - 1] - gap_extend)).abs() < ep {
                    state = TracebackState::GapInRepeat;
                } else if (cur - (mm[i][j - 1] - gap_open)).abs() < ep {
                    state = TracebackState::Match;
                } else {
                    state = TracebackState::GapInSpacer;
                }
                j -= 1;
            }
            TracebackState::GapInSpacer => {
                if i == 0 {
                    break;
                }
                total += 1;
                gaps += 1;
                let cur = gb[i][j];
                if j == 0 {
                    // Free leading gap
                    i -= 1;
                    continue;
                }
                if (cur - (gb[i - 1][j] - gap_extend)).abs() < ep {
                    state = TracebackState::GapInSpacer;
                } else if (cur - (mm[i - 1][j] - gap_open)).abs() < ep {
                    state = TracebackState::Match;
                } else {
                    state = TracebackState::GapInRepeat;
                }
                i -= 1;
            }
        }
    }

    // Remaining leading gaps
    gaps += i + j;
    total += i + j;

    if total == 0 {
        return None;
    }
    let gap_frac = gaps as f64 / total as f64;
    log::debug!(
        "check_short_crispr: dr_len={} sp_len={} align_len={} gaps={} gap%={:.1} score={:.1}",
        m,
        n,
        total,
        gaps,
        gap_frac * 100.0,
        best_score
    );
    Some(gap_frac)
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::filler;

    fn default_params() -> DetectionParams {
        DetectionParams::default()
    }

    #[test]
    fn orientation_accepts_gc_only_flank_as_real_measurement() {
        let sequence = b"GGGGCCCCATAT";
        assert_eq!(infer_orientation(sequence, 5, 8, 4), Orientation::Reverse);
    }

    #[test]
    fn orientation_is_unknown_when_a_flank_is_empty() {
        let sequence = b"CCCCATAT";
        assert_eq!(infer_orientation(sequence, 1, 4, 4), Orientation::Unknown);
    }

    // ── is_hamming_distance_within ───────────────────────────────────────────

    #[test]
    fn hamming_distance_identical() {
        assert!(is_hamming_distance_within(
            b"ACGTACGTACGT",
            b"ACGTACGTACGT",
            0
        ));
    }

    #[test]
    fn hamming_distance_one_mismatch_within_budget() {
        // last base differs: T vs G — 1 mismatch allowed
        assert!(is_hamming_distance_within(b"ACGT", b"ACGG", 1));
    }

    #[test]
    fn hamming_distance_one_mismatch_over_budget() {
        // 1 mismatch but 0 allowed
        assert!(!is_hamming_distance_within(b"ACGT", b"ACGG", 0));
    }

    #[test]
    fn hamming_distance_two_mismatches_one_allowed() {
        // C→C (ok), G→T, T→C — 2 mismatches, only 1 allowed
        assert!(!is_hamming_distance_within(b"ACGT", b"ACTC", 1));
    }

    #[test]
    fn hamming_distance_all_different_zero_allowed() {
        assert!(!is_hamming_distance_within(b"AAAA", b"CCCC", 0));
    }

    // ── slide_search ──────────────────────────────────────────────────────────

    #[test]
    fn slide_search_exact_match_found() {
        let hits = slide_search(b"NNNNACGTNNNN", b"ACGT", 0);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0], (4, 0));
    }

    #[test]
    fn slide_search_multiple_exact_matches() {
        // "ACGT" appears at positions 0 and 8
        let hits = slide_search(b"ACGTNNNACGT", b"ACGT", 0);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[1].0, 7);
    }

    #[test]
    fn slide_search_one_mismatch_found() {
        // "ACGG" at pos 4: 1 mismatch from "ACGT"
        let hits = slide_search(b"NNNNACGGNNNN", b"ACGT", 1);
        assert!(hits.iter().any(|&(p, m)| p == 4 && m == 1));
    }

    #[test]
    fn slide_search_no_match_strict() {
        // No window matches "ACGT" with 0 mismatches
        let hits = slide_search(b"TTTTTTTTTTTT", b"ACGT", 0);
        assert!(hits.is_empty());
    }

    #[test]
    fn slide_search_pattern_longer_than_text() {
        let hits = slide_search(b"ACG", b"ACGTACGT", 0);
        assert!(hits.is_empty());
    }

    #[test]
    fn slide_search_empty_pattern() {
        let hits = slide_search(b"ACGTACGT", b"", 0);
        assert!(hits.is_empty());
    }

    #[test]
    fn slide_search_ignores_case() {
        assert_eq!(slide_search(b"nnnnacGTnnnn", b"ACGT", 0), [(4, 0)]);
        assert_eq!(slide_search(b"NNNNACGTNNNN", b"acgt", 0), [(4, 0)]);
        // A base in the other case is not a mismatch; another base is.
        assert_eq!(slide_search(b"NNNNacgGNNNN", b"ACGT", 1), [(4, 1)]);
    }

    // ── cluster_repeats ───────────────────────────────────────────────────────

    fn make_hit(pos: usize) -> RepeatHit {
        RepeatHit {
            repeat_length: 28,
            pos1: pos,
            repeat_sequence: "A".repeat(28),
        }
    }

    #[test]
    fn cluster_repeats_empty_input() {
        assert!(cluster_repeats(&[], 200).is_empty());
    }

    #[test]
    fn cluster_repeats_all_in_one_cluster() {
        // Positions 100, 160, 220 — gaps of 60 each, max_gap = 200 → one cluster
        let hits = vec![make_hit(100), make_hit(160), make_hit(220)];
        let clusters = cluster_repeats(&hits, 200);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], (100, 220));
    }

    #[test]
    fn cluster_repeats_two_distinct_clusters() {
        // First cluster: 100,160; second: 5000,5060 (gap >> 200)
        let hits = vec![make_hit(100), make_hit(160), make_hit(5000), make_hit(5060)];
        let clusters = cluster_repeats(&hits, 200);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0], (100, 160));
        assert_eq!(clusters[1], (5000, 5060));
    }

    #[test]
    fn cluster_repeats_unsorted_input_is_handled() {
        // cluster_repeats sorts internally, so order shouldn't matter
        let hits = vec![make_hit(220), make_hit(100), make_hit(160)];
        let clusters = cluster_repeats(&hits, 200);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], (100, 220));
    }

    #[test]
    fn cluster_repeats_handles_coordinate_overflow_safely() {
        let hits = vec![make_hit(usize::MAX - 1), make_hit(usize::MAX)];
        assert_eq!(
            cluster_repeats(&hits, usize::MAX),
            vec![(usize::MAX - 1, usize::MAX)]
        );
    }

    // ── find_direct_repeats: core pipeline ────────────────────────────────────────
    //
    // Synthetic CRISPR layout (1-based coords after 100-bp flank):
    //   [flank 100 bp] DR1(28) SP1(32) DR2(28) SP2(32) DR3(28) [flank 100 bp]
    //
    // DR = 28 bp  (within default min_repeat_length=23 .. max_repeat_length=55)
    // Spacer = 32 bp (within default min_spacer_length=25 .. max_spacer_length=60)

    /// DR sequence used across CRISPR integration tests (28 bp, realistic motif).
    const DR: &[u8] = b"GTTCACTGCCGTATAGGCAGCTAAGAAA";

    fn build_crispr_seq(dr_copies: &[&[u8]], spacers: &[&[u8]], flank_len: usize) -> Vec<u8> {
        let flank: Vec<u8> = b"TGCATGCATGCATGCATGCA"
            .iter()
            .cycle()
            .take(flank_len)
            .cloned()
            .collect();
        let mut seq = flank.clone();
        for (i, dr) in dr_copies.iter().enumerate() {
            seq.extend_from_slice(dr);
            if i < spacers.len() {
                seq.extend_from_slice(spacers[i]);
            }
        }
        seq.extend(flank);
        seq
    }

    #[test]
    fn hits_are_the_left_copies_of_the_repeats() {
        // Four copies of the repeat give three repeats of two neighbouring
        // copies each; the last copy is never the left one.
        let sequence = [
            filler(100, 1).as_bytes(),
            DR,
            b"CAGTTTCAGCGATACGATCGATCGATCGTTTT",
            DR,
            b"AAAACCCCGGGGTTTTACGATCGATCGATCGA",
            DR,
            b"TTGACCATGGCATTACGGATCCAATGCCGTAC",
            DR,
            b"C",
            filler(100, 2).as_bytes(),
        ]
        .concat();
        let hits = find_direct_repeats(&sequence, &default_params());
        let found: Vec<(usize, &str)> = hits
            .iter()
            .map(|hit| (hit.pos1, hit.repeat_sequence.as_str()))
            .collect();
        let repeat = std::str::from_utf8(DR).unwrap();
        assert_eq!(found, [(101, repeat), (161, repeat), (221, repeat)]);
    }

    #[test]
    fn find_direct_repeats_one_mismatch_in_third_copy() {
        // Third DR has 1 mismatch at its last base (T→C)
        let dr_mut = b"GTTCACTGCCGTATAGGCAGCTAAGAAC"; // 1 mismatch
        let sp1 = b"CAGTTTCAGCGATACGATCGATCGATCGTTTT"; // 32 bp
        let sp2 = b"AAAACCCCGGGGTTTTACGATCGATCGATCGA"; // 32 bp
        let seq = build_crispr_seq(&[DR, DR, dr_mut], &[sp1, sp2], 100);

        let params = default_params(); // no_mismatch = false → 1 mismatch allowed
        let hits = find_direct_repeats(&seq, &params);

        assert!(
            hits.len() >= 2,
            "expected ≥2 hits with 1 mismatch allowed, got {}: {:?}",
            hits.len(),
            hits
        );
    }

    #[test]
    fn find_direct_repeats_no_mismatch_flag_rejects_mismatched_copy() {
        // With --noMism, the 1-mismatch third copy must not pair with the others.
        let dr_mut = b"GTTCACTGCCGTATAGGCAGCTAAGAAC"; // 1 mismatch (last base T→C)
        let sp1 = b"CAGTTTCAGCGATACGATCGATCGATCGTTTT"; // 32 bp
        let sp2 = b"AAAACCCCGGGGTTTTACGATCGATCGATCGA"; // 32 bp
        let seq = build_crispr_seq(&[DR, DR, dr_mut], &[sp1, sp2], 100);

        let mut params = default_params();
        params.no_mismatch = true;

        let hits = find_direct_repeats(&seq, &params);
        // The mutant DR sequence must not appear in any hit (it can't form a 0-mismatch pair)
        let dr_mut_str = String::from_utf8(dr_mut.to_vec()).unwrap();
        let mut_hits: Vec<_> = hits
            .iter()
            .filter(|h| h.repeat_sequence == dr_mut_str)
            .collect();
        assert!(
            mut_hits.is_empty(),
            "mutant DR should not be reported with no_mismatch=true, but got: {:?}",
            mut_hits
        );
    }

    fn hits_at(seq: &[u8], starts: &[usize]) -> Vec<RepeatHit> {
        starts
            .iter()
            .map(|&start| RepeatHit {
                repeat_length: DR.len(),
                pos1: start,
                repeat_sequence: String::from_utf8_lossy(&seq[start - 1..start - 1 + DR.len()])
                    .to_string(),
            })
            .collect()
    }

    #[test]
    fn truncated_terminal_repeat_is_added_after_the_array() {
        let sp1 = b"CAGTTTCAGCGATACGATCGATCGATCGTTTT";
        let sp2 = b"AAAACCCCGGGGTTTTACGATCGATCGATCGA";
        let sp3 = b"GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA";
        let mut truncated = DR[..14].to_vec();
        truncated.extend_from_slice(b"CCCCCCCCCCCCCC");
        let seq = build_crispr_seq(&[DR, DR, DR, &truncated], &[sp1, sp2, sp3], 100);
        let consensus = std::str::from_utf8(DR).unwrap();
        let mut repeats = hits_at(&seq, &[101, 161, 221]);

        let added = extend_with_truncated_repeats(consensus, &mut repeats, &seq, &default_params());

        assert_eq!(added, 1);
        assert_eq!(repeats.last().unwrap().pos1, 281);
    }

    #[test]
    fn truncated_leading_repeat_must_match_the_half_nearest_the_array() {
        let sp1 = b"CAGTTTCAGCGATACGATCGATCGATCGTTTT";
        let sp2 = b"AAAACCCCGGGGTTTTACGATCGATCGATCGA";
        let sp3 = b"GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA";
        let mut truncated = b"CCCCCCCCCCCCCC".to_vec();
        truncated.extend_from_slice(&DR[14..]);
        let seq = build_crispr_seq(&[&truncated, DR, DR, DR], &[sp1, sp2, sp3], 100);
        let consensus = std::str::from_utf8(DR).unwrap();
        let mut repeats = hits_at(&seq, &[161, 221, 281]);

        let added = extend_with_truncated_repeats(consensus, &mut repeats, &seq, &default_params());

        assert_eq!(added, 1);
        assert_eq!(repeats[0].pos1, 101);
    }

    #[test]
    fn complete_array_gains_no_truncated_repeat() {
        let sp1 = b"CAGTTTCAGCGATACGATCGATCGATCGTTTT";
        let sp2 = b"AAAACCCCGGGGTTTTACGATCGATCGATCGA";
        let seq = build_crispr_seq(&[DR, DR, DR], &[sp1, sp2], 100);
        let consensus = std::str::from_utf8(DR).unwrap();
        let mut repeats = hits_at(&seq, &[101, 161, 221]);

        let added = extend_with_truncated_repeats(consensus, &mut repeats, &seq, &default_params());

        assert_eq!(added, 0);
        assert_eq!(repeats.len(), 3);
    }

    #[test]
    fn cluster_region_reaches_500_bases_past_the_hits() {
        // Positions are 1-based, the range 0-based.
        assert_eq!(cluster_region((1001, 1301), 10_000), 500..1801);
        // Hits at one position only: 60 bases more.
        assert_eq!(cluster_region((1001, 1001), 10_000), 500..1561);
        assert_eq!(cluster_region((101, 201), 400), 0..400);
        assert_eq!(cluster_region((101, 201), 701), 0..701);
        assert_eq!(cluster_region((101, 201), 702), 0..701);
    }

    #[test]
    fn candidates_are_the_most_frequent_hit_sequences() {
        let hit = |pos1: usize, sequence: &str| RepeatHit {
            repeat_length: sequence.len(),
            pos1,
            repeat_sequence: sequence.to_string(),
        };
        // AAAC occurs once, and once more as GTTT on the other strand; CCGA
        // twice; ACGT, its own reverse complement, once.
        let hits = [
            hit(10, "ACGT"),
            hit(20, "CCGA"),
            hit(30, "AAAC"),
            hit(40, "CCGA"),
            hit(5000, "GTTT"),
        ];
        let occurrences = repeat_occurrences(&hits);
        assert_eq!(
            repeat_candidates(&hits[..4], &occurrences),
            ["CCGA", "AAAC"]
        );
        assert_eq!(repeat_candidates(&hits[..1], &occurrences), ["ACGT"]);
        assert!(repeat_candidates(&[], &occurrences).is_empty());
    }

    #[test]
    fn copy_mismatches_follow_the_original_rounding() {
        let params = default_params();
        // 30 x 33.3 % is 9.99: nine mismatches, not ten.
        assert_eq!(allowed_copy_mismatches(30, &params), 9);
        assert_eq!(allowed_copy_mismatches(41, &params), 13);
        assert_eq!(allowed_copy_mismatches(23, &params), 7);
    }

    #[test]
    fn candidate_score_charges_inexact_copies() {
        let spacer: &[u8] = b"CAGTTTCAGCGATACGATCGATCGATCGTTTT";
        let seq = build_crispr_seq(&[DR, DR, DR], &[spacer, spacer], 100);
        let region = &seq[100..];
        let mut params = default_params();

        let (score, share) = candidate_score(&[(0, 0), (60, 0), (120, 0)], region, 28, &params);
        assert_eq!((score, share), (0.0, 0.0));

        let (score, share) = candidate_score(&[(0, 0), (60, 2), (120, 7)], region, 28, &params);
        assert_eq!(score, (1.0 + 2.0 / 28.0) + (1.0 + 7.0 / 28.0));
        // Nine mismatches over three copies, counted as 26 lines.
        assert_eq!(share, 9.0 / (26.0 * 28.0));

        // DR starts with G and ends with AAA: two fostered properties a copy.
        params.force_detection = true;
        let (score, _) = candidate_score(&[(0, 0), (60, 0), (120, 0)], region, 28, &params);
        assert_eq!(score, -12.0);
        params.foster_repeat_length = 28;
        let (score, _) = candidate_score(&[(0, 0), (60, 0), (120, 0)], region, 28, &params);
        assert_eq!(score, -18.0);
    }

    #[test]
    fn query_repeats_are_found_on_both_strands_with_a_mismatch() {
        let spacer = b"CAGTTTCAGCGATACGATCGATCGATCGTTTT";
        let mut mutated = DR.to_vec();
        mutated[5] = b'T';
        let seq = build_crispr_seq(&[DR, &mutated, DR], &[spacer, spacer], 100);
        let reverse_query = String::from_utf8(reverse_complement(DR)).unwrap();

        let exact = find_query_repeats(&seq, std::slice::from_ref(&reverse_query), 0);
        assert_eq!(
            exact.iter().map(|hit| hit.pos1).collect::<Vec<_>>(),
            [101, 221]
        );

        let tolerant = find_query_repeats(&seq, &[reverse_query], 1);
        assert_eq!(
            tolerant.iter().map(|hit| hit.pos1).collect::<Vec<_>>(),
            [101, 161, 221]
        );
        assert_eq!(tolerant[1].repeat_sequence.as_bytes(), mutated.as_slice());
    }

    #[test]
    fn find_direct_repeats_sequence_too_short_returns_empty() {
        // Sequence shorter than 2*min_repeat_length + min_spacer_length = 2*23 + 25 = 71 bp
        let seq = vec![b'A'; 50];
        let params = default_params();
        assert!(find_direct_repeats(&seq, &params).is_empty());
    }

    #[test]
    fn find_direct_repeats_no_panic_on_random_sequence() {
        // A non-repetitive sequence should not panic, even if it produces hits
        let params = default_params();
        let seq: Vec<u8> = b"GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\
                              GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\
                              GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\
                              GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
            .to_vec();
        let _ = find_direct_repeats(&seq, &params);
    }

    #[test]
    fn spacer_bounds_are_rounded_inwards() {
        // 23 x 0.6 = 13.8 and 23 x 2.5 = 57.5: a 13-bp or a 58-bp spacer is
        // outside the range CRISPRCasFinder accepts.
        let params = default_params();
        assert_eq!(spacer_length_bounds(23, &params), (14, 57));
        assert_eq!(spacer_length_bounds(25, &params), (15, 62));
        assert_eq!(spacer_length_bounds(30, &params), (18, 75));
    }

    /// Spacers as `(start, length, divides)`.
    fn spacers_of(copies: &[(usize, usize)]) -> Vec<(usize, usize, bool)> {
        define_spacers(copies, 28, &default_params())
            .into_iter()
            .map(|(spacer, divides)| (spacer.start, spacer.length, divides))
            .collect()
    }

    #[test]
    fn spacers_lie_between_consecutive_copies() {
        assert_eq!(
            spacers_of(&[(0, 0), (60, 0), (120, 1)]),
            [(29, 32, false), (89, 32, false)]
        );
        assert!(spacers_of(&[(0, 0)]).is_empty());
        assert!(spacers_of(&[]).is_empty());
    }

    #[test]
    fn copy_too_close_to_the_previous_one_is_skipped() {
        // 28 x 0.6 = 16.8: a copy 16 bases after the previous one is not a
        // repeat of the array, one 17 bases after is.
        assert_eq!(spacers_of(&[(0, 0), (44, 0), (60, 0)]), [(29, 32, false)]);
        assert_eq!(
            spacers_of(&[(0, 0), (45, 0), (105, 0)]),
            [(29, 17, false), (74, 32, false)]
        );
        // Overlapping copies.
        assert_eq!(spacers_of(&[(0, 0), (10, 0), (60, 0)]), [(29, 32, false)]);
    }

    #[test]
    fn spacer_too_long_divides_the_locus() {
        // 28 x 2.5 = 70.
        assert_eq!(
            spacers_of(&[(0, 0), (60, 0), (158, 0), (218, 0)]),
            [(29, 32, false), (89, 70, false), (187, 32, false)]
        );
        assert_eq!(
            spacers_of(&[(0, 0), (60, 0), (159, 0), (219, 0)]),
            [(29, 32, false), (89, 71, true), (188, 32, false)]
        );
    }

    #[test]
    fn spacers_too_long_at_either_end_are_dropped() {
        // A stray copy 100 bases before the array and one 100 bases after.
        assert_eq!(
            spacers_of(&[(0, 0), (128, 0), (188, 0), (248, 0), (376, 0)]),
            [(157, 32, false), (217, 32, false)]
        );
        // Two in a row at the end, the first of which is not the last copy.
        assert_eq!(
            spacers_of(&[(0, 0), (60, 0), (188, 0), (316, 0)]),
            [(29, 32, false)]
        );
        assert!(spacers_of(&[(0, 0), (128, 0)]).is_empty());
    }

    #[test]
    fn copies_too_unlike_the_repeat_give_no_spacers() {
        // Nine mismatches in two of three copies: 21 % on average.
        assert!(spacers_of(&[(0, 0), (60, 9), (120, 9)]).is_empty());
        // Those of the first copy are not counted.
        assert_eq!(spacers_of(&[(0, 9), (60, 0), (120, 9)]).len(), 2);
        // Those of a skipped copy replace those of the copy before it.
        assert_eq!(spacers_of(&[(0, 0), (60, 9), (70, 0), (120, 9)]).len(), 2);
    }

    #[test]
    fn needle_gap_share_is_that_of_emboss_needle() {
        // Gaps and alignment length reported by EMBOSS needle 6.6 with
        // -gapopen 10 -gapextend 0.5 for the same sequences.
        for (repeat, spacer, gaps, length) in [
            (
                "CATAAGTGACACTAGCAATTAAAAATTGCAGTG",
                "AAAAAGAGGAACGCCTGAGCAAAGCTCAAG",
                17,
                40,
            ),
            (
                "AAAAATGCGAGCACGAACGGTCCCCTCT",
                "TCCTTGAGGGGTGAGGAATCCCCAA",
                15,
                34,
            ),
            (
                "AAAACTAAAGATGCCTTAAGTGACACTGG",
                "CGATGCTGTAGCTACTAAAACTAAGAAAGCATACTAT",
                28,
                47,
            ),
            (
                "AAAGTGCTAAAGCACGTCGCCAAC",
                "AAACAAACTTCACATGCGTTCAGTTTATTCTGTTGCTA",
                46,
                54,
            ),
        ] {
            let share = needle_gap_share(repeat, spacer).unwrap();
            assert!(
                (share - gaps as f64 / length as f64).abs() < 1e-9,
                "{repeat} {spacer}: {share}"
            );
            assert_eq!(check_short_crispr(repeat, spacer), 2 * gaps > length);
        }
        assert_eq!(needle_gap_share("", "ACGT"), None);
    }

    #[test]
    fn scaffold_gaps_are_not_repeats() {
        // Runs of N spaced like the repeats of an array.
        let gap = "N".repeat(30);
        let seq = [
            filler(100, 5),
            gap.clone(),
            "CAGTTTCAGCGATACGATCGATCGATCGTTTT".into(),
            gap.clone(),
            "AAAACCCCGGGGTTTTACGATCGATCGATCGA".into(),
            gap.clone(),
            "TTGACCATGGCATTACGGATCCAATGCCGTAA".into(),
            gap,
            filler(100, 6),
        ]
        .concat();
        assert!(find_direct_repeats(seq.as_bytes(), &default_params()).is_empty());
    }

    #[test]
    fn repeat_fitting_the_shortest_spacer_only_when_cut_short_is_not_a_seed() {
        // Two copies of a 23-bp unit start 48 bp apart, which leaves exactly
        // the shortest spacer between them. When the copies agree for two
        // more bases, the repeat vmatch would report is 25 bp long and its
        // spacer too short.
        let unit = "GCAGCAGCGGCTTTTTCAGCGGC";
        let locus = |after_second: &str| {
            [
                &filler(99, 7),
                "T",
                unit,
                "ACCTTTCTCTGCCGCTGCCTTCTCG",
                unit,
                after_second,
                &filler(100, 8),
            ]
            .concat()
        };
        let params = default_params();

        let starts: Vec<usize> = find_direct_repeats(locus("GT").as_bytes(), &params)
            .iter()
            .map(|hit| hit.pos1)
            .collect();
        assert_eq!(starts, [101]);
        assert!(find_direct_repeats(locus("AC").as_bytes(), &params).is_empty());
    }
}
