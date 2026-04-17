use crate::types::{DetectionParams, Orientation};
use bio::alignment::AlignmentOperation;
use bio::alignment::pairwise::{Aligner, Scoring};
use std::collections::{HashMap, HashSet};

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

    if left_at == 0.0 || right_at == 0.0 || (left_at - right_at).abs() < f64::EPSILON {
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

/// Reverse-complement of a DNA sequence (ACGT only; unknown bases pass through)
fn revcomp(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b {
            b'A' | b'a' => b'T',
            b'T' | b't' => b'A',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            other => other,
        })
        .collect()
}

/// Represents a direct repeat hit from vmatch output
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RepeatHit {
    pub seq_id: String,
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
    // Sort hits by genomic start position
    let mut sorted = hits.to_vec();
    sorted.sort_by_key(|h| h.pos1);
    // Initialize first cluster
    let mut start = sorted[0].pos1;
    let mut end = sorted[0].pos1;
    for hit in &sorted[1..] {
        if hit.pos1 <= end + max_gap {
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
// Hashing helpers  (replaces mkvtree index building)
// ─────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash — fast with good distribution for short DNA k-mers.
#[inline]
fn fnv1a(data: &[u8]) -> u64 {
    const BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut h = BASIS;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Build a position index:  hash(seq[i .. i+seed_len]) → sorted list of positions
fn build_seed_index(seq: &[u8], seed_len: usize) -> HashMap<u64, Vec<usize>> {
    let n = seq.len();
    let mut index: HashMap<u64, Vec<usize>> = HashMap::new();
    if n < seed_len {
        return index;
    }
    for i in 0..=(n - seed_len) {
        index
            .entry(fnv1a(&seq[i..i + seed_len]))
            .or_default()
            .push(i);
    }
    for v in index.values_mut() {
        v.sort_unstable();
    }
    index
}

// ─────────────────────────────────────────────────────────────────────────────
// Hamming helpers  (replaces vmatch -e mismatch filter + sel392v2.so)
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` when `a` and `b` differ in **at most** `max_mism` positions.
/// Bails early as soon as `max_mism` is exceeded.
#[inline]
pub fn hamming_le(a: &[u8], b: &[u8], max_mism: usize) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut count = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x != y {
            count += 1;
            if count > max_mism {
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
/// `max_mism` mismatches.  Returns `(position, mismatch_count)` pairs.
pub fn slide_search(text: &[u8], pattern: &[u8], max_mism: usize) -> Vec<(usize, usize)> {
    let plen = pattern.len();
    let mut results = Vec::new();
    if plen == 0 || text.len() < plen {
        return results;
    }
    for i in 0..=(text.len() - plen) {
        let mut mism = 0usize;
        for (&a, &b) in text[i..i + plen].iter().zip(pattern.iter()) {
            if a != b {
                mism += 1;
            }
            if mism > max_mism {
                break;
            }
        }
        if mism <= max_mism {
            results.push((i, mism));
        }
    }
    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Core CRISPR Direct Repeat finder
// Pure-Rust replacement for  mkvtree + vmatch + sel392v2.so
//
// Strategy — seed-and-extend with two half-length hash indexes:
//   Left-half  index: hash(seq[i .. i+seed])
//   Right-half index: hash(seq[i + min_repeat_length − seed .. i + min_repeat_length])
//
// A DR pair with exactly 1 mismatch will have a matching half on at least one
// side, so the union of both lookups ensures complete recall at ≤1 mismatch.
// Every candidate is verified with a full, early-exit Hamming check.
// ─────────────────────────────────────────────────────────────────────────────

/// Check if the repeat match at `(a, b)` with length `dr_len` can be extended
/// beyond `limit` while staying within `max_mism` total mismatches.
/// Mimics vmatch + sel392v2.so: vmatch finds the longest local match at the
/// same offset, sel392 rejects when length > max_repeat_length.
///
/// Two-pass approach:
///   1. Extend left/right from the detected boundaries (cheap).
///   2. If that fails, run a two-pointer scan in a tight neighbourhood to
///      catch shifted matches (vmatch can start at a different position).
fn match_extends_beyond(
    seq: &[u8],
    a: usize,
    b: usize,
    dr_len: usize,
    max_mism: usize,
    limit: usize,
) -> bool {
    let n = seq.len();
    // Count actual mismatches in the core
    let mut core_mm = 0usize;
    for k in 0..dr_len {
        if seq[a + k] != seq[b + k] {
            core_mm += 1;
        }
    }
    let budget = max_mism - core_mm;

    // Pass 1: simple extension from fixed boundaries
    let mut right_ext = 0usize;
    let mut right_mm_used = 0usize;
    loop {
        let pa = a + dr_len + right_ext;
        let pb = b + dr_len + right_ext;
        if pa >= n || pb >= n {
            break;
        }
        if seq[pa] != seq[pb] {
            if right_mm_used >= budget {
                break;
            }
            right_mm_used += 1;
        }
        right_ext += 1;
    }

    let mut left_ext = 0usize;
    let mut left_mm_used = 0usize;
    loop {
        if left_ext >= a || left_ext >= b {
            break;
        }
        let pa = a - left_ext - 1;
        let pb = b - left_ext - 1;
        if seq[pa] != seq[pb] {
            if left_mm_used >= budget {
                break;
            }
            left_mm_used += 1;
        }
        left_ext += 1;
    }

    if dr_len + left_ext + right_ext > limit {
        return true;
    }

    // Pass 2: two-pointer sliding window to catch shifted matches.
    // Scan a tight neighbourhood (limit + small margin on each side) at
    // the same offset b - a.
    let offset = b - a;
    let margin = limit / 2 + 1;
    let scan_start = a.saturating_sub(margin);
    let scan_end = (a + dr_len + margin).min(n.saturating_sub(offset));
    if scan_start >= scan_end {
        return false;
    }

    let mut left = scan_start;
    let mut mm = 0usize;
    for right in scan_start..scan_end {
        if seq[right] != seq[right + offset] {
            mm += 1;
        }
        while mm > max_mism {
            if seq[left] != seq[left + offset] {
                mm -= 1;
            }
            left += 1;
        }
        if right - left + 1 > limit {
            return true;
        }
    }
    false
}

/// Scan `index` for valid DR pairs ahead of each position `i`.
/// `seed_offset` = 0 for left-half index; `min_repeat_length − seed` for right-half.
fn find_pairs_with_index(
    seq: &[u8],
    index: &HashMap<u64, Vec<usize>>,
    seed_offset: usize,
    seed: usize,
    min_repeat_length: usize,
    max_repeat_length: usize,
    min_spacer_length: usize,
    max_spacer_length: usize,
    max_mism: usize,
) -> Vec<(usize, usize, usize)> {
    let n = seq.len();
    let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
    if n < min_repeat_length {
        return pairs;
    }
    for i in 0..=(n - min_repeat_length) {
        let seed_end = i + seed_offset + seed;
        if seed_end > n {
            break;
        }
        let h = fnv1a(&seq[i + seed_offset..seed_end]);
        let positions = match index.get(&h) {
            Some(v) => v,
            None => continue,
        };
        let j_min = i + min_repeat_length + min_spacer_length;
        let j_max =
            (i + max_repeat_length + max_spacer_length).min(n.saturating_sub(min_repeat_length));
        if j_min > j_max {
            continue;
        }
        let lo = positions.partition_point(|&p| p < j_min);
        let hi = positions.partition_point(|&p| p <= j_max);
        for &j in &positions[lo..hi] {
            let gap = j - i;
            let dr_lo = gap.saturating_sub(max_spacer_length).max(min_repeat_length);
            let dr_hi = match gap.checked_sub(min_spacer_length) {
                Some(v) => v.min(max_repeat_length),
                None => continue,
            };
            if dr_lo > dr_hi {
                continue;
            }
            // Take the longest valid DR length to maximise information
            'dr: for dr_len in (dr_lo..=dr_hi).rev() {
                if j + dr_len > n {
                    continue;
                }
                if hamming_le(&seq[i..i + dr_len], &seq[j..j + dr_len], max_mism) {
                    // Mimic vmatch + sel392v2.so: reject if the maximum
                    // matchable region at this offset (shifting start/end
                    // within the mismatch budget) exceeds max_repeat_length.  vmatch
                    // reports the longest match and sel392 filters on
                    // `length <= max_repeat_length`.
                    if max_mism > 0
                        && match_extends_beyond(seq, i, j, dr_len, max_mism, max_repeat_length)
                    {
                        break 'dr; // entire (i,j) pair rejected
                    }
                    pairs.push((i, j, dr_len));
                    break 'dr;
                }
            }
        }
    }
    pairs
}

/// Find all direct repeat occurrences in `seq` satisfying the CRISPR parameters in `params`.
/// Pure-Rust replacement for `mkvtree` + `vmatch` + `sel392v2.so`.
pub fn find_direct_repeats(seq: &[u8], seq_id: &str, params: &DetectionParams) -> Vec<RepeatHit> {
    let n = seq.len();
    let min_repeat_length = params.min_repeat_length;
    let max_repeat_length = params.max_repeat_length;
    let min_spacer_length = params.min_spacer_length;
    let max_spacer_length = params.max_spacer_length;
    let max_mism: usize = if params.no_mismatch { 0 } else { 1 };
    let seed = (min_repeat_length / 2).max(1);

    if n < 2 * min_repeat_length + min_spacer_length {
        return Vec::new();
    }

    // Left-half index
    let left_index = build_seed_index(seq, seed);

    // Right-half index
    let right_index: HashMap<u64, Vec<usize>> = {
        let mut idx: HashMap<u64, Vec<usize>> = HashMap::new();
        if n >= min_repeat_length {
            let roff = min_repeat_length - seed;
            for i in 0..=(n - min_repeat_length) {
                idx.entry(fnv1a(&seq[i + roff..i + min_repeat_length]))
                    .or_default()
                    .push(i);
            }
            for v in idx.values_mut() {
                v.sort_unstable();
            }
        }
        idx
    };

    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut hits: Vec<RepeatHit> = Vec::new();

    let record = |hits: &mut Vec<RepeatHit>,
                  seen: &mut HashSet<(usize, usize)>,
                  pos: usize,
                  dr_len: usize| {
        if seen.insert((pos, dr_len)) {
            hits.push(RepeatHit {
                seq_id: seq_id.to_string(),
                repeat_length: dr_len,
                pos1: pos + 1, // 1-based
                repeat_sequence: String::from_utf8_lossy(&seq[pos..pos + dr_len]).to_string(),
            });
        }
    };

    // Pass 1 — left-half seeds (catches mismatches in the right half)
    for (i, j, dr_len) in find_pairs_with_index(
        seq,
        &left_index,
        0,
        seed,
        min_repeat_length,
        max_repeat_length,
        min_spacer_length,
        max_spacer_length,
        max_mism,
    ) {
        record(&mut hits, &mut seen, i, dr_len);
        record(&mut hits, &mut seen, j, dr_len);
    }

    // Pass 2 — right-half seeds (catches mismatches in the left half)
    if max_mism > 0 {
        for (i, j, dr_len) in find_pairs_with_index(
            seq,
            &right_index,
            min_repeat_length - seed,
            seed,
            min_repeat_length,
            max_repeat_length,
            min_spacer_length,
            max_spacer_length,
            max_mism,
        ) {
            record(&mut hits, &mut seen, i, dr_len);
            record(&mut hits, &mut seen, j, dr_len);
        }
    }

    // Deduplicate overlapping hits: keep the leftmost in each group, preferring
    // the longer DR on position ties.  Without this, every 1-bp shift of a
    // window that spans two DRs looks like a valid pair (1-mismatch budget is
    // consumed by the boundary byte), inflating DR/spacer counts ~dr_len×.
    hits.sort_unstable_by(|a, b| {
        a.pos1
            .cmp(&b.pos1)
            .then(b.repeat_length.cmp(&a.repeat_length))
    });
    let mut deduped: Vec<RepeatHit> = Vec::new();
    for hit in hits {
        if deduped.last().is_none_or(|prev: &RepeatHit| {
            hit.pos1 >= prev.pos1 + prev.repeat_length
        }) {
            deduped.push(hit);
        }
    }

    // ── Array extension ───────────────────────────────────────────────────────
    // The 1-mismatch pair-finding step discovers the core of each array, but
    // highly divergent flanking DRs can't always form a valid pair within the
    // search window.  Walk outward from each core cluster at its dominant period
    // using the full params.repeat_mismatch_percent budget — matching what CRISPRCasFinder does
    // when growing an array from a seed pair.
    let max_mism_ext = if params.no_mismatch {
        0
    } else {
        ((min_repeat_length as f64 * params.repeat_mismatch_percent / 100.0).floor() as usize)
            .max(1)
    };
    if max_mism_ext > max_mism {
        let internal_gap = max_repeat_length + max_spacer_length;
        let core_clusters = cluster_repeats(&deduped, internal_gap);
        let n = seq.len();
        let mut extra: Vec<RepeatHit> = Vec::new();

        for &(cs, ce) in &core_clusters {
            let mut sorted: Vec<&RepeatHit> = deduped
                .iter()
                .filter(|h| h.pos1 >= cs && h.pos1 <= ce)
                .collect();
            if sorted.len() < 2 {
                continue;
            }
            sorted.sort_by_key(|h| h.pos1);

            // Modal inter-DR spacing = period
            let period = {
                let mut freq: HashMap<usize, usize> = HashMap::new();
                for w in sorted.windows(2) {
                    *freq.entry(w[1].pos1 - w[0].pos1).or_default() += 1;
                }
                match freq.into_iter().max_by_key(|&(_, c)| c) {
                    Some((s, _)) => s,
                    None => continue,
                }
            };
            // Skip clusters whose period is outside the valid CRISPR-array range.
            // This filters tandem repeats (period < min_repeat_length+min_spacer_length) and other noise.
            if period < min_repeat_length + min_spacer_length
                || period > max_repeat_length + max_spacer_length
            {
                continue;
            }

            // Consensus DR = most frequent dr_seq
            let consensus_seq: String = {
                let mut freq: HashMap<&str, usize> = HashMap::new();
                for h in &sorted {
                    *freq.entry(h.repeat_sequence.as_str()).or_default() += 1;
                }
                match freq.into_iter().max_by_key(|&(_, c)| c) {
                    Some((s, _)) => s.to_string(),
                    None => continue,
                }
            };
            if consensus_seq.is_empty() {
                continue;
            }
            let dr_len = consensus_seq.len();
            let cons = consensus_seq.as_bytes();

            // Extend backward (before first DR in core cluster)
            // We search in a small window around the expected `pos - period`
            // to absorb positional jitter from the pair-finder (its DR
            // boundaries can be a few bp off from the reference tool's).
            let half_search: usize = ((dr_len / 4) + 2).min(10);
            let mut pos = sorted[0].pos1;
            loop {
                if pos <= period + half_search {
                    break;
                }
                let nominal = pos - period;
                let lo = nominal.saturating_sub(half_search).max(1);
                let hi = (nominal + half_search)
                    .min(pos.saturating_sub(min_repeat_length + min_spacer_length));
                if lo > hi {
                    break;
                }
                // Pick the position with the fewest mismatches against consensus
                let best = (lo..=hi)
                    .filter(|&p| p >= 1 && p.saturating_sub(1) + dr_len <= n)
                    .map(|p| {
                        let w = &seq[p - 1..p - 1 + dr_len];
                        let mm = w.iter().zip(cons.iter()).filter(|(a, b)| a != b).count();
                        (mm, p)
                    })
                    .min_by_key(|&(mm, _)| mm);
                match best {
                    Some((mm, new_pos)) if mm <= max_mism_ext => {
                        extra.push(RepeatHit {
                            seq_id: seq_id.to_string(),
                            repeat_length: dr_len,
                            pos1: new_pos,
                            repeat_sequence: String::from_utf8_lossy(
                                &seq[new_pos - 1..new_pos - 1 + dr_len],
                            )
                            .to_string(),
                        });
                        pos = new_pos;
                    }
                    _ => break,
                }
            }

            // Extend forward (after last DR in core cluster)
            // Same windowed search to handle positional jitter.
            let mut pos = sorted.last().unwrap().pos1;
            loop {
                let nominal = match pos.checked_add(period) {
                    Some(v) => v,
                    None => break,
                };
                let lo = nominal
                    .saturating_sub(half_search)
                    .max(pos + min_spacer_length + min_repeat_length);
                let hi = nominal + half_search;
                if lo > hi {
                    break;
                }
                let best = (lo..=hi)
                    .filter(|&p| p >= 1 && p.saturating_sub(1) + dr_len <= n)
                    .map(|p| {
                        let w = &seq[p - 1..p - 1 + dr_len];
                        let mm = w.iter().zip(cons.iter()).filter(|(a, b)| a != b).count();
                        (mm, p)
                    })
                    .min_by_key(|&(mm, _)| mm);
                match best {
                    Some((mm, new_pos)) if mm <= max_mism_ext => {
                        extra.push(RepeatHit {
                            seq_id: seq_id.to_string(),
                            repeat_length: dr_len,
                            pos1: new_pos,
                            repeat_sequence: String::from_utf8_lossy(
                                &seq[new_pos - 1..new_pos - 1 + dr_len],
                            )
                            .to_string(),
                        });
                        pos = new_pos;
                    }
                    _ => break,
                }
            }
        }

        if !extra.is_empty() {
            // Merge extension hits into the existing deduped list, giving
            // deduped hits strict priority: an extension hit is inserted only
            // if it fits in the gap between two consecutive deduped hits
            // without overlapping either one.  This prevents extension
            // artifacts (e.g. tandem-repeat cluster extensions) from
            // displacing legitimately-found CRISPR DRs.
            extra.sort_unstable_by_key(|h| h.pos1);
            let mut result = deduped;
            for ext_hit in extra {
                let idx = result.partition_point(|h| h.pos1 < ext_hit.pos1);
                let prev_ok = idx == 0
                    || ext_hit.pos1 >= result[idx - 1].pos1 + result[idx - 1].repeat_length;
                let next_ok =
                    idx == result.len() || ext_hit.pos1 + ext_hit.repeat_length <= result[idx].pos1;
                if prev_ok && next_ok {
                    result.insert(idx, ext_hit);
                }
            }
            return result;
        }
    }

    deduped
}

/// Extract spacer sequences from a cluster of DR hits.
pub fn extract_spacers(
    hits: &[RepeatHit],
    cluster: (usize, usize),
    seq: &[u8],
    params: &DetectionParams,
) -> Vec<String> {
    let mut dr_hits: Vec<&RepeatHit> = hits
        .iter()
        .filter(|h| h.pos1 >= cluster.0 && h.pos1 <= cluster.1)
        .collect();
    if dr_hits.len() < 2 {
        return Vec::new();
    }
    dr_hits.sort_by_key(|h| h.pos1);

    let dr_len = {
        let mut freq: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for h in &dr_hits {
            *freq.entry(h.repeat_length).or_default() += 1;
        }
        freq.into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(l, _)| l)
            .unwrap_or(params.min_repeat_length)
    };
    let sp_min = (dr_len as f64 * params.min_spacer_to_repeat_ratio).floor() as usize;
    let sp_max = (dr_len as f64 * params.max_spacer_to_repeat_ratio).ceil() as usize;

    let mut spacers = Vec::new();
    for win in dr_hits.windows(2) {
        let first = win[0];
        let second = win[1];
        let dr_end = first.pos1 + first.repeat_length - 1;
        let sp_start = dr_end + 1;
        let sp_end = second.pos1.saturating_sub(1);
        if sp_start > sp_end || sp_end > seq.len() {
            continue;
        }
        let sp_len = sp_end - sp_start + 1;
        if sp_len < sp_min || sp_len > sp_max {
            continue;
        }
        spacers.push(String::from_utf8_lossy(&seq[(sp_start - 1)..sp_end]).to_string());
    }
    spacers
}

/// Refine a cluster using a consensus DR sequence.
pub fn refine_cluster_with_consensus(
    consensus: &str,
    cluster: (usize, usize),
    seq_id: &str,
    seq: &[u8],
    params: &DetectionParams,
) -> Vec<RepeatHit> {
    if consensus.is_empty() || seq.is_empty() {
        return Vec::new();
    }

    let dr_len = consensus.len();
    let max_mism = ((dr_len as f64 * params.truncated_repeat_mismatch_percent / 100.0).floor()
        as usize)
        .max(1);
    let sp_min = (dr_len as f64 * params.min_spacer_to_repeat_ratio).floor() as usize;
    let sp_max = (dr_len as f64 * params.max_spacer_to_repeat_ratio).ceil() as usize;

    let region_start = cluster.0.saturating_sub(501);
    let region_end = (cluster.1 + dr_len + 500).min(seq.len());
    let region = &seq[region_start..region_end];

    let raw_hits = slide_search(region, consensus.as_bytes(), max_mism);

    let mut refined: Vec<RepeatHit> = Vec::new();
    for (rel_pos, _mism) in raw_hits {
        let abs_pos = region_start + rel_pos + 1; // 1-based
        if abs_pos + dr_len - 1 > seq.len() {
            continue;
        }
        let repeat_sequence =
            String::from_utf8_lossy(&seq[abs_pos - 1..abs_pos - 1 + dr_len]).to_string();
        refined.push(RepeatHit {
            seq_id: seq_id.to_string(),
            repeat_length: dr_len,
            pos1: abs_pos,
            repeat_sequence,
        });
    }

    refined.sort_by_key(|h| h.pos1);
    let mut kept: Vec<RepeatHit> = Vec::new();
    for hit in refined {
        if let Some(prev) = kept.last() {
            let sp_start = prev.pos1 + prev.repeat_length;
            let sp_end = hit.pos1.saturating_sub(1);
            if sp_end < sp_start {
                continue;
            }
            let sp_len = sp_end - sp_start + 1;
            if sp_len < sp_min || sp_len > sp_max {
                continue;
            }
        }
        kept.push(hit);
    }
    kept
}

/// Re-scan the cluster region using the consensus DR (fuzznuc-style).
///
/// Mirrors the Perl pipeline's second pass: after choosing the consensus DR
/// via `choose_consensus_repeat`, Perl runs fuzznuc through the ±500 bp cluster
/// window using `err = floor(DRlength / DRtrunMism)` mismatches, then
/// re-derives the DR hit list and spacers from those fresh positions.
/// Spacers whose length falls outside `[min_spacer_to_repeat_ratio × DRlen, max_spacer_to_repeat_ratio ×
/// DRlen]` are dropped, matching Perl's `definespacers` filter.
///
/// Compute pairwise percent identity with global alignment
fn percent_identity(a: &str, b: &str) -> f64 {
    let scoring = Scoring::from_scores(-5, -1, 1, -1);
    let mut aligner = Aligner::with_scoring(scoring);
    let al = aligner.global(a.as_bytes(), b.as_bytes());
    let matches = al
        .operations
        .iter()
        .filter(|&&op| op == AlignmentOperation::Match)
        .count();
    let len = al.operations.len();
    if len == 0 {
        0.0
    } else {
        matches as f64 / len as f64 * 100.0
    }
}

/// Check that no two spacers exceed similarity threshold; returns true if pass
pub fn check_spacer_similarity(spacers: &[String], threshold: f64) -> bool {
    for i in 0..spacers.len() {
        for j in (i + 1)..spacers.len() {
            let pid = percent_identity(&spacers[i], &spacers[j]);
            if pid >= threshold {
                return false;
            }
        }
    }
    true
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
    let a = dr.as_bytes();
    let b = spacer.as_bytes();
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return false;
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
    for j in 1..=n {
        ga[0][j] = 0.0; // j bases of b unmatched at start = free
    }
    for i in 1..=m {
        gb[i][0] = 0.0; // i bases of a unmatched at start = free
    }

    for i in 1..=m {
        for j in 1..=n {
            let s = if a[i - 1] == b[j - 1] {
                match_sc
            } else {
                mismatch_sc
            };
            mm[i][j] = s + mm[i - 1][j - 1].max(ga[i - 1][j - 1]).max(gb[i - 1][j - 1]);

            ga[i][j] = (mm[i][j - 1] - gap_open - gap_extend)
                .max(ga[i][j - 1] - gap_extend)
                .max(gb[i][j - 1] - gap_open - gap_extend);

            gb[i][j] = (mm[i - 1][j] - gap_open - gap_extend)
                .max(ga[i - 1][j] - gap_open - gap_extend)
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
    enum St {
        Mm,
        Ga,
        Gb,
    }

    let mut state = St::Mm;
    {
        let mut bv = mm[best_i][best_j];
        if ga[best_i][best_j] > bv {
            bv = ga[best_i][best_j];
            state = St::Ga;
        }
        if gb[best_i][best_j] > bv {
            state = St::Gb;
        }
    }

    let mut i = best_i;
    let mut j = best_j;
    let mut gaps = (trailing_a_gaps + trailing_b_gaps);
    let mut total = gaps;
    let ep = 1e-6;

    while i > 0 || j > 0 {
        match state {
            St::Mm => {
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
                    state = St::Mm;
                } else if (prev - ga[i - 1][j - 1]).abs() < ep {
                    state = St::Ga;
                } else {
                    state = St::Gb;
                }
                i -= 1;
                j -= 1;
            }
            St::Ga => {
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
                    state = St::Ga;
                } else if (cur - (mm[i][j - 1] - gap_open - gap_extend)).abs() < ep {
                    state = St::Mm;
                } else {
                    state = St::Gb;
                }
                j -= 1;
            }
            St::Gb => {
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
                    state = St::Gb;
                } else if (cur - (mm[i - 1][j] - gap_open - gap_extend)).abs() < ep {
                    state = St::Mm;
                } else {
                    state = St::Ga;
                }
                i -= 1;
            }
        }
    }

    // Remaining leading gaps
    gaps += i + j;
    total += i + j;

    if total == 0 {
        return false;
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
    gap_frac > 0.5
}

/// Trim poorly-conserved positions from the edges of a consensus DR.
///
/// Given a set of refined repeat hits and the current consensus, computes
/// per-position conservation and trims positions from left and right where
/// fewer than `threshold` fraction of instances match the consensus base.
/// This corrects for over-extension in the initial pair-finding step.
pub fn trim_consensus(consensus: &str, refined: &[RepeatHit], min_len: usize) -> String {
    if refined.len() < 2 || consensus.is_empty() {
        return consensus.to_string();
    }
    let cons_bytes = consensus.as_bytes();
    let dr_len = cons_bytes.len();
    let threshold = 0.80;
    let n = refined.len() as f64;

    // Per-position conservation
    let mut conservation: Vec<f64> = vec![0.0; dr_len];
    for hit in refined {
        let hit_bytes = hit.repeat_sequence.as_bytes();
        for (k, &cb) in cons_bytes.iter().enumerate() {
            if k < hit_bytes.len() && hit_bytes[k] == cb {
                conservation[k] += 1.0;
            }
        }
    }
    for v in conservation.iter_mut() {
        *v /= n;
    }

    // Trim from left
    let mut left_trim = 0;
    while left_trim < dr_len && conservation[left_trim] < threshold {
        left_trim += 1;
    }
    // Trim from right
    let mut right_trim = 0;
    while right_trim < dr_len && conservation[dr_len - 1 - right_trim] < threshold {
        right_trim += 1;
    }

    if left_trim + right_trim >= dr_len || dr_len - left_trim - right_trim < min_len {
        return consensus.to_string(); // trimming would be too aggressive
    }

    if left_trim == 0 && right_trim == 0 {
        return consensus.to_string();
    }

    consensus[left_trim..dr_len - right_trim].to_string()
}

/// Choose the best DR consensus for a CRISPR cluster.
/// Scores each candidate by counting how many times it appears in the cluster
/// Returns all unique candidate DR sequences for a cluster, ranked by global
/// occurrence count (including reverse complement), then by length ascending.
/// Mirrors Perl's `write_clusters` → `DR_occ_rev` candidate ranking.
pub fn get_repeat_candidates(hits: &[RepeatHit], cluster: (usize, usize)) -> Vec<String> {
    let cluster_hits: Vec<&RepeatHit> = hits
        .iter()
        .filter(|h| h.pos1 >= cluster.0 && h.pos1 <= cluster.1)
        .collect();
    if cluster_hits.is_empty() {
        return Vec::new();
    }

    // Collect unique candidate DR sequences from this cluster
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut cands: Vec<String> = Vec::new();
    for h in &cluster_hits {
        if seen.insert(h.repeat_sequence.clone()) {
            cands.push(h.repeat_sequence.clone());
        }
    }

    // Score each candidate by global occurrence (self + revcomp) across all hits
    let scored: Vec<(String, usize)> = cands
        .into_iter()
        .map(|cand| {
            let rc = revcomp(cand.as_bytes());
            let rc_str = String::from_utf8_lossy(&rc).into_owned();
            let count = hits
                .iter()
                .filter(|h| h.repeat_sequence == cand || h.repeat_sequence == rc_str)
                .count();
            (cand, count)
        })
        .collect();

    // Sort: global count descending, then shorter DR preferred (Perl tie-breaking)
    let mut sorted = scored;
    sorted.sort_unstable_by_key(|(s, count)| (usize::MAX - count, s.len()));
    sorted.into_iter().map(|(s, _)| s).collect()
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> DetectionParams {
        DetectionParams::default()
    }

    // ── hamming_le ────────────────────────────────────────────────────────────

    #[test]
    fn hamming_le_identical() {
        assert!(hamming_le(b"ACGTACGTACGT", b"ACGTACGTACGT", 0));
    }

    #[test]
    fn hamming_le_one_mismatch_within_budget() {
        // last base differs: T vs G — 1 mismatch allowed
        assert!(hamming_le(b"ACGT", b"ACGG", 1));
    }

    #[test]
    fn hamming_le_one_mismatch_over_budget() {
        // 1 mismatch but 0 allowed
        assert!(!hamming_le(b"ACGT", b"ACGG", 0));
    }

    #[test]
    fn hamming_le_two_mismatches_one_allowed() {
        // C→C (ok), G→T, T→C — 2 mismatches, only 1 allowed
        assert!(!hamming_le(b"ACGT", b"ACTC", 1));
    }

    #[test]
    fn hamming_le_all_different_zero_allowed() {
        assert!(!hamming_le(b"AAAA", b"CCCC", 0));
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

    // ── cluster_repeats ───────────────────────────────────────────────────────

    fn make_hit(pos: usize) -> RepeatHit {
        RepeatHit {
            seq_id: "contig1".into(),
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

    // ── check_spacer_similarity ───────────────────────────────────────────────

    #[test]
    fn spacer_similarity_distinct_spacers_pass() {
        // Completely different sequences → low identity → passes
        let spacers = vec![
            "ACGTACGTACGTACGTACGTACGTACGT".to_string(),
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTT".to_string(),
        ];
        assert!(check_spacer_similarity(&spacers, 60.0));
    }

    #[test]
    fn spacer_similarity_identical_spacers_fail() {
        // 100 % identity → above any threshold → fails
        let spacers = vec![
            "ACGTACGTACGTACGT".to_string(),
            "ACGTACGTACGTACGT".to_string(),
        ];
        assert!(!check_spacer_similarity(&spacers, 60.0));
    }

    #[test]
    fn spacer_similarity_single_spacer_passes() {
        // Nothing to compare against → trivially passes
        let spacers = vec!["ACGTACGTACGTACGT".to_string()];
        assert!(check_spacer_similarity(&spacers, 60.0));
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
    fn find_direct_repeats_three_identical_drs_found() {
        // Three identical DRs → all three occurrences must be reported.
        // The pair-finder can return DR boundaries shifted by ±1-2 bp due to
        // tandem-repeat flanks consuming a shared boundary byte, so we use a
        // small positional tolerance here (same approach as compare_ecoli_*).
        let sp1 = b"CAGTTTCAGCGATACGATCGATCGATCGTTTT"; // 32 bp
        let sp2 = b"AAAACCCCGGGGTTTTACGATCGATCGATCGA"; // 32 bp
        let seq = build_crispr_seq(&[DR, DR, DR], &[sp1, sp2], 100);

        let params = default_params();
        let hits = find_direct_repeats(&seq, "contig1", &params);

        // Positions are 1-based: DR1 starts at byte 100 → pos1=101, etc.
        // Allow ±2 bp tolerance for boundary jitter.
        let dr1_pos: usize = 101;
        let dr2_pos: usize = 101 + 28 + 32; // 161
        let dr3_pos: usize = 101 + 28 + 32 + 28 + 32; // 221
        const TOL: usize = 2;
        for expected in [dr1_pos, dr2_pos, dr3_pos] {
            assert!(
                hits.iter()
                    .any(|h| { (h.pos1 as isize - expected as isize).unsigned_abs() <= TOL }),
                "DR near pos1={} (±{TOL}) not found; all hits: {:?}",
                expected,
                hits
            );
        }
    }

    #[test]
    fn find_direct_repeats_one_mismatch_in_third_copy() {
        // Third DR has 1 mismatch at its last base (T→C)
        let dr_mut = b"GTTCACTGCCGTATAGGCAGCTAAGAAC"; // 1 mismatch
        let sp1 = b"CAGTTTCAGCGATACGATCGATCGATCGTTTT"; // 32 bp
        let sp2 = b"AAAACCCCGGGGTTTTACGATCGATCGATCGA"; // 32 bp
        let seq = build_crispr_seq(&[DR, DR, dr_mut], &[sp1, sp2], 100);

        let params = default_params(); // no_mismatch = false → 1 mismatch allowed
        let hits = find_direct_repeats(&seq, "contig1", &params);

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

        let hits = find_direct_repeats(&seq, "contig1", &params);
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

    #[test]
    fn find_direct_repeats_four_dr_array() {
        // 4-DR array with 3 spacers
        let sp1 = b"CAGTTTCAGCGATACGATCGATCGATCGTTTT"; // 32 bp
        let sp2 = b"AAAACCCCGGGGTTTTACGATCGATCGATCGA"; // 32 bp
        let sp3 = b"GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"; // 32 bp
        let seq = build_crispr_seq(&[DR, DR, DR, DR], &[sp1, sp2, sp3], 100);

        let params = default_params();
        let hits = find_direct_repeats(&seq, "contig1", &params);

        assert!(
            hits.len() >= 4,
            "expected ≥4 DR hits for 4-repeat array, got {}: {:?}",
            hits.len(),
            hits
        );
    }

    #[test]
    fn find_direct_repeats_sequence_too_short_returns_empty() {
        // Sequence shorter than 2*min_repeat_length + min_spacer_length = 2*23 + 25 = 71 bp
        let seq = vec![b'A'; 50];
        let params = default_params();
        assert!(find_direct_repeats(&seq, "tiny", &params).is_empty());
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
        let _ = find_direct_repeats(&seq, "rand", &params);
    }

    // ── E. coli comparison against CRISPRCasFinder v4.2.30 reference ─────────
    //
    // Reference: tests/data/639085007760113337/result.json
    // Run with:  cargo test compare_ecoli -- --nocapture
    //
    // Exact DR start positions taken directly from result.json "Regions" blocks.
    #[test]
    fn compare_ecoli_against_reference() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let fasta = std::path::Path::new(manifest).join("tests/data/ecoli.fasta");
        if !fasta.exists() {
            eprintln!("Skipping: tests/data/ecoli.fasta not found");
            return;
        }

        let params = default_params();
        let gap_threshold = 1500usize;

        // Run detection over all contigs
        let reader = bio::io::fasta::Reader::from_file(&fasta).expect("open ecoli.fasta");
        let mut all_hits: Vec<RepeatHit> = Vec::new();
        for result in reader.records() {
            let record = result.expect("read FASTA record");
            let hits = find_direct_repeats(record.seq(), record.id(), &params);
            all_hits.extend(hits);
        }
        let clusters = cluster_repeats(&all_hits, gap_threshold);

        // ── Reference data hard-coded from result.json ────────────────────────
        struct RefArray {
            label: &'static str,
            // 1-based start position of every DR (from JSON "Start" field of DR regions)
            dr_starts: &'static [usize],
            dr_len: usize,
            evidence: u8,
        }
        let reference = [
            RefArray {
                label: "CRISPR_1",
                evidence: 1,
                dr_len: 43,
                dr_starts: &[411076, 411178],
            },
            RefArray {
                label: "CRISPR_2",
                evidence: 1,
                dr_len: 27,
                dr_starts: &[551366, 551436],
            },
            RefArray {
                label: "CRISPR_3",
                evidence: 1,
                dr_len: 39,
                dr_starts: &[2347201, 2347289],
            },
            RefArray {
                label: "CRISPR_4",
                evidence: 4,
                dr_len: 29,
                dr_starts: &[
                    2877701, 2877762, 2877823, 2877884, 2877945, 2878006, 2878067, 2878128,
                    2878190, 2878252, 2878313, 2878374, 2878435,
                ],
            },
            RefArray {
                label: "CRISPR_5",
                evidence: 4,
                dr_len: 28,
                dr_starts: &[
                    2904014, 2904075, 2904136, 2904197, 2904258, 2904319, 2904380,
                ],
            },
        ];

        // Tolerance: a cluster must overlap within this many bp of the ref array
        // boundaries, and each ref DR position must have a hit within dr_pos_tol.
        let cluster_tol = 200usize;
        let dr_pos_tol = 10usize;

        println!();
        println!(
            "E. coli — CRISPRCasFinder v4.2.30 vs our implementation  ({} total clusters)",
            clusters.len()
        );
        println!("{}", "─".repeat(112));
        println!(
            "{:<12}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}%  {}",
            "Array", "RefDRs", "OurDRs", "RefSPs", "OurSPs", "DRhit", "Notes"
        );
        println!("{}", "─".repeat(112));

        let mut failures: Vec<String> = Vec::new();

        for exp in &reference {
            let ref_start = exp.dr_starts[0];
            let ref_end = exp.dr_starts.last().unwrap() + exp.dr_len - 1;
            let ref_spacers = exp.dr_starts.len() - 1;

            // Find the cluster whose range overlaps the reference array
            let matched = clusters
                .iter()
                .find(|&&(cs, ce)| cs <= ref_end + cluster_tol && ce + cluster_tol >= ref_start);

            match matched {
                None => {
                    println!(
                        "{:<12}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}%  NOT FOUND",
                        exp.label,
                        exp.dr_starts.len(),
                        "—",
                        ref_spacers,
                        "—",
                        0
                    );
                    failures.push(format!("{}: cluster not found", exp.label));
                }
                Some(&(cs, ce)) => {
                    // Collect our hits inside (or just outside) the matched cluster
                    let mut our_hits: Vec<&RepeatHit> = all_hits
                        .iter()
                        .filter(|h| {
                            h.pos1 >= cs.saturating_sub(cluster_tol)
                                && h.pos1 <= ce + cluster_tol + exp.dr_len
                        })
                        .collect();
                    our_hits.sort_by_key(|h| h.pos1);
                    let our_drs = our_hits.len();
                    let our_spacers = our_drs.saturating_sub(1);

                    // Check every reference DR position has a close hit
                    let mut missing: Vec<usize> = Vec::new();
                    for &ref_pos in exp.dr_starts {
                        let found = our_hits.iter().any(|h| {
                            (h.pos1 as isize - ref_pos as isize).abs() <= dr_pos_tol as isize
                        });
                        if !found {
                            missing.push(ref_pos);
                        }
                    }
                    let matched_pct =
                        100 * (exp.dr_starts.len() - missing.len()) / exp.dr_starts.len();

                    println!(
                        "{:<12}  {:>6}  {:>6}  {:>6}  {:>6}  {:>5}%  {}",
                        exp.label,
                        exp.dr_starts.len(),
                        our_drs,
                        ref_spacers,
                        our_spacers,
                        matched_pct,
                        if missing.is_empty() {
                            format!(
                                "all {} DRs matched (±{} bp)",
                                exp.dr_starts.len(),
                                dr_pos_tol
                            )
                        } else {
                            format!("MISSING DR starts: {:?}", missing)
                        }
                    );

                    if !missing.is_empty() {
                        failures.push(format!(
                            "{}: {} DR positions unmatched: {:?}",
                            exp.label,
                            missing.len(),
                            missing
                        ));
                    }
                    // For high-confidence arrays check spacer count is within ±2
                    if exp.evidence >= 4 {
                        let diff = (our_spacers as isize - ref_spacers as isize).abs();
                        if diff > 2 {
                            failures.push(format!(
                                "{}: spacer count mismatch — ref={} ours={}",
                                exp.label, ref_spacers, our_spacers
                            ));
                        }
                    }
                }
            }
        }

        println!("{}", "─".repeat(112));
        if failures.is_empty() {
            println!("All reference arrays reproduced correctly.");
        } else {
            for f in &failures {
                println!("FAIL: {}", f);
            }
        }

        assert!(
            failures.is_empty(),
            "Detection diverged from CRISPRCasFinder reference:\n{}",
            failures.join("\n")
        );
    }
}
