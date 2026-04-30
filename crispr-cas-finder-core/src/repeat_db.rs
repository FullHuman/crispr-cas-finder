//! CRISPR repeat database lookup for orientation inference.
//!
//! This module embeds two supplementary files from the original CRISPRCasFinder:
//!
//! * `Repeat_List.csv` — maps known CRISPR consensus repeat sequences to their
//!   canonical repeat ID (e.g., `R10`) and expert-annotated orientation.
//! * `repeatDirection.tsv` — maps repeat IDs to statistical orientations computed
//!   by the CRISPRDirection tool, with a confidence level.
//!
//! Together they provide a `lookup_repeat` function that, given a consensus
//! repeat sequence, returns the repeat ID and a normalised CRISPR direction
//! (`"+"`, `"-"`, or `"ND"`).

use std::collections::HashMap;
use std::sync::OnceLock;

/// Raw `Repeat_List.csv` embedded at compile time.
static REPEAT_LIST_CSV: &str = include_str!("../data/Repeat_List.csv");

/// Raw `repeatDirection.tsv` embedded at compile time.
static REPEAT_DIRECTION_TSV: &str = include_str!("../data/repeatDirection.tsv");

// ---------------------------------------------------------------------------
// Lazy-initialised lookup tables
// ---------------------------------------------------------------------------

/// Maps trimmed uppercase repeat sequence → repeat ID (e.g. "R10").
static SEQ_TO_ID: OnceLock<HashMap<String, String>> = OnceLock::new();

/// Maps repeat ID → raw direction string from `repeatDirection.tsv`
/// (e.g. `"F [0.37,0   Confidence: MEDIUM]"`).
static ID_TO_DIRECTION: OnceLock<HashMap<String, String>> = OnceLock::new();

fn seq_to_id_map() -> &'static HashMap<String, String> {
    SEQ_TO_ID.get_or_init(|| {
        let mut map = HashMap::new();
        for line in REPEAT_LIST_CSV.lines() {
            // Skip the header and blank lines
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(';').collect();
            if parts.len() < 2 {
                continue;
            }
            let seq = parts[0].trim().to_uppercase();
            let id = parts[1].trim().to_string();
            if !seq.is_empty() && !id.is_empty() {
                map.insert(seq, id);
            }
        }
        map
    })
}

fn id_to_direction_map() -> &'static HashMap<String, String> {
    ID_TO_DIRECTION.get_or_init(|| {
        let mut map = HashMap::new();
        for line in REPEAT_DIRECTION_TSV.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let mut iter = line.splitn(2, '\t');
            let id = match iter.next() {
                Some(s) => s.trim().to_string(),
                None => continue,
            };
            let direction = match iter.next() {
                Some(s) => s.trim().to_string(),
                None => continue,
            };
            if !id.is_empty() {
                map.insert(id, direction);
            }
        }
        map
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of a repeat-database lookup.
#[derive(Debug, Clone)]
pub struct RepeatLookup {
    /// The canonical repeat ID (e.g. `"R10"`), or `"Unknown"` if not found.
    pub repeat_id: String,
    /// Normalised orientation from CRISPRDirection: `"+"`, `"-"`, or `"ND"`.
    pub crispr_direction: String,
}

/// Reverse-complement a DNA string (uppercase ACGT; unknown bases pass through).
fn revcomp_str(seq: &str) -> String {
    seq.chars()
        .rev()
        .map(|c| match c {
            'A' => 'T',
            'T' => 'A',
            'C' => 'G',
            'G' => 'C',
            other => other,
        })
        .collect()
}

/// Flip a normalised direction symbol: `"+"` ↔ `"-"`, `"ND"` stays `"ND"`.
fn flip_direction(dir: &str) -> String {
    match dir {
        "+" => "-".to_string(),
        "-" => "+".to_string(),
        other => other.to_string(),
    }
}

/// Look up a consensus repeat sequence in the embedded database.
///
/// The sequence is normalised to uppercase before matching.  Both the
/// forward sequence and its reverse complement are tried:
/// * forward hit → direction as stored in the database
/// * RC hit      → direction flipped (the array is on the opposite strand)
///
/// If neither is found the function returns `repeat_id = "Unknown"` and
/// `crispr_direction = "ND"`.
pub fn lookup_repeat(consensus_repeat: &str) -> RepeatLookup {
    let key = consensus_repeat.trim().to_uppercase();
    let seq_map = seq_to_id_map();
    let dir_map = id_to_direction_map();

    // ── Forward match ──────────────────────────────────────────────────────
    if let Some(repeat_id) = seq_map.get(&key).cloned() {
        let crispr_direction = match dir_map.get(&repeat_id) {
            Some(raw) => normalise_direction(raw),
            None => "ND".to_string(),
        };
        return RepeatLookup {
            repeat_id,
            crispr_direction,
        };
    }

    // ── Reverse-complement match ────────────────────────────────────────────
    // The detection algorithm may produce a consensus that is the RC of the
    // canonical sequence stored in Repeat_List.csv.  In that case the array
    // is on the opposite strand, so the direction must be flipped.
    let rc_key = revcomp_str(&key);
    if let Some(repeat_id) = seq_map.get(&rc_key).cloned() {
        let crispr_direction = match dir_map.get(&repeat_id) {
            Some(raw) => flip_direction(&normalise_direction(raw)),
            None => "ND".to_string(),
        };
        return RepeatLookup {
            repeat_id,
            crispr_direction,
        };
    }

    RepeatLookup {
        repeat_id: "Unknown".to_string(),
        crispr_direction: "ND".to_string(),
    }
}

/// Convert the raw direction string from `repeatDirection.tsv` to the
/// canonical single-character symbol used in the original Perl script:
/// * starts with `"F"` → `"+"`
/// * starts with `"R"` → `"-"`
/// * anything else (including `"NA"`) → `"ND"`
fn normalise_direction(raw: &str) -> String {
    let raw = raw.trim();
    if raw.starts_with('F') {
        "+".to_string()
    } else if raw.starts_with('R') {
        "-".to_string()
    } else {
        "ND".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_repeat_is_found() {
        // R10 is listed in Repeat_List.csv with sequence "AAAAACCGCATCACTTATGATATGGA"
        // and in repeatDirection.tsv as "F [0.37,0   Confidence: MEDIUM]" → "+"
        let result = lookup_repeat("AAAAACCGCATCACTTATGATATGGA");
        assert_eq!(result.repeat_id, "R10");
        assert_eq!(result.crispr_direction, "+");
    }

    #[test]
    fn unknown_repeat_returns_defaults() {
        let result = lookup_repeat("ACGTACGTACGTACGT");
        assert_eq!(result.repeat_id, "Unknown");
        assert_eq!(result.crispr_direction, "ND");
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let upper = lookup_repeat("AAAAACCGCATCACTTATGATATGGA");
        let lower = lookup_repeat("aaaaaccgcatcacttatgatatgga");
        assert_eq!(upper.repeat_id, lower.repeat_id);
        assert_eq!(upper.crispr_direction, lower.crispr_direction);
    }

    #[test]
    fn reverse_repeat_returns_minus() {
        // R20: sequence "AAAAAGTGTTTCACTTTTGTCGTGCACTTTT" in Repeat_List.csv
        // R20 in repeatDirection.tsv: "R [0,0.74   Confidence: HIGH]" → "-"
        let result = lookup_repeat("AAAAAGTGTTTCACTTTTGTCGTGCACTTTT");
        assert_eq!(result.repeat_id, "R20");
        assert_eq!(result.crispr_direction, "-");
    }

    #[test]
    fn rc_of_forward_repeat_returns_minus() {
        // R10 canonical sequence is "AAAAACCGCATCACTTATGATATGGA" → direction "+"
        // If we supply its RC the lookup should still find R10 but return "-"
        // (the array was detected on the opposite strand).
        let canonical = "AAAAACCGCATCACTTATGATATGGA";
        let rc: String = canonical
            .chars()
            .rev()
            .map(|c| match c {
                'A' => 'T',
                'T' => 'A',
                'C' => 'G',
                'G' => 'C',
                other => other,
            })
            .collect();
        let result = lookup_repeat(&rc);
        assert_eq!(result.repeat_id, "R10");
        assert_eq!(result.crispr_direction, "-");
    }

    #[test]
    fn rc_of_reverse_repeat_returns_plus() {
        // R20 canonical sequence → direction "-".
        // Its RC should return "+" (flipped).
        let canonical = "AAAAAGTGTTTCACTTTTGTCGTGCACTTTT";
        let rc: String = canonical
            .chars()
            .rev()
            .map(|c| match c {
                'A' => 'T',
                'T' => 'A',
                'C' => 'G',
                'G' => 'C',
                other => other,
            })
            .collect();
        let result = lookup_repeat(&rc);
        assert_eq!(result.repeat_id, "R20");
        assert_eq!(result.crispr_direction, "+");
    }
}
