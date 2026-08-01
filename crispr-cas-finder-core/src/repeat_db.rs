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

fn sequence_to_repeat_id_map() -> &'static HashMap<String, String> {
    SEQ_TO_ID.get_or_init(|| {
        let mut repeat_id_by_sequence = HashMap::new();
        for line in REPEAT_LIST_CSV.lines() {
            // Skip the header and blank lines
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let columns: Vec<&str> = line.split(';').collect();
            if columns.len() < 2 {
                continue;
            }
            let repeat_sequence = columns[0].trim().to_uppercase();
            let repeat_id = columns[1].trim().to_string();
            if !repeat_sequence.is_empty() && !repeat_id.is_empty() {
                repeat_id_by_sequence.insert(repeat_sequence, repeat_id);
            }
        }
        repeat_id_by_sequence
    })
}

fn repeat_id_to_direction_map() -> &'static HashMap<String, String> {
    ID_TO_DIRECTION.get_or_init(|| {
        let mut direction_by_repeat_id = HashMap::new();
        for line in REPEAT_DIRECTION_TSV.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let mut columns = line.splitn(2, '\t');
            let repeat_id = match columns.next() {
                Some(s) => s.trim().to_string(),
                None => continue,
            };
            let direction = match columns.next() {
                Some(s) => s.trim().to_string(),
                None => continue,
            };
            if !repeat_id.is_empty() {
                direction_by_repeat_id.insert(repeat_id, direction);
            }
        }
        direction_by_repeat_id
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

/// Flip a normalised direction symbol: `"+"` ↔ `"-"`, `"ND"` stays `"ND"`.
fn flipped_direction_symbol(direction: &str) -> String {
    match direction {
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
    let normalized_repeat_sequence = consensus_repeat.trim().to_uppercase();
    let repeat_id_by_sequence = sequence_to_repeat_id_map();
    let direction_by_repeat_id = repeat_id_to_direction_map();

    // ── Forward match ──────────────────────────────────────────────────────
    if let Some(repeat_id) = repeat_id_by_sequence
        .get(&normalized_repeat_sequence)
        .cloned()
    {
        let crispr_direction = match direction_by_repeat_id.get(&repeat_id) {
            Some(raw_direction) => normalized_direction_symbol(raw_direction),
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
    let reverse_complement_sequence = String::from_utf8(crate::dna::reverse_complement(
        normalized_repeat_sequence.as_bytes(),
    ))
    .expect("normalized DNA repeat is ASCII");
    if let Some(repeat_id) = repeat_id_by_sequence
        .get(&reverse_complement_sequence)
        .cloned()
    {
        let crispr_direction = match direction_by_repeat_id.get(&repeat_id) {
            Some(raw_direction) => {
                flipped_direction_symbol(&normalized_direction_symbol(raw_direction))
            }
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
fn normalized_direction_symbol(raw_direction: &str) -> String {
    let raw_direction = raw_direction.trim();
    if raw_direction.starts_with('F') {
        "+".to_string()
    } else if raw_direction.starts_with('R') {
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
