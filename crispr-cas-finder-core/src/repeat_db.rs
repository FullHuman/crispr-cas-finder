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

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Raw `Repeat_List.csv` embedded at compile time.
static REPEAT_LIST_CSV: &str = include_str!("../data/Repeat_List.csv");

/// Raw `repeatDirection.tsv` embedded at compile time.
static REPEAT_DIRECTION_TSV: &str = include_str!("../data/repeatDirection.tsv");

static EMBEDDED: OnceLock<RepeatDatabase> = OnceLock::new();

#[derive(Debug, Clone)]
struct RepeatEntry {
    id: String,
    crisprdb_count: usize,
}

/// Known consensus repeats with their IDs, CRISPRdb counts and directions.
#[derive(Debug, Clone, Default)]
pub struct RepeatDatabase {
    /// Uppercase repeat sequence → entry.
    by_sequence: HashMap<String, RepeatEntry>,
    /// Repeat ID → raw direction string (e.g. `"F [0.37,0   Confidence: MEDIUM]"`).
    direction_by_id: HashMap<String, String>,
}

impl RepeatDatabase {
    /// Parse a `Repeat_List.csv` (`sequence;id;count;orientation`) and a
    /// `repeatDirection.tsv` (`id<TAB>direction`). Header, blank and
    /// malformed lines are skipped.
    pub fn parse(repeat_list: &str, directions: &str) -> Self {
        let mut database = Self::default();
        for line in repeat_list.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let columns: Vec<&str> = line.split(';').collect();
            if columns.len() < 2 {
                continue;
            }
            let sequence: String = columns[0]
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>()
                .to_uppercase();
            let id = columns[1].trim().to_string();
            if sequence.is_empty() || id.is_empty() {
                continue;
            }
            let crisprdb_count = columns
                .get(2)
                .and_then(|count| count.trim().parse().ok())
                .unwrap_or(0);
            database
                .by_sequence
                .insert(sequence, RepeatEntry { id, crisprdb_count });
        }
        for line in directions.lines() {
            let Some((id, direction)) = line.split_once('\t') else {
                continue;
            };
            let id = id.trim();
            if !id.is_empty() {
                database
                    .direction_by_id
                    .insert(id.to_string(), direction.trim().to_string());
            }
        }
        database
    }

    /// The database shipped with CRISPRCasFinder.
    pub fn embedded() -> &'static Self {
        EMBEDDED.get_or_init(|| Self::parse(REPEAT_LIST_CSV, REPEAT_DIRECTION_TSV))
    }

    /// Number of repeat sequences in the database.
    pub fn len(&self) -> usize {
        self.by_sequence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_sequence.is_empty()
    }

    /// Look up a consensus repeat, case-insensitively.
    ///
    /// Both the sequence and its reverse complement are tried:
    /// * forward hit → direction as stored in the database
    /// * reverse-complement hit → direction flipped (the array is on the
    ///   opposite strand)
    ///
    /// If neither is found, `repeat_id` is `"Unknown"`, `crispr_direction`
    /// is `"ND"` and `crisprdb_count` is 0.
    pub fn lookup(&self, consensus_repeat: &str) -> RepeatLookup {
        let forward = consensus_repeat.trim().to_uppercase();
        let direction_of = |entry: &RepeatEntry| {
            self.direction_by_id
                .get(&entry.id)
                .map_or_else(|| "ND".to_string(), |raw| normalized_direction_symbol(raw))
        };
        if let Some(entry) = self.by_sequence.get(&forward) {
            return RepeatLookup {
                repeat_id: entry.id.clone(),
                crispr_direction: direction_of(entry),
                crisprdb_count: entry.crisprdb_count,
            };
        }
        // The detector may report the reverse complement of the stored
        // sequence; the array is then on the opposite strand.
        let reverse = String::from_utf8_lossy(&crate::dna::reverse_complement(forward.as_bytes()))
            .into_owned();
        if let Some(entry) = self.by_sequence.get(&reverse) {
            return RepeatLookup {
                repeat_id: entry.id.clone(),
                crispr_direction: flipped_direction_symbol(&direction_of(entry)),
                crisprdb_count: entry.crisprdb_count,
            };
        }
        RepeatLookup {
            repeat_id: "Unknown".to_string(),
            crispr_direction: "ND".to_string(),
            crisprdb_count: 0,
        }
    }
}

/// Result of a repeat-database lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepeatLookup {
    /// The canonical repeat ID (e.g. `"R10"`), or `"Unknown"` if not found.
    pub repeat_id: String,
    /// Normalised orientation from CRISPRDirection: `"+"`, `"-"`, or `"ND"`.
    pub crispr_direction: String,
    /// Number of CRISPRdb arrays sharing this repeat; 0 if not found.
    pub crisprdb_count: usize,
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
pub fn lookup_repeat(consensus_repeat: &str) -> RepeatLookup {
    RepeatDatabase::embedded().lookup(consensus_repeat)
}

/// Accession of a sequence as CRISPRdb stores it: the last `|`-separated
/// field without its version suffix (`gi|1|ref|NC_000913.3|` → `NC_000913`).
pub fn crisprdb_accession(seq_id: &str) -> &str {
    let last_field = seq_id
        .split('|')
        .rev()
        .find(|field| !field.is_empty())
        .unwrap_or(seq_id);
    last_field.split(['.', ':']).next().unwrap_or(last_field)
}

/// CRISPR arrays already recorded in CRISPRdb, from a `CRISPR_crisprdb.csv`.
#[derive(Debug, Clone, Default)]
pub struct KnownCrisprs {
    /// (accession, start, end)
    arrays: HashSet<(String, usize, usize)>,
}

impl KnownCrisprs {
    /// Parse the tab-separated CRISPRdb export, whose columns 2, 5 and 6 are
    /// the accession, array start and array end.
    pub fn parse(content: &str) -> Self {
        let mut known = Self::default();
        for line in content.lines() {
            let columns: Vec<&str> = line.split('\t').collect();
            if columns.len() < 6 {
                continue;
            }
            let (Ok(start), Ok(end)) = (
                columns[4].trim().parse::<usize>(),
                columns[5].trim().parse::<usize>(),
            ) else {
                continue;
            };
            known
                .arrays
                .insert((columns[1].trim().to_string(), start, end));
        }
        known
    }

    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    /// Whether an array with this start, and this end give or take one base,
    /// is recorded for the sequence.
    pub fn contains(&self, seq_id: &str, start: usize, end: usize) -> bool {
        let accession = crisprdb_accession(seq_id).to_string();
        [end.saturating_sub(1), end, end + 1]
            .into_iter()
            .any(|end| self.arrays.contains(&(accession.clone(), start, end)))
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

    #[test]
    fn lookup_reports_the_crisprdb_count() {
        let database = RepeatDatabase::parse(
            "#header\nACGTAC ;R1;7;\nGGGTTT;R2;x;\n",
            "R1\tF [0.3,0 Confidence: HIGH]\nR2\tR [0,0.3]\n",
        );
        assert_eq!(database.len(), 2);
        assert_eq!(
            database.lookup("acgtac"),
            RepeatLookup {
                repeat_id: "R1".into(),
                crispr_direction: "+".into(),
                crisprdb_count: 7,
            }
        );
        // Reverse complement of GGGTTT, stored as reverse: flipped to "+".
        assert_eq!(database.lookup("AAACCC").crispr_direction, "+");
        assert_eq!(database.lookup("AAACCC").crisprdb_count, 0);
        assert_eq!(database.lookup("TTTT").repeat_id, "Unknown");
    }

    #[test]
    fn accessions_drop_separators_and_versions() {
        assert_eq!(crisprdb_accession("NC_000913.3"), "NC_000913");
        assert_eq!(crisprdb_accession("gi|1|ref|NC_000913.3|"), "NC_000913");
        assert_eq!(crisprdb_accession("NC_000913.3:1-4641652"), "NC_000913");
        assert_eq!(crisprdb_accession("contig7"), "contig7");
    }

    #[test]
    fn known_arrays_match_start_and_end_within_one_base() {
        let known = KnownCrisprs::parse(
            "Specie\tRefSeq\tCRISPR_id\tHypothetical\tCrisprBegin\tCrisprEnd\n\
             E. coli\tNC_000913\tNC_000913_1\t0\t2877701\t2878463\n",
        );
        assert_eq!(known.len(), 1);
        assert!(known.contains("NC_000913.3", 2877701, 2878463));
        assert!(known.contains("NC_000913.3", 2877701, 2878464));
        assert!(!known.contains("NC_000913.3", 2877702, 2878463));
        assert!(!known.contains("NC_000914.1", 2877701, 2878463));
    }
}
