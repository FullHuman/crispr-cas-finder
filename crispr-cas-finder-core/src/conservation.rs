//! Multiple alignment and conservation scores for repeats and spacers.
//!
//! Pure-Rust replacement for the MUSCLE + BioPerl step of CRISPRCasFinder,
//! which aligns the repeats and the spacers of each array and derives the
//! scores behind evidence levels 2 to 4.

use bio::alignment::AlignmentOperation;
use bio::alignment::pairwise::{Aligner, Scoring};
use serde::{Deserialize, Serialize};

const GAP: u8 = b'-';

/// Conservation scores and length statistics for one CRISPR array.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ArrayStatistics {
    /// Entropy-based conservation of the repeats, 0 to 100 (100 = identical).
    pub repeat_conservation: f64,
    /// Percentage of repeat alignment columns identical in every repeat.
    pub repeat_identity: f64,
    /// Entropy-based conservation of the spacers, 0 to 100.
    pub spacer_conservation: f64,
    /// Percentage of spacer alignment columns identical in every spacer.
    pub spacer_identity: f64,
    /// Average pairwise identity of the aligned spacers, 0 to 100.
    pub spacer_average_identity: f64,
    /// Mean spacer length in bp.
    pub mean_spacer_length: f64,
    /// Sample standard deviation of the spacer lengths.
    pub spacer_length_std_dev: f64,
    /// Number of repeats identical to the consensus.
    pub exact_repeat_count: usize,
    /// Whether at least two spacers have the same sequence.
    pub has_duplicated_spacers: bool,
}

/// Minimum repeat conservation for evidence levels 3 and 4.
pub const MIN_REPEAT_CONSERVATION: f64 = 70.0;
/// Maximum spacer identity for evidence level 4.
pub const MAX_SPACER_IDENTITY: f64 = 8.0;
/// Arrays with at most this many spacers are evidence level 1.
pub const MAX_SMALL_ARRAY_SPACERS: usize = 3;

/// Evidence level as defined by CRISPRCasFinder.
///
/// * 1: at most three spacers
/// * 2: poorly conserved repeats
/// * 3: conserved repeats but similar spacers
/// * 4: conserved repeats and dissimilar spacers
pub fn evidence_level(spacer_count: usize, statistics: &ArrayStatistics) -> usize {
    if spacer_count <= MAX_SMALL_ARRAY_SPACERS {
        1
    } else if statistics.repeat_conservation < MIN_REPEAT_CONSERVATION {
        2
    } else if statistics.spacer_identity > MAX_SPACER_IDENTITY {
        3
    } else {
        4
    }
}

/// Compute all statistics for an array from its repeats and spacers.
pub fn array_statistics(consensus: &str, repeats: &[&str], spacers: &[&str]) -> ArrayStatistics {
    let repeat_alignment = align_sequences(repeats);
    let spacer_alignment = align_sequences(spacers);
    let spacer_lengths: Vec<f64> = spacers.iter().map(|spacer| spacer.len() as f64).collect();
    let mean_spacer_length = mean(&spacer_lengths);

    let mut sorted_spacers: Vec<String> = spacers
        .iter()
        .map(|spacer| spacer.to_ascii_uppercase())
        .collect();
    sorted_spacers.sort_unstable();
    let has_duplicated_spacers = sorted_spacers.windows(2).any(|pair| pair[0] == pair[1]);

    ArrayStatistics {
        repeat_conservation: entropy_conservation(&repeat_alignment),
        repeat_identity: conserved_column_percent(&repeat_alignment),
        spacer_conservation: entropy_conservation(&spacer_alignment),
        spacer_identity: conserved_column_percent(&spacer_alignment),
        spacer_average_identity: average_identity(&spacer_alignment),
        mean_spacer_length,
        spacer_length_std_dev: sample_std_dev(&spacer_lengths, mean_spacer_length),
        exact_repeat_count: repeats
            .iter()
            .filter(|repeat| repeat.eq_ignore_ascii_case(consensus))
            .count(),
        has_duplicated_spacers,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_std_dev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let squared: f64 = values.iter().map(|value| (value - mean).powi(2)).sum();
    (squared / (values.len() - 1) as f64).sqrt()
}

/// Align sequences into equal-length uppercase rows padded with `-`.
///
/// Sequences of equal length are stacked without gaps, which is what repeats
/// of one array always are. Otherwise a center-star alignment is built: every
/// sequence is aligned globally to the sequence of median length and the
/// pairwise alignments are merged on the center's coordinates.
pub fn align_sequences(sequences: &[&str]) -> Vec<Vec<u8>> {
    let rows: Vec<Vec<u8>> = sequences
        .iter()
        .map(|sequence| sequence.as_bytes().to_ascii_uppercase())
        .collect();
    let Some(first_length) = rows.first().map(Vec::len) else {
        return rows;
    };
    if rows.iter().all(|row| row.len() == first_length) {
        return rows;
    }

    let mut by_length: Vec<usize> = (0..rows.len()).collect();
    by_length.sort_by_key(|&index| (rows[index].len(), index));
    let center_index = by_length[by_length.len() / 2];
    let center = &rows[center_index];

    // Gaps are deliberately expensive. Spacers of one array are unrelated, and
    // cheap gaps let the aligner manufacture identical columns between random
    // sequences, which would push small arrays over `MAX_SPACER_IDENTITY`.
    // Real duplicates differing by an indel still align across the gap.
    let scoring = Scoring::from_scores(-40, -10, 5, -4);
    let mut aligner = Aligner::with_scoring(scoring);

    // For each sequence: residues aligned to each center position, and
    // residues inserted before each center position (index `center.len()`
    // holds insertions after the last one).
    let mut aligned: Vec<Vec<u8>> = Vec::with_capacity(rows.len());
    let mut insertions: Vec<Vec<Vec<u8>>> = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let mut on_center = vec![GAP; center.len()];
        let mut inserted = vec![Vec::new(); center.len() + 1];
        if index == center_index {
            on_center.copy_from_slice(center);
        } else {
            let alignment = aligner.global(row, center);
            let (mut row_position, mut center_position) = (0, 0);
            for operation in alignment.operations {
                match operation {
                    AlignmentOperation::Match | AlignmentOperation::Subst => {
                        on_center[center_position] = row[row_position];
                        row_position += 1;
                        center_position += 1;
                    }
                    AlignmentOperation::Ins => {
                        inserted[center_position].push(row[row_position]);
                        row_position += 1;
                    }
                    AlignmentOperation::Del => center_position += 1,
                    AlignmentOperation::Xclip(_) | AlignmentOperation::Yclip(_) => {}
                }
            }
        }
        aligned.push(on_center);
        insertions.push(inserted);
    }

    let insertion_widths: Vec<usize> = (0..=center.len())
        .map(|position| {
            insertions
                .iter()
                .map(|inserted| inserted[position].len())
                .max()
                .unwrap_or(0)
        })
        .collect();

    aligned
        .iter()
        .zip(&insertions)
        .map(|(on_center, inserted)| {
            let mut row = Vec::new();
            for position in 0..=center.len() {
                row.extend_from_slice(&inserted[position]);
                row.resize(
                    row.len() + insertion_widths[position] - inserted[position].len(),
                    GAP,
                );
                if position < center.len() {
                    row.push(on_center[position]);
                }
            }
            row
        })
        .collect()
}

fn column_counts(alignment: &[Vec<u8>], column: usize) -> [usize; 26] {
    let mut counts = [0usize; 26];
    for row in alignment {
        if let Some(residue) = row
            .get(column)
            .filter(|residue| residue.is_ascii_uppercase())
        {
            counts[(residue - b'A') as usize] += 1;
        }
    }
    counts
}

/// Conservation as `100 * (1 - mean column entropy in bits)`, floored at 0.
///
/// Probabilities are taken over all rows, so gaps lower the score, exactly as
/// in the upstream `entropy` routine.
pub fn entropy_conservation(alignment: &[Vec<u8>]) -> f64 {
    let width = alignment.first().map_or(0, Vec::len);
    if width == 0 {
        return 0.0;
    }
    let row_count = alignment.len() as f64;
    let total: f64 = (0..width)
        .map(|column| {
            let entropy: f64 = column_counts(alignment, column)
                .iter()
                .filter(|&&count| count > 0)
                .map(|&count| {
                    let count = count as f64;
                    -(count / row_count) * (count / row_count).log2()
                })
                .sum();
            1.0 - entropy
        })
        .sum();
    (total / width as f64 * 100.0).max(0.0)
}

/// Percentage of columns in which every row carries the same residue.
pub fn conserved_column_percent(alignment: &[Vec<u8>]) -> f64 {
    let width = alignment.first().map_or(0, Vec::len);
    if width == 0 {
        return 0.0;
    }
    let conserved = (0..width)
        .filter(|&column| column_counts(alignment, column).contains(&alignment.len()))
        .count();
    conserved as f64 / width as f64 * 100.0
}

/// Average pairwise identity over all columns.
///
/// A residue facing a gap counts as a mismatch; two gaps are ignored.
/// BioPerl ignores residue-gap pairs, which lets an aligner raise the score
/// of unrelated sequences simply by opening gaps.
pub fn average_identity(alignment: &[Vec<u8>]) -> f64 {
    let width = alignment.first().map_or(0, Vec::len);
    let row_count = alignment.len();
    let (mut identical_pairs, mut compared_pairs) = (0usize, 0usize);
    for column in 0..width {
        let counts = column_counts(alignment, column);
        let residues: usize = counts.iter().sum();
        let gaps = row_count - residues;
        identical_pairs += counts
            .iter()
            .map(|&count| count * count.saturating_sub(1))
            .sum::<usize>();
        compared_pairs += row_count * row_count.saturating_sub(1) - gaps * gaps.saturating_sub(1);
    }
    if compared_pairs == 0 {
        return 0.0;
    }
    identical_pairs as f64 / compared_pairs as f64 * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(alignment: &[Vec<u8>]) -> Vec<String> {
        alignment
            .iter()
            .map(|row| String::from_utf8(row.clone()).unwrap())
            .collect()
    }

    #[test]
    fn equal_length_sequences_are_stacked_without_gaps() {
        let alignment = align_sequences(&["acgt", "ACGA"]);
        assert_eq!(rows(&alignment), ["ACGT", "ACGA"]);
    }

    #[test]
    fn center_star_alignment_is_flush_and_preserves_residues() {
        let sequences = ["ACGTACGTAC", "ACGTTACGTAC", "ACGACGTAC", "ACGTACGTACGG"];
        let alignment = align_sequences(&sequences);
        let width = alignment[0].len();
        for (row, sequence) in alignment.iter().zip(sequences) {
            assert_eq!(row.len(), width);
            let ungapped: Vec<u8> = row.iter().copied().filter(|&base| base != GAP).collect();
            assert_eq!(ungapped, sequence.as_bytes());
        }
    }

    #[test]
    fn identical_rows_are_fully_conserved() {
        let alignment = align_sequences(&["ACGT", "ACGT", "ACGT"]);
        assert_eq!(entropy_conservation(&alignment), 100.0);
        assert_eq!(conserved_column_percent(&alignment), 100.0);
        assert_eq!(average_identity(&alignment), 100.0);
    }

    #[test]
    fn half_split_column_costs_one_bit() {
        // One of two columns is split evenly: entropy 1 bit, conservation 0.
        let alignment = align_sequences(&["AA", "AC"]);
        assert_eq!(entropy_conservation(&alignment), 50.0);
        assert_eq!(conserved_column_percent(&alignment), 50.0);
        assert_eq!(average_identity(&alignment), 50.0);
    }

    #[test]
    fn residue_facing_a_gap_counts_as_a_mismatch() {
        let alignment = vec![b"AC-".to_vec(), b"ACG".to_vec()];
        assert!((average_identity(&alignment) - 200.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn duplicates_with_an_indel_still_align() {
        let alignment = align_sequences(&[
            "ACGTTGCAAGCTTGCATGCCTGCAGGTCGACT",
            "ACGTTGCAAGCTTGCTGCCTGCAGGTCGACT",
        ]);
        assert!(average_identity(&alignment) > 95.0);
    }

    #[test]
    fn evidence_levels_follow_upstream_thresholds() {
        let conserved = ArrayStatistics {
            repeat_conservation: 95.0,
            spacer_identity: 0.0,
            ..ArrayStatistics::default()
        };
        assert_eq!(evidence_level(3, &conserved), 1);
        assert_eq!(evidence_level(4, &conserved), 4);
        let similar_spacers = ArrayStatistics {
            spacer_identity: 8.5,
            ..conserved.clone()
        };
        assert_eq!(evidence_level(4, &similar_spacers), 3);
        let degenerate_repeats = ArrayStatistics {
            repeat_conservation: 69.9,
            ..conserved
        };
        assert_eq!(evidence_level(4, &degenerate_repeats), 2);
    }

    #[test]
    fn statistics_report_lengths_duplicates_and_exact_repeats() {
        let statistics = array_statistics(
            "ACGT",
            &["ACGT", "ACGT", "ACGA"],
            &["TTTTGG", "ttttgg", "CCCCAAAA"],
        );
        assert_eq!(statistics.exact_repeat_count, 2);
        assert!(statistics.has_duplicated_spacers);
        assert!((statistics.mean_spacer_length - 20.0 / 3.0).abs() < 1e-12);
        assert!((statistics.spacer_length_std_dev - (4.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }
}
