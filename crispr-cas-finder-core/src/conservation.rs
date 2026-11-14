//! Multiple alignment and conservation scores for repeats and spacers.
//!
//! CRISPRCasFinder aligns the repeats and the spacers of each locus with
//! MUSCLE and reads its thresholds off those alignments with BioPerl: whether
//! the spacers are too alike for an array, and the scores behind evidence
//! levels 2 to 4. The alignments here are those of MUSCLE 5.3, computed by
//! [`crate::muscle5`], so that a locus near a threshold falls on the same
//! side of it.

use serde::{Deserialize, Serialize};

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

/// An alignment of spacers already made, with the spacers it is of.
pub struct SpacerAlignment<'a> {
    pub spacers: Vec<&'a str>,
    pub rows: Vec<Vec<u8>>,
}

/// Compute all statistics for an array from its repeats and spacers.
pub fn array_statistics(consensus: &str, repeats: &[&str], spacers: &[&str]) -> ArrayStatistics {
    array_statistics_with(consensus, repeats, spacers, None)
}

/// [`array_statistics`], reusing `known` in place of aligning the spacers
/// again when it is the alignment of exactly these spacers.
pub fn array_statistics_with(
    consensus: &str,
    repeats: &[&str],
    spacers: &[&str],
    known: Option<&SpacerAlignment<'_>>,
) -> ArrayStatistics {
    let repeat_alignment = align_sequences(repeats);
    let aligned_again;
    let spacer_alignment = match known.filter(|known| known.spacers == spacers) {
        Some(known) => &known.rows,
        None => {
            aligned_again = align_sequences(spacers);
            &aligned_again
        }
    };
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
        spacer_conservation: entropy_conservation(spacer_alignment),
        spacer_identity: conserved_column_percent(spacer_alignment),
        spacer_average_identity: average_identity(spacer_alignment),
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

/// Align sequences as MUSCLE 5.3 does: equal-length uppercase rows padded
/// with `-`, in the order the sequences were given.
///
/// The order matters to MUSCLE, and CRISPRCasFinder hands it the repeats or
/// the spacers of a locus from left to right.
pub fn align_sequences(sequences: &[&str]) -> Vec<Vec<u8>> {
    let sequences: Vec<&[u8]> = sequences
        .iter()
        .map(|sequence| sequence.as_bytes())
        .collect();
    crate::muscle5::align(&sequences)
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

/// Average pairwise identity over the pairs of residues facing each other.
///
/// A residue facing a gap is left out, as in BioPerl's
/// `average_percentage_identity`, which CRISPRCasFinder applies to the
/// spacers of a locus to tell a tandem repeat from an array: a short spacer
/// that is part of a longer one counts as identical to it.
pub fn aligned_residue_identity(alignment: &[Vec<u8>]) -> f64 {
    let width = alignment.first().map_or(0, Vec::len);
    let (mut identical_pairs, mut compared_pairs) = (0usize, 0usize);
    for column in 0..width {
        let counts = column_counts(alignment, column);
        let residues: usize = counts.iter().sum();
        identical_pairs += counts
            .iter()
            .map(|&count| count * count.saturating_sub(1))
            .sum::<usize>();
        compared_pairs += residues * residues.saturating_sub(1);
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
    fn rows_are_uppercase_and_in_the_order_given() {
        let alignment = align_sequences(&["acgt", "ACGA"]);
        assert_eq!(rows(&alignment), ["ACGT", "ACGA"]);
        assert!(align_sequences(&[]).is_empty());
        assert_eq!(rows(&align_sequences(&["acgt"])), ["ACGT"]);
    }

    #[test]
    fn alignment_is_the_one_muscle_writes() {
        let sequences = ["ACGTACGTAC", "ACGTTACGTAC", "ACGACGTAC", "ACGTACGTACGG"];
        assert_eq!(
            rows(&align_sequences(&sequences)),
            [
                "ACGT-ACGTA--C",
                "ACGTTACGTA--C",
                "ACG--ACGTA--C",
                "ACGT-ACGTACGG"
            ]
        );
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
    fn duplicates_with_an_indel_align_across_it() {
        let alignment = align_sequences(&[
            "ACGTTGCAAGCTTGCATGCCTGCAGGTCGACT",
            "ACGTTGCAAGCTTGCTGCCTGCAGGTCGACT",
        ]);
        assert_eq!(aligned_residue_identity(&alignment), 100.0);
        assert_eq!(conserved_column_percent(&alignment), 96.875);
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
        assert_eq!(statistics.repeat_identity, 75.0);
        assert_eq!(statistics.spacer_identity, 0.0);
    }

    /// Identity MUSCLE 5.3 and BioPerl give CRISPRCasFinder for the spacers
    /// of three loci, the first two of which it drops as tandem repeats.
    #[test]
    fn spacers_are_as_identical_as_muscle_finds_them() {
        for (spacers, expected) in [
            // One spacer is the other shifted by a base.
            (
                vec![
                    "GTCCCTCAAGGGAGAGGGGATAAGTCCGCTGCGATGTTCTTCTCCCCA",
                    "CATCCCTCAAGGGAGAGGGGATAAGTCCGCTGCGATGTTCTTCTCCCG",
                ],
                95.7447,
            ),
            // The short spacer is two pieces of the long one.
            (
                vec![
                    "GAAACTAAAAAGGCTGAAGTTAAACCTTTAAAACAAGAAACTAAAAAAGTTGAAGTTAAAC",
                    "ACTAAAAAGGTTGAAGCTAGGATTGAAACTAAGA",
                ],
                73.5294,
            ),
            // MUSCLE 3.8 found these at 69.64, above the limit of 60;
            // MUSCLE 5.3 leaves them below it.
            (
                vec![
                    "TTGCCAGTTGTATTTCCACC",
                    "TCTTCAGTGGCTGCGGATGAAGAAGT",
                    "TCTGCAGTGGCTATGGATGAAGAAGT",
                ],
                56.0606,
            ),
        ] {
            let identity = aligned_residue_identity(&align_sequences(&spacers));
            assert!(
                (identity - expected).abs() < 0.0001,
                "{spacers:?}: {identity}"
            );
        }
    }

    #[test]
    fn spacers_of_one_length_are_aligned_too() {
        // The second spacer is the first shifted by a base: a gap at either
        // end lines them up, although they have the same length.
        let alignment = align_sequences(&[
            "GTCCCTCAAGGGAGAGGGGATAAGTCCGCTGCGATGTTCTTCTCCCCA",
            "CATCCCTCAAGGGAGAGGGGATAAGTCCGCTGCGATGTTCTTCTCCCG",
        ]);
        assert_eq!(alignment[0].len(), 49);
        assert_eq!(alignment[0][0], b'-');
        assert_eq!(alignment[1][45], b'-');
    }

    #[test]
    fn residue_identity_leaves_gaps_out() {
        let alignment = vec![b"ACGT--".to_vec(), b"ACGAAC".to_vec()];
        // Three of the four facing pairs agree.
        assert_eq!(aligned_residue_identity(&alignment), 75.0);
        // Counting the two bases facing gaps as mismatches gives half.
        assert_eq!(average_identity(&alignment), 50.0);
        assert_eq!(aligned_residue_identity(&[]), 0.0);
    }
}
