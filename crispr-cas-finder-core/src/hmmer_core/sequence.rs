// sequence.rs — Digitized biological sequence
//
// Replaces the C-ported EslSq with an idiomatic Rust type:
//   - 0-based residue storage (no sentinel padding)
//   - Option<String> for optional metadata
//   - Removed unused fields (seq, start, end, c, w)

use crate::hmmer_core::alphabet::{Alphabet, Dsq};

/// A digitized biological sequence.
///
/// Residues are stored 0-based in `dsq[0..len]` as digital codes
/// produced by [`Alphabet::digitize`]. No sentinel values.
#[derive(Debug, Clone)]
pub struct DigitalSequence {
    /// Sequence name / identifier.
    pub name: String,
    /// Accession number (if any).
    pub accession: Option<String>,
    /// Description line (if any).
    pub description: Option<String>,
    /// Digitized residue codes, 0-based: `dsq[0..len]`.
    pub residues: Vec<Dsq>,
    /// Source sequence length (may differ from `dsq.len()` for subsequences).
    pub source_length: i64,
    /// Sequence index in a database (for hit reporting).
    pub database_index: i64,
}

impl DigitalSequence {
    /// Create a digital sequence from raw ASCII bytes.
    ///
    /// Each byte is digitized through the alphabet's lookup table.
    pub fn from_bytes(
        name: &str,
        description: &str,
        sequence_bytes: &[u8],
        alphabet: &Alphabet,
    ) -> Self {
        let digitized_residues: Vec<Dsq> = sequence_bytes
            .iter()
            .map(|&byte| alphabet.digitize(byte))
            .collect();
        let sequence_length = digitized_residues.len();
        DigitalSequence {
            name: name.to_string(),
            accession: None,
            description: if description.is_empty() {
                None
            } else {
                Some(description.to_string())
            },
            residues: digitized_residues,
            source_length: sequence_length as i64,
            database_index: -1,
        }
    }

    /// Number of residues.
    #[inline]
    pub fn len(&self) -> usize {
        self.residues.len()
    }

    /// Whether the sequence is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.residues.is_empty()
    }

    /// Get the residue at position `i` (0-based).
    #[inline]
    pub fn residue(&self, i: usize) -> Dsq {
        self.residues[i]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmmer_core::alphabet::Alphabet;

    #[test]
    fn test_from_bytes_amino() {
        let amino_alphabet = Alphabet::amino();
        let digital_sequence =
            DigitalSequence::from_bytes("test", "a protein", b"ACDEF", &amino_alphabet);
        assert_eq!(digital_sequence.name, "test");
        assert_eq!(digital_sequence.description.as_deref(), Some("a protein"));
        assert_eq!(digital_sequence.accession, None);
        assert_eq!(digital_sequence.len(), 5);
        assert!(!digital_sequence.is_empty());
        assert_eq!(digital_sequence.residue(0), amino_alphabet.digitize(b'A'));
        assert_eq!(digital_sequence.residue(4), amino_alphabet.digitize(b'F'));
        assert_eq!(digital_sequence.source_length, 5);
        assert_eq!(digital_sequence.database_index, -1);
    }

    #[test]
    fn test_from_bytes_dna() {
        let dna_alphabet = Alphabet::dna();
        let digital_sequence = DigitalSequence::from_bytes("dna1", "", b"ACGT", &dna_alphabet);
        assert_eq!(digital_sequence.len(), 4);
        assert_eq!(digital_sequence.description, None);
        assert_eq!(digital_sequence.residue(0), 0); // A
        assert_eq!(digital_sequence.residue(1), 1); // C
        assert_eq!(digital_sequence.residue(2), 2); // G
        assert_eq!(digital_sequence.residue(3), 3); // T
    }

    #[test]
    fn test_empty_sequence() {
        let amino_alphabet = Alphabet::amino();
        let digital_sequence = DigitalSequence::from_bytes("empty", "", b"", &amino_alphabet);
        assert!(digital_sequence.is_empty());
        assert_eq!(digital_sequence.len(), 0);
    }
}
