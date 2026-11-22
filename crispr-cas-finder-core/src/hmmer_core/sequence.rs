// sequence.rs — Digitized biological sequence
//
// Replaces the C-ported EslSq with an idiomatic Rust type:
//   - 0-based residue storage (no sentinel padding)

use crate::hmmer_core::alphabet::{Alphabet, Dsq};

/// A digitized biological sequence.
///
/// Residues are stored 0-based in `dsq[0..len]` as digital codes
/// produced by [`Alphabet::digitize`]. No sentinel values.
#[derive(Debug, Clone)]
pub struct DigitalSequence {
    /// Sequence name / identifier.
    pub name: String,
    /// Digitized residue codes, 0-based: `dsq[0..len]`.
    pub residues: Vec<Dsq>,
}

impl DigitalSequence {
    /// Create a digital sequence from raw ASCII bytes.
    ///
    /// Each byte is digitized through the alphabet's lookup table.
    pub fn from_bytes(name: &str, sequence_bytes: &[u8], alphabet: &Alphabet) -> Self {
        DigitalSequence {
            name: name.to_string(),
            residues: sequence_bytes
                .iter()
                .map(|&byte| alphabet.digitize(byte))
                .collect(),
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
        let digital_sequence = DigitalSequence::from_bytes("test", b"ACDEF", &amino_alphabet);
        assert_eq!(digital_sequence.name, "test");
        assert_eq!(digital_sequence.len(), 5);
        assert!(!digital_sequence.is_empty());
        assert_eq!(digital_sequence.residues[0], amino_alphabet.digitize(b'A'));
        assert_eq!(digital_sequence.residues[4], amino_alphabet.digitize(b'F'));
    }

    #[test]
    fn test_from_bytes_dna() {
        let dna_alphabet = Alphabet::dna();
        let digital_sequence = DigitalSequence::from_bytes("dna1", b"ACGT", &dna_alphabet);
        assert_eq!(digital_sequence.residues, [0, 1, 2, 3]); // A, C, G, T
    }

    #[test]
    fn test_empty_sequence() {
        let amino_alphabet = Alphabet::amino();
        let digital_sequence = DigitalSequence::from_bytes("empty", b"", &amino_alphabet);
        assert!(digital_sequence.is_empty());
        assert_eq!(digital_sequence.len(), 0);
    }
}
