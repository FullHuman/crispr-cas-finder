// sequence.rs — Digitized biological sequence
//
// Replaces the C-ported EslSq with an idiomatic Rust type:
//   - 0-based residue storage (no sentinel padding)
//   - Option<String> for optional metadata
//   - Removed unused fields (seq, start, end, c, w)

use crate::alphabet::{Alphabet, Dsq};

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
    pub fn from_bytes(name: &str, desc: &str, seq_bytes: &[u8], abc: &Alphabet) -> Self {
        let dsq: Vec<Dsq> = seq_bytes.iter().map(|&b| abc.digitize(b)).collect();
        let n = dsq.len();
        DigitalSequence {
            name: name.to_string(),
            accession: None,
            description: if desc.is_empty() {
                None
            } else {
                Some(desc.to_string())
            },
            residues: dsq,
            source_length: n as i64,
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
    use crate::alphabet::Alphabet;

    #[test]
    fn test_from_bytes_amino() {
        let abc = Alphabet::amino();
        let seq = DigitalSequence::from_bytes("test", "a protein", b"ACDEF", &abc);
        assert_eq!(seq.name, "test");
        assert_eq!(seq.description.as_deref(), Some("a protein"));
        assert_eq!(seq.accession, None);
        assert_eq!(seq.len(), 5);
        assert!(!seq.is_empty());
        assert_eq!(seq.residue(0), abc.digitize(b'A'));
        assert_eq!(seq.residue(4), abc.digitize(b'F'));
        assert_eq!(seq.source_length, 5);
        assert_eq!(seq.database_index, -1);
    }

    #[test]
    fn test_from_bytes_dna() {
        let abc = Alphabet::dna();
        let seq = DigitalSequence::from_bytes("dna1", "", b"ACGT", &abc);
        assert_eq!(seq.len(), 4);
        assert_eq!(seq.description, None);
        assert_eq!(seq.residue(0), 0); // A
        assert_eq!(seq.residue(1), 1); // C
        assert_eq!(seq.residue(2), 2); // G
        assert_eq!(seq.residue(3), 3); // T
    }

    #[test]
    fn test_empty_sequence() {
        let abc = Alphabet::amino();
        let seq = DigitalSequence::from_bytes("empty", "", b"", &abc);
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
    }
}
