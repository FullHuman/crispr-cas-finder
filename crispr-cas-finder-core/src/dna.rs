//! Shared DNA sequence helpers.

/// Return the reverse complement of a DNA sequence.
///
/// Canonical bases are normalized to uppercase; unrecognized bytes pass
/// through unchanged in reverse order.
pub(crate) fn reverse_complement(sequence: &[u8]) -> Vec<u8> {
    sequence
        .iter()
        .rev()
        .map(|base| match base {
            b'A' => b'T',
            b'a' => b'T',
            b'T' => b'A',
            b't' => b'A',
            b'C' => b'G',
            b'c' => b'G',
            b'G' => b'C',
            b'g' => b'C',
            other => *other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_complement_handles_case_and_unknown_bases() {
        assert_eq!(reverse_complement(b"ACGTNacgtn"), b"nACGTNACGT");
    }
}
