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

/// Return whether `base` is covered by the IUPAC nucleotide code `code`.
///
/// `.` is accepted as a synonym for `N` so upstream-style patterns such as
/// `AA.` keep working. Matching is case-insensitive.
pub(crate) fn iupac_matches(code: u8, base: u8) -> bool {
    let allowed: &[u8] = match code.to_ascii_uppercase() {
        b'A' => b"A",
        b'C' => b"C",
        b'G' => b"G",
        b'T' | b'U' => b"T",
        b'R' => b"AG",
        b'Y' => b"CT",
        b'S' => b"CG",
        b'W' => b"AT",
        b'K' => b"GT",
        b'M' => b"AC",
        b'B' => b"CGT",
        b'D' => b"AGT",
        b'H' => b"ACT",
        b'V' => b"ACG",
        b'N' | b'.' => b"ACGT",
        _ => return false,
    };
    allowed.contains(&base.to_ascii_uppercase())
}

/// Return whether every byte of `pattern` is an IUPAC nucleotide code or `.`.
pub(crate) fn is_iupac_pattern(pattern: &str) -> bool {
    pattern
        .bytes()
        .all(|code| b"ACGTURYSWKMBDHVN.".contains(&code.to_ascii_uppercase()))
}

/// Return whether `sequence` starts with the IUPAC `pattern`.
pub(crate) fn starts_with_iupac(sequence: &[u8], pattern: &[u8]) -> bool {
    sequence.len() >= pattern.len()
        && pattern
            .iter()
            .zip(sequence)
            .all(|(&code, &base)| iupac_matches(code, base))
}

/// Return whether `sequence` ends with the IUPAC `pattern`.
pub(crate) fn ends_with_iupac(sequence: &[u8], pattern: &[u8]) -> bool {
    sequence.len() >= pattern.len()
        && starts_with_iupac(&sequence[sequence.len() - pattern.len()..], pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iupac_patterns_match_prefixes_and_suffixes() {
        assert!(starts_with_iupac(b"gTTT", b"G"));
        assert!(!starts_with_iupac(b"ATTT", b"G"));
        assert!(ends_with_iupac(b"GTTAAC", b"AA."));
        assert!(ends_with_iupac(b"GTTAAC", b"RAN"));
        assert!(!ends_with_iupac(b"GTTCAC", b"AA."));
        assert!(!ends_with_iupac(b"AC", b"AA."));
        assert!(is_iupac_pattern("AA."));
        assert!(!is_iupac_pattern("A[CG]"));
    }

    #[test]
    fn reverse_complement_handles_case_and_unknown_bases() {
        assert_eq!(reverse_complement(b"ACGTNacgtn"), b"nACGTNACGT");
    }
}
