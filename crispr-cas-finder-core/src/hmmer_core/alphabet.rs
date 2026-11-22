use crate::hmmer_core::config::{MAX_CANONICAL_ALPHABET, MAX_FULL_ALPHABET};

/// Digitized residue code (0-based index into the alphabet).
pub type Dsq = u8;

/// Maximum canonical alphabet size (amino = 20).
pub const MAX_CANONICAL_SIZE: usize = MAX_CANONICAL_ALPHABET; // 20

/// Maximum full alphabet size including degeneracies (amino = 29).
pub const MAX_FULL_SIZE: usize = MAX_FULL_ALPHABET; // 29

// ---------------------------------------------------------------------------
// Alphabet
// ---------------------------------------------------------------------------

/// Biological sequence alphabet.
///
/// Knows how to digitize residue characters (case-insensitively) and
/// provides degeneracy information for scoring ambiguous residues.
#[derive(Debug, Clone)]
pub struct Alphabet {
    /// Which kind of alphabet this is.
    pub kind: AlphabetKind,

    /// Number of canonical residues (4 for DNA, 20 for amino).
    pub canonical_size: usize,

    /// Full alphabet size including gap, degeneracies, nonresidue, missing.
    pub full_size: usize,

    /// Symbol table: `sym[code]` gives the ASCII character for a digital code.
    pub symbols: [u8; MAX_FULL_SIZE],

    /// Flat degeneracy table stored row-major: `degen[x * K_MAX + a]`.
    /// True if digital code `x` includes canonical residue `a`.
    pub degen: [bool; MAX_FULL_SIZE * MAX_CANONICAL_SIZE],

    /// ASCII → digital code lookup (128 entries, case-folding).
    /// Unknown characters map to `kp - 1` (missing data).
    input_map: [Dsq; 128],
}

/// The kind of biological alphabet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphabetKind {
    Amino,
    Dna,
    Rna,
}

impl Alphabet {
    /// Create a standard amino acid alphabet (K=20, Kp=29).
    pub fn amino() -> Self {
        let k = 20;
        let kp = 29;
        // Canonical: A C D E F G H I K L M N P Q R S T V W Y
        // Then: gap(-) B J Z O U X nonresidue(*) missing(~)
        let sym = padded_sym(b"ACDEFGHIKLMNPQRSTVWY-BJZOUX*~", kp);

        let mut degen = [false; MAX_FULL_SIZE * MAX_CANONICAL_SIZE];
        // Canonical residues: each is itself
        for i in 0..k {
            degen[i * MAX_CANONICAL_SIZE + i] = true;
        }
        // B = D(2) | N(11)
        set_degen(&mut degen, 21, &[2, 11]);
        // J = I(7) | L(9)
        set_degen(&mut degen, 22, &[7, 9]);
        // Z = E(3) | Q(13)
        set_degen(&mut degen, 23, &[3, 13]);
        // O = K(8) (pyrrolysine)
        set_degen(&mut degen, 24, &[8]);
        // U = C(1) (selenocysteine)
        set_degen(&mut degen, 25, &[1]);
        // X = any
        for a in 0..k {
            degen[26 * MAX_CANONICAL_SIZE + a] = true;
        }

        let inmap = build_inmap(&sym, kp);

        Alphabet {
            kind: AlphabetKind::Amino,
            canonical_size: k,
            full_size: kp,
            symbols: sym,
            degen,
            input_map: inmap,
        }
    }

    /// Create a standard DNA alphabet (K=4, Kp=18).
    pub fn dna() -> Self {
        let k = 4;
        let kp = 18;
        // Canonical: A C G T
        // Then: gap(-) R Y M K S W H B V D N nonresidue(*) missing(~)
        let sym = padded_sym(b"ACGT-RYMKSWHBVDN*~", kp);

        let mut degen = [false; MAX_FULL_SIZE * MAX_CANONICAL_SIZE];
        for i in 0..k {
            degen[i * MAX_CANONICAL_SIZE + i] = true;
        }
        // R = A|G
        set_degen(&mut degen, 5, &[0, 2]);
        // Y = C|T
        set_degen(&mut degen, 6, &[1, 3]);
        // M = A|C
        set_degen(&mut degen, 7, &[0, 1]);
        // K = G|T
        set_degen(&mut degen, 8, &[2, 3]);
        // S = G|C
        set_degen(&mut degen, 9, &[1, 2]);
        // W = A|T
        set_degen(&mut degen, 10, &[0, 3]);
        // H = A|C|T
        set_degen(&mut degen, 11, &[0, 1, 3]);
        // B = C|G|T
        set_degen(&mut degen, 12, &[1, 2, 3]);
        // V = A|C|G
        set_degen(&mut degen, 13, &[0, 1, 2]);
        // D = A|G|T
        set_degen(&mut degen, 14, &[0, 2, 3]);
        // N = any
        for a in 0..k {
            degen[15 * MAX_CANONICAL_SIZE + a] = true;
        }

        let inmap = build_inmap(&sym, kp);

        Alphabet {
            kind: AlphabetKind::Dna,
            canonical_size: k,
            full_size: kp,
            symbols: sym,
            degen,
            input_map: inmap,
        }
    }

    /// Create an RNA alphabet with U as its fourth canonical residue.
    pub fn rna() -> Self {
        let mut abc = Self::dna();
        abc.kind = AlphabetKind::Rna;
        abc.symbols[3] = b'U';
        abc.input_map = build_inmap(&abc.symbols, abc.full_size);
        abc
    }

    /// Digitize a single ASCII character to a residue code.
    ///
    /// Case-insensitive. Unknown characters map to `kp - 1`.
    #[inline]
    pub fn digitize(&self, c: u8) -> Dsq {
        if (c as usize) < 128 {
            self.input_map[c as usize]
        } else {
            (self.full_size - 1) as Dsq
        }
    }

    /// Get the degeneracy row for code `x` as a slice of K booleans.
    #[inline]
    pub fn degen_row(&self, x: usize) -> &[bool] {
        &self.degen[x * MAX_CANONICAL_SIZE..x * MAX_CANONICAL_SIZE + self.canonical_size]
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Copy a byte-string into a fixed-size array, padding with 0.
fn padded_sym(src: &[u8], kp: usize) -> [u8; MAX_FULL_SIZE] {
    let mut arr = [0u8; MAX_FULL_SIZE];
    let n = src.len().min(kp).min(MAX_FULL_SIZE);
    arr[..n].copy_from_slice(&src[..n]);
    arr
}

/// Set degeneracy flags: code `x` includes each residue in `residues`.
fn set_degen(degen: &mut [bool; MAX_FULL_SIZE * MAX_CANONICAL_SIZE], x: usize, residues: &[usize]) {
    for &a in residues {
        degen[x * MAX_CANONICAL_SIZE + a] = true;
    }
}

/// Build the ASCII → digital code lookup table.
/// Maps both upper and lower case to the same code.
fn build_inmap(sym: &[u8; MAX_FULL_SIZE], kp: usize) -> [Dsq; 128] {
    let fallback = (kp - 1) as Dsq; // missing data / unknown
    let mut map = [fallback; 128];
    for (code, &c) in sym[..kp].iter().enumerate() {
        if c == 0 {
            continue;
        }
        let upper = c.to_ascii_uppercase();
        let lower = c.to_ascii_lowercase();
        if (upper as usize) < 128 {
            map[upper as usize] = code as Dsq;
        }
        if (lower as usize) < 128 {
            map[lower as usize] = code as Dsq;
        }
    }
    // DNA and RNA accept both spellings of the fourth nucleotide.
    if sym.starts_with(b"ACGT") || sym.starts_with(b"ACGU") {
        for c in b"TtUu" {
            map[*c as usize] = 3;
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// How many canonical residues a degeneracy row stands for.
    fn count_of(row: &[bool]) -> usize {
        row.iter().filter(|&&included| included).count()
    }

    #[test]
    fn nucleotide_alphabets_accept_uracil_and_thymine() {
        for abc in [Alphabet::dna(), Alphabet::rna()] {
            for base in b"TtUu" {
                assert_eq!(abc.digitize(*base), 3);
            }
            assert_eq!(abc.degen_row(3), &[false, false, false, true]);
        }
        assert_eq!(Alphabet::rna().symbols[3], b'U');
        assert_eq!(Alphabet::amino().digitize(b'U'), 25);
    }

    #[test]
    fn test_amino_basics() {
        let amino_alphabet = Alphabet::amino();
        assert_eq!(amino_alphabet.canonical_size, 20);
        assert_eq!(amino_alphabet.full_size, 29);
        assert_eq!(amino_alphabet.kind, AlphabetKind::Amino);
    }

    #[test]
    fn test_dna_basics() {
        let dna_alphabet = Alphabet::dna();
        assert_eq!(dna_alphabet.canonical_size, 4);
        assert_eq!(dna_alphabet.full_size, 18);
        assert_eq!(dna_alphabet.kind, AlphabetKind::Dna);
    }

    #[test]
    fn test_digitize_case_insensitive() {
        let amino_alphabet = Alphabet::amino();
        assert_eq!(amino_alphabet.digitize(b'A'), 0);
        assert_eq!(amino_alphabet.digitize(b'a'), 0);
        assert_eq!(amino_alphabet.digitize(b'C'), 1);
        assert_eq!(amino_alphabet.digitize(b'c'), 1);
        assert_eq!(amino_alphabet.digitize(b'Y'), 19);
        assert_eq!(amino_alphabet.digitize(b'y'), 19);
    }

    #[test]
    fn test_digitize_unknown() {
        let amino_alphabet = Alphabet::amino();
        // Unknown characters map to kp-1
        assert_eq!(
            amino_alphabet.digitize(b'?'),
            (amino_alphabet.full_size - 1) as Dsq
        );
        assert_eq!(
            amino_alphabet.digitize(b'1'),
            (amino_alphabet.full_size - 1) as Dsq
        );
    }

    #[test]
    fn test_digitize_dna() {
        let dna_alphabet = Alphabet::dna();
        assert_eq!(dna_alphabet.digitize(b'A'), 0);
        assert_eq!(dna_alphabet.digitize(b'C'), 1);
        assert_eq!(dna_alphabet.digitize(b'G'), 2);
        assert_eq!(dna_alphabet.digitize(b'T'), 3);
        assert_eq!(dna_alphabet.digitize(b't'), 3);
        // N = any = code 15
        assert_eq!(dna_alphabet.digitize(b'N'), 15);
        assert_eq!(dna_alphabet.digitize(b'n'), 15);
    }

    #[test]
    fn test_degeneracy_amino() {
        let amino_alphabet = Alphabet::amino();
        // B = D(2) | N(11)
        assert!(amino_alphabet.degen_row(21)[2]);
        assert!(amino_alphabet.degen_row(21)[11]);
        assert!(!amino_alphabet.degen_row(21)[0]);
        assert_eq!(count_of(amino_alphabet.degen_row(21)), 2);
        // X = any (26)
        assert_eq!(count_of(amino_alphabet.degen_row(26)), 20);
    }

    #[test]
    fn test_degeneracy_dna() {
        let dna_alphabet = Alphabet::dna();
        // R = A(0) | G(2)
        assert!(dna_alphabet.degen_row(5)[0]);
        assert!(dna_alphabet.degen_row(5)[2]);
        assert!(!dna_alphabet.degen_row(5)[1]);
        assert_eq!(count_of(dna_alphabet.degen_row(5)), 2);
        // N = any (15)
        assert_eq!(count_of(dna_alphabet.degen_row(15)), 4);
    }
}
