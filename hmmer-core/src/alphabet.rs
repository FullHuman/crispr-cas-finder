// alphabet.rs — Biological sequence alphabets for HMMER
//
// Replaces the C-ported EslAlphabet with an idiomatic Rust design:
//   - Enum-based: Amino, Dna, Rna (no unused Coins/Dice/Custom/Unknown)
//   - O(1) case-folding digitization via 128-byte lookup table
//   - Flat degeneracy table
//   - Background frequencies removed (belong in BackgroundModel)

use crate::config::{MAX_CANONICAL_ALPHABET, MAX_FULL_ALPHABET};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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

    /// Number of canonical residues represented by each code.
    pub degeneracy_counts: [usize; MAX_FULL_SIZE],

    /// ASCII → digital code lookup (128 entries, case-folding).
    /// Unknown characters map to `kp - 1` (missing data).
    input_map: [Dsq; 128],
}

/// The kind of biological alphabet.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
        // Then: gap(-) B J Z O U X nonresidue(~) missing(*)
        let sym = padded_sym(b"ACDEFGHIKLMNPQRSTVWY-BJZOUX~*", kp);

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

        let ndegen = compute_ndegen(&degen, kp, k);
        let inmap = build_inmap(&sym, kp);

        Alphabet {
            kind: AlphabetKind::Amino,
            canonical_size: k,
            full_size: kp,
            symbols: sym,
            degen,
            degeneracy_counts: ndegen,
            input_map: inmap,
        }
    }

    /// Create a standard DNA alphabet (K=4, Kp=18).
    pub fn dna() -> Self {
        let k = 4;
        let kp = 18;
        // Canonical: A C G T
        // Then: gap(-) R Y M K S W H B V D N nonresidue(~) missing(*)
        let sym = padded_sym(b"ACGT-RYMKSWHBVDN~*", kp);

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

        let ndegen = compute_ndegen(&degen, kp, k);
        let inmap = build_inmap(&sym, kp);

        Alphabet {
            kind: AlphabetKind::Dna,
            canonical_size: k,
            full_size: kp,
            symbols: sym,
            degen,
            degeneracy_counts: ndegen,
            input_map: inmap,
        }
    }

    /// Create an RNA alphabet (identical to DNA — HMMER treats them the same).
    pub fn rna() -> Self {
        let mut abc = Self::dna();
        abc.kind = AlphabetKind::Rna;
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

    /// Check whether digital code `x` is a degenerate (ambiguous) residue,
    /// i.e. not one of the K canonical residues and not gap/special.
    #[inline]
    pub fn is_degenerate(&self, x: Dsq) -> bool {
        let x = x as usize;
        x > self.canonical_size && x < self.full_size.saturating_sub(2)
    }

    /// Check whether digital code `x` is a canonical residue (0..K-1).
    #[inline]
    pub fn is_canonical(&self, x: Dsq) -> bool {
        (x as usize) < self.canonical_size
    }

    /// Get the degeneracy row for code `x` as a slice of K booleans.
    #[inline]
    pub fn degen_row(&self, x: usize) -> &[bool] {
        &self.degen[x * MAX_CANONICAL_SIZE..x * MAX_CANONICAL_SIZE + self.canonical_size]
    }
}

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
struct AlphabetSerde {
    kind: AlphabetKind,
    canonical_size: usize,
    full_size: usize,
    symbols: Vec<u8>,
    degen: Vec<bool>,
    degeneracy_counts: Vec<usize>,
}

#[cfg(feature = "serde")]
impl Serialize for Alphabet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        AlphabetSerde {
            kind: self.kind,
            canonical_size: self.canonical_size,
            full_size: self.full_size,
            symbols: self.symbols.to_vec(),
            degen: self.degen.to_vec(),
            degeneracy_counts: self.degeneracy_counts.to_vec(),
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Alphabet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        let raw = AlphabetSerde::deserialize(deserializer)?;
        if raw.canonical_size > MAX_CANONICAL_SIZE {
            return Err(D::Error::custom(
                "canonical_size exceeds maximum alphabet size",
            ));
        }
        if raw.full_size > MAX_FULL_SIZE {
            return Err(D::Error::custom("full_size exceeds maximum alphabet size"));
        }
        if raw.symbols.len() != MAX_FULL_SIZE {
            return Err(D::Error::custom("symbols has incorrect serialized length"));
        }
        if raw.degen.len() != MAX_FULL_SIZE * MAX_CANONICAL_SIZE {
            return Err(D::Error::custom("degen has incorrect serialized length"));
        }
        if raw.degeneracy_counts.len() != MAX_FULL_SIZE {
            return Err(D::Error::custom(
                "degeneracy_counts has incorrect serialized length",
            ));
        }

        let mut symbols = [0u8; MAX_FULL_SIZE];
        symbols.copy_from_slice(&raw.symbols);

        let mut degen = [false; MAX_FULL_SIZE * MAX_CANONICAL_SIZE];
        degen.copy_from_slice(&raw.degen);

        let mut degeneracy_counts = [0usize; MAX_FULL_SIZE];
        degeneracy_counts.copy_from_slice(&raw.degeneracy_counts);

        Ok(Self {
            kind: raw.kind,
            canonical_size: raw.canonical_size,
            full_size: raw.full_size,
            symbols,
            degen,
            degeneracy_counts,
            input_map: build_inmap(&symbols, raw.full_size),
        })
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

/// Compute ndegen counts from the flat degen table.
fn compute_ndegen(
    degen: &[bool; MAX_FULL_SIZE * MAX_CANONICAL_SIZE],
    kp: usize,
    k: usize,
) -> [usize; MAX_FULL_SIZE] {
    let mut nd = [0usize; MAX_FULL_SIZE];
    for x in 0..kp {
        nd[x] = degen[x * MAX_CANONICAL_SIZE..x * MAX_CANONICAL_SIZE + k]
            .iter()
            .filter(|&&b| b)
            .count();
    }
    nd
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
    map
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amino_basics() {
        let abc = Alphabet::amino();
        assert_eq!(abc.canonical_size, 20);
        assert_eq!(abc.full_size, 29);
        assert_eq!(abc.kind, AlphabetKind::Amino);
    }

    #[test]
    fn test_dna_basics() {
        let abc = Alphabet::dna();
        assert_eq!(abc.canonical_size, 4);
        assert_eq!(abc.full_size, 18);
        assert_eq!(abc.kind, AlphabetKind::Dna);
    }

    #[test]
    fn test_digitize_case_insensitive() {
        let abc = Alphabet::amino();
        assert_eq!(abc.digitize(b'A'), 0);
        assert_eq!(abc.digitize(b'a'), 0);
        assert_eq!(abc.digitize(b'C'), 1);
        assert_eq!(abc.digitize(b'c'), 1);
        assert_eq!(abc.digitize(b'Y'), 19);
        assert_eq!(abc.digitize(b'y'), 19);
    }

    #[test]
    fn test_digitize_unknown() {
        let abc = Alphabet::amino();
        // Unknown characters map to kp-1
        assert_eq!(abc.digitize(b'?'), (abc.full_size - 1) as Dsq);
        assert_eq!(abc.digitize(b'1'), (abc.full_size - 1) as Dsq);
    }

    #[test]
    fn test_digitize_dna() {
        let abc = Alphabet::dna();
        assert_eq!(abc.digitize(b'A'), 0);
        assert_eq!(abc.digitize(b'C'), 1);
        assert_eq!(abc.digitize(b'G'), 2);
        assert_eq!(abc.digitize(b'T'), 3);
        assert_eq!(abc.digitize(b't'), 3);
        // N = any = code 15
        assert_eq!(abc.digitize(b'N'), 15);
        assert_eq!(abc.digitize(b'n'), 15);
    }

    #[test]
    fn test_degeneracy_amino() {
        let abc = Alphabet::amino();
        // B = D(2) | N(11)
        assert!(abc.degen_row(21)[2]);
        assert!(abc.degen_row(21)[11]);
        assert!(!abc.degen_row(21)[0]);
        assert_eq!(abc.degeneracy_counts[21], 2);
        // X = any (26)
        assert_eq!(abc.degeneracy_counts[26], 20);
    }

    #[test]
    fn test_degeneracy_dna() {
        let abc = Alphabet::dna();
        // R = A(0) | G(2)
        assert!(abc.degen_row(5)[0]);
        assert!(abc.degen_row(5)[2]);
        assert!(!abc.degen_row(5)[1]);
        assert_eq!(abc.degeneracy_counts[5], 2);
        // N = any (15)
        assert_eq!(abc.degeneracy_counts[15], 4);
    }

    #[test]
    fn test_is_canonical() {
        let abc = Alphabet::amino();
        assert!(abc.is_canonical(0));
        assert!(abc.is_canonical(19));
        assert!(!abc.is_canonical(20)); // gap
        assert!(!abc.is_canonical(26)); // X
    }

    #[test]
    fn test_is_degenerate() {
        let abc = Alphabet::amino();
        assert!(!abc.is_degenerate(0)); // canonical
        assert!(!abc.is_degenerate(20)); // gap
        assert!(abc.is_degenerate(21)); // B
        assert!(abc.is_degenerate(26)); // X
        assert!(!abc.is_degenerate(27)); // nonresidue
        assert!(!abc.is_degenerate(28)); // missing
    }
}
