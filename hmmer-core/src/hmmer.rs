// hmmer.rs - General routines used throughout HMMER
//
// Port of src/hmmer.c
//
// Contents:
//   1. Miscellaneous functions for H3
//   2. Unit tests

use crate::config;
use std::io::{self, Write};

/// Print the standard HMMER command line application banner.
///
/// Constructs the banner from `progname` and a short one-line `banner` description.
pub fn banner(fp: &mut dyn Write, progname: &str, banner: &str) -> io::Result<()> {
    let appname = std::path::Path::new(progname)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| progname.to_string());

    writeln!(fp, "# {} :: {}", appname, banner)?;
    writeln!(
        fp,
        "# HMMER {} ({}); {}",
        config::HMMER_VERSION,
        config::HMMER_DATE,
        config::HMMER_URL
    )?;
    writeln!(fp, "# {}", config::HMMER_COPYRIGHT)?;
    writeln!(fp, "# {}", config::HMMER_LICENSE)?;
    writeln!(
        fp,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )?;
    Ok(())
}

/// Return amino acid background frequencies in [A..Y] alphabetic order.
///
/// These are from Swiss-Prot 50.8 (Oct 2006), counting over 85956127 (86.0M) residues.
pub fn amino_frequencies(f: &mut [f32]) {
    assert!(f.len() >= 20);
    f[0] = 0.0787945; // A
    f[1] = 0.0151600; // C
    f[2] = 0.0535222; // D
    f[3] = 0.0668298; // E
    f[4] = 0.0397062; // F
    f[5] = 0.0695071; // G
    f[6] = 0.0229198; // H
    f[7] = 0.0590092; // I
    f[8] = 0.0594422; // K
    f[9] = 0.0963728; // L
    f[10] = 0.0237718; // M
    f[11] = 0.0414386; // N
    f[12] = 0.0482904; // P
    f[13] = 0.0395639; // Q
    f[14] = 0.0540978; // R
    f[15] = 0.0683364; // S
    f[16] = 0.0540687; // T
    f[17] = 0.0673417; // V
    f[18] = 0.0114135; // W
    f[19] = 0.0304133; // Y
}

// ---------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;

    #[test]
    fn test_amino_frequencies() {
        let mut f = [0.0f32; 20];
        amino_frequencies(&mut f);

        // Verify they sum to ~1.0
        let sum: f32 = f.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.001,
            "amino frequencies sum to {} (expected ~1.0)",
            sum
        );

        // Verify all are positive
        for (i, &freq) in f.iter().enumerate() {
            assert!(freq > 0.0, "frequency[{}] = {} is not positive", i, freq);
        }
    }

    #[test]
    fn test_banner() {
        let mut buf = Vec::new();
        banner(
            &mut buf,
            "hmmsearch",
            "search profile(s) against a sequence database",
        )
        .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("hmmsearch"));
        assert!(output.contains("HMMER"));
    }

    #[test]
    fn test_alphabet_config() {
        // Verify that our alphabet sizes are compatible with HMMER's MAXABET/MAXCODE
        let abc = Alphabet::amino();
        assert!(
            abc.canonical_size <= config::MAX_CANONICAL_ALPHABET,
            "amino K={} > P7_MAXABET={}",
            abc.canonical_size,
            config::MAX_CANONICAL_ALPHABET
        );
        assert!(
            abc.full_size <= config::MAX_FULL_ALPHABET,
            "amino Kp={} > P7_MAXCODE={}",
            abc.full_size,
            config::MAX_FULL_ALPHABET
        );

        let abc = Alphabet::dna();
        assert!(abc.canonical_size <= config::MAX_CANONICAL_ALPHABET);
        assert!(abc.full_size <= config::MAX_FULL_ALPHABET);
    }
}
