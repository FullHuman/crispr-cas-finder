// h2_io.rs - HMMER2 file format I/O (backwards compatibility)
//
// Port of src/h2_io.c

use hmmer_core::alphabet::Alphabet;
use hmmer_core::hmm::Hmm;

/// Read an old HMMER2 format HMM.
pub fn h2_read_hmm(_filename: &str) -> Result<(Alphabet, Hmm), String> {
    // TODO: Full HMMER2 format parser
    Err("HMMER2 format reading not yet implemented".to_string())
}
