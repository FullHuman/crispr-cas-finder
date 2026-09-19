//! HMMER3 ASCII model I/O and FASTA sequence readers.

pub mod hmmfile;
pub mod seq_reader;

// Convenience re-exports
pub use hmmfile::HmmFile;
pub use hmmfile::read_hmm;
pub use seq_reader::{FastaReader, read_fasta_digital_sequences};

pub use crate::hmmer_core::errors::HmmerError;
