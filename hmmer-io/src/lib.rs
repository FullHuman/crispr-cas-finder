// hmmer-io: File I/O for HMM files, sequence databases, and alignments.

pub mod hmmfile;
pub mod seq_reader;

// Convenience re-exports
pub use hmmfile::HmmFile;
pub use hmmfile::read_hmm;
pub use seq_reader::{FastaReader, sqfile_open_digital};
