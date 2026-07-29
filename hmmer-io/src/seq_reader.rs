// seq_reader.rs - FASTA sequence file I/O
//
// Uses rust-bio for FASTA parsing, returning DigitalSequence.

use std::io;

use bio::io::fasta::{self, FastaRead};

use hmmer_core::alphabet::Alphabet;
use hmmer_core::errors::HmmerError;
use hmmer_core::sequence::DigitalSequence;

/// Read all sequences from a FASTA file, returning digitized sequences.
pub fn sqfile_open_digital(
    abc: &Alphabet,
    filename: &str,
) -> Result<Vec<DigitalSequence>, HmmerError> {
    let mut reader = fasta::Reader::from_file(filename)
        .map_err(|e| HmmerError::NotFound(format!("Failed to open {}: {}", filename, e)))?;
    let mut record = fasta::Record::new();
    let mut seqs = Vec::new();

    loop {
        reader
            .read(&mut record)
            .map_err(|e| HmmerError::Internal(format!("Read error: {}", e)))?;
        if record.is_empty() {
            return Ok(seqs);
        }

        seqs.push(DigitalSequence::from_bytes(
            record.id(),
            record.desc().unwrap_or(""),
            record.seq(),
            abc,
        ));
    }
}

/// Streaming FASTA reader that yields one digitized sequence at a time.
pub struct FastaReader {
    reader: fasta::Reader<io::BufReader<std::fs::File>>,
    record: fasta::Record,
    abc: Alphabet,
}

impl FastaReader {
    pub fn open_digital(abc: &Alphabet, filename: &str) -> Result<Self, HmmerError> {
        let reader = fasta::Reader::from_file(filename)
            .map_err(|e| HmmerError::NotFound(format!("Failed to open {}: {}", filename, e)))?;
        Ok(FastaReader {
            reader,
            record: fasta::Record::new(),
            abc: abc.clone(),
        })
    }

    /// Read the next sequence. Returns None at EOF.
    pub fn read(&mut self) -> Result<Option<DigitalSequence>, HmmerError> {
        self.reader
            .read(&mut self.record)
            .map_err(|e| HmmerError::Internal(format!("Read error: {}", e)))?;

        if self.record.is_empty() {
            return Ok(None);
        }

        Ok(Some(DigitalSequence::from_bytes(
            self.record.id(),
            self.record.desc().unwrap_or(""),
            self.record.seq(),
            &self.abc,
        )))
    }
}
