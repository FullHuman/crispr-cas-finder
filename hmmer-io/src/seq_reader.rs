// seq_reader.rs - FASTA sequence file I/O
//
// Uses rust-bio for FASTA parsing, returning DigitalSequence.

use std::io;

use bio::io::fasta;

use hmmer_core::alphabet::Alphabet;
use hmmer_core::errors::HmmerError;
use hmmer_core::sequence::DigitalSequence;

/// Read all sequences from a FASTA file, returning digitized sequences.
pub fn sqfile_open_digital(
    abc: &Alphabet,
    filename: &str,
) -> Result<Vec<DigitalSequence>, HmmerError> {
    let reader = fasta::Reader::from_file(filename)
        .map_err(|e| HmmerError::NotFound(format!("Failed to open {}: {}", filename, e)))?;
    let mut seqs = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| HmmerError::Internal(format!("Read error: {}", e)))?;
        let desc = record.desc().unwrap_or("");
        let sq = DigitalSequence::from_bytes(record.id(), desc, record.seq(), abc);
        seqs.push(sq);
    }
    Ok(seqs)
}

/// Streaming FASTA reader that yields one digitized sequence at a time.
pub struct FastaReader {
    records: fasta::Records<io::BufReader<std::fs::File>>,
    abc: Alphabet,
}

impl FastaReader {
    pub fn open_digital(abc: &Alphabet, filename: &str) -> Result<Self, HmmerError> {
        let reader = fasta::Reader::from_file(filename)
            .map_err(|e| HmmerError::NotFound(format!("Failed to open {}: {}", filename, e)))?;
        Ok(FastaReader {
            records: reader.records(),
            abc: abc.clone(),
        })
    }

    /// Read the next sequence. Returns None at EOF.
    pub fn read(&mut self) -> Result<Option<DigitalSequence>, HmmerError> {
        match self.records.next() {
            Some(Ok(record)) => {
                let desc = record.desc().unwrap_or("");
                let sq = DigitalSequence::from_bytes(record.id(), desc, record.seq(), &self.abc);
                Ok(Some(sq))
            }
            Some(Err(e)) => Err(HmmerError::Internal(format!("Read error: {}", e))),
            None => Ok(None),
        }
    }
}
