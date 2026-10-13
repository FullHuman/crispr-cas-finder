//! Print the repeats `vmatch -l 23 25 60 -e 1 -evalue 1 -selfun sel392v2.so 55`
//! would report for each sequence of a FASTA file, to compare with Vmatch
//! itself (see benchmarks/vmatch_oracle.py).
//!
//! One line per repeat: sequence number, then length and 0-based start of the
//! left copy, then of the right copy.

use crispr_cas_finder_core::parse_fasta;
use crispr_cas_finder_core::vmatch::{RepeatSearch, find_repeats};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: repeat_matches <genome.fasta>")?;
    let search = RepeatSearch {
        min_length: 23,
        min_gap: 25,
        max_gap: 60,
        max_length: 55,
        max_differences: 1,
    };
    for (number, record) in parse_fasta(std::fs::File::open(path)?)?.iter().enumerate() {
        for repeat in find_repeats(&record.sequence, &search) {
            println!(
                "{number}\t{}\t{}\t{}\t{}",
                repeat.left_length, repeat.left_start, repeat.right_length, repeat.right_start
            );
        }
    }
    Ok(())
}
