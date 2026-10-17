//! Print the share of gaps in the alignment of a repeat with a spacer, the
//! figure CRISPRCasFinder reads from EMBOSS `needle` for a locus with a
//! single spacer: `cargo run --example test_align -- <repeat> <spacer>`.

use crispr_cas_finder_core::detect::needle_gap_share;

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (repeat, spacer) = match arguments.as_slice() {
        [repeat, spacer] => (repeat.as_str(), spacer.as_str()),
        // EMBOSS needle: 28 gaps in 64 columns, 43.8 %.
        _ => (
            "ATGCCTGATGCGACGCTATGCGCGTCTTATCAGGCCTACGG",
            "TTTATGGGCGAAGTGTAGACCGGATAAGGCGTTCACGCCGCATCCGGCAGTCGTGCGCC",
        ),
    };
    match needle_gap_share(repeat, spacer) {
        Some(share) => println!(
            "gaps: {:.1} % ({})",
            share * 100.0,
            if share > 0.5 { "kept" } else { "dropped" }
        ),
        None => println!("nothing to align"),
    }
}
