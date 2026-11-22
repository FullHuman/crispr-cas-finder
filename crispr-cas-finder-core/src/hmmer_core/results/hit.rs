// p7_hit.rs - Per-target hit data structure
//
// Port of src/p7_hit.c

use crate::hmmer_core::domain::Domain;

/// A single target hit.
#[derive(Debug, Clone, Default)]
pub struct Hit {
    pub name: String,
    /// Sequence score in bits, null corrected.
    pub score: f32,
    /// log(P-value) of the sequence score.
    pub log_pvalue: f64,

    /// What `hmmsearch` lists as `reg`, `clu`, `ov` and `env`: the regions
    /// found, how many of them were split by sampling, the envelopes that
    /// overlap the one before, and the envelopes scored.
    pub num_regions: usize,
    pub num_clustered: usize,
    pub num_overlaps: usize,
    pub num_envelopes: usize,

    pub domains: Vec<Domain>,
}
