// p7_hit.rs - Per-target hit data structure
//
// Port of src/p7_hit.c

use crate::hmmer_core::domain::Domain;
use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct HitFlags: u32 {
        const INCLUDED  = 1 << 0;
        const REPORTED  = 1 << 1;
        const NEW       = 1 << 2;
        const DROPPED   = 1 << 3;
        const DUPLICATE = 1 << 4;
    }
}

/// A single target hit.
#[derive(Debug, Clone)]
pub struct Hit {
    pub name: String,
    pub accession: Option<String>,
    pub description: Option<String>,
    pub window_length: i32,
    pub sortkey: f64,

    pub score: f32,
    pub pre_score: f32,
    pub sum_score: f32,

    pub log_pvalue: f64,
    pub pre_log_pvalue: f64,
    pub sum_log_pvalue: f64,

    pub num_expected: f32,
    pub num_regions: usize,
    pub num_clustered: usize,
    pub num_overlaps: usize,
    pub num_envelopes: usize,
    pub num_domains: usize,

    pub flags: HitFlags,
    pub num_reported: usize,
    pub num_included: usize,
    pub best_domain: Option<usize>,

    pub seqidx: Option<usize>,
    pub subseq_start: i64,

    pub domains: Vec<Domain>,
    pub offset: i64,
}

impl Hit {
    pub fn is_included(&self) -> bool {
        self.flags.contains(HitFlags::INCLUDED)
    }

    pub fn is_reported(&self) -> bool {
        self.flags.contains(HitFlags::REPORTED)
    }

    pub fn is_new(&self) -> bool {
        self.flags.contains(HitFlags::NEW)
    }

    pub fn is_dropped(&self) -> bool {
        self.flags.contains(HitFlags::DROPPED)
    }

    pub fn is_duplicate(&self) -> bool {
        self.flags.contains(HitFlags::DUPLICATE)
    }
}

impl Default for Hit {
    fn default() -> Self {
        Hit {
            name: String::new(),
            accession: None,
            description: None,
            window_length: 0,
            sortkey: 0.0,
            score: 0.0,
            pre_score: 0.0,
            sum_score: 0.0,
            log_pvalue: 0.0,
            pre_log_pvalue: 0.0,
            sum_log_pvalue: 0.0,
            num_expected: 0.0,
            num_regions: 0,
            num_clustered: 0,
            num_overlaps: 0,
            num_envelopes: 0,
            num_domains: 0,
            flags: HitFlags::empty(),
            num_reported: 0,
            num_included: 0,
            best_domain: None,
            seqidx: None,
            subseq_start: 0,
            domains: Vec::new(),
            offset: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let hit = Hit::default();
        assert_eq!(hit.flags, HitFlags::empty());
        assert!(!hit.is_included());
        assert!(hit.seqidx.is_none());
        assert!(hit.best_domain.is_none());
    }

    #[test]
    fn test_flags() {
        let mut hit = Hit::default();
        hit.flags |= HitFlags::INCLUDED | HitFlags::REPORTED;
        assert!(hit.is_included());
        assert!(hit.is_reported());
        assert!(!hit.is_new());
    }
}
