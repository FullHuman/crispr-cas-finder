// p7_tophits.rs - Ranked list of top-scoring hits
//
// Port of src/p7_tophits.c

use crate::constants::results::tophits::DEFAULT_INITIAL_CAPACITY;
use crate::hit::{Hit, HitFlags};

/// Top hits list — maintains both sorted and unsorted access to hits.
///
/// Hits are stored in insertion order in `unsrt`. The `hit_order` vec holds
/// indices into `unsrt` and provides logical sort order without moving hits.
#[derive(Debug, Clone)]
pub struct TopHits {
    pub hits: Vec<Hit>,
    pub hit_order: Vec<usize>,
    pub num_reported: u64,
    pub num_included: u64,
    pub is_sorted_by_sortkey: bool,
    pub is_sorted_by_seqidx: bool,
}

impl TopHits {
    /// Create a new empty top hits list.
    pub fn new() -> Self {
        TopHits {
            hits: Vec::with_capacity(DEFAULT_INITIAL_CAPACITY),
            hit_order: Vec::with_capacity(DEFAULT_INITIAL_CAPACITY),
            num_reported: 0,
            num_included: 0,
            is_sorted_by_sortkey: true, // vacuously true for empty list
            is_sorted_by_seqidx: false,
        }
    }

    /// Total number of hits.
    pub fn len(&self) -> usize {
        self.hits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }

    /// Add a new hit to the list, return a mutable reference to it.
    pub fn create_next_hit(&mut self) -> &mut Hit {
        let idx = self.hits.len();
        self.hits.push(Hit::default());
        self.hit_order.push(idx);
        self.is_sorted_by_sortkey = false;
        self.is_sorted_by_seqidx = false;
        &mut self.hits[idx]
    }

    /// Add an owned hit to the list.
    pub fn push_hit(&mut self, hit: Hit) {
        let idx = self.hits.len();
        self.hits.push(hit);
        self.hit_order.push(idx);
        self.is_sorted_by_sortkey = false;
        self.is_sorted_by_seqidx = false;
    }

    /// Sort hits by sortkey (E-value / bit score).
    pub fn sort_by_sortkey(&mut self) {
        if self.is_sorted_by_sortkey {
            return;
        }
        let unsrt = &self.hits;
        self.hit_order.sort_by(|&a, &b| {
            unsrt[b]
                .sortkey
                .partial_cmp(&unsrt[a].sortkey)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.is_sorted_by_sortkey = true;
        self.is_sorted_by_seqidx = false;
    }

    /// Sort hits by sequence index (database order).
    pub fn sort_by_seqidx(&mut self) {
        if self.is_sorted_by_seqidx {
            return;
        }
        let unsrt = &self.hits;
        self.hit_order
            .sort_by(|&a, &b| unsrt[a].seqidx.cmp(&unsrt[b].seqidx));
        self.is_sorted_by_seqidx = true;
        self.is_sorted_by_sortkey = false;
    }

    /// Get the hit in sorted position `i`.
    pub fn get_hit(&self, i: usize) -> Option<&Hit> {
        self.hit_order.get(i).map(|&idx| &self.hits[idx])
    }

    /// Get a mutable reference to the hit in sorted position `i`.
    pub fn get_hit_mut(&mut self, i: usize) -> Option<&mut Hit> {
        let idx = *self.hit_order.get(i)?;
        Some(&mut self.hits[idx])
    }

    /// Apply thresholds and count reported/included hits.
    pub fn threshold(
        &mut self,
        pipeline_e: f64,
        _pipeline_dom_e: f64,
        pipeline_inc_e: f64,
        _pipeline_incdom_e: f64,
        use_bit_cutoffs: bool,
    ) {
        self.num_reported = 0;
        self.num_included = 0;

        for hit in self.hits.iter_mut() {
            if !use_bit_cutoffs && hit.log_pvalue <= pipeline_e.ln() {
                hit.flags |= HitFlags::REPORTED;
                self.num_reported += 1;
            }
            if !use_bit_cutoffs && hit.log_pvalue <= pipeline_inc_e.ln() {
                hit.flags |= HitFlags::INCLUDED;
                self.num_included += 1;
            }
        }
    }

    /// Merge another tophits list into this one, consuming it.
    pub fn merge(&mut self, mut other: TopHits) {
        let base = self.hits.len();
        self.hits.append(&mut other.hits);
        self.hit_order
            .extend(other.hit_order.iter().map(|&idx| idx + base));
        self.is_sorted_by_sortkey = false;
        self.is_sorted_by_seqidx = false;
    }
}

impl Default for TopHits {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create() {
        let th = TopHits::new();
        assert_eq!(th.len(), 0);
        assert!(th.is_empty());
    }

    #[test]
    fn test_add_and_sort() {
        let mut th = TopHits::new();

        let hit1 = th.create_next_hit();
        hit1.name = "seq3".to_string();
        hit1.sortkey = 3.0;

        let hit2 = th.create_next_hit();
        hit2.name = "seq1".to_string();
        hit2.sortkey = 1.0;

        let hit3 = th.create_next_hit();
        hit3.name = "seq2".to_string();
        hit3.sortkey = 2.0;

        assert_eq!(th.len(), 3);

        th.sort_by_sortkey();
        assert_eq!(th.get_hit(0).unwrap().name, "seq3");
        assert_eq!(th.get_hit(1).unwrap().name, "seq2");
        assert_eq!(th.get_hit(2).unwrap().name, "seq1");
    }
}
