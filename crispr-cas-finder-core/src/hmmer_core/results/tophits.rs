// p7_tophits.rs - Ranked list of top-scoring hits
//
// Port of src/p7_tophits.c

use crate::hmmer_core::constants::results::tophits::DEFAULT_INITIAL_CAPACITY;
use crate::hmmer_core::hit::{Hit, HitFlags};
use std::cmp::Ordering;

/// Top hits list — maintains both sorted and unsorted access to hits.
///
/// Hits are stored in insertion order in `hits`. The `hit_order` vector holds
/// indices into `hits` and provides logical sort order without moving hits.
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
    fn debug_assert_valid_order(&self) {
        debug_assert_eq!(self.hits.len(), self.hit_order.len());
        debug_assert!(self.hit_order.iter().all(|&index| index < self.hits.len()));
        #[cfg(debug_assertions)]
        {
            let mut seen = vec![false; self.hits.len()];
            for &index in &self.hit_order {
                debug_assert!(!seen[index]);
                seen[index] = true;
            }
        }
    }

    fn compare_sortkey(hits: &[Hit], a: usize, b: usize) -> Ordering {
        let a_hit = &hits[a];
        let b_hit = &hits[b];
        let by_sortkey = match (a_hit.sortkey.is_nan(), b_hit.sortkey.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => b_hit.sortkey.total_cmp(&a_hit.sortkey),
        };

        by_sortkey
            .then_with(|| a_hit.name.cmp(&b_hit.name))
            .then_with(|| a.cmp(&b))
    }

    fn recount_thresholds(&mut self) {
        self.num_reported = 0;
        self.num_included = 0;

        for hit in &mut self.hits {
            hit.num_reported = hit
                .domains
                .iter()
                .filter(|domain| domain.is_reported)
                .count();
            hit.num_included = hit
                .domains
                .iter()
                .filter(|domain| domain.is_included)
                .count();
            if hit.is_reported() {
                self.num_reported += 1;
            }
            if hit.is_included() {
                self.num_included += 1;
            }
        }
    }

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
        self.debug_assert_valid_order();
        self.hits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.debug_assert_valid_order();
        self.hits.is_empty()
    }

    /// Add a new hit to the list, return a mutable reference to it.
    pub fn create_next_hit(&mut self) -> &mut Hit {
        self.debug_assert_valid_order();
        let idx = self.hits.len();
        self.hits.push(Hit::default());
        self.hit_order.push(idx);
        self.is_sorted_by_sortkey = false;
        self.is_sorted_by_seqidx = false;
        self.debug_assert_valid_order();
        &mut self.hits[idx]
    }

    /// Add an owned hit to the list.
    pub fn push_hit(&mut self, hit: Hit) {
        self.debug_assert_valid_order();
        let idx = self.hits.len();
        self.hits.push(hit);
        self.hit_order.push(idx);
        self.is_sorted_by_sortkey = false;
        self.is_sorted_by_seqidx = false;
        self.debug_assert_valid_order();
    }

    /// Sort hits by sortkey (E-value / bit score).
    pub fn sort_by_sortkey(&mut self) {
        self.debug_assert_valid_order();
        if self.is_sorted_by_sortkey {
            return;
        }
        let unsrt = &self.hits;
        self.hit_order
            .sort_by(|&a, &b| Self::compare_sortkey(unsrt, a, b));
        self.is_sorted_by_sortkey = true;
        self.is_sorted_by_seqidx = false;
        self.debug_assert_valid_order();
    }

    /// Sort hits by sequence index (database order).
    pub fn sort_by_seqidx(&mut self) {
        self.debug_assert_valid_order();
        if self.is_sorted_by_seqidx {
            return;
        }
        let unsrt = &self.hits;
        self.hit_order.sort_by(|&a, &b| {
            match (unsrt[a].seqidx, unsrt[b].seqidx) {
                (Some(a_idx), Some(b_idx)) => a_idx.cmp(&b_idx),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            }
            .then_with(|| a.cmp(&b))
        });
        self.is_sorted_by_seqidx = true;
        self.is_sorted_by_sortkey = false;
        self.debug_assert_valid_order();
    }

    /// Get the hit in sorted position `i`.
    pub fn get_hit(&self, i: usize) -> Option<&Hit> {
        self.debug_assert_valid_order();
        let idx = *self.hit_order.get(i)?;
        self.hits.get(idx)
    }

    /// Get a mutable reference to the hit in sorted position `i`.
    pub fn get_hit_mut(&mut self, i: usize) -> Option<&mut Hit> {
        self.debug_assert_valid_order();
        let idx = *self.hit_order.get(i)?;
        self.is_sorted_by_sortkey = false;
        self.is_sorted_by_seqidx = false;
        self.hits.get_mut(idx)
    }

    /// Apply P-value thresholds and count reported/included targets and domains.
    ///
    /// As in upstream HMMER, model-specific bit-score cutoffs must be applied
    /// while the model is available. When `bit_cutoffs_preapplied` is true,
    /// this method preserves those precomputed flags and only recounts them.
    pub fn threshold(
        &mut self,
        pipeline_e: f64,
        pipeline_dom_e: f64,
        pipeline_inc_e: f64,
        pipeline_incdom_e: f64,
        bit_cutoffs_preapplied: bool,
    ) {
        self.debug_assert_valid_order();
        if !bit_cutoffs_preapplied {
            let report_threshold = pipeline_e.ln();
            let domain_report_threshold = pipeline_dom_e.ln();
            let inclusion_threshold = pipeline_inc_e.ln();
            let domain_inclusion_threshold = pipeline_incdom_e.ln();

            for hit in &mut self.hits {
                hit.flags.remove(HitFlags::REPORTED | HitFlags::INCLUDED);
                for domain in &mut hit.domains {
                    domain.is_reported = false;
                    domain.is_included = false;
                }

                if hit.is_duplicate() || hit.log_pvalue > report_threshold {
                    continue;
                }

                hit.flags |= HitFlags::REPORTED;
                if hit.log_pvalue <= inclusion_threshold {
                    hit.flags |= HitFlags::INCLUDED;
                }

                let hit_is_included = hit.is_included();
                for domain in &mut hit.domains {
                    if domain.log_pvalue <= domain_report_threshold {
                        domain.is_reported = true;
                        if hit_is_included && domain.log_pvalue <= domain_inclusion_threshold {
                            domain.is_included = true;
                        }
                    }
                }
            }
        }

        self.recount_thresholds();
    }

    /// Merge another top-hits list into this one, consuming it.
    ///
    /// Both inputs are sorted by sortkey first. The merged list remains sorted
    /// by sortkey, matching `p7_tophits_Merge()`.
    pub fn merge(&mut self, mut other: TopHits) {
        self.debug_assert_valid_order();
        other.debug_assert_valid_order();
        self.sort_by_sortkey();
        if other.is_empty() {
            self.recount_thresholds();
            return;
        }

        other.sort_by_sortkey();

        let base = self.hits.len();
        self.hits.append(&mut other.hits);
        let left_order = std::mem::take(&mut self.hit_order);
        let right_order: Vec<usize> = other
            .hit_order
            .into_iter()
            .map(|index| index + base)
            .collect();
        let mut merged_order = Vec::with_capacity(self.hits.len());
        let (mut left, mut right) = (0, 0);

        while left < left_order.len() && right < right_order.len() {
            if Self::compare_sortkey(&self.hits, left_order[left], right_order[right])
                != Ordering::Greater
            {
                merged_order.push(left_order[left]);
                left += 1;
            } else {
                merged_order.push(right_order[right]);
                right += 1;
            }
        }
        merged_order.extend_from_slice(&left_order[left..]);
        merged_order.extend_from_slice(&right_order[right..]);
        self.hit_order = merged_order;
        self.is_sorted_by_sortkey = true;
        self.is_sorted_by_seqidx = false;
        self.recount_thresholds();
        self.debug_assert_valid_order();
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
    use crate::hmmer_core::domain::Domain;

    fn hit(name: &str, sortkey: f64, seqidx: Option<usize>) -> Hit {
        Hit {
            name: name.to_string(),
            sortkey,
            seqidx,
            ..Hit::default()
        }
    }

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

    #[test]
    fn sort_by_sortkey_places_nan_last_deterministically() {
        let mut hits = TopHits::new();
        hits.push_hit(hit("nan-b", f64::NAN, None));
        hits.push_hit(hit("low", 1.0, None));
        hits.push_hit(hit("nan-a", f64::NAN, None));
        hits.push_hit(hit("high", 2.0, None));

        hits.sort_by_sortkey();

        let names: Vec<&str> = (0..hits.len())
            .map(|index| hits.get_hit(index).unwrap().name.as_str())
            .collect();
        assert_eq!(names, ["high", "low", "nan-a", "nan-b"]);
    }

    #[test]
    fn sort_by_seqidx_places_unassigned_hits_last() {
        let mut hits = TopHits::new();
        hits.push_hit(hit("unassigned-a", 0.0, None));
        hits.push_hit(hit("second", 0.0, Some(2)));
        hits.push_hit(hit("first", 0.0, Some(1)));
        hits.push_hit(hit("unassigned-b", 0.0, None));

        hits.sort_by_seqidx();

        let names: Vec<&str> = (0..hits.len())
            .map(|index| hits.get_hit(index).unwrap().name.as_str())
            .collect();
        assert_eq!(names, ["first", "second", "unassigned-a", "unassigned-b"]);
    }

    #[test]
    fn mutable_sorted_access_invalidates_cached_sort_state() {
        let mut hits = TopHits::new();
        hits.push_hit(hit("first", 2.0, Some(0)));
        hits.push_hit(hit("second", 1.0, Some(1)));
        hits.sort_by_sortkey();

        hits.get_hit_mut(0).unwrap().sortkey = 0.0;

        assert!(!hits.is_sorted_by_sortkey);
        assert!(!hits.is_sorted_by_seqidx);
        hits.sort_by_sortkey();
        assert_eq!(hits.get_hit(0).unwrap().name, "second");
    }

    #[test]
    fn threshold_applies_target_and_domain_pvalue_hierarchy() {
        let mut hits = TopHits::new();
        let mut passing = hit("passing", 1.0, Some(0));
        passing.log_pvalue = 0.005f64.ln();
        passing.domains = vec![
            Domain {
                log_pvalue: 0.02f64.ln(),
                ..Domain::default()
            },
            Domain {
                log_pvalue: 0.001f64.ln(),
                ..Domain::default()
            },
        ];
        hits.push_hit(passing);

        let mut failing = hit("failing", 0.0, Some(1));
        failing.log_pvalue = 0.5f64.ln();
        hits.push_hit(failing);

        let mut duplicate = hit("duplicate", 2.0, Some(2));
        duplicate.log_pvalue = 0.0001f64.ln();
        duplicate.flags |= HitFlags::DUPLICATE;
        hits.push_hit(duplicate);

        hits.threshold(0.1, 0.1, 0.01, 0.01, false);

        assert_eq!(hits.num_reported, 1);
        assert_eq!(hits.num_included, 1);
        assert!(hits.hits[0].is_reported());
        assert!(hits.hits[0].is_included());
        assert_eq!(hits.hits[0].num_reported, 2);
        assert_eq!(hits.hits[0].num_included, 1);
        assert!(hits.hits[0].domains[0].is_reported);
        assert!(!hits.hits[0].domains[0].is_included);
        assert!(hits.hits[0].domains[1].is_included);
        assert!(!hits.hits[1].is_reported());
        assert!(!hits.hits[2].is_reported());

        hits.threshold(0.001, 0.001, 0.0001, 0.0001, false);
        assert_eq!(hits.num_reported, 0);
        assert_eq!(hits.num_included, 0);
        assert_eq!(hits.hits[0].num_reported, 0);
        assert_eq!(hits.hits[0].num_included, 0);
    }

    #[test]
    fn bit_cutoff_thresholding_preserves_and_counts_precomputed_flags() {
        let mut hits = TopHits::new();
        let mut included = hit("included", 2.0, Some(0));
        included.flags |= HitFlags::REPORTED | HitFlags::INCLUDED;
        included.domains.push(Domain {
            is_reported: true,
            is_included: true,
            ..Domain::default()
        });
        hits.push_hit(included);

        let mut reported = hit("reported", 1.0, Some(1));
        reported.flags |= HitFlags::REPORTED;
        hits.push_hit(reported);

        hits.threshold(0.0, 0.0, 0.0, 0.0, true);

        assert_eq!(hits.num_reported, 2);
        assert_eq!(hits.num_included, 1);
        assert_eq!(hits.hits[0].num_reported, 1);
        assert_eq!(hits.hits[0].num_included, 1);
    }

    #[test]
    fn merge_preserves_sortkey_order_and_recounts_flags() {
        let mut left = TopHits::new();
        left.push_hit(hit("five", 5.0, Some(0)));
        left.push_hit(hit("one", 1.0, Some(1)));

        let mut right = TopHits::new();
        let mut four = hit("four", 4.0, Some(2));
        four.flags |= HitFlags::REPORTED;
        right.push_hit(four);
        right.push_hit(hit("two", 2.0, Some(3)));
        right.push_hit(hit("nan", f64::NAN, None));

        left.merge(right);

        let names: Vec<&str> = (0..left.len())
            .map(|index| left.get_hit(index).unwrap().name.as_str())
            .collect();
        assert_eq!(names, ["five", "four", "two", "one", "nan"]);
        assert!(left.is_sorted_by_sortkey);
        assert!(!left.is_sorted_by_seqidx);
        assert_eq!(left.num_reported, 1);
    }
}
