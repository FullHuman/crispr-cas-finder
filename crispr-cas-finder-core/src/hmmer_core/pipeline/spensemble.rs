// spensemble.rs - Domain envelopes from an ensemble of sampled segment pairs
//
// Port of src/p7_spensemble.c (HMMER 3.4) and of the single-linkage
// clustering it calls, esl_cluster_SingleLinkage() (Easel 0.49).
//
// Every stochastic traceback of a region contributes the domains it
// contains, each a segment pair: a stretch i..j of the sequence aligned to
// a stretch k..m of the model. Segment pairs that overlap enough on both
// and lie on nearly the same diagonal are linked; a cluster of linked pairs
// that occurs in enough of the tracebacks is a domain, and its envelope is
// the widest start and end that enough of its members share.

/// One domain of one sampled traceback, or the consensus of a cluster.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SegmentPair {
    /// The traceback the pair comes from; for a consensus, its cluster.
    pub index: usize,
    /// First and last residue on the sequence, 1-based.
    pub i: i32,
    pub j: i32,
    /// First and last node on the model, 1-based.
    pub k: i32,
    pub m: i32,
    /// Share of the tracebacks with a member of the cluster; consensus only.
    pub probability: f32,
}

/// The linkage and significance rules of `p7_spensemble_Cluster()`.
#[derive(Debug, Clone, Copy)]
pub struct ClusterRules {
    /// Smallest overlap, on the sequence and on the model, of two linked
    /// pairs, as a share of the shorter (or longer) of the two.
    pub min_overlap: f32,
    /// Whether that share is of the shorter pair.
    pub of_smaller: bool,
    /// Two linked pairs start or end within this many diagonals.
    pub max_diagdiff: i32,
    /// Smallest share of the tracebacks a cluster must occur in.
    pub min_posterior: f32,
    /// Smallest share of a cluster's members an endpoint must be found in.
    pub min_endpointp: f32,
}

/// The segment pairs sampled for one region, and the buffers clustering
/// them needs; reused from region to region.
#[derive(Debug, Default)]
pub struct SegmentPairEnsemble {
    pairs: Vec<SegmentPair>,
    num_samples: usize,
    /// Stacks of the pairs not yet clustered and of those to extend from.
    available: Vec<usize>,
    pending: Vec<usize>,
    /// Cluster of each pair.
    assignment: Vec<usize>,
    /// Occurrences of each endpoint within a cluster's window.
    endpoint_counts: Vec<u32>,
    significant: Vec<SegmentPair>,
    dominated: Vec<bool>,
}

impl SegmentPairEnsemble {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget the previous region.
    pub fn reuse(&mut self) {
        self.pairs.clear();
        self.num_samples = 0;
        self.significant.clear();
    }

    /// Add the pair `i..j`, `k..m` of traceback `sample`. Tracebacks are
    /// added in order, all the pairs of one before those of the next.
    pub fn add(&mut self, sample: usize, i: i32, j: i32, k: i32, m: i32) {
        debug_assert!(sample <= self.num_samples);
        if sample == self.num_samples {
            self.num_samples += 1;
        }
        self.pairs.push(SegmentPair {
            index: sample,
            i,
            j,
            k,
            m,
            probability: 0.0,
        });
    }

    /// Number of pairs added.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The consensus pairs found by [`Self::cluster`], by start.
    pub fn significant(&self) -> &[SegmentPair] {
        &self.significant
    }

    /// `esl_cluster_SingleLinkage()` over the pairs with `link_spsamples()`:
    /// fills `assignment` and returns the number of clusters.
    ///
    /// The order in which pairs are visited decides the numbering of the
    /// clusters, and is Easel's.
    fn single_linkage(&mut self, rules: &ClusterRules) -> usize {
        let n = self.pairs.len();
        self.available.clear();
        self.available.extend((0..n).rev());
        self.pending.clear();
        self.assignment.clear();
        self.assignment.resize(n, 0);
        let mut num_clusters = 0;
        while let Some(first) = self.available.pop() {
            self.pending.push(first);
            while let Some(v) = self.pending.pop() {
                self.assignment[v] = num_clusters;
                let mut position = self.available.len();
                while position > 0 {
                    position -= 1;
                    let w = self.available[position];
                    if linked(&self.pairs[v], &self.pairs[w], rules) {
                        self.available.swap_remove(position);
                        self.pending.push(w);
                    }
                }
            }
            num_clusters += 1;
        }
        num_clusters
    }

    /// `p7_spensemble_Cluster()`: cluster the pairs and define the consensus
    /// pair of every significant cluster, ordered by start on the sequence.
    /// Returns their number.
    pub fn cluster(&mut self, rules: &ClusterRules) -> usize {
        self.significant.clear();
        let num_clusters = self.single_linkage(rules);
        let num_samples = self.num_samples as f32;

        for cluster in 0..num_clusters {
            let members = || {
                self.pairs
                    .iter()
                    .zip(&self.assignment)
                    .filter(move |&(_, &assigned)| assigned == cluster)
                    .map(|(pair, _)| pair)
            };

            // Tracebacks with a member in the cluster: one of them can
            // contribute several, which count once.
            let mut included = 0usize;
            let mut last_sample = usize::MAX;
            for pair in members() {
                if pair.index != last_sample {
                    included += 1;
                }
                last_sample = pair.index;
            }
            let probability = included as f32 / num_samples;
            if probability < rules.min_posterior {
                continue;
            }

            let mut bounds: Option<[(i32, i32); 4]> = None;
            for pair in members() {
                let coordinates = [pair.i, pair.j, pair.k, pair.m];
                bounds = Some(match bounds {
                    None => coordinates.map(|c| (c, c)),
                    Some(bounds) => {
                        let mut widened = bounds;
                        for (bound, c) in widened.iter_mut().zip(coordinates) {
                            *bound = (bound.0.min(c), bound.1.max(c));
                        }
                        widened
                    }
                });
            }
            let Some([i_bounds, j_bounds, k_bounds, m_bounds]) = bounds else {
                continue;
            };

            // An endpoint needs this many occurrences: the share, rounded up.
            let threshold = (included as f32 * rules.min_endpointp).ceil() as i64;
            let counts = &mut self.endpoint_counts;
            let mut consensus =
                |(low, high): (i32, i32), coordinate: fn(&SegmentPair) -> i32, leftmost: bool| {
                    let width = (high - low + 1) as usize;
                    counts.clear();
                    counts.resize(width, 0);
                    for (pair, _) in self
                        .pairs
                        .iter()
                        .zip(&self.assignment)
                        .filter(|&(_, &assigned)| assigned == cluster)
                    {
                        counts[(coordinate(pair) - low) as usize] += 1;
                    }
                    let reaches = |&count: &u32| i64::from(count) >= threshold;
                    let found = if leftmost {
                        counts.iter().position(reaches)
                    } else {
                        counts.iter().rposition(reaches)
                    };
                    // No endpoint reaches it: the most frequent one, the first
                    // of them on a tie (esl_vec_IArgMax).
                    let offset = found.unwrap_or_else(|| {
                        let mut best = 0;
                        for (offset, &count) in counts.iter().enumerate() {
                            if count > counts[best] {
                                best = offset;
                            }
                        }
                        best
                    });
                    low + offset as i32
                };
            let best_i = consensus(i_bounds, |pair| pair.i, true);
            let best_k = consensus(k_bounds, |pair| pair.k, true);
            let best_j = consensus(j_bounds, |pair| pair.j, false);
            let best_m = consensus(m_bounds, |pair| pair.m, false);

            // Without a well-defined domain (a stretch of biased
            // composition, say) the endpoints can cross: no domain.
            if best_i > best_j || best_k > best_m {
                continue;
            }
            self.significant.push(SegmentPair {
                index: cluster,
                i: best_i,
                j: best_j,
                k: best_k,
                m: best_m,
                probability,
            });
        }

        // HMMER sorts with qsort() on the start alone. Equal starts keep
        // their order here, as they do in the C libraries' sorts of so few
        // elements.
        self.significant.sort_by_key(|pair| pair.i);
        self.significant.len()
    }

    /// The end of `region_trace_ensemble()` in p7_domaindef.c: drop the
    /// consensus pairs "dominated" by another one that covers 80% of the
    /// shorter of the two on the sequence and is more probable. Returns the
    /// number left.
    pub fn remove_dominated(&mut self) -> usize {
        let pairs = &self.significant;
        let n = pairs.len();
        self.dominated.clear();
        self.dominated.resize(n, false);
        for d in 0..n {
            for d2 in d + 1..n {
                let overlap = pairs[d].j.min(pairs[d2].j) - pairs[d].i.max(pairs[d2].i) + 1;
                // As in HMMER: an overlap of exactly zero residues, two
                // adjacent envelopes, ends the comparisons of `d`; a gap
                // between two envelopes does not.
                if overlap == 0 {
                    break;
                }
                let shorter = (pairs[d].j - pairs[d].i + 1).min(pairs[d2].j - pairs[d2].i + 1);
                if overlap as f32 / shorter as f32 >= 0.8 {
                    if pairs[d].probability > pairs[d2].probability {
                        self.dominated[d2] = true;
                    } else {
                        self.dominated[d] = true;
                    }
                }
            }
        }
        let mut index = 0;
        let dominated = &self.dominated;
        self.significant.retain(|_| {
            let keep = !dominated[index];
            index += 1;
            keep
        });
        self.significant.len()
    }
}

/// `link_spsamples()`: whether two segment pairs belong to one domain.
fn linked(a: &SegmentPair, b: &SegmentPair, rules: &ClusterRules) -> bool {
    let pick = |x: i32, y: i32| {
        if rules.of_smaller { x.min(y) } else { x.max(y) }
    };

    // Overlap on the sequence.
    let overlap = a.j.min(b.j) - a.i.max(b.i) + 1;
    let length = pick(a.j - a.i + 1, b.j - b.i + 1);
    if (overlap as f32 / length as f32) < rules.min_overlap {
        return false;
    }

    // Overlap on the model; HMMER counts it one short of the sequence's.
    let overlap = a.m.min(b.m) - a.k.max(b.k);
    let length = pick(a.m - a.k + 1, b.m - b.k + 1);
    if (overlap as f32 / length as f32) < rules.min_overlap {
        return false;
    }

    // Starts or ends on nearby diagonals.
    ((a.i - a.k) - (b.i - b.k)).abs() <= rules.max_diagdiff
        || ((a.j - a.m) - (b.j - b.m)).abs() <= rules.max_diagdiff
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULES: ClusterRules = ClusterRules {
        min_overlap: 0.8,
        of_smaller: true,
        max_diagdiff: 4,
        min_posterior: 0.25,
        min_endpointp: 0.02,
    };

    #[test]
    fn two_domains_sampled_together_give_two_envelopes() {
        let mut ensemble = SegmentPairEnsemble::new();
        // Every traceback has both domains; one in ten starts the first a
        // residue early, and a single one starts it far too early.
        for sample in 0..100 {
            let start = match sample {
                7 => 2,
                s if s % 10 == 0 => 9,
                _ => 10,
            };
            ensemble.add(sample, start, 60, 1, 50);
            ensemble.add(sample, 80, 130, 1, 50);
        }
        assert_eq!(ensemble.len(), 200);
        assert_eq!(ensemble.cluster(&RULES), 2);
        let found = ensemble.significant();
        // The widest start that 2% of the members share: 9, not 2.
        assert_eq!(
            (found[0].i, found[0].j, found[0].k, found[0].m),
            (9, 60, 1, 50)
        );
        assert_eq!((found[1].i, found[1].j), (80, 130));
        assert_eq!(found[0].probability, 1.0);
        assert_eq!(ensemble.remove_dominated(), 2);
    }

    #[test]
    fn rare_clusters_are_not_domains() {
        let mut ensemble = SegmentPairEnsemble::new();
        for sample in 0..100 {
            ensemble.add(sample, 10, 60, 1, 50);
            if sample < 24 {
                ensemble.add(sample, 80, 130, 1, 50);
            }
        }
        assert_eq!(ensemble.cluster(&RULES), 1);
        ensemble.reuse();
        assert!(ensemble.is_empty());
        assert_eq!(ensemble.cluster(&RULES), 0);
    }

    #[test]
    fn linkage_needs_overlap_on_both_axes_and_a_shared_diagonal() {
        let pair = |i, j, k, m| SegmentPair {
            i,
            j,
            k,
            m,
            ..SegmentPair::default()
        };
        let a = pair(10, 60, 1, 51);
        assert!(linked(&a, &pair(12, 60, 3, 51), &RULES));
        // Same sequence stretch, another part of the model.
        assert!(!linked(&a, &pair(10, 60, 40, 90), &RULES));
        // Overlapping on both, but ten diagonals apart at both ends.
        assert!(!linked(&a, &pair(20, 70, 1, 51), &RULES));
    }

    #[test]
    fn a_less_probable_envelope_inside_another_is_dropped() {
        let mut ensemble = SegmentPairEnsemble::new();
        ensemble.significant = vec![
            SegmentPair {
                index: 0,
                i: 10,
                j: 100,
                k: 1,
                m: 90,
                probability: 0.9,
            },
            SegmentPair {
                index: 1,
                i: 20,
                j: 60,
                k: 40,
                m: 80,
                probability: 0.3,
            },
            SegmentPair {
                index: 2,
                i: 150,
                j: 200,
                k: 1,
                m: 50,
                probability: 0.5,
            },
        ];
        assert_eq!(ensemble.remove_dominated(), 2);
        assert_eq!(ensemble.significant()[0].index, 0);
        assert_eq!(ensemble.significant()[1].index, 2);
    }
}
