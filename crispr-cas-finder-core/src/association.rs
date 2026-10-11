//! Relating CRISPR arrays to Cas systems on the same sequence.

use crate::cas_types::RepliconTopology;
use crate::casparser::CasCluster;
use crate::types::CrisprArray;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A CRISPR array lying within the vicinity of a Cas system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrisprCasAssociation {
    pub seq_id: String,
    pub crispr_id: String,
    pub cas_id: String,
    /// Model of the Cas system, e.g. `CASFinder/CAS-TypeIE`.
    pub cas_system: String,
    /// Bases between the array and the system; 0 when they overlap.
    pub distance: usize,
}

/// Kind of element in a [`CrisprCasLocus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocusElementKind {
    Crispr,
    Cas,
}

/// One array or Cas system inside a [`CrisprCasLocus`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocusElement {
    pub kind: LocusElementKind,
    pub id: String,
    /// Consensus repeat for an array, model name for a Cas system.
    pub label: String,
    pub start: usize,
    pub end: usize,
}

/// Arrays and Cas systems chained by proximity on one sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrisprCasLocus {
    /// `<seq_id>_locus_<n>` with `n` counted per sequence.
    pub id: String,
    pub seq_id: String,
    pub start: usize,
    pub end: usize,
    pub crispr_count: usize,
    pub cas_count: usize,
    pub elements: Vec<LocusElement>,
}

fn linear_gap(first: (usize, usize), second: (usize, usize)) -> usize {
    if first.1 < second.0 {
        second.0 - first.1 - 1
    } else if second.1 < first.0 {
        first.0 - second.1 - 1
    } else {
        0
    }
}

/// Bases between two 1-based inclusive intervals, 0 when they overlap.
///
/// On a circular sequence of known length the shorter way around is used.
pub fn interval_distance(
    first: (usize, usize),
    second: (usize, usize),
    topology: RepliconTopology,
    sequence_length: Option<usize>,
) -> usize {
    let linear = linear_gap(first, second);
    let (RepliconTopology::Circular, Some(length)) = (topology, sequence_length) else {
        return linear;
    };
    let (left, right) = if first.0 <= second.0 {
        (first, second)
    } else {
        (second, first)
    };
    // From the end of the rightmost interval through the origin to the
    // start of the leftmost one.
    let around = length.saturating_sub(right.1) + left.0.saturating_sub(1);
    linear.min(around)
}

/// List every array within `vicinity` bp of a Cas system on its sequence.
///
/// Upstream tests whether the array boundary shifted by the vicinity falls
/// inside the Cas system, which misses arrays next to short systems; this
/// uses the plain distance between the two elements instead.
pub fn associate_crisprs_with_cas(
    arrays: &[CrisprArray],
    cas_systems: &[CasCluster],
    vicinity: usize,
    topology: RepliconTopology,
    sequence_lengths: &HashMap<String, usize>,
) -> Vec<CrisprCasAssociation> {
    let mut associations = Vec::new();
    for array in arrays {
        for system in cas_systems
            .iter()
            .filter(|system| system.seq_id == array.seq_id)
        {
            let distance = interval_distance(
                (array.start, array.end),
                (system.start, system.end),
                topology,
                sequence_lengths.get(&array.seq_id).copied(),
            );
            if distance <= vicinity {
                associations.push(CrisprCasAssociation {
                    seq_id: array.seq_id.clone(),
                    crispr_id: array.id.clone(),
                    cas_id: system.id.clone(),
                    cas_system: system.system.clone(),
                    distance,
                });
            }
        }
    }
    associations
}

/// Chain arrays and Cas systems closer than `max_distance` bp into loci.
///
/// Elements are ordered by start; an element joins the current locus when
/// it starts within `max_distance` of the locus end. Loci holding a single
/// element are not reported. Chaining does not cross the sequence origin.
pub fn cluster_loci(
    arrays: &[CrisprArray],
    cas_systems: &[CasCluster],
    max_distance: usize,
) -> Vec<CrisprCasLocus> {
    let mut sequence_order: Vec<&str> = Vec::new();
    let mut elements_by_sequence: HashMap<&str, Vec<LocusElement>> = HashMap::new();
    for array in arrays {
        let entry = elements_by_sequence
            .entry(array.seq_id.as_str())
            .or_insert_with(|| {
                sequence_order.push(array.seq_id.as_str());
                Vec::new()
            });
        entry.push(LocusElement {
            kind: LocusElementKind::Crispr,
            id: array.id.clone(),
            label: array.consensus_repeat.clone(),
            start: array.start,
            end: array.end,
        });
    }
    for system in cas_systems {
        let entry = elements_by_sequence
            .entry(system.seq_id.as_str())
            .or_insert_with(|| {
                sequence_order.push(system.seq_id.as_str());
                Vec::new()
            });
        entry.push(LocusElement {
            kind: LocusElementKind::Cas,
            id: system.id.clone(),
            label: system.system.clone(),
            start: system.start,
            end: system.end,
        });
    }

    let mut loci = Vec::new();
    for seq_id in sequence_order {
        let mut elements = elements_by_sequence.remove(seq_id).unwrap_or_default();
        elements.sort_by(|left, right| {
            (left.start, left.end, &left.id).cmp(&(right.start, right.end, &right.id))
        });
        let mut locus_count = 0;
        let mut current: Vec<LocusElement> = Vec::new();
        let mut current_end = 0usize;
        let mut flush = |current: &mut Vec<LocusElement>, current_end: usize| {
            if current.len() > 1 {
                locus_count += 1;
                loci.push(CrisprCasLocus {
                    id: format!("{seq_id}_locus_{locus_count}"),
                    seq_id: seq_id.to_string(),
                    start: current[0].start,
                    end: current_end,
                    crispr_count: current
                        .iter()
                        .filter(|element| element.kind == LocusElementKind::Crispr)
                        .count(),
                    cas_count: current
                        .iter()
                        .filter(|element| element.kind == LocusElementKind::Cas)
                        .count(),
                    elements: std::mem::take(current),
                });
            } else {
                current.clear();
            }
        };
        for element in elements {
            if !current.is_empty() && element.start > current_end.saturating_add(max_distance) {
                flush(&mut current, current_end);
                current_end = 0;
            }
            current_end = current_end.max(element.end);
            current.push(element);
        }
        flush(&mut current, current_end);
    }
    loci
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array(seq_id: &str, number: usize, start: usize, end: usize) -> CrisprArray {
        CrisprArray {
            id: format!("{seq_id}_CRISPR_{number}"),
            seq_id: seq_id.to_string(),
            start,
            end,
            consensus_repeat: "GTTT".to_string(),
            repeats: Vec::new(),
            spacers: Vec::new(),
            evidence_level: 4,
            statistics: Default::default(),
            orientation: crate::types::Orientation::Unknown,
            repeat_id: "Unknown".to_string(),
            crispr_direction: "ND".to_string(),
            crisprdb_repeat_count: 0,
            known_in_crisprdb: None,
            reclassified_from: None,
        }
    }

    fn system(seq_id: &str, number: usize, start: usize, end: usize) -> CasCluster {
        CasCluster {
            id: format!("{seq_id}_Cas_{number}"),
            seq_id: seq_id.to_string(),
            system: "CASFinder/CAS-TypeIE".to_string(),
            start,
            end,
            ..CasCluster::default()
        }
    }

    #[test]
    fn distance_is_zero_for_overlap_and_counts_bases_between() {
        let linear = RepliconTopology::Linear;
        assert_eq!(interval_distance((10, 20), (15, 30), linear, None), 0);
        assert_eq!(interval_distance((10, 20), (21, 30), linear, None), 0);
        assert_eq!(interval_distance((10, 20), (25, 30), linear, None), 4);
        assert_eq!(interval_distance((25, 30), (10, 20), linear, None), 4);
    }

    #[test]
    fn circular_distance_takes_the_shorter_way_around() {
        let circular = RepliconTopology::Circular;
        assert_eq!(
            interval_distance((1, 100), (9_901, 10_000), circular, Some(10_000)),
            0
        );
        assert_eq!(
            interval_distance((11, 100), (9_901, 9_990), circular, Some(10_000)),
            20
        );
        assert_eq!(
            interval_distance((11, 100), (9_901, 9_990), circular, None),
            9_800
        );
    }

    #[test]
    fn associations_respect_vicinity_and_sequence() {
        let arrays = [array("a", 1, 1_000, 1_500), array("b", 1, 1_000, 1_500)];
        let systems = [system("a", 1, 2_000, 9_000), system("a", 2, 20_000, 25_000)];
        let lengths = HashMap::from([("a".to_string(), 1_000_000)]);

        let associations = associate_crisprs_with_cas(
            &arrays,
            &systems,
            600,
            RepliconTopology::Circular,
            &lengths,
        );

        assert_eq!(associations.len(), 1);
        assert_eq!(associations[0].crispr_id, "a_CRISPR_1");
        assert_eq!(associations[0].cas_id, "a_Cas_1");
        assert_eq!(associations[0].distance, 499);
    }

    #[test]
    fn loci_chain_neighbours_and_skip_singletons() {
        let arrays = [
            array("a", 1, 1_000, 1_500),
            array("a", 2, 12_000, 12_400),
            array("a", 3, 500_000, 500_300),
        ];
        let systems = [system("a", 1, 2_000, 9_000)];

        let loci = cluster_loci(&arrays, &systems, 5_000);

        assert_eq!(loci.len(), 1);
        assert_eq!(loci[0].id, "a_locus_1");
        assert_eq!((loci[0].start, loci[0].end), (1_000, 12_400));
        assert_eq!((loci[0].crispr_count, loci[0].cas_count), (2, 1));
        assert_eq!(
            loci[0]
                .elements
                .iter()
                .map(|element| element.id.as_str())
                .collect::<Vec<_>>(),
            ["a_CRISPR_1", "a_Cas_1", "a_CRISPR_2"]
        );
    }
}
