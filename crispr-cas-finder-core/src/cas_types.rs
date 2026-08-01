//! Local re-implementations of types previously provided by
//! `genomic-system-finder-core`.  Keeping them here removes the external
//! dependency while preserving the same API surface used by `casfinder`,
//! `casparser`, and the WASM wrapper.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RepliconTopology
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepliconTopology {
    #[default]
    Circular,
    Linear,
}

// ---------------------------------------------------------------------------
// HmmerOptions (only the fields actually used)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmerOptions {
    pub coverage_profile: f64,
}

impl Default for HmmerOptions {
    fn default() -> Self {
        Self {
            coverage_profile: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Hit types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmerHit {
    pub id: String,
    pub gene_name: String,
    pub seq_len: u32,
    pub i_evalue: f64,
    pub score: f64,
    pub profile_coverage: f64,
    pub seq_coverage: f64,
    pub begin_match: u32,
    pub end_match: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneStatus {
    Mandatory,
    Accessory,
    Forbidden,
    Neutral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHit {
    pub hit: HmmerHit,
    pub position: usize,
    pub gene_ref: String,
    pub gene_status: GeneStatus,
    pub model_fully_qualified_name: String,
    pub is_exchangeable: bool,
    pub locus_num: usize,
    pub counterpart: String,
    pub used_in: Vec<String>,
}

// ---------------------------------------------------------------------------
// SequenceIndex
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct SequenceIndex {
    id_to_position: HashMap<String, usize>,
}

impl SequenceIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_ids(ids: &[&str]) -> Self {
        let id_to_position_map = ids
            .iter()
            .enumerate()
            .map(|(position_index, sequence_id)| (sequence_id.to_string(), position_index))
            .collect();
        Self {
            id_to_position: id_to_position_map,
        }
    }

    pub fn position(&self, id: &str) -> Option<usize> {
        self.id_to_position.get(id).copied()
    }
}

// ---------------------------------------------------------------------------
// Gene / Model definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneDefinition {
    pub name: String,
    pub status: GeneStatus,
    pub loner: bool,
    pub multi_system: bool,
    pub system_ref: Option<String>,
    pub exchangeables: Vec<String>,
    pub inter_gene_max_space: Option<u32>,
    pub multi_model: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemModel {
    pub fully_qualified_name: String,
    pub family: String,
    pub name: String,
    pub version: Option<String>,
    pub genes: Vec<GeneDefinition>,
    pub inter_gene_max_space: u32,
    pub min_mandatory_genes_required: u32,
    pub min_genes_required: u32,
    pub multi_loci: bool,
}

impl SystemModel {
    pub fn mandatory_genes(&self) -> impl Iterator<Item = &GeneDefinition> {
        self.genes
            .iter()
            .filter(|g| g.status == GeneStatus::Mandatory)
    }

    pub fn accessory_genes(&self) -> impl Iterator<Item = &GeneDefinition> {
        self.genes
            .iter()
            .filter(|g| g.status == GeneStatus::Accessory)
    }

    pub fn forbidden_genes(&self) -> impl Iterator<Item = &GeneDefinition> {
        self.genes
            .iter()
            .filter(|g| g.status == GeneStatus::Forbidden)
    }

    pub fn mandatory_count(&self) -> usize {
        self.mandatory_genes().count()
    }

    pub fn accessory_count(&self) -> usize {
        self.accessory_genes().count()
    }

    pub fn all_profile_names(&self) -> Vec<&str> {
        let mut profile_names: Vec<&str> = self
            .genes
            .iter()
            .flat_map(|g| {
                std::iter::once(g.name.as_str()).chain(g.exchangeables.iter().map(|e| e.as_str()))
            })
            .collect();
        profile_names.sort_unstable();
        profile_names.dedup();
        profile_names
    }
}

// ---------------------------------------------------------------------------
// ModelRegistry
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct ModelRegistry {
    models: HashMap<String, SystemModel>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, model: SystemModel) {
        self.models
            .insert(model.fully_qualified_name.clone(), model);
    }

    pub fn get(&self, fully_qualified_model_name: &str) -> Option<&SystemModel> {
        self.models.get(fully_qualified_model_name)
    }
}

// ---------------------------------------------------------------------------
// Clustering
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    pub hits: Vec<SystemHit>,
    pub wraps_origin: bool,
}

impl Default for Cluster {
    fn default() -> Self {
        Self::new()
    }
}

impl Cluster {
    pub fn new() -> Self {
        Self {
            hits: Vec::new(),
            wraps_origin: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }
}

/// Cluster hits by genomic proximity. Hits are sorted by `position`, then
/// consecutive hits whose gap (exclusive positions between them) exceeds
/// `inter_gene_max_space` start a new cluster.  For circular replicons the
/// first and last clusters may be merged across the origin.
pub fn cluster_hits(
    hits: &mut [SystemHit],
    inter_gene_max_space: u32,
    replicon_length: usize,
    replicon_topology: RepliconTopology,
) -> Vec<Cluster> {
    if hits.is_empty() {
        return Vec::new();
    }

    hits.sort_by_key(|h| h.position);

    let mut clusters: Vec<Cluster> = Vec::new();
    let mut current_cluster = Cluster::new();
    current_cluster.hits.push(hits[0].clone());

    for hit in &hits[1..] {
        let previous_position = current_cluster.hits.last().unwrap().position;
        let gap_between_hits = if hit.position > previous_position {
            hit.position - previous_position - 1
        } else {
            0
        };
        if gap_between_hits > inter_gene_max_space as usize {
            clusters.push(current_cluster);
            current_cluster = Cluster::new();
        }
        current_cluster.hits.push(hit.clone());
    }
    clusters.push(current_cluster);

    // Circular wrap-around merge
    if replicon_topology == RepliconTopology::Circular && clusters.len() > 1 && replicon_length > 0
    {
        let first_start = clusters.first().unwrap().hits.first().unwrap().position;
        let last_end = clusters.last().unwrap().hits.last().unwrap().position;
        let wrap_gap = (replicon_length.saturating_sub(last_end).saturating_sub(1)) + first_start;
        if wrap_gap <= inter_gene_max_space as usize {
            let mut last_cluster = clusters.pop().unwrap();
            last_cluster.hits.extend(clusters.remove(0).hits);
            last_cluster.wraps_origin = true;
            clusters.insert(0, last_cluster);
        }
    }

    clusters
}

// ---------------------------------------------------------------------------
// DetectedSystem
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedSystem {
    pub id: String,
    pub replicon: String,
    pub model_fully_qualified_name: String,
    pub score: f64,
    pub wholeness: f64,
    pub loci_count: usize,
    pub occurrence: usize,
    pub hits: Vec<SystemHit>,
    pub state: String,
    pub mandatory_found: Vec<String>,
    pub accessory_found: Vec<String>,
    pub forbidden_found: Vec<String>,
}

impl DetectedSystem {
    /// Two systems overlap if they share any hit (by sequence id).
    fn overlaps_with(&self, other: &DetectedSystem) -> bool {
        use std::collections::HashSet;
        let this_system_hit_ids: HashSet<&str> =
            self.hits.iter().map(|h| h.hit.id.as_str()).collect();
        other
            .hits
            .iter()
            .any(|h| this_system_hit_ids.contains(h.hit.id.as_str()))
    }
}

/// Greedy maximum-weight independent set on the conflict graph.
/// Systems that share hits conflict; we keep the highest-scoring
/// non-overlapping subset.
pub fn select_best_solution(systems: Vec<DetectedSystem>) -> Vec<DetectedSystem> {
    if systems.len() <= 1 {
        return systems;
    }

    // Sort by score desc, tie-break by wholeness desc
    let mut indexed_systems: Vec<(usize, &DetectedSystem)> = systems.iter().enumerate().collect();
    indexed_systems.sort_by(|left, right| {
        right
            .1
            .score
            .partial_cmp(&left.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .1
                    .wholeness
                    .partial_cmp(&left.1.wholeness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let system_count = systems.len();
    let mut is_excluded = vec![false; system_count];
    let mut selected_indices = Vec::new();

    for (system_index, _system) in &indexed_systems {
        if is_excluded[*system_index] {
            continue;
        }
        selected_indices.push(*system_index);
        // Exclude all overlapping systems
        for (other_system_index, other_system) in &indexed_systems {
            if !is_excluded[*other_system_index]
                && *other_system_index != *system_index
                && systems[*system_index].overlaps_with(other_system)
            {
                is_excluded[*other_system_index] = true;
            }
        }
    }

    selected_indices.sort_unstable();
    selected_indices
        .into_iter()
        .map(|index| systems[index].clone())
        .collect()
}

// ---------------------------------------------------------------------------
// SearchResults
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SearchResults {
    pub systems: Vec<DetectedSystem>,
    pub rejected: Vec<String>,
    pub skipped_replicons: Vec<String>,
}
