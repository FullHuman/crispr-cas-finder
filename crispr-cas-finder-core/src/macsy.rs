//! System search following MacSyFinder 2, used for model definitions in the
//! MacSyFinder 2 format (`<model vers="2.0">`), such as CasFinder 3.
//!
//! The steps mirror MacSyLib: keep the best-scoring profile for each
//! protein, cluster the hits of a model by gene distance, check each cluster
//! against the model's quorum, score it, and finally pick the combination of
//! non-conflicting systems with the highest total score.
//!
//! `loner`, `multi_system` and `multi_loci` are not implemented; models using
//! them are rejected by [`check_supported`] rather than evaluated wrongly.

use crate::annotation::RepliconProteins;
use crate::cas_types::{
    DetectedSystem, GeneDefinition, GeneStatus, HmmerHit, RepliconTopology, SystemHit, SystemModel,
};
use quick_xml::Reader;
use quick_xml::events::Event;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Weights used to score clusters and systems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HitWeights {
    pub mandatory: f64,
    pub accessory: f64,
    pub neutral: f64,
    /// Factor applied when a function is fulfilled by an exchangeable gene.
    pub exchangeable: f64,
    /// Penalty per extra cluster fulfilling the same function.
    pub redundancy_penalty: f64,
}

impl Default for HitWeights {
    /// MacSyFinder's defaults.
    fn default() -> Self {
        Self {
            mandatory: 1.0,
            accessory: 0.5,
            neutral: 0.0,
            exchangeable: 0.8,
            redundancy_penalty: 1.5,
        }
    }
}

/// Thresholds deciding which HMM hits are kept.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HitFiltering {
    /// Maximum sequence E-value for profiles searched without GA cutoffs.
    pub e_value_search: f64,
    /// Maximum independent E-value of a domain.
    pub i_evalue_sel: f64,
    /// Minimum fraction of the profile covered by a domain.
    pub coverage_profile: f64,
    /// Use the profile's gathering (GA) bit-score cutoffs when it has them.
    pub cut_ga: bool,
}

impl Default for HitFiltering {
    /// MacSyFinder's defaults.
    fn default() -> Self {
        Self {
            e_value_search: 0.1,
            i_evalue_sel: 0.001,
            coverage_profile: 0.5,
            cut_ga: true,
        }
    }
}

/// Settings shipped with a model package in `model_conf.xml`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ModelConfig {
    pub weights: HitWeights,
    pub filtering: HitFiltering,
}

impl ModelConfig {
    /// Parse a `model_conf.xml`; settings it does not mention keep
    /// MacSyFinder's defaults.
    pub fn parse(content: &str) -> Result<Self, String> {
        let mut config = Self::default();
        let mut reader = Reader::from_str(content);
        reader.config_mut().trim_text(true);
        let mut path: Vec<String> = Vec::new();
        loop {
            match reader.read_event() {
                Ok(Event::Start(start)) => path.push(
                    std::str::from_utf8(start.name().as_ref())
                        .map_err(|error| format!("invalid element name: {error}"))?
                        .to_string(),
                ),
                Ok(Event::End(_)) => {
                    path.pop();
                }
                Ok(Event::Text(text)) => {
                    let value = std::str::from_utf8(text.as_ref())
                        .map_err(|error| format!("invalid text in model_conf.xml: {error}"))?;
                    let value = value.trim();
                    let [.., section, key] = path.as_slice() else {
                        continue;
                    };
                    let number = || {
                        value.parse::<f64>().map_err(|error| {
                            format!("invalid number for <{key}> in model_conf.xml: {error}")
                        })
                    };
                    match (section.as_str(), key.as_str()) {
                        ("weights", "mandatory") => config.weights.mandatory = number()?,
                        ("weights", "accessory") => config.weights.accessory = number()?,
                        ("weights", "neutral") => config.weights.neutral = number()?,
                        ("weights", "exchangeable") => config.weights.exchangeable = number()?,
                        ("weights", "redundancy_penalty") => {
                            config.weights.redundancy_penalty = number()?
                        }
                        ("filtering", "e_value_search") => {
                            config.filtering.e_value_search = number()?
                        }
                        ("filtering", "i_evalue_sel") => config.filtering.i_evalue_sel = number()?,
                        ("filtering", "coverage_profile") => {
                            config.filtering.coverage_profile = number()?
                        }
                        ("filtering", "cut_ga") => {
                            config.filtering.cut_ga = match value {
                                "True" | "true" | "1" => true,
                                "False" | "false" | "0" => false,
                                other => {
                                    return Err(format!(
                                        "invalid boolean for <cut_ga> in model_conf.xml: '{other}'"
                                    ));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(error) => return Err(format!("invalid model_conf.xml: {error}")),
            }
        }
        Ok(config)
    }
}

/// Reject models relying on rules this implementation does not apply.
pub fn check_supported(model: &SystemModel) -> Result<(), String> {
    if model.multi_loci {
        return Err(format!(
            "model {} uses multi_loci, which is not supported",
            model.fully_qualified_name
        ));
    }
    for gene in &model.genes {
        if gene.loner || gene.multi_system {
            return Err(format!(
                "gene {} of model {} uses loner or multi_system, which is not supported",
                gene.name, model.fully_qualified_name
            ));
        }
    }
    Ok(())
}

/// Keep, for each protein, only its best-scoring hit over all profiles.
///
/// Ties are broken by profile name so the result does not depend on the
/// order in which profiles were searched.
pub fn best_hit_per_protein(
    all_hits: &HashMap<String, Vec<HmmerHit>>,
) -> HashMap<String, HmmerHit> {
    let mut best: HashMap<String, HmmerHit> = HashMap::new();
    for hit in all_hits.values().flatten() {
        match best.get(&hit.id) {
            Some(current)
                if current.score > hit.score
                    || (current.score == hit.score && current.gene_name <= hit.gene_name) => {}
            _ => {
                best.insert(hit.id.clone(), hit.clone());
            }
        }
    }
    best
}

/// A hit placed in a model, with what clustering and scoring need.
#[derive(Debug, Clone)]
struct ModelHit<'a> {
    hit: &'a HmmerHit,
    /// Rank of the protein on its replicon, starting at 0.
    position: usize,
    /// Gene of the model whose function the hit fulfils.
    function: &'a GeneDefinition,
    is_exchangeable: bool,
}

fn model_hits<'a>(
    model: &'a SystemModel,
    best_hits: &'a HashMap<String, HmmerHit>,
    replicon: &RepliconProteins,
) -> Vec<ModelHit<'a>> {
    let mut profile_to_gene: HashMap<&str, (&GeneDefinition, bool)> = HashMap::new();
    for gene in &model.genes {
        profile_to_gene.entry(&gene.name).or_insert((gene, false));
        for exchangeable in &gene.exchangeables {
            profile_to_gene.entry(exchangeable).or_insert((gene, true));
        }
    }
    replicon
        .protein_ids
        .iter()
        .enumerate()
        .filter_map(|(position, protein_id)| {
            let hit = best_hits.get(protein_id)?;
            let &(function, is_exchangeable) = profile_to_gene.get(hit.gene_name.as_str())?;
            Some(ModelHit {
                hit,
                position,
                function,
                is_exchangeable,
            })
        })
        .collect()
}

/// Whether `second` follows `first` closely enough to share a cluster.
fn colocates(
    first: &ModelHit<'_>,
    second: &ModelHit<'_>,
    model: &SystemModel,
    replicon_length: usize,
    topology: RepliconTopology,
) -> bool {
    let max_space = match (
        first.function.inter_gene_max_space,
        second.function.inter_gene_max_space,
    ) {
        (None, None) => model.inter_gene_max_space,
        (Some(space), None) | (None, Some(space)) => space,
        (Some(left), Some(right)) => left.min(right),
    } as i64;
    let distance = second.position as i64 - first.position as i64 - 1;
    if (0..=max_space).contains(&distance) {
        return true;
    }
    if distance <= 0 && topology == RepliconTopology::Circular && replicon_length > 0 {
        // Number of proteins between `first` and `second` going through the origin.
        let around = replicon_length as i64 - 1 - first.position as i64 + second.position as i64;
        return around <= max_space;
    }
    false
}

/// A scaffold is a cluster when it holds at least two different profiles
/// that are not all neutral, or a single non-neutral profile if the model
/// needs only one gene.
fn is_cluster(scaffold: &[ModelHit<'_>], model: &SystemModel) -> bool {
    let profiles: HashSet<&str> = scaffold
        .iter()
        .map(|hit| hit.hit.gene_name.as_str())
        .collect();
    let all_neutral = scaffold
        .iter()
        .all(|hit| hit.function.status == GeneStatus::Neutral);
    if profiles.len() > 1 {
        !all_neutral
    } else {
        model.min_genes_required == 1 && !all_neutral
    }
}

fn build_clusters<'a>(
    hits: Vec<ModelHit<'a>>,
    model: &SystemModel,
    replicon_length: usize,
    topology: RepliconTopology,
) -> Vec<Vec<ModelHit<'a>>> {
    let Some(first_hit) = hits.first().cloned() else {
        return Vec::new();
    };
    let near = |left: &ModelHit<'_>, right: &ModelHit<'_>| {
        colocates(left, right, model, replicon_length, topology)
    };

    let mut clusters: Vec<Vec<ModelHit<'a>>> = Vec::new();
    let mut scaffold = vec![first_hit.clone()];
    for hit in hits.into_iter().skip(1) {
        if near(scaffold.last().expect("scaffold is never empty"), &hit) {
            scaffold.push(hit);
        } else {
            if is_cluster(&scaffold, model) {
                clusters.push(std::mem::take(&mut scaffold));
            }
            scaffold = vec![hit];
        }
    }

    let circular = topology == RepliconTopology::Circular;
    if is_cluster(&scaffold, model) {
        clusters.push(scaffold);
    } else if circular {
        // The last hits may continue into the first cluster, or into the
        // first hit alone, across the origin.
        let last = scaffold.last().expect("scaffold is never empty").clone();
        if !clusters.is_empty() && near(&last, &clusters[0][0]) {
            scaffold.append(&mut clusters[0]);
            clusters[0] = scaffold;
        } else if near(&last, &first_hit) {
            scaffold.push(first_hit);
            if is_cluster(&scaffold, model) {
                clusters.push(scaffold);
            }
        }
    }
    if circular && clusters.len() > 1 {
        let last_cluster = clusters.last().expect("more than one cluster");
        if near(
            last_cluster.last().expect("clusters are never empty"),
            &clusters[0][0],
        ) {
            let mut merged = clusters.pop().expect("more than one cluster");
            merged.append(&mut clusters[0]);
            clusters[0] = merged;
        }
    }
    clusters
}

/// Check a cluster against the model's quorum and turn it into a system.
fn match_cluster(
    cluster: &[ModelHit<'_>],
    model: &SystemModel,
    weights: &HitWeights,
    replicon: &str,
) -> Option<DetectedSystem> {
    let functions_with = |status: GeneStatus| -> BTreeSet<&str> {
        cluster
            .iter()
            .filter(|hit| hit.function.status == status)
            .map(|hit| hit.function.name.as_str())
            .collect()
    };
    let mandatory = functions_with(GeneStatus::Mandatory);
    let accessory = functions_with(GeneStatus::Accessory);
    let forbidden = functions_with(GeneStatus::Forbidden);
    if !forbidden.is_empty()
        || mandatory.len() < model.min_mandatory_genes_required as usize
        || mandatory.len() + accessory.len() < model.min_genes_required as usize
    {
        return None;
    }

    // Each function counts once, with its best-weighted hit.
    let mut score_by_function: HashMap<&str, f64> = HashMap::new();
    for hit in cluster {
        let status_weight = match hit.function.status {
            GeneStatus::Mandatory => weights.mandatory,
            GeneStatus::Accessory => weights.accessory,
            GeneStatus::Neutral => weights.neutral,
            GeneStatus::Forbidden => 0.0,
        };
        let hit_score = status_weight
            * if hit.is_exchangeable {
                weights.exchangeable
            } else {
                1.0
            };
        let best = score_by_function
            .entry(hit.function.name.as_str())
            .or_insert(hit_score);
        *best = best.max(hit_score);
    }
    // Sum in a fixed order so equal systems get bit-identical scores.
    let mut function_scores: Vec<(&str, f64)> = score_by_function.into_iter().collect();
    function_scores.sort_by(|left, right| left.0.cmp(right.0));
    let score: f64 = function_scores.iter().map(|(_, score)| score).sum();

    let complete_gene_count = model.complete_gene_count();
    let wholeness = if complete_gene_count > 0 {
        (mandatory.len() + accessory.len()) as f64 / complete_gene_count as f64
    } else {
        1.0
    };
    let mandatory_occurrences: Vec<usize> = model
        .mandatory_genes()
        .map(|gene| {
            cluster
                .iter()
                .filter(|hit| std::ptr::eq(hit.function, gene))
                .count()
        })
        .collect();

    Some(DetectedSystem {
        id: String::new(),
        replicon: replicon.to_string(),
        model_fully_qualified_name: model.fully_qualified_name.clone(),
        score,
        wholeness,
        loci_count: 1,
        occurrence: median_occurrence(mandatory_occurrences),
        hits: cluster
            .iter()
            .map(|hit| SystemHit {
                hit: hit.hit.clone(),
                position: hit.position,
                gene_ref: hit.function.name.clone(),
                gene_status: hit.function.status,
                model_fully_qualified_name: model.fully_qualified_name.clone(),
                is_exchangeable: hit.is_exchangeable,
                locus_num: 1,
                counterpart: String::new(),
                used_in: Vec::new(),
            })
            .collect(),
        state: "single_locus".to_string(),
        mandatory_found: mandatory.iter().map(|name| name.to_string()).collect(),
        accessory_found: accessory.iter().map(|name| name.to_string()).collect(),
        forbidden_found: Vec::new(),
    })
}

/// Predicted number of biological systems in one cluster: the rounded
/// median number of hits per mandatory gene, at least one.
fn median_occurrence(mut occurrences: Vec<usize>) -> usize {
    if occurrences.is_empty() {
        return 1;
    }
    occurrences.sort_unstable();
    let middle = occurrences.len() / 2;
    let median = if occurrences.len() % 2 == 1 {
        occurrences[middle] as f64
    } else {
        (occurrences[middle - 1] + occurrences[middle]) as f64 / 2.0
    };
    // Python's round() rounds halves to even.
    (median.round_ties_even() as usize).max(1)
}

/// Find every candidate system of every model on every replicon.
///
/// `models` must be sorted the way results should be ordered for ties;
/// MacSyFinder sorts them by name.
pub fn find_systems(
    models: &[&SystemModel],
    all_hits: &HashMap<String, Vec<HmmerHit>>,
    replicons: &[RepliconProteins],
    topology: RepliconTopology,
    weights: &HitWeights,
) -> Vec<DetectedSystem> {
    let best_hits = best_hit_per_protein(all_hits);
    let mut systems = Vec::new();
    for replicon in replicons {
        for model in models {
            let hits = model_hits(model, &best_hits, replicon);
            for cluster in build_clusters(hits, model, replicon.protein_ids.len(), topology) {
                systems.extend(match_cluster(&cluster, model, weights, &replicon.name));
            }
        }
    }
    systems
}

/// Whether two systems can both be reported: they share no protein, except
/// proteins whose gene is `multi_model` in both models.
fn compatible(
    left: &DetectedSystem,
    right: &DetectedSystem,
    multi_model_genes: &HashSet<(&str, &str)>,
) -> bool {
    if left.replicon != right.replicon {
        return true;
    }
    let same_model = left.model_fully_qualified_name == right.model_fully_qualified_name;
    left.hits.iter().all(|left_hit| {
        right
            .hits
            .iter()
            .filter(|right_hit| right_hit.hit.id == left_hit.hit.id)
            .all(|right_hit| {
                !same_model
                    && multi_model_genes.contains(&(
                        left.model_fully_qualified_name.as_str(),
                        left_hit.gene_ref.as_str(),
                    ))
                    && multi_model_genes.contains(&(
                        right.model_fully_qualified_name.as_str(),
                        right_hit.gene_ref.as_str(),
                    ))
            })
    })
}

/// Ranking of a set of systems: total score, then MacSyFinder's tie-breakers
/// (number of hits, number of systems, wholeness, hit positions).
#[derive(Debug, PartialEq, PartialOrd)]
struct SolutionRank {
    score: f64,
    hit_count: usize,
    system_count: usize,
    wholeness: f64,
    positions: Vec<usize>,
}

fn rank(systems: &[DetectedSystem], members: &[usize]) -> SolutionRank {
    let mut ordered: Vec<&DetectedSystem> = members.iter().map(|&index| &systems[index]).collect();
    ordered.sort_by(|left, right| {
        let positions = |system: &DetectedSystem| -> Vec<usize> {
            system.hits.iter().map(|hit| hit.position).collect()
        };
        positions(left)
            .cmp(&positions(right))
            .then_with(|| {
                left.model_fully_qualified_name
                    .cmp(&right.model_fully_qualified_name)
            })
            .then_with(|| right.score.total_cmp(&left.score))
    });
    SolutionRank {
        score: ordered.iter().map(|system| system.score).sum(),
        hit_count: ordered.iter().map(|system| system.hits.len()).sum(),
        system_count: ordered.len(),
        wholeness: ordered.iter().map(|system| system.wholeness).sum(),
        positions: ordered
            .iter()
            .flat_map(|system| system.hits.iter().map(|hit| hit.position))
            .collect(),
    }
}

/// Enumerate maximal sets of mutually compatible systems (Bron-Kerbosch
/// with pivoting) and keep the best-ranked one.
fn best_clique(
    systems: &[DetectedSystem],
    adjacency: &[Vec<bool>],
    current: &mut Vec<usize>,
    mut candidates: Vec<usize>,
    mut excluded: Vec<usize>,
    best: &mut Option<(SolutionRank, Vec<usize>)>,
) {
    if candidates.is_empty() {
        if excluded.is_empty() {
            let candidate_rank = rank(systems, current);
            if best
                .as_ref()
                .is_none_or(|(best_rank, _)| candidate_rank > *best_rank)
            {
                *best = Some((candidate_rank, current.clone()));
            }
        }
        return;
    }
    let pivot = candidates
        .iter()
        .chain(&excluded)
        .copied()
        .max_by_key(|&vertex| {
            candidates
                .iter()
                .filter(|&&other| adjacency[vertex][other])
                .count()
        })
        .expect("candidates is not empty");
    let branch: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&vertex| !adjacency[pivot][vertex])
        .collect();
    for vertex in branch {
        current.push(vertex);
        best_clique(
            systems,
            adjacency,
            current,
            candidates
                .iter()
                .copied()
                .filter(|&other| adjacency[vertex][other])
                .collect(),
            excluded
                .iter()
                .copied()
                .filter(|&other| adjacency[vertex][other])
                .collect(),
            best,
        );
        current.pop();
        candidates.retain(|&other| other != vertex);
        excluded.push(vertex);
    }
}

/// Choose the non-conflicting systems maximising the total score.
///
/// MacSyFinder enumerates the maximal cliques of the compatibility graph of
/// a whole replicon. Systems only compete with those they conflict with, so
/// each group of directly or indirectly conflicting systems is solved on
/// its own here, which gives the same choice without the combinatorial
/// growth on replicons carrying many systems.
pub fn best_solution(systems: Vec<DetectedSystem>, models: &[&SystemModel]) -> Vec<DetectedSystem> {
    let multi_model_genes: HashSet<(&str, &str)> = models
        .iter()
        .flat_map(|model| {
            model
                .genes
                .iter()
                .filter(|gene| gene.multi_model)
                .map(|gene| (model.fully_qualified_name.as_str(), gene.name.as_str()))
        })
        .collect();
    let count = systems.len();
    let adjacency: Vec<Vec<bool>> = (0..count)
        .map(|left| {
            (0..count)
                .map(|right| {
                    left != right && compatible(&systems[left], &systems[right], &multi_model_genes)
                })
                .collect()
        })
        .collect();

    // Group systems connected through conflicts.
    let mut group_of: Vec<Option<usize>> = vec![None; count];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for start in 0..count {
        if group_of[start].is_some() {
            continue;
        }
        let group_index = groups.len();
        let mut members = Vec::new();
        let mut stack = vec![start];
        group_of[start] = Some(group_index);
        while let Some(vertex) = stack.pop() {
            members.push(vertex);
            for other in 0..count {
                if other != vertex && !adjacency[vertex][other] && group_of[other].is_none() {
                    group_of[other] = Some(group_index);
                    stack.push(other);
                }
            }
        }
        members.sort_unstable();
        groups.push(members);
    }

    let mut selected: Vec<usize> = Vec::new();
    for members in groups {
        let mut best = None;
        best_clique(
            &systems,
            &adjacency,
            &mut Vec::new(),
            members,
            Vec::new(),
            &mut best,
        );
        if let Some((_, chosen)) = best {
            selected.extend(chosen);
        }
    }
    selected.sort_unstable();
    let mut keep = vec![false; count];
    for index in selected {
        keep[index] = true;
    }
    systems
        .into_iter()
        .zip(keep)
        .filter_map(|(system, keep)| keep.then_some(system))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gene(name: &str, status: GeneStatus, exchangeables: &[&str]) -> GeneDefinition {
        GeneDefinition {
            name: name.to_string(),
            status,
            loner: false,
            multi_system: false,
            system_ref: None,
            exchangeables: exchangeables.iter().map(|name| name.to_string()).collect(),
            inter_gene_max_space: None,
            multi_model: false,
        }
    }

    fn model(
        name: &str,
        genes: Vec<GeneDefinition>,
        space: u32,
        mandatory: u32,
        total: u32,
    ) -> SystemModel {
        SystemModel {
            fully_qualified_name: format!("Test/{name}"),
            family: "Test".to_string(),
            name: name.to_string(),
            version: Some("2.0".to_string()),
            genes,
            inter_gene_max_space: space,
            min_mandatory_genes_required: mandatory,
            min_genes_required: total,
            multi_loci: false,
            max_nb_genes: None,
        }
    }

    fn hit(protein: &str, profile: &str, score: f64) -> HmmerHit {
        HmmerHit {
            id: protein.to_string(),
            gene_name: profile.to_string(),
            seq_len: 100,
            i_evalue: 1e-20,
            score,
            profile_coverage: 0.9,
            seq_coverage: 0.9,
            begin_match: 1,
            end_match: 90,
        }
    }

    fn hits(entries: &[(&str, &str, f64)]) -> HashMap<String, Vec<HmmerHit>> {
        let mut map: HashMap<String, Vec<HmmerHit>> = HashMap::new();
        for &(protein, profile, score) in entries {
            map.entry(profile.to_string())
                .or_default()
                .push(hit(protein, profile, score));
        }
        map
    }

    fn replicon(count: usize) -> RepliconProteins {
        RepliconProteins {
            name: "rep".to_string(),
            protein_ids: (0..count).map(|index| format!("p{index}")).collect(),
        }
    }

    fn abc_model() -> SystemModel {
        model(
            "abc",
            vec![
                gene("a", GeneStatus::Mandatory, &["a2"]),
                gene("b", GeneStatus::Mandatory, &[]),
                gene("c", GeneStatus::Accessory, &[]),
                gene("n", GeneStatus::Neutral, &[]),
            ],
            2,
            1,
            2,
        )
    }

    #[test]
    fn model_configuration_overrides_defaults_it_mentions() {
        let config = ModelConfig::parse(
            "<model_config><weights><accessory>1</accessory><exchangeable>1</exchangeable>\
             <redundancy_penalty>1</redundancy_penalty></weights><filtering>\
             <i_evalue_sel>0.01</i_evalue_sel><coverage_profile>0.4</coverage_profile>\
             <cut_ga>True</cut_ga></filtering></model_config>",
        )
        .unwrap();
        assert_eq!(config.weights.accessory, 1.0);
        assert_eq!(config.weights.mandatory, 1.0);
        assert_eq!(config.weights.redundancy_penalty, 1.0);
        assert_eq!(config.filtering.i_evalue_sel, 0.01);
        assert_eq!(config.filtering.e_value_search, 0.1);
        assert!(config.filtering.cut_ga);
        assert!(
            ModelConfig::parse(
                "<model_config><filtering><cut_ga>maybe</cut_ga></filtering></model_config>"
            )
            .is_err()
        );
    }

    #[test]
    fn each_protein_keeps_its_best_scoring_profile() {
        let best = best_hit_per_protein(&hits(&[
            ("p1", "a", 50.0),
            ("p1", "b", 80.0),
            ("p2", "b", 10.0),
            ("p2", "a", 10.0),
        ]));
        assert_eq!(best["p1"].gene_name, "b");
        assert_eq!(best["p2"].gene_name, "a");
    }

    #[test]
    fn cluster_scores_count_each_function_once_with_weights() {
        let abc = abc_model();
        let all_hits = hits(&[
            ("p3", "a2", 90.0),
            ("p4", "b", 90.0),
            ("p5", "b", 70.0),
            ("p6", "c", 90.0),
            ("p7", "n", 90.0),
        ]);
        let systems = find_systems(
            &[&abc],
            &all_hits,
            &[replicon(20)],
            RepliconTopology::Linear,
            &HitWeights::default(),
        );
        assert_eq!(systems.len(), 1);
        // a via exchangeable 0.8 + b 1 + c 0.5 + neutral 0
        assert!((systems[0].score - 2.3).abs() < 1e-12);
        assert_eq!(systems[0].wholeness, 1.0);
        assert_eq!(systems[0].hits.len(), 5);
        assert_eq!(systems[0].hits[0].gene_ref, "a");
        assert!(systems[0].hits[0].is_exchangeable);
    }

    #[test]
    fn distance_quorum_and_neutral_rules_decide_clusters() {
        let abc = abc_model();
        let find = |entries: &[(&str, &str, f64)], topology| {
            find_systems(
                &[&abc],
                &hits(entries),
                &[replicon(20)],
                topology,
                &HitWeights::default(),
            )
        };
        let linear = RepliconTopology::Linear;
        // Two genes at most two proteins apart form a system.
        assert_eq!(find(&[("p1", "a", 9.0), ("p4", "b", 9.0)], linear).len(), 1);
        // Three proteins apart: two singletons, no system.
        assert!(find(&[("p1", "a", 9.0), ("p5", "b", 9.0)], linear).is_empty());
        // The same profile twice is not a cluster.
        assert!(find(&[("p1", "a", 9.0), ("p2", "a", 9.0)], linear).is_empty());
        // An accessory gene and a neutral gene: no mandatory gene, rejected.
        assert!(find(&[("p1", "c", 9.0), ("p2", "n", 9.0)], linear).is_empty());
        // Across the origin only on a circular replicon.
        let across = [("p19", "a", 9.0), ("p1", "b", 9.0)];
        assert!(find(&across, linear).is_empty());
        let wrapped = find(&across, RepliconTopology::Circular);
        assert_eq!(wrapped.len(), 1);
        assert_eq!(
            wrapped[0]
                .hits
                .iter()
                .map(|hit| hit.position)
                .collect::<Vec<_>>(),
            [19, 1]
        );
    }

    #[test]
    fn gene_specific_spacing_overrides_the_model_spacing() {
        let mut abc = abc_model();
        abc.genes[1].inter_gene_max_space = Some(0);
        let systems = find_systems(
            &[&abc],
            &hits(&[("p1", "a", 9.0), ("p3", "b", 9.0)]),
            &[replicon(20)],
            RepliconTopology::Linear,
            &HitWeights::default(),
        );
        assert!(systems.is_empty());
    }

    #[test]
    fn forbidden_genes_reject_a_cluster() {
        let mut abc = abc_model();
        abc.genes.push(gene("f", GeneStatus::Forbidden, &[]));
        let systems = find_systems(
            &[&abc],
            &hits(&[("p1", "a", 9.0), ("p2", "b", 9.0), ("p3", "f", 9.0)]),
            &[replicon(20)],
            RepliconTopology::Linear,
            &HitWeights::default(),
        );
        assert!(systems.is_empty());
    }

    #[test]
    fn best_solution_keeps_the_higher_scoring_of_two_overlapping_systems() {
        let specific = model(
            "specific",
            vec![
                gene("a", GeneStatus::Mandatory, &[]),
                gene("b", GeneStatus::Mandatory, &[]),
                gene("c", GeneStatus::Mandatory, &[]),
            ],
            2,
            1,
            2,
        );
        let general = model(
            "general",
            vec![
                gene("a", GeneStatus::Mandatory, &[]),
                gene("b", GeneStatus::Mandatory, &[]),
            ],
            2,
            1,
            2,
        );
        let models = [&general, &specific];
        let all_hits = hits(&[
            ("p1", "a", 9.0),
            ("p2", "b", 9.0),
            ("p3", "c", 9.0),
            ("p10", "a", 9.0),
            ("p11", "b", 9.0),
        ]);
        let systems = find_systems(
            &models,
            &all_hits,
            &[replicon(20)],
            RepliconTopology::Linear,
            &HitWeights::default(),
        );
        assert_eq!(systems.len(), 4);

        let chosen = best_solution(systems, &models);
        let names: Vec<(&str, usize)> = chosen
            .iter()
            .map(|system| {
                (
                    system.model_fully_qualified_name.as_str(),
                    system.hits[0].position,
                )
            })
            .collect();
        // First locus: the three-gene model wins. Second locus: both models
        // score 2 with the same hits and wholeness decides (2/2 vs 2/3).
        assert_eq!(names, [("Test/general", 10), ("Test/specific", 1)]);
    }

    #[test]
    fn multi_model_genes_may_be_shared_between_models() {
        let mut left = model(
            "left",
            vec![
                gene("a", GeneStatus::Mandatory, &[]),
                gene("shared", GeneStatus::Accessory, &[]),
            ],
            2,
            1,
            2,
        );
        let mut right = model(
            "right",
            vec![
                gene("b", GeneStatus::Mandatory, &[]),
                gene("shared", GeneStatus::Accessory, &[]),
            ],
            2,
            1,
            2,
        );
        let all_hits = hits(&[("p1", "a", 9.0), ("p2", "shared", 9.0), ("p3", "b", 9.0)]);
        let run = |left: &SystemModel, right: &SystemModel| {
            let models = [left, right];
            let systems = find_systems(
                &models,
                &all_hits,
                &[replicon(20)],
                RepliconTopology::Linear,
                &HitWeights::default(),
            );
            best_solution(systems, &models).len()
        };
        assert_eq!(run(&left, &right), 1);
        left.genes[1].multi_model = true;
        right.genes[1].multi_model = true;
        assert_eq!(run(&left, &right), 2);
    }

    #[test]
    fn unsupported_model_features_are_reported() {
        let mut abc = abc_model();
        assert!(check_supported(&abc).is_ok());
        abc.genes[0].loner = true;
        assert!(check_supported(&abc).unwrap_err().contains("loner"));
    }

    #[test]
    fn occurrence_is_the_rounded_median_of_mandatory_gene_counts() {
        assert_eq!(median_occurrence(vec![]), 1);
        assert_eq!(median_occurrence(vec![0, 0, 1]), 1);
        assert_eq!(median_occurrence(vec![2, 2, 1]), 2);
        assert_eq!(median_occurrence(vec![1, 2]), 2);
        assert_eq!(median_occurrence(vec![2, 3]), 2);
    }
}
