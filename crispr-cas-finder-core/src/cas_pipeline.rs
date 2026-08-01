use crate::cas_types::{
    Cluster, DetectedSystem, GeneDefinition, GeneStatus, HmmerHit, ModelRegistry, SequenceIndex,
    SystemHit, SystemModel,
};
use crate::casparser::GeneCoordinates;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// In-memory CAS model definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub name: String,
    pub family: String,
    /// XML content of the model definition.
    pub content: String,
}

/// A translated gene record used as an in-memory input to the Cas pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneRecord {
    pub id: String,
    pub protein: String,
    pub start: usize,
    pub end: usize,
    pub strand: String,
}

impl GeneRecord {
    pub fn coordinates(&self) -> GeneCoordinates {
        GeneCoordinates {
            start: self.start,
            end: self.end,
            strand: self.strand.clone(),
        }
    }
}

pub fn gene_coordinates_from_records(records: &[GeneRecord]) -> HashMap<String, GeneCoordinates> {
    records
        .iter()
        .map(|record| (record.id.clone(), record.coordinates()))
        .collect()
}

pub fn build_faa_content(genes: &[GeneRecord]) -> String {
    let mut fasta_amino_acid_content = String::new();
    for gene in genes {
        let strand_numeric_value: i32 = if gene.strand == "-" { -1 } else { 1 };
        fasta_amino_acid_content.push_str(&format!(
            ">{id} # {start} # {end} # {strand_int} # ID={id}\n",
            id = gene.id,
            start = gene.start,
            end = gene.end,
            strand_int = strand_numeric_value,
        ));
        for chunk in gene.protein.as_bytes().chunks(60) {
            fasta_amino_acid_content.push_str(std::str::from_utf8(chunk).unwrap_or(""));
            fasta_amino_acid_content.push('\n');
        }
    }
    fasta_amino_acid_content
}

/// Build a model registry from in-memory XML definitions.
pub fn build_model_registry(models: &[ModelDefinition]) -> Result<ModelRegistry, String> {
    let mut registry = ModelRegistry::new();
    for model_def in models {
        let model = parse_cas_model_xml(&model_def.content, &model_def.name, &model_def.family)?;
        registry.add(model);
    }
    Ok(registry)
}

/// Parse a CAS model definition from XML content.
/// Handles the CRISPRCasFinder format which uses `<system>` tags instead of `<model>`.
pub fn parse_cas_model_xml(
    content: &str,
    model_name: &str,
    family: &str,
) -> Result<SystemModel, String> {
    let converted = convert_cas_xml(content);

    let fully_qualified_model_name = format!("{}/{}", family, model_name);
    let mut inter_gene_max_space: u32 = 20;
    let mut min_mandatory_genes_required: u32 = 0;
    let mut min_genes_required: u32 = 0;
    let mut multi_loci = false;
    let mut genes: Vec<GeneDefinition> = Vec::new();

    for line in converted.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("<model ") || trimmed.starts_with("<model>") {
            if let Some(val) = extract_xml_attr(trimmed, "inter_gene_max_space") {
                inter_gene_max_space = val.parse().unwrap_or(20);
            }
            if let Some(val) = extract_xml_attr(trimmed, "min_mandatory_genes_required") {
                min_mandatory_genes_required = val.parse().unwrap_or(0);
            }
            if let Some(val) = extract_xml_attr(trimmed, "min_genes_required") {
                min_genes_required = val.parse().unwrap_or(0);
            }
            if let Some(val) = extract_xml_attr(trimmed, "multi_loci") {
                multi_loci = val == "True" || val == "true" || val == "1";
            }
        } else if trimmed.starts_with("<gene ") {
            let name = extract_xml_attr(trimmed, "name").unwrap_or_default();
            let presence = extract_xml_attr(trimmed, "presence").unwrap_or_default();
            let status = match presence.as_str() {
                "mandatory" => GeneStatus::Mandatory,
                "accessory" => GeneStatus::Accessory,
                "forbidden" => GeneStatus::Forbidden,
                _ => GeneStatus::Mandatory,
            };
            let loner = extract_xml_attr(trimmed, "loner")
                .is_some_and(|v| v == "True" || v == "true" || v == "1");
            let multi_system = extract_xml_attr(trimmed, "multi_system")
                .is_some_and(|v| v == "True" || v == "true" || v == "1");
            let system_ref = extract_xml_attr(trimmed, "system_ref");

            genes.push(GeneDefinition {
                name,
                status,
                loner,
                multi_system,
                system_ref,
                exchangeables: Vec::new(),
                inter_gene_max_space: None,
                multi_model: false,
            });
        }
    }

    Ok(SystemModel {
        fully_qualified_name: fully_qualified_model_name,
        family: family.to_string(),
        name: model_name.to_string(),
        version: None,
        genes,
        inter_gene_max_space,
        min_mandatory_genes_required,
        min_genes_required,
        multi_loci,
    })
}

pub fn assign_hits_to_model(
    model: &SystemModel,
    hmmer_hits: &HashMap<String, Vec<HmmerHit>>,
    seq_index: &SequenceIndex,
) -> Vec<SystemHit> {
    let mut best_by_protein: HashMap<&str, (SystemHit, bool)> = HashMap::new();

    for gene in &model.genes {
        let gene_names_to_check: Vec<(&str, bool)> = std::iter::once((gene.name.as_str(), false))
            .chain(gene.exchangeables.iter().map(|e| (e.as_str(), true)))
            .collect();

        for (gene_name, is_exchangeable) in gene_names_to_check {
            if let Some(hits) = hmmer_hits.get(gene_name) {
                for hit in hits {
                    if let Some(position) = seq_index.position(&hit.id) {
                        let candidate = SystemHit {
                            hit: hit.clone(),
                            position,
                            gene_ref: gene.name.clone(),
                            gene_status: gene.status,
                            model_fully_qualified_name: model.fully_qualified_name.clone(),
                            is_exchangeable,
                            locus_num: 1,
                            counterpart: String::new(),
                            used_in: Vec::new(),
                        };
                        let inherited = gene.system_ref.is_some();

                        match best_by_protein.get_mut(hit.id.as_str()) {
                            Some((best_hit, best_inherited)) => {
                                if should_replace_assignment(
                                    &candidate,
                                    inherited,
                                    best_hit,
                                    *best_inherited,
                                ) {
                                    *best_hit = candidate;
                                    *best_inherited = inherited;
                                }
                            }
                            None => {
                                best_by_protein.insert(hit.id.as_str(), (candidate, inherited));
                            }
                        }
                    }
                }
            }
        }
    }

    best_by_protein.into_values().map(|(hit, _)| hit).collect()
}

pub fn evaluate_cluster(cluster: &Cluster, model: &SystemModel) -> Option<DetectedSystem> {
    let mandatory_names: HashSet<&str> = model.mandatory_genes().map(|g| g.name.as_str()).collect();
    let accessory_names: HashSet<&str> = model.accessory_genes().map(|g| g.name.as_str()).collect();
    let forbidden_names: HashSet<&str> = model.forbidden_genes().map(|g| g.name.as_str()).collect();

    let found_genes: HashSet<&str> = cluster.hits.iter().map(|h| h.gene_ref.as_str()).collect();

    let mandatory_found: Vec<String> = mandatory_names
        .iter()
        .filter(|g| found_genes.contains(*g))
        .map(|g| g.to_string())
        .collect();
    let accessory_found: Vec<String> = accessory_names
        .iter()
        .filter(|g| found_genes.contains(*g))
        .map(|g| g.to_string())
        .collect();
    let forbidden_found: Vec<String> = forbidden_names
        .iter()
        .filter(|g| found_genes.contains(*g))
        .map(|g| g.to_string())
        .collect();

    if !forbidden_found.is_empty() {
        return None;
    }

    let min_mandatory = model.min_mandatory_genes_required as usize;
    let min_genes = model.min_genes_required as usize;
    let total_found = mandatory_found.len() + accessory_found.len();

    if mandatory_found.len() < min_mandatory || total_found < min_genes {
        return None;
    }

    let score: f64 = cluster.hits.iter().map(|h| h.hit.score).sum();
    let max_nb_genes = model.mandatory_count() + model.accessory_count();
    let wholeness = if max_nb_genes > 0 {
        total_found as f64 / max_nb_genes as f64
    } else {
        1.0
    };

    Some(DetectedSystem {
        id: String::new(),
        replicon: String::new(),
        model_fully_qualified_name: model.fully_qualified_name.clone(),
        score,
        wholeness,
        loci_count: 1,
        occurrence: 0,
        hits: cluster.hits.clone(),
        state: "single_locus".to_string(),
        mandatory_found,
        accessory_found,
        forbidden_found,
    })
}

pub fn reverse_complement_dna(seq: &[u8]) -> Vec<u8> {
    crate::dna::reverse_complement(seq)
}

pub fn translate_dna(seq: &[u8], table: &HashMap<[u8; 3], u8>) -> String {
    let mut protein = String::with_capacity(seq.len() / 3);
    for codon in seq.chunks(3) {
        if codon.len() < 3 {
            break;
        }
        let key = [
            codon[0].to_ascii_uppercase(),
            codon[1].to_ascii_uppercase(),
            codon[2].to_ascii_uppercase(),
        ];
        let aa = table.get(&key).copied().unwrap_or(b'X');
        if aa == b'*' {
            break;
        }
        protein.push(aa as char);
    }
    protein
}

pub fn codon_table(genetic_code: usize) -> HashMap<[u8; 3], u8> {
    let codons = b"TTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGG";
    let second = b"TTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGG";
    let third = b"TCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAG";
    let standard = b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";
    let table11 = b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";

    let aa_table: &[u8] = match genetic_code {
        11 => table11,
        _ => standard,
    };

    let mut map = HashMap::new();
    for i in 0..64 {
        map.insert([codons[i], second[i], third[i]], aa_table[i]);
    }
    map
}

fn convert_cas_xml(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        let converted = line
            .replace("<system ", "<model ")
            .replace("<system>", "<model>")
            .replace("</system>", "</model>");
        out.push_str(&converted);
        out.push('\n');
    }
    out
}

fn extract_xml_attr(tag: &str, attr_name: &str) -> Option<String> {
    let pattern = format!("{}=\"", attr_name);
    if let Some(start) = tag.find(&pattern) {
        let value_start = start + pattern.len();
        if let Some(end) = tag[value_start..].find('"') {
            return Some(tag[value_start..value_start + end].to_string());
        }
    }
    None
}

fn should_replace_assignment(
    candidate: &SystemHit,
    candidate_inherited: bool,
    current: &SystemHit,
    current_inherited: bool,
) -> bool {
    if candidate_inherited != current_inherited {
        return !candidate_inherited;
    }

    candidate
        .hit
        .score
        .total_cmp(&current.hit.score)
        .then_with(|| {
            candidate
                .hit
                .profile_coverage
                .total_cmp(&current.hit.profile_coverage)
        })
        .then_with(|| {
            candidate
                .hit
                .seq_coverage
                .total_cmp(&current.hit.seq_coverage)
        })
        .then_with(|| current.is_exchangeable.cmp(&candidate.is_exchangeable))
        .is_gt()
}
