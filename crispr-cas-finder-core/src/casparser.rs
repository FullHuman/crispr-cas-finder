use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneCoordinates {
    pub start: usize,
    pub end: usize,
    pub strand: String,
}

/// A Cas-associated gene
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CasGene {
    /// Protein identifier.
    pub id: String,
    /// Gene of the model the protein was assigned to.
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub strand: String,
    /// HMM profile that matched (differs from `name` for exchangeable genes).
    #[serde(default)]
    pub profile: String,
    /// Role of the gene in the model: `mandatory`, `accessory`, `neutral`
    /// or `forbidden`.
    #[serde(default)]
    pub status: String,
    /// Bit score of the best domain.
    #[serde(default)]
    pub score: f64,
    /// Independent E-value of the best domain.
    #[serde(default)]
    pub evalue: f64,
    /// Fraction of the HMM profile covered by the domain.
    #[serde(default)]
    pub profile_coverage: f64,
}

/// A Cas system cluster
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CasCluster {
    /// Stable identifier, `<seq_id>_Cas_<n>` with `n` counted per sequence.
    #[serde(default)]
    pub id: String,
    /// Identifier of the sequence carrying the system.
    #[serde(default)]
    pub seq_id: String,
    /// Fully qualified model name, e.g. `CASFinder/CAS-TypeIE`.
    pub system: String,
    pub genes: Vec<CasGene>,
    pub start: usize,
    pub end: usize,
    /// Sum of the gene bit scores.
    #[serde(default)]
    pub score: f64,
    /// Fraction of the model's mandatory and accessory genes that were found.
    #[serde(default)]
    pub wholeness: f64,
}

pub fn from_search_results_with_gene_map(
    results: &crate::cas_types::SearchResults,
    gene_map: &std::collections::HashMap<String, GeneCoordinates>,
) -> Vec<CasCluster> {
    clusters_from_search_results(results, Some(gene_map))
}

/// Convert search results into CasCluster format.
///
/// `faa_content` contains the proteome FASTA whose headers encode genomic
/// coordinates in Orphos/Prodigal format:
///   >seqid_N # start # end # strand # ID=N
pub fn from_search_results(
    results: &crate::cas_types::SearchResults,
    faa_content: Option<&str>,
) -> Vec<CasCluster> {
    let gene_coordinates_by_id = faa_content.map(parse_gene_coordinates_from_faa);
    clusters_from_search_results(results, gene_coordinates_by_id.as_ref())
}

fn clusters_from_search_results(
    results: &crate::cas_types::SearchResults,
    gene_map: Option<&std::collections::HashMap<String, GeneCoordinates>>,
) -> Vec<CasCluster> {
    use crate::cas_types::GeneStatus;

    let mut clusters = Vec::new();
    for detected_system in &results.systems {
        let genes: Vec<CasGene> = detected_system
            .hits
            .iter()
            .map(|hit| {
                let (start, end, strand) = gene_map
                    .and_then(|coordinates| coordinates.get(hit.hit.id.as_str()))
                    .map(|coordinates| {
                        (
                            coordinates.start,
                            coordinates.end,
                            coordinates.strand.clone(),
                        )
                    })
                    .unwrap_or((
                        hit.hit.begin_match as usize,
                        hit.hit.end_match as usize,
                        ".".to_string(),
                    ));
                CasGene {
                    id: hit.hit.id.clone(),
                    name: hit.gene_ref.clone(),
                    start,
                    end,
                    strand,
                    profile: hit.hit.gene_name.clone(),
                    status: match hit.gene_status {
                        GeneStatus::Mandatory => "mandatory",
                        GeneStatus::Accessory => "accessory",
                        GeneStatus::Forbidden => "forbidden",
                        GeneStatus::Neutral => "neutral",
                    }
                    .to_string(),
                    score: hit.hit.score,
                    evalue: hit.hit.i_evalue,
                    profile_coverage: hit.hit.profile_coverage,
                }
            })
            .collect();
        clusters.push(CasCluster {
            id: String::new(),
            seq_id: detected_system.replicon.clone(),
            system: detected_system.model_fully_qualified_name.clone(),
            start: genes.iter().map(|gene| gene.start).min().unwrap_or(0),
            end: genes.iter().map(|gene| gene.end).max().unwrap_or(0),
            genes,
            score: detected_system.score,
            wholeness: detected_system.wholeness,
        });
    }

    let mut per_sequence: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for cluster in &mut clusters {
        let count = per_sequence.entry(cluster.seq_id.clone()).or_default();
        *count += 1;
        cluster.id = format!("{}_Cas_{count}", cluster.seq_id);
    }
    clusters
}

/// Parse gene coordinates from FASTA headers in Orphos/Prodigal format:
///   >seqid_N # start # end # strand_int # ID=N
fn parse_gene_coordinates_from_faa(
    faa_content: &str,
) -> std::collections::HashMap<String, GeneCoordinates> {
    let mut gene_coordinates_by_id = std::collections::HashMap::new();
    for line in faa_content.lines() {
        if !line.starts_with('>') {
            continue;
        }
        // Format: >id # start # end # strand_int # attrs
        let header = &line[1..]; // strip '>'
        let header_parts: Vec<&str> = header.split(" # ").collect();
        if header_parts.len() < 4 {
            continue;
        }
        let id = header_parts[0].trim().to_string();
        let start = header_parts[1].trim().parse::<usize>().unwrap_or(0);
        let end = header_parts[2].trim().parse::<usize>().unwrap_or(0);
        let strand = match header_parts[3].trim() {
            "-1" => "-".to_string(),
            "1" => "+".to_string(),
            s => s.to_string(),
        };
        gene_coordinates_by_id.insert(id, GeneCoordinates { start, end, strand });
    }
    gene_coordinates_by_id
}

fn parse_gene_coordinates_from_gff(
    gff_content: &str,
) -> std::collections::HashMap<String, GeneCoordinates> {
    let mut gene_coordinates_by_id = std::collections::HashMap::new();
    for line in gff_content.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let columns: Vec<&str> = line.split('\t').collect();
        if columns.len() < 9 {
            continue;
        }
        let feature_type = columns[2];
        if feature_type != "CDS" && feature_type != "gene" {
            continue;
        }
        let start = columns[3].parse::<usize>().unwrap_or(0);
        let end = columns[4].parse::<usize>().unwrap_or(0);
        let strand = columns[6].to_string();
        let attributes = columns[8];
        for attribute in attributes.split(';') {
            if let Some(value) = attribute.strip_prefix("ID=") {
                gene_coordinates_by_id.insert(
                    value.to_string(),
                    GeneCoordinates {
                        start,
                        end,
                        strand: strand.clone(),
                    },
                );
                break;
            }
        }
    }
    gene_coordinates_by_id
}

pub fn parse_cas_from_strings(best_solution_tsv: &str, ann_gff: &str) -> Result<Vec<CasCluster>> {
    let gene_coordinates_by_id = parse_gene_coordinates_from_gff(ann_gff);
    let mut genes_by_system_label: std::collections::HashMap<String, Vec<CasGene>> =
        std::collections::HashMap::new();

    for line in best_solution_tsv.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with("replicon\t") || line.is_empty() {
            continue;
        }
        let columns: Vec<&str> = line.split('\t').collect();
        if columns.len() < 6 {
            continue;
        }
        let gene_id = columns[1];
        let gene_name = columns[2].to_string();
        let system_label = columns[5].to_string();
        if let Some(coords) = gene_coordinates_by_id.get(gene_id) {
            let gene = CasGene {
                id: gene_id.to_string(),
                name: gene_name,
                start: coords.start,
                end: coords.end,
                strand: coords.strand.clone(),
                ..CasGene::default()
            };
            genes_by_system_label
                .entry(system_label)
                .or_default()
                .push(gene);
        }
    }

    let mut result = Vec::new();
    for (system, genes) in genes_by_system_label {
        if genes.is_empty() {
            continue;
        }
        let mut min_start = usize::MAX;
        let mut max_end = 0;
        for g in &genes {
            if g.start < min_start {
                min_start = g.start;
            }
            if g.end > max_end {
                max_end = g.end;
            }
        }
        result.push(CasCluster {
            system,
            genes,
            start: min_start,
            end: max_end,
            ..CasCluster::default()
        });
    }
    Ok(result)
}

/// Parse the MacSyFinder best_solution.tsv and the annotation GFF to extract Cas clusters
pub fn parse_cas_tsv(cas_dir: &Path, ann_gff: &Path) -> Result<Vec<CasCluster>> {
    let tsv_path = cas_dir.join("best_solution.tsv");
    let tsv_content = std::fs::read_to_string(&tsv_path)
        .with_context(|| format!("Failed to read TSV: {:?}", tsv_path))?;
    let gff_content = std::fs::read_to_string(ann_gff)
        .with_context(|| format!("Failed to read annotation GFF: {:?}", ann_gff))?;
    parse_cas_from_strings(&tsv_content, &gff_content)
}
