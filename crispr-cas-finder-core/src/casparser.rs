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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CasGene {
    pub id: String,
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub strand: String,
}

/// A Cas system cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CasCluster {
    pub system: String,
    pub genes: Vec<CasGene>,
    pub start: usize,
    pub end: usize,
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
    let mut clusters = Vec::new();
    for detected_system in &results.systems {
        let mut cluster_genes = Vec::new();
        let mut min_start = usize::MAX;
        let mut max_end = 0usize;

        for hit in &detected_system.hits {
            let (start, end, strand) = if let Some(coordinate_map) = gene_map {
                coordinate_map
                    .get(hit.hit.id.as_str())
                    .cloned()
                    .map(|coords| (coords.start, coords.end, coords.strand))
                    .unwrap_or((
                        hit.hit.begin_match as usize,
                        hit.hit.end_match as usize,
                        ".".to_string(),
                    ))
            } else {
                (
                    hit.hit.begin_match as usize,
                    hit.hit.end_match as usize,
                    ".".to_string(),
                )
            };

            if start < min_start {
                min_start = start;
            }
            if end > max_end {
                max_end = end;
            }

            cluster_genes.push(CasGene {
                id: hit.hit.id.clone(),
                name: hit.gene_ref.clone(),
                start,
                end,
                strand,
            });
        }

        clusters.push(CasCluster {
            system: detected_system.model_fully_qualified_name.clone(),
            genes: cluster_genes,
            start: if min_start == usize::MAX {
                0
            } else {
                min_start
            },
            end: max_end,
        });
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
