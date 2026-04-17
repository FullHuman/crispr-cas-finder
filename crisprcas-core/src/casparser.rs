use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;

/// A Cas-associated gene
#[derive(Serialize)]
pub struct CasGene {
    pub id: String,
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub strand: String,
}

/// A Cas system cluster
#[derive(Serialize)]
pub struct CasCluster {
    pub system: String,
    pub genes: Vec<CasGene>,
    pub start: usize,
    pub end: usize,
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
    let gene_map = faa_content.map(parse_gene_map_from_faa);

    let mut clusters = Vec::new();
    for sys in &results.systems {
        let mut genes = Vec::new();
        let mut min_start = usize::MAX;
        let mut max_end = 0usize;

        for hit in &sys.hits {
            let (start, end, strand) = if let Some(ref gmap) = gene_map {
                gmap.get(hit.hit.id.as_str()).cloned().unwrap_or((
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

            genes.push(CasGene {
                id: hit.hit.id.clone(),
                name: hit.gene_ref.clone(),
                start,
                end,
                strand,
            });
        }

        clusters.push(CasCluster {
            system: sys.model_fqn.clone(),
            genes,
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
fn parse_gene_map_from_faa(
    faa_content: &str,
) -> std::collections::HashMap<String, (usize, usize, String)> {
    let mut gene_map = std::collections::HashMap::new();
    for line in faa_content.lines() {
        if !line.starts_with('>') {
            continue;
        }
        // Format: >id # start # end # strand_int # attrs
        let header = &line[1..]; // strip '>'
        let parts: Vec<&str> = header.split(" # ").collect();
        if parts.len() < 4 {
            continue;
        }
        let id = parts[0].trim().to_string();
        let start = parts[1].trim().parse::<usize>().unwrap_or(0);
        let end = parts[2].trim().parse::<usize>().unwrap_or(0);
        let strand = match parts[3].trim() {
            "-1" => "-".to_string(),
            "1" => "+".to_string(),
            s => s.to_string(),
        };
        gene_map.insert(id, (start, end, strand));
    }
    gene_map
}

fn parse_gene_map_from_gff(
    gff_content: &str,
) -> std::collections::HashMap<String, (usize, usize, String)> {
    let mut gene_map = std::collections::HashMap::new();
    for line in gff_content.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 9 {
            continue;
        }
        let feature_type = cols[2];
        if feature_type != "CDS" && feature_type != "gene" {
            continue;
        }
        let start = cols[3].parse::<usize>().unwrap_or(0);
        let end = cols[4].parse::<usize>().unwrap_or(0);
        let strand = cols[6].to_string();
        let attrs = cols[8];
        for attr in attrs.split(';') {
            if let Some(value) = attr.strip_prefix("ID=") {
                gene_map.insert(value.to_string(), (start, end, strand.clone()));
                break;
            }
        }
    }
    gene_map
}

pub fn parse_cas_from_strings(best_solution_tsv: &str, ann_gff: &str) -> Result<Vec<CasCluster>> {
    let gene_map = parse_gene_map_from_gff(ann_gff);
    let mut clusters: std::collections::HashMap<String, Vec<CasGene>> =
        std::collections::HashMap::new();

    for line in best_solution_tsv.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with("replicon\t") || line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 6 {
            continue;
        }
        let gene_id = cols[1];
        let gene_name = cols[2].to_string();
        let system_label = cols[5].to_string();
        if let Some(&(start, end, ref strand)) = gene_map.get(gene_id) {
            let gene = CasGene {
                id: gene_id.to_string(),
                name: gene_name,
                start,
                end,
                strand: strand.clone(),
            };
            clusters.entry(system_label).or_default().push(gene);
        }
    }

    let mut result = Vec::new();
    for (system, genes) in clusters {
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
