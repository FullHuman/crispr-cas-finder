//! Protein-coding gene annotation used as input to Cas detection.
//!
//! A [`Proteome`] is either predicted with Orphos (the Rust port of
//! Prodigal) or built from a user-supplied GFF3 annotation and protein FASTA.

use crate::cas_pipeline::{
    GeneRecord, codon_table, reverse_complement_dna, translate_dna, validate_genetic_code,
};
use anyhow::{Context, Result, bail};
use bio::bio_types::strand::Strand;
use orphos_core::{OrphosAnalyzer, config::OrphosConfig};
use std::collections::{HashMap, HashSet};

/// Sequences shorter than this are annotated in metagenomic mode, as in
/// CRISPRCasFinder: they are too short to train a gene model on.
pub const AUTO_METAGENOME_BELOW: usize = 100_000;

/// Protein IDs in genomic order for one input replicon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepliconProteins {
    pub name: String,
    pub protein_ids: Vec<String>,
}

/// Proteins of all annotated sequences, in input and genomic order.
#[derive(Debug, Clone, Default)]
pub struct Proteome {
    pub genes: Vec<GeneRecord>,
    pub replicons: Vec<RepliconProteins>,
}

impl Proteome {
    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }
}

/// How gene prediction treats each sequence.
#[derive(Debug, Clone, Copy)]
pub struct GenePredictionOptions {
    pub genetic_code: usize,
    /// Force metagenomic mode for every sequence.
    pub metagenome: bool,
    /// Use metagenomic mode for sequences shorter than this; `0` disables it.
    pub auto_metagenome_below: usize,
}

impl Default for GenePredictionOptions {
    fn default() -> Self {
        Self {
            genetic_code: 11,
            metagenome: false,
            auto_metagenome_below: AUTO_METAGENOME_BELOW,
        }
    }
}

impl GenePredictionOptions {
    /// Whether a sequence of `length` bp is annotated in metagenomic mode.
    pub fn uses_metagenome_mode(&self, length: usize) -> bool {
        self.metagenome || length < self.auto_metagenome_below
    }
}

/// Predict and translate protein-coding genes for each sequence separately.
///
/// Every sequence gets its own gene model, as upstream runs Prodigal once
/// per sequence. `include` restricts annotation to the listed sequence IDs.
pub fn predict_proteome(
    sequences: &[(String, Vec<u8>)],
    options: &GenePredictionOptions,
    include: Option<&HashSet<String>>,
) -> Result<Proteome> {
    let translation_table_id = validate_genetic_code(options.genetic_code)?;
    let translation_table = codon_table(options.genetic_code)?;
    let analyzer_for = |metagenomic: bool| {
        OrphosAnalyzer::new(OrphosConfig {
            metagenomic,
            closed_ends: true,
            quiet: true,
            translation_table: Some(translation_table_id),
            ..OrphosConfig::default()
        })
    };
    let single_analyzer = analyzer_for(false);
    let metagenome_analyzer = analyzer_for(true);

    let mut proteome = Proteome::default();
    for (sequence_id, sequence) in sequences {
        if include.is_some_and(|ids| !ids.contains(sequence_id)) {
            continue;
        }
        let analyzer = if options.uses_metagenome_mode(sequence.len()) {
            &metagenome_analyzer
        } else {
            &single_analyzer
        };
        let results = analyzer
            .analyze_sequence_bytes(sequence, sequence_id.clone(), None)
            .map_err(|error| {
                anyhow::anyhow!("Gene prediction failed for '{sequence_id}': {error}")
            })?;

        let mut protein_ids = Vec::with_capacity(results.genes.len());
        for gene in &results.genes {
            let (begin, end) = (gene.coordinates.begin, gene.coordinates.end);
            if begin == 0 || begin > end || end > sequence.len() {
                bail!(
                    "Gene prediction returned invalid coordinates {begin}..{end} for \
                     '{sequence_id}' (length {})",
                    sequence.len()
                );
            }
            let coding_sequence = &sequence[begin - 1..end];
            let (coding_sequence, strand) = match gene.coordinates.strand {
                Strand::Forward => (coding_sequence.to_vec(), "+"),
                Strand::Reverse => (reverse_complement_dna(coding_sequence), "-"),
                Strand::Unknown => {
                    bail!("Gene prediction returned an unknown strand for '{sequence_id}'")
                }
            };
            let mut protein = translate_dna(&coding_sequence, &translation_table);
            if protein.is_empty() {
                continue;
            }
            // Genes are predicted with closed ends, so each one begins at a
            // start codon, which is read as methionine whatever its bases.
            protein.replace_range(..1, "M");
            let protein_id = format!("{sequence_id}_{}", proteome.genes.len() + 1);
            protein_ids.push(protein_id.clone());
            proteome.genes.push(GeneRecord {
                id: protein_id,
                seq_id: sequence_id.clone(),
                protein,
                start: begin,
                end,
                strand: strand.to_string(),
            });
        }
        proteome.replicons.push(RepliconProteins {
            name: sequence_id.clone(),
            protein_ids,
        });
    }
    Ok(proteome)
}

/// Build a proteome from a GFF3 annotation and its protein FASTA.
///
/// A protein is matched to a CDS feature through the feature's `ID`,
/// `protein_id`, `locus_tag` or `Name` attribute. Prodigal-style GFF IDs
/// (`<contig number>_<gene number>`) are also matched to FASTA IDs of the
/// form `<sequence id>_<gene number>`. Proteins without a CDS feature are
/// ignored; it is an error if none can be placed.
pub fn proteome_from_annotation(gff_content: &str, faa_content: &str) -> Result<Proteome> {
    let mut proteins: HashMap<String, String> = HashMap::new();
    let reader = bio::io::fasta::Reader::new(faa_content.as_bytes());
    for record in reader.records() {
        let record = record.context("Reading protein FASTA")?;
        let protein = String::from_utf8(record.seq().to_vec())
            .context("Protein FASTA contains non-UTF-8 residues")?
            .trim_end_matches('*')
            .to_ascii_uppercase();
        if protein.is_empty() {
            continue;
        }
        if proteins.insert(record.id().to_string(), protein).is_some() {
            bail!("Duplicate protein identifier in FASTA: {}", record.id());
        }
    }
    if proteins.is_empty() {
        bail!("Protein FASTA contains no sequences");
    }

    let mut sequence_order: Vec<String> = Vec::new();
    let mut genes_by_sequence: HashMap<String, Vec<GeneRecord>> = HashMap::new();
    let mut placed: HashSet<String> = HashSet::new();
    for line in gff_content.lines() {
        if line.starts_with("##FASTA") {
            break;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let columns: Vec<&str> = line.split('\t').collect();
        if columns.len() < 9 || columns[2] != "CDS" {
            continue;
        }
        let sequence_id = columns[0];
        let (Ok(start), Ok(end)) = (columns[3].parse::<usize>(), columns[4].parse::<usize>())
        else {
            bail!("Invalid CDS coordinates in GFF line: {line}");
        };
        let Some(protein_id) = matching_protein_id(sequence_id, columns[8], &proteins) else {
            continue;
        };
        if !placed.insert(protein_id.clone()) {
            // Later rows of a multi-part CDS extend the first one.
            if let Some(gene) = genes_by_sequence
                .get_mut(sequence_id)
                .and_then(|genes| genes.iter_mut().find(|gene| gene.id == protein_id))
            {
                gene.start = gene.start.min(start);
                gene.end = gene.end.max(end);
            }
            continue;
        }
        if !genes_by_sequence.contains_key(sequence_id) {
            sequence_order.push(sequence_id.to_string());
        }
        genes_by_sequence
            .entry(sequence_id.to_string())
            .or_default()
            .push(GeneRecord {
                protein: proteins[&protein_id].clone(),
                id: protein_id,
                seq_id: sequence_id.to_string(),
                start,
                end,
                strand: if columns[6] == "-" { "-" } else { "+" }.to_string(),
            });
    }
    if placed.is_empty() {
        bail!(
            "No protein in the FASTA matches a CDS feature in the GFF \
             (compared ID, protein_id, locus_tag and Name attributes)"
        );
    }
    if placed.len() < proteins.len() {
        log::warn!(
            "{} of {} proteins have no CDS feature in the GFF and are ignored",
            proteins.len() - placed.len(),
            proteins.len()
        );
    }

    let mut proteome = Proteome::default();
    for sequence_id in sequence_order {
        let mut genes = genes_by_sequence.remove(&sequence_id).unwrap_or_default();
        genes.sort_by(|left, right| {
            (left.start, left.end, &left.id).cmp(&(right.start, right.end, &right.id))
        });
        proteome.replicons.push(RepliconProteins {
            name: sequence_id,
            protein_ids: genes.iter().map(|gene| gene.id.clone()).collect(),
        });
        proteome.genes.extend(genes);
    }
    Ok(proteome)
}

fn matching_protein_id(
    sequence_id: &str,
    attributes: &str,
    proteins: &HashMap<String, String>,
) -> Option<String> {
    let attribute = |key: &str| {
        attributes.split(';').find_map(|attribute| {
            let (name, value) = attribute.trim().split_once('=')?;
            (name == key).then_some(value)
        })
    };
    for key in ["ID", "protein_id", "locus_tag", "Name"] {
        let Some(value) = attribute(key) else {
            continue;
        };
        if proteins.contains_key(value) {
            return Some(value.to_string());
        }
        // NCBI-style `ID=cds-WP_000000001.1`.
        if let Some(stripped) = value.strip_prefix("cds-")
            && proteins.contains_key(stripped)
        {
            return Some(stripped.to_string());
        }
    }
    // Prodigal: `ID=<contig number>_<gene number>` vs `><sequence id>_<gene number>`.
    let (contig_number, gene_number) = attribute("ID")?.split_once('_')?;
    if contig_number.bytes().all(|byte| byte.is_ascii_digit())
        && !gene_number.is_empty()
        && gene_number.bytes().all(|byte| byte.is_ascii_digit())
    {
        let candidate = format!("{sequence_id}_{gene_number}");
        if proteins.contains_key(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// Write predicted genes as GFF3 `CDS` features.
pub fn gff_content(proteome: &Proteome) -> String {
    let mut content = String::from("##gff-version 3\n");
    for gene in &proteome.genes {
        content.push_str(&format!(
            "{}\tcrispr-cas-finder\tCDS\t{}\t{}\t.\t{}\t0\tID={}\n",
            gene.seq_id, gene.start, gene.end, gene.strand, gene.id
        ));
    }
    content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_sequences_switch_to_metagenome_mode() {
        let options = GenePredictionOptions::default();
        assert!(options.uses_metagenome_mode(99_999));
        assert!(!options.uses_metagenome_mode(100_000));
        let forced = GenePredictionOptions {
            metagenome: true,
            ..options
        };
        assert!(forced.uses_metagenome_mode(5_000_000));
        let disabled = GenePredictionOptions {
            auto_metagenome_below: 0,
            ..options
        };
        assert!(!disabled.uses_metagenome_mode(10));
    }

    #[test]
    fn annotation_matches_ids_and_orders_genes_by_position() {
        let gff = "##gff-version 3\n\
            chr1\tsrc\tgene\t1\t90\t.\t+\t.\tID=gene1\n\
            chr1\tsrc\tCDS\t200\t290\t.\t-\t0\tID=cds-WP_2.1;Parent=gene2\n\
            chr1\tsrc\tCDS\t1\t90\t.\t+\t0\tID=x;locus_tag=b0001\n\
            chr2\tsrc\tCDS\t10\t40\t.\t+\t0\tID=unknown\n";
        let faa = ">b0001 first\nMKT*\n>WP_2.1\nMAA\n>orphan\nMCC\n";

        let proteome = proteome_from_annotation(gff, faa).unwrap();

        assert_eq!(proteome.replicons.len(), 1);
        assert_eq!(proteome.replicons[0].name, "chr1");
        assert_eq!(proteome.replicons[0].protein_ids, ["b0001", "WP_2.1"]);
        assert_eq!(proteome.genes[0].protein, "MKT");
        assert_eq!(proteome.genes[1].strand, "-");
        assert_eq!((proteome.genes[1].start, proteome.genes[1].end), (200, 290));
    }

    #[test]
    fn prodigal_gff_ids_match_sequence_prefixed_proteins() {
        let gff = "contigA\tProdigal_v2.6.3\tCDS\t3\t98\t5.1\t+\t0\tID=1_1;partial=00\n\
            contigA\tProdigal_v2.6.3\tCDS\t150\t300\t7.2\t-\t0\tID=1_2;partial=00\n";
        let faa =
            ">contigA_1 # 3 # 98 # 1 # ID=1_1\nMKK\n>contigA_2 # 150 # 300 # -1 # ID=1_2\nMLL\n";

        let proteome = proteome_from_annotation(gff, faa).unwrap();

        assert_eq!(
            proteome.replicons[0].protein_ids,
            ["contigA_1", "contigA_2"]
        );
    }

    #[test]
    fn annotation_without_any_match_is_an_error() {
        let gff = "chr1\tsrc\tCDS\t1\t90\t.\t+\t0\tID=a\n";
        let error = proteome_from_annotation(gff, ">b\nMKT\n").unwrap_err();
        assert!(error.to_string().contains("No protein"));
    }

    #[test]
    fn predicted_genes_are_written_as_gff_cds_features() {
        let proteome = Proteome {
            genes: vec![GeneRecord {
                id: "s_1".into(),
                seq_id: "s".into(),
                protein: "MK".into(),
                start: 4,
                end: 12,
                strand: "-".into(),
            }],
            replicons: Vec::new(),
        };
        assert_eq!(
            gff_content(&proteome),
            "##gff-version 3\ns\tcrispr-cas-finder\tCDS\t4\t12\t.\t-\t0\tID=s_1\n"
        );
    }
}
