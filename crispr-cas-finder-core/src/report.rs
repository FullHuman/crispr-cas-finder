//! The complete result of an analysis and the entry point that produces it.

use crate::annotation::Proteome;
use crate::association::{
    CrisprCasAssociation, CrisprCasLocus, associate_crisprs_with_cas, cluster_loci,
};
use crate::cas_types::RepliconTopology;
use crate::casfinder::detect_cas_systems;
use crate::casparser::CasCluster;
use crate::detect::at_percent;
use crate::types::{CasFinderConfig, CrisprArray, DetectionParams};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Default distance, in bp, within which an array is tied to a Cas system.
pub const DEFAULT_VICINITY: usize = 600;

/// One input sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceRecord {
    pub id: String,
    /// Text after the identifier on the FASTA header line; may be empty.
    pub description: String,
    pub sequence: Vec<u8>,
}

/// Parse FASTA text into sequence records, rejecting duplicate identifiers.
pub fn parse_fasta(reader: impl std::io::Read) -> Result<Vec<SequenceRecord>> {
    let mut records = Vec::new();
    let mut seen = HashSet::new();
    for record in bio::io::fasta::Reader::new(reader).records() {
        let record = record?;
        if !seen.insert(record.id().to_string()) {
            anyhow::bail!("Duplicate FASTA identifier: {}", record.id());
        }
        records.push(SequenceRecord {
            id: record.id().to_string(),
            description: record.desc().unwrap_or_default().to_string(),
            sequence: record.seq().to_vec(),
        });
    }
    Ok(records)
}

/// Everything an analysis can be asked to do.
#[derive(Debug, Clone)]
pub struct AnalysisOptions {
    pub detection: DetectionParams,
    /// Cas detection settings; `None` skips Cas detection.
    pub cas: Option<CasFinderConfig>,
    /// Search Cas systems only on sequences carrying at least one array.
    pub cas_only_with_crispr: bool,
    /// Distance, in bp, within which an array is tied to a Cas system.
    pub vicinity: usize,
    /// Chain arrays and Cas systems closer than this many bp into loci.
    pub locus_distance: Option<usize>,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            detection: DetectionParams::default(),
            cas: None,
            cas_only_with_crispr: false,
            vicinity: DEFAULT_VICINITY,
            locus_distance: None,
        }
    }
}

/// Name and version of the program that wrote a report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub version: String,
}

impl Default for ToolInfo {
    fn default() -> Self {
        Self {
            name: "crispr-cas-finder".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Cas settings recorded in a report. Paths and thread counts are left out
/// so reports from different machines and runtimes stay comparable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CasParameters {
    pub definition: String,
    pub genetic_code: usize,
    pub metagenome: bool,
    pub auto_metagenome_below: usize,
    pub user_annotation: bool,
    pub topology: RepliconTopology,
    pub min_best_hit_score: f64,
    pub only_with_crispr: bool,
}

/// Settings an analysis ran with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportParameters {
    pub detection: DetectionParams,
    pub cas: Option<CasParameters>,
    pub vicinity: usize,
    pub locus_distance: Option<usize>,
}

impl ReportParameters {
    pub fn from_options(options: &AnalysisOptions) -> Self {
        Self {
            detection: options.detection.clone(),
            cas: options.cas.as_ref().map(|cas| CasParameters {
                definition: cas.definition.clone(),
                genetic_code: cas.genetic_code,
                metagenome: cas.metagenome,
                auto_metagenome_below: cas.auto_metagenome_below,
                user_annotation: cas.annotation_gff.is_some(),
                topology: cas.replicon_topology,
                min_best_hit_score: cas.min_best_hit_score,
                only_with_crispr: options.cas_only_with_crispr,
            }),
            vicinity: options.vicinity,
            locus_distance: options.locus_distance,
        }
    }
}

/// Per-sequence overview.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceSummary {
    pub id: String,
    pub description: String,
    pub length: usize,
    /// Percentage of A and T bases.
    pub at_percent: f64,
    /// Whether the sequence met `min_sequence_length` and was analysed.
    pub analyzed: bool,
    pub crispr_count: usize,
    pub cas_system_count: usize,
}

/// Totals over the whole input.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportSummary {
    pub sequence_count: usize,
    pub crispr_count: usize,
    /// Number of arrays at evidence levels 1, 2, 3 and 4.
    pub evidence_levels: [usize; 4],
    pub cas_system_count: usize,
    /// Number of systems per model name.
    pub cas_systems_by_type: BTreeMap<String, usize>,
    pub association_count: usize,
    pub locus_count: usize,
}

/// Complete, self-describing result of one analysis.
///
/// `crisprs` and `cas_clusters` are flat lists; every element carries the
/// `seq_id` of its sequence and a stable `id` that `associations` and
/// `loci` refer to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub tool: ToolInfo,
    pub parameters: ReportParameters,
    pub sequences: Vec<SequenceSummary>,
    pub crisprs: Vec<CrisprArray>,
    pub cas_clusters: Vec<CasCluster>,
    /// Arrays lying within `parameters.vicinity` of a Cas system.
    pub associations: Vec<CrisprCasAssociation>,
    /// Arrays and Cas systems chained by `parameters.locus_distance`.
    pub loci: Vec<CrisprCasLocus>,
    pub summary: ReportSummary,
}

/// Lengths and descriptions of the input, without the sequences themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceInfo {
    pub id: String,
    pub description: String,
    pub length: usize,
    pub at_percent: f64,
}

impl SequenceInfo {
    pub fn from_record(record: &SequenceRecord) -> Self {
        Self {
            id: record.id.clone(),
            description: record.description.clone(),
            length: record.sequence.len(),
            at_percent: at_percent(&record.sequence),
        }
    }
}

/// Assemble a report from already computed arrays and Cas systems.
///
/// Used directly by callers that run the stages themselves, such as the
/// browser worker, so every runtime produces the same structure.
pub fn build_report(
    sequences: &[SequenceInfo],
    crisprs: Vec<CrisprArray>,
    cas_clusters: Vec<CasCluster>,
    parameters: ReportParameters,
) -> AnalysisReport {
    let topology = parameters
        .cas
        .as_ref()
        .map_or(RepliconTopology::Circular, |cas| cas.topology);
    let lengths: HashMap<String, usize> = sequences
        .iter()
        .map(|sequence| (sequence.id.clone(), sequence.length))
        .collect();
    let associations = associate_crisprs_with_cas(
        &crisprs,
        &cas_clusters,
        parameters.vicinity,
        topology,
        &lengths,
    );
    let loci = parameters
        .locus_distance
        .map(|distance| cluster_loci(&crisprs, &cas_clusters, distance))
        .unwrap_or_default();

    let mut summary = ReportSummary {
        sequence_count: sequences.len(),
        crispr_count: crisprs.len(),
        cas_system_count: cas_clusters.len(),
        association_count: associations.len(),
        locus_count: loci.len(),
        ..ReportSummary::default()
    };
    for array in &crisprs {
        if let Some(count) = summary
            .evidence_levels
            .get_mut(array.evidence_level.wrapping_sub(1))
        {
            *count += 1;
        }
    }
    for system in &cas_clusters {
        *summary
            .cas_systems_by_type
            .entry(system.system.clone())
            .or_default() += 1;
    }

    let min_sequence_length = parameters.detection.min_sequence_length;
    let sequences = sequences
        .iter()
        .map(|sequence| SequenceSummary {
            id: sequence.id.clone(),
            description: sequence.description.clone(),
            length: sequence.length,
            at_percent: sequence.at_percent,
            analyzed: sequence.length >= min_sequence_length,
            crispr_count: crisprs
                .iter()
                .filter(|array| array.seq_id == sequence.id)
                .count(),
            cas_system_count: cas_clusters
                .iter()
                .filter(|system| system.seq_id == sequence.id)
                .count(),
        })
        .collect();

    AnalysisReport {
        tool: ToolInfo::default(),
        parameters,
        sequences,
        crisprs,
        cas_clusters,
        associations,
        loci,
        summary,
    }
}

/// A report together with the proteins searched for Cas systems.
#[derive(Debug)]
pub struct Analysis {
    pub report: AnalysisReport,
    /// Proteins predicted or supplied for Cas detection; `None` without it.
    pub proteome: Option<Proteome>,
}

/// Detect CRISPR arrays and, when configured, Cas systems, and relate them.
pub fn analyze(sequences: &[SequenceRecord], options: &AnalysisOptions) -> Result<Analysis> {
    options.detection.validate().map_err(anyhow::Error::msg)?;
    let sequence_refs: Vec<(&str, &[u8])> = sequences
        .iter()
        .map(|record| (record.id.as_str(), record.sequence.as_slice()))
        .collect();
    let crisprs = crate::detect_crisprs(&sequence_refs, &options.detection);

    let mut proteome = None;
    let mut cas_clusters = Vec::new();
    if let Some(cas_config) = &options.cas {
        let min_sequence_length = options.detection.min_sequence_length;
        let with_crispr: HashSet<&str> =
            crisprs.iter().map(|array| array.seq_id.as_str()).collect();
        let included: HashSet<String> = sequences
            .iter()
            .filter(|record| record.sequence.len() >= min_sequence_length)
            .filter(|record| {
                !options.cas_only_with_crispr || with_crispr.contains(record.id.as_str())
            })
            .map(|record| record.id.clone())
            .collect();
        let owned_sequences: Vec<(String, Vec<u8>)> = sequences
            .iter()
            .filter(|record| included.contains(&record.id))
            .map(|record| (record.id.clone(), record.sequence.clone()))
            .collect();
        let analysis = detect_cas_systems(&owned_sequences, cas_config, Some(&included))?;
        cas_clusters = analysis.clusters;
        proteome = Some(analysis.proteome);
    }

    let sequence_info: Vec<SequenceInfo> =
        sequences.iter().map(SequenceInfo::from_record).collect();
    let report = build_report(
        &sequence_info,
        crisprs,
        cas_clusters,
        ReportParameters::from_options(options),
    );
    Ok(Analysis { report, proteome })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fasta_records_keep_descriptions_and_reject_duplicates() {
        let records = parse_fasta(">a first one\nACGT\n>b\nTT\n".as_bytes()).unwrap();
        assert_eq!(records[0].description, "first one");
        assert_eq!(records[1].description, "");
        assert_eq!(records[0].sequence, b"ACGT");
        assert!(parse_fasta(">a\nAC\n>a\nGT\n".as_bytes()).is_err());
    }

    #[test]
    fn report_without_findings_still_describes_the_input() {
        let records = parse_fasta(">a plasmid\nAATTGGCC\n".as_bytes()).unwrap();
        let analysis = analyze(&records, &AnalysisOptions::default()).unwrap();
        let report = analysis.report;

        assert!(analysis.proteome.is_none());
        assert_eq!(report.tool.name, "crispr-cas-finder");
        assert_eq!(report.sequences.len(), 1);
        assert_eq!(report.sequences[0].length, 8);
        assert_eq!(report.sequences[0].at_percent, 50.0);
        assert_eq!(report.summary.sequence_count, 1);
        assert_eq!(report.summary.evidence_levels, [0; 4]);
        assert!(report.parameters.cas.is_none());
    }

    #[test]
    fn report_round_trips_through_json() {
        let records = parse_fasta(">a\nAATTGGCC\n".as_bytes()).unwrap();
        let report = analyze(&records, &AnalysisOptions::default())
            .unwrap()
            .report;
        let json = serde_json::to_string(&report).unwrap();
        assert_eq!(
            serde_json::from_str::<AnalysisReport>(&json).unwrap(),
            report
        );
    }

    #[test]
    fn invalid_detection_parameters_are_rejected() {
        let options = AnalysisOptions {
            detection: DetectionParams {
                min_repeat_length: 0,
                ..DetectionParams::default()
            },
            ..AnalysisOptions::default()
        };
        assert!(analyze(&[], &options).is_err());
    }
}
