//! Writers for every output format, all driven by an [`AnalysisReport`].

use crate::annotation::Proteome;
use crate::report::AnalysisReport;
use crate::types::CrisprArray;
use anyhow::Result;
use std::collections::HashMap;
use std::io::{BufWriter, Write};

const GFF_SOURCE: &str = "crispr-cas-finder";
const FASTA_LINE_WIDTH: usize = 80;

/// Write the complete report as pretty-printed JSON.
pub fn write_report_json(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    serde_json::to_writer_pretty(&mut writer, report)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

/// Write arrays, repeats, spacers, flanking regions and Cas genes as GFF3.
///
/// Flanks are `parameters.detection.flank` bp on each side of an array,
/// clipped to the sequence. Cas systems are `gene_group` features whose
/// children are the `CDS` features of their genes.
pub fn write_gff(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    writeln!(writer, "##gff-version 3")?;
    for sequence in &report.sequences {
        if sequence.length > 0 {
            writeln!(
                writer,
                "##sequence-region {} 1 {}",
                escape_gff(&sequence.id),
                sequence.length
            )?;
        }
    }
    let lengths: HashMap<&str, usize> = report
        .sequences
        .iter()
        .map(|sequence| (sequence.id.as_str(), sequence.length))
        .collect();
    let flank = report.parameters.detection.flank;

    for (index, array) in report.crisprs.iter().enumerate() {
        let fallback_id = format!("CRISPR_{}", index + 1);
        let array_id = escape_gff(if array.id.is_empty() {
            &fallback_id
        } else {
            &array.id
        });
        let strand = array.orientation.to_string();
        let seq_id = escape_gff(&array.seq_id);
        let statistics = &array.statistics;
        writeln!(
            writer,
            "{seq_id}\t{GFF_SOURCE}\tCRISPR\t{}\t{}\t.\t{strand}\t.\tID={array_id};Name={array_id};\
             evidence_level={};consensus_repeat={};repeat_length={};repeat_count={};spacer_count={};\
             Repeat_ID={};CRISPRDirection={};repeat_conservation={:.2};spacer_conservation={:.2};\
             spacer_identity={:.2};duplicated_spacers={}",
            array.start,
            array.end,
            array.evidence_level,
            escape_gff(&array.consensus_repeat),
            array.consensus_repeat.len(),
            array.repeats.len(),
            array.spacers.len(),
            escape_gff(&array.repeat_id),
            escape_gff(&array.crispr_direction),
            statistics.repeat_conservation,
            statistics.spacer_conservation,
            statistics.spacer_identity,
            statistics.has_duplicated_spacers,
        )?;

        let mut flanks = Vec::new();
        if flank > 0 && array.start > 1 {
            flanks.push((
                "left",
                array.start.saturating_sub(flank).max(1),
                array.start - 1,
            ));
        }
        let sequence_length = lengths.get(array.seq_id.as_str()).copied().unwrap_or(0);
        if flank > 0 && array.end < sequence_length {
            flanks.push((
                "right",
                array.end + 1,
                (array.end + flank).min(sequence_length),
            ));
        }
        for (side, start, end) in flanks {
            writeln!(
                writer,
                "{seq_id}\t{GFF_SOURCE}\tflanking_region\t{start}\t{end}\t.\t{strand}\t.\t\
                 ID={array_id}_{side}_flank;Parent={array_id}"
            )?;
        }

        for (number, repeat) in array.repeats.iter().enumerate() {
            writeln!(
                writer,
                "{seq_id}\t{GFF_SOURCE}\tdirect_repeat\t{}\t{}\t.\t{strand}\t.\t\
                 ID={array_id}_repeat_{};Parent={array_id};sequence={}",
                repeat.start,
                repeat.end,
                number + 1,
                escape_gff(&repeat.sequence)
            )?;
            // Spacer follows the repeat of the same rank, if it exists.
            if let Some(spacer) = array.spacers.get(number) {
                writeln!(
                    writer,
                    "{seq_id}\t{GFF_SOURCE}\tspacer\t{}\t{}\t.\t{strand}\t.\t\
                     ID={array_id}_spacer_{};Parent={array_id};sequence={}",
                    spacer.start,
                    spacer.end,
                    number + 1,
                    escape_gff(&spacer.sequence)
                )?;
            }
        }
    }

    for (index, system) in report.cas_clusters.iter().enumerate() {
        let fallback_id = format!("Cas_{}", index + 1);
        let system_id = escape_gff(if system.id.is_empty() {
            &fallback_id
        } else {
            &system.id
        });
        let seq_id = escape_gff(&system.seq_id);
        // A system spanning the origin of a circular sequence has no single
        // start..end range; each gene is then written without a parent span.
        let spans_origin = system
            .genes
            .windows(2)
            .any(|pair| pair[1].start < pair[0].start);
        if !spans_origin {
            writeln!(
                writer,
                "{seq_id}\t{GFF_SOURCE}\tgene_group\t{}\t{}\t{:.1}\t.\t.\tID={system_id};Name={};\
                 gene_count={};wholeness={:.2}",
                system.start,
                system.end,
                system.score,
                escape_gff(&system.system),
                system.genes.len(),
                system.wholeness,
            )?;
        }
        for gene in &system.genes {
            let strand = if gene.strand == "+" || gene.strand == "-" {
                gene.strand.as_str()
            } else {
                "."
            };
            let parent = if spans_origin {
                format!("cas_system={system_id}")
            } else {
                format!("Parent={system_id}")
            };
            writeln!(
                writer,
                "{seq_id}\t{GFF_SOURCE}\tCDS\t{}\t{}\t{:.1}\t{strand}\t0\tID={system_id}.{};{parent};\
                 Name={};profile={};status={};evalue={:.2e}",
                gene.start,
                gene.end,
                gene.score,
                escape_gff(&gene.id),
                escape_gff(&gene.name),
                escape_gff(&gene.profile),
                escape_gff(&gene.status),
                gene.evalue,
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Tab-separated table with one row per CRISPR array.
pub fn write_crisprs_tsv(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    writeln!(
        writer,
        "crispr_id\tseq_id\tstart\tend\tlength\torientation\tcrispr_direction\tevidence_level\t\
         consensus_repeat\trepeat_length\trepeat_id\trepeat_count\tspacer_count\texact_repeat_count\t\
         repeat_conservation\trepeat_identity\tspacer_conservation\tspacer_identity\t\
         spacer_average_identity\tmean_spacer_length\tspacer_length_std_dev\tduplicated_spacers"
    )?;
    for array in &report.crisprs {
        let statistics = &array.statistics;
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{}",
            tsv_field(&array.id),
            tsv_field(&array.seq_id),
            array.start,
            array.end,
            array.end.saturating_sub(array.start) + 1,
            array.orientation,
            tsv_field(&array.crispr_direction),
            array.evidence_level,
            tsv_field(&array.consensus_repeat),
            array.consensus_repeat.len(),
            tsv_field(&array.repeat_id),
            array.repeats.len(),
            array.spacers.len(),
            statistics.exact_repeat_count,
            statistics.repeat_conservation,
            statistics.repeat_identity,
            statistics.spacer_conservation,
            statistics.spacer_identity,
            statistics.spacer_average_identity,
            statistics.mean_spacer_length,
            statistics.spacer_length_std_dev,
            statistics.has_duplicated_spacers,
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// Tab-separated table with one row per gene of each Cas system.
pub fn write_cas_tsv(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    writeln!(
        writer,
        "cas_id\tseq_id\tsystem\tsystem_start\tsystem_end\tsystem_score\twholeness\t\
         protein_id\tgene\tprofile\tstatus\tstart\tend\tstrand\tscore\tevalue\tprofile_coverage"
    )?;
    for system in &report.cas_clusters {
        for gene in &system.genes {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.2e}\t{:.2}",
                tsv_field(&system.id),
                tsv_field(&system.seq_id),
                tsv_field(&system.system),
                system.start,
                system.end,
                system.score,
                system.wholeness,
                tsv_field(&gene.id),
                tsv_field(&gene.name),
                tsv_field(&gene.profile),
                tsv_field(&gene.status),
                gene.start,
                gene.end,
                tsv_field(&gene.strand),
                gene.score,
                gene.evalue,
                gene.profile_coverage,
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Tab-separated table with one row per input sequence.
pub fn write_sequences_tsv(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    writeln!(
        writer,
        "seq_id\tdescription\tlength\tat_percent\tanalyzed\tcrispr_count\tcrisprs\t\
         cas_system_count\tcas_systems"
    )?;
    for sequence in &report.sequences {
        let crisprs: Vec<String> = report
            .crisprs
            .iter()
            .filter(|array| array.seq_id == sequence.id)
            .map(|array| {
                format!(
                    "{}[{}..{};level={}]",
                    array.id, array.start, array.end, array.evidence_level
                )
            })
            .collect();
        let cas_systems: Vec<String> = report
            .cas_clusters
            .iter()
            .filter(|system| system.seq_id == sequence.id)
            .map(|system| format!("{}[{}..{}]", system.system, system.start, system.end))
            .collect();
        writeln!(
            writer,
            "{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\t{}\t{}",
            tsv_field(&sequence.id),
            tsv_field(&sequence.description),
            sequence.length,
            sequence.at_percent,
            sequence.analyzed,
            sequence.crispr_count,
            tsv_field(&crisprs.join(",")),
            sequence.cas_system_count,
            tsv_field(&cas_systems.join(",")),
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// Tab-separated table of arrays lying in the vicinity of a Cas system.
pub fn write_associations_tsv(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    writeln!(writer, "seq_id\tcrispr_id\tcas_id\tcas_system\tdistance")?;
    for association in &report.associations {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}",
            tsv_field(&association.seq_id),
            tsv_field(&association.crispr_id),
            tsv_field(&association.cas_id),
            tsv_field(&association.cas_system),
            association.distance,
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// Tab-separated table of loci chaining arrays and Cas systems.
pub fn write_loci_tsv(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    writeln!(
        writer,
        "locus_id\tseq_id\tstart\tend\tcrispr_count\tcas_count\telements"
    )?;
    for locus in &report.loci {
        let elements: Vec<String> = locus
            .elements
            .iter()
            .map(|element| format!("{}[{}..{}]", element.id, element.start, element.end))
            .collect();
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv_field(&locus.id),
            tsv_field(&locus.seq_id),
            locus.start,
            locus.end,
            locus.crispr_count,
            locus.cas_count,
            tsv_field(&elements.join(",")),
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// FASTA of every spacer, named `<crispr_id>_spacer_<n>`.
pub fn write_spacers_fasta(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    for array in &report.crisprs {
        for (number, spacer) in array.spacers.iter().enumerate() {
            write_fasta_record(
                &mut writer,
                &format!(
                    "{}_spacer_{} {}:{}-{}",
                    array.id,
                    number + 1,
                    array.seq_id,
                    spacer.start,
                    spacer.end
                ),
                spacer.sequence.as_bytes(),
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// FASTA of the consensus repeat of every array.
pub fn write_repeats_fasta(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    for array in &report.crisprs {
        write_fasta_record(
            &mut writer,
            &format!(
                "{} {}:{}-{} repeats={} repeat_id={}",
                array.id,
                array.seq_id,
                array.start,
                array.end,
                array.repeats.len(),
                array.repeat_id
            ),
            array.consensus_repeat.as_bytes(),
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// FASTA of every array, from its first to its last repeat.
pub fn write_arrays_fasta(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    for array in &report.crisprs {
        write_fasta_record(
            &mut writer,
            &format!(
                "{} {}:{}-{} evidence_level={}",
                array.id, array.seq_id, array.start, array.end, array.evidence_level
            ),
            array_sequence(array).as_bytes(),
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// Sequence of an array, rebuilt from its alternating repeats and spacers.
pub fn array_sequence(array: &CrisprArray) -> String {
    let mut sequence = String::new();
    for (number, repeat) in array.repeats.iter().enumerate() {
        sequence.push_str(&repeat.sequence);
        if let Some(spacer) = array.spacers.get(number) {
            sequence.push_str(&spacer.sequence);
        }
    }
    sequence
}

/// Protein FASTA of the genes assigned to Cas systems.
pub fn write_cas_proteins_fasta(
    report: &AnalysisReport,
    proteome: &Proteome,
    writer: &mut impl Write,
) -> Result<()> {
    let proteins: HashMap<&str, &str> = proteome
        .genes
        .iter()
        .map(|gene| (gene.id.as_str(), gene.protein.as_str()))
        .collect();
    let mut writer = BufWriter::new(writer);
    for system in &report.cas_clusters {
        for gene in &system.genes {
            let Some(protein) = proteins.get(gene.id.as_str()) else {
                continue;
            };
            write_fasta_record(
                &mut writer,
                &format!(
                    "{} {} {} {}:{}-{}({})",
                    gene.id, gene.name, system.id, system.seq_id, gene.start, gene.end, gene.strand
                ),
                protein.as_bytes(),
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Protein FASTA of every annotated gene, with Prodigal-style headers.
pub fn write_proteome_fasta(proteome: &Proteome, writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    for gene in &proteome.genes {
        write_fasta_record(
            &mut writer,
            &format!(
                "{} # {} # {} # {} # seq_id={}",
                gene.id,
                gene.start,
                gene.end,
                if gene.strand == "-" { -1 } else { 1 },
                gene.seq_id
            ),
            gene.protein.as_bytes(),
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn write_fasta_record(writer: &mut impl Write, header: &str, sequence: &[u8]) -> Result<()> {
    writeln!(writer, ">{}", header.replace(['\n', '\r'], " "))?;
    for line in sequence.chunks(FASTA_LINE_WIDTH) {
        writer.write_all(line)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

/// Tabs and line breaks would break the table layout; replace them.
fn tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

// GFF3 column/attribute delimiters and control bytes must be percent-escaped.
fn escape_gff(value: &str) -> String {
    let mut escaped = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || b".:^*$@!+_?-|".contains(&byte) {
            escaped.push(char::from(byte));
        } else {
            use std::fmt::Write;
            write!(escaped, "%{byte:02X}").expect("writing to a String cannot fail");
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::casparser::{CasCluster, CasGene};
    use crate::report::{ReportParameters, SequenceInfo, build_report};
    use crate::{AnalysisOptions, Orientation, Repeat, Spacer};

    fn sample_array(seq_id: &str) -> CrisprArray {
        CrisprArray {
            id: format!("{seq_id}_CRISPR_1"),
            seq_id: seq_id.into(),
            start: 11,
            end: 22,
            repeats: vec![
                Repeat {
                    start: 11,
                    end: 14,
                    sequence: "ACGT".into(),
                },
                Repeat {
                    start: 19,
                    end: 22,
                    sequence: "ACGA".into(),
                },
            ],
            spacers: vec![Spacer {
                start: 15,
                end: 18,
                sequence: "TTGG".into(),
            }],
            consensus_repeat: "ACGT".into(),
            orientation: Orientation::Forward,
            evidence_level: 1,
            statistics: Default::default(),
            repeat_id: "db=x;y".into(),
            crispr_direction: "+".into(),
        }
    }

    fn sample_report(seq_id: &str, length: usize) -> AnalysisReport {
        let system = CasCluster {
            id: format!("{seq_id}_Cas_1"),
            seq_id: seq_id.into(),
            system: "CASFinder/CAS-TypeIE".into(),
            genes: vec![CasGene {
                id: "p1".into(),
                name: "Cas1_0_IE".into(),
                start: 30,
                end: 60,
                strand: "-".into(),
                profile: "Cas1_0_IE".into(),
                status: "mandatory".into(),
                score: 120.5,
                evalue: 1e-30,
                profile_coverage: 0.9,
            }],
            start: 30,
            end: 60,
            score: 120.5,
            wholeness: 0.5,
        };
        build_report(
            &[SequenceInfo {
                id: seq_id.into(),
                description: "a\tb".into(),
                length,
                at_percent: 50.0,
            }],
            vec![sample_array(seq_id)],
            vec![system],
            ReportParameters::from_options(&AnalysisOptions {
                locus_distance: Some(100),
                ..AnalysisOptions::default()
            }),
        )
    }

    fn written(write: impl Fn(&AnalysisReport, &mut Vec<u8>) -> Result<()>) -> String {
        let mut output = Vec::new();
        write(&sample_report("chr1", 70), &mut output).unwrap();
        String::from_utf8(output).unwrap()
    }

    #[test]
    fn gff_escapes_identifiers_and_attributes() {
        let mut output = Vec::new();
        write_gff(&sample_report("chr 1;\t", 70), &mut output).unwrap();
        let text = String::from_utf8(output).unwrap();
        for row in text.lines().filter(|row| !row.starts_with('#')) {
            assert_eq!(row.split('\t').count(), 9, "{row}");
        }
        assert!(text.contains("chr%201%3B%09"));
        assert!(text.contains("Repeat_ID=db%3Dx%3By"));
    }

    #[test]
    fn gff_lists_flanks_children_and_cas_genes() {
        let text = written(write_gff);
        assert!(text.contains("##sequence-region chr1 1 70"));
        // 100 bp flanks are clipped to the 70 bp sequence.
        assert!(text.contains("\tflanking_region\t1\t10\t"));
        assert!(text.contains("\tflanking_region\t23\t70\t"));
        assert!(text.contains("ID=chr1_CRISPR_1_spacer_1;Parent=chr1_CRISPR_1;sequence=TTGG"));
        assert!(text.contains("\tgene_group\t30\t60\t"));
        assert!(text.contains("ID=chr1_Cas_1.p1;Parent=chr1_Cas_1;Name=Cas1_0_IE"));
    }

    #[test]
    fn tables_have_one_header_and_consistent_columns() {
        for text in [
            written(write_crisprs_tsv),
            written(write_cas_tsv),
            written(write_sequences_tsv),
            written(write_associations_tsv),
            written(write_loci_tsv),
        ] {
            let rows: Vec<&str> = text.lines().collect();
            assert_eq!(rows.len(), 2, "{text}");
            assert_eq!(
                rows[0].split('\t').count(),
                rows[1].split('\t').count(),
                "{text}"
            );
        }
    }

    #[test]
    fn fasta_outputs_name_and_rebuild_sequences() {
        assert_eq!(
            written(write_spacers_fasta),
            ">chr1_CRISPR_1_spacer_1 chr1:15-18\nTTGG\n"
        );
        assert!(written(write_arrays_fasta).ends_with("\nACGTTTGGACGA\n"));
        assert!(written(write_repeats_fasta).ends_with("\nACGT\n"));
    }

    #[test]
    fn cas_protein_fasta_uses_the_supplied_proteome() {
        let proteome = Proteome {
            genes: vec![crate::GeneRecord {
                id: "p1".into(),
                seq_id: "chr1".into(),
                protein: "MKT".into(),
                start: 30,
                end: 60,
                strand: "-".into(),
            }],
            replicons: Vec::new(),
        };
        let mut output = Vec::new();
        write_cas_proteins_fasta(&sample_report("chr1", 70), &proteome, &mut output).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            ">p1 Cas1_0_IE chr1_Cas_1 chr1:30-60(-)\nMKT\n"
        );
    }

    #[test]
    fn writers_propagate_buffered_write_errors() {
        struct FailingWriter;
        impl std::io::Write for FailingWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk full"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let report = sample_report("chr1", 70);
        assert!(write_gff(&report, &mut FailingWriter).is_err());
        assert!(write_report_json(&report, &mut FailingWriter).is_err());
        assert!(write_crisprs_tsv(&report, &mut FailingWriter).is_err());
    }
}
