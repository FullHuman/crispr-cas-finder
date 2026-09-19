use crate::types::CrisprArray;
use anyhow::Result;
use std::io::{BufWriter, Write};

/// Write CRISPR arrays as a GFF3 file.
pub fn write_gff(arrays: &[CrisprArray], writer: &mut impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    writeln!(writer, "##gff-version 3")?;
    for (i, array) in arrays.iter().enumerate() {
        let crispr_id = format!("crispr{}", i + 1);
        let crispr_name = format!("CRISPR_{}", i + 1);
        let strand = array.orientation.to_string();
        let seq_id = escape_gff(&array.seq_id);
        writeln!(
            writer,
            "{seqid}\tCRISPRCasFinder\tCRISPR\t{start}\t{end}\t.\t{strand}\t.\tID={id};Name={name};Repeat_ID={repeat_id};CRISPRDirection={crispr_direction}",
            seqid = seq_id,
            start = array.start,
            end = array.end,
            strand = strand,
            id = crispr_id,
            name = crispr_name,
            repeat_id = escape_gff(&array.repeat_id),
            crispr_direction = escape_gff(&array.crispr_direction),
        )?;
        for (j, repeat) in array.repeats.iter().enumerate() {
            let dr_id = format!("{}_dr{}", crispr_id, j + 1);
            writeln!(
                writer,
                "{seqid}\tCRISPRCasFinder\tdirect_repeat\t{start}\t{end}\t.\t{strand}\t.\tID={id};Parent={parent};Note={seq}",
                seqid = seq_id,
                start = repeat.start,
                end = repeat.end,
                strand = strand,
                id = dr_id,
                parent = crispr_id,
                seq = escape_gff(&repeat.sequence)
            )?;
            // Spacer follows the j-th repeat (if it exists)
            if let Some(spacer) = array.spacers.get(j) {
                let spacer_id = format!("{}_spacer{}", crispr_id, j + 1);
                writeln!(
                    writer,
                    "{seqid}\tCRISPRCasFinder\tspacer\t{start}\t{end}\t.\t{strand}\t.\tID={id};Parent={parent};Note={seq}",
                    seqid = seq_id,
                    start = spacer.start,
                    end = spacer.end,
                    strand = strand,
                    id = spacer_id,
                    parent = crispr_id,
                    seq = escape_gff(&spacer.sequence)
                )?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

/// Write CRISPR arrays as a JSON file.
pub fn write_json(arrays: &[CrisprArray], writer: &mut impl Write) -> Result<()> {
    serde_json::to_writer_pretty(writer, arrays)?;
    Ok(())
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
    use crate::{Orientation, Repeat};

    #[test]
    fn gff_escapes_identifiers_and_attributes() {
        let array = CrisprArray {
            seq_id: "chr 1;\t".into(),
            start: 1,
            end: 4,
            repeats: vec![Repeat {
                start: 1,
                end: 4,
                sequence: "A;=%".into(),
            }],
            spacers: vec![],
            consensus_repeat: "ACGT".into(),
            orientation: Orientation::Unknown,
            evidence_level: 1,
            repeat_id: "db=x;y".into(),
            crispr_direction: "+".into(),
        };
        let mut output = Vec::new();
        write_gff(&[array], &mut output).unwrap();
        let text = String::from_utf8(output).unwrap();
        for row in text.lines().skip(1) {
            assert_eq!(row.split('\t').count(), 9);
        }
        assert!(text.contains("chr%201%3B%09"));
        assert!(text.contains("Repeat_ID=db%3Dx%3By"));
        assert!(text.contains("Note=A%3B%3D%25"));
    }

    #[test]
    fn gff_propagates_buffered_write_errors() {
        struct FailingWriter;
        impl std::io::Write for FailingWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk full"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        assert!(write_gff(&[], &mut FailingWriter).is_err());
    }
}
