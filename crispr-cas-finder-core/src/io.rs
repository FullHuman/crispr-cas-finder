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
        writeln!(
            writer,
            "{seqid}\tCRISPRCasFinder\tCRISPR\t{start}\t{end}\t.\t{strand}\t.\tID={id};Name={name};Repeat_ID={repeat_id};CRISPRDirection={crispr_direction}",
            seqid = array.seq_id,
            start = array.start,
            end = array.end,
            strand = strand,
            id = crispr_id,
            name = crispr_name,
            repeat_id = array.repeat_id,
            crispr_direction = array.crispr_direction,
        )?;
        for (j, repeat) in array.repeats.iter().enumerate() {
            let dr_id = format!("{}_dr{}", crispr_id, j + 1);
            writeln!(
                writer,
                "{seqid}\tCRISPRCasFinder\tdirect_repeat\t{start}\t{end}\t.\t{strand}\t.\tID={id};Parent={parent};Note={seq}",
                seqid = array.seq_id,
                start = repeat.start,
                end = repeat.end,
                strand = strand,
                id = dr_id,
                parent = crispr_id,
                seq = repeat.sequence
            )?;
            // Spacer follows the j-th repeat (if it exists)
            if let Some(spacer) = array.spacers.get(j) {
                let spacer_id = format!("{}_spacer{}", crispr_id, j + 1);
                writeln!(
                    writer,
                    "{seqid}\tCRISPRCasFinder\tspacer\t{start}\t{end}\t.\t{strand}\t.\tID={id};Parent={parent};Note={seq}",
                    seqid = array.seq_id,
                    start = spacer.start,
                    end = spacer.end,
                    strand = strand,
                    id = spacer_id,
                    parent = crispr_id,
                    seq = spacer.sequence
                )?;
            }
        }
    }
    Ok(())
}

/// Write CRISPR arrays as a JSON file.
pub fn write_json(arrays: &[CrisprArray], writer: &mut impl Write) -> Result<()> {
    serde_json::to_writer_pretty(writer, arrays)?;
    Ok(())
}
