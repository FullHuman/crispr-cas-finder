//! Self-contained HTML rendering of an [`AnalysisReport`].
//!
//! The page embeds its own style sheet and needs no network access, unlike
//! the upstream viewer, which depends on a separate `crispr.css`.

use crate::report::AnalysisReport;
use crate::types::CrisprArray;
use anyhow::Result;
use std::io::{BufWriter, Write};

const STYLE: &str = "
:root{--ink:#1c2430;--muted:#5d6b7a;--line:#d8dee6;--panel:#f5f7fa;--page:#fff;
--repeat:#0b6e99;--spacer:#b25d00;--cas:#3f7d3a;--crispr:#7a4fb5}
@media (prefers-color-scheme:dark){:root{--ink:#e6ebf1;--muted:#9aa7b5;--line:#33404d;
--panel:#1a222b;--page:#11171d;--repeat:#6cc4ea;--spacer:#f0a95a;--cas:#8fce88;--crispr:#c3a3f0}}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px 48px;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto}
a{color:var(--repeat)}
h1{font-size:1.6rem;margin:0 0 4px}
h2{font-size:1.2rem;margin:32px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:20px 0 6px}
p.meta{color:var(--muted);margin:0 0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.card b{display:block;font-size:1.5rem}
.card span{color:var(--muted);font-size:.85rem}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;padding:5px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
code,.seq{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85rem}
.seq{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;
overflow-x:auto;white-space:pre;line-height:1.6}
.repeat{color:var(--repeat);font-weight:600}
.spacer{color:var(--spacer)}
.level{display:inline-block;min-width:1.6em;text-align:center;border-radius:4px;
border:1px solid var(--line);padding:0 4px;font-weight:600}
svg.map{width:100%;height:auto;display:block;margin:8px 0}
svg.map text{font:11px system-ui,sans-serif;fill:var(--muted)}
details{margin:8px 0}
summary{cursor:pointer}
";

/// Escape text for use in HTML content and attribute values.
fn escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Group digits by thousands for readability.
fn grouped(value: usize) -> String {
    let digits = value.to_string();
    let mut output = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(digit);
    }
    output
}

/// Write the report as one HTML page with no external resources.
pub fn write_html(report: &AnalysisReport, writer: &mut impl Write) -> Result<()> {
    let mut w = BufWriter::new(writer);
    let summary = &report.summary;
    writeln!(
        w,
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>CRISPR-Cas report</title>\n<style>{STYLE}</style>\n</head>\n<body>\n<main>"
    )?;
    writeln!(
        w,
        "<h1>CRISPR-Cas report</h1>\n<p class=\"meta\">{} {}</p>",
        escape(&report.tool.name),
        escape(&report.tool.version)
    )?;

    writeln!(w, "<div class=\"cards\">")?;
    for (value, label) in [
        (summary.sequence_count, "sequences"),
        (summary.crispr_count, "CRISPR arrays"),
        (summary.evidence_levels[3], "arrays at level 4"),
        (summary.cas_system_count, "Cas systems"),
        (summary.association_count, "arrays near a Cas system"),
    ] {
        writeln!(
            w,
            "<div class=\"card\"><b>{}</b><span>{label}</span></div>",
            grouped(value)
        )?;
    }
    writeln!(w, "</div>")?;

    writeln!(
        w,
        "<h2>Sequences</h2>\n<div class=\"scroll\"><table>\n<tr><th>Sequence</th><th>Description</th>\
         <th class=\"num\">Length</th><th class=\"num\">AT%</th><th class=\"num\">Arrays</th>\
         <th class=\"num\">Cas systems</th></tr>"
    )?;
    for sequence in &report.sequences {
        writeln!(
            w,
            "<tr><td><a href=\"#seq-{0}\">{0}</a></td><td>{1}</td><td class=\"num\">{2}</td>\
             <td class=\"num\">{3:.1}</td><td class=\"num\">{4}</td><td class=\"num\">{5}</td></tr>",
            escape(&sequence.id),
            escape(&sequence.description),
            grouped(sequence.length),
            sequence.at_percent,
            sequence.crispr_count,
            sequence.cas_system_count,
        )?;
    }
    writeln!(w, "</table></div>")?;

    for sequence in &report.sequences {
        if sequence.crispr_count == 0 && sequence.cas_system_count == 0 {
            continue;
        }
        writeln!(w, "<h2 id=\"seq-{0}\">{0}</h2>", escape(&sequence.id))?;
        write_map(&mut w, report, &sequence.id, sequence.length)?;

        let arrays: Vec<&CrisprArray> = report
            .crisprs
            .iter()
            .filter(|array| array.seq_id == sequence.id)
            .collect();
        if !arrays.is_empty() {
            writeln!(
                w,
                "<h3>CRISPR arrays</h3>\n<div class=\"scroll\"><table>\n<tr><th>Array</th>\
                 <th class=\"num\">Start</th><th class=\"num\">End</th><th>Strand</th><th>Level</th>\
                 <th class=\"num\">Spacers</th><th>Consensus repeat</th><th>Repeat ID</th>\
                 <th class=\"num\">Repeat conservation</th><th class=\"num\">Spacer identity</th>\
                 <th>Cas nearby</th></tr>"
            )?;
            for array in &arrays {
                let nearby: Vec<String> = report
                    .associations
                    .iter()
                    .filter(|association| association.crispr_id == array.id)
                    .map(|association| {
                        format!(
                            "{} ({} bp)",
                            escape(&association.cas_system),
                            grouped(association.distance)
                        )
                    })
                    .collect();
                writeln!(
                    w,
                    "<tr><td><a href=\"#{0}\">{0}</a></td><td class=\"num\">{1}</td>\
                     <td class=\"num\">{2}</td><td>{3}</td><td><span class=\"level\">{4}</span></td>\
                     <td class=\"num\">{5}</td><td><code>{6}</code></td><td>{7}</td>\
                     <td class=\"num\">{8:.1}</td><td class=\"num\">{9:.1}</td><td>{10}</td></tr>",
                    escape(&array.id),
                    grouped(array.start),
                    grouped(array.end),
                    array.orientation,
                    array.evidence_level,
                    array.spacers.len(),
                    escape(&array.consensus_repeat),
                    escape(&array.repeat_id),
                    array.statistics.repeat_conservation,
                    array.statistics.spacer_identity,
                    if nearby.is_empty() {
                        "—".to_string()
                    } else {
                        nearby.join(", ")
                    },
                )?;
            }
            writeln!(w, "</table></div>")?;
            for array in &arrays {
                writeln!(
                    w,
                    "<details id=\"{0}\"><summary>{0}: repeats and spacers</summary>\n<div class=\"seq\">",
                    escape(&array.id)
                )?;
                for (number, repeat) in array.repeats.iter().enumerate() {
                    write!(
                        w,
                        "{:>10}  <span class=\"repeat\">{}</span>",
                        repeat.start,
                        escape(&repeat.sequence)
                    )?;
                    if let Some(spacer) = array.spacers.get(number) {
                        write!(
                            w,
                            "  <span class=\"spacer\">{}</span>",
                            escape(&spacer.sequence)
                        )?;
                    }
                    writeln!(w)?;
                }
                writeln!(w, "</div></details>")?;
            }
        }

        let systems: Vec<_> = report
            .cas_clusters
            .iter()
            .filter(|system| system.seq_id == sequence.id)
            .collect();
        if !systems.is_empty() {
            writeln!(w, "<h3>Cas systems</h3>")?;
            for system in systems {
                writeln!(
                    w,
                    "<p><b>{}</b> · {} – {} · score {:.1} · {:.0}% of model genes</p>\n\
                     <div class=\"scroll\"><table>\n<tr><th>Gene</th><th>Profile</th><th>Role</th>\
                     <th>Protein</th><th class=\"num\">Start</th><th class=\"num\">End</th>\
                     <th>Strand</th><th class=\"num\">Score</th><th class=\"num\">E-value</th></tr>",
                    escape(&system.system),
                    grouped(system.start),
                    grouped(system.end),
                    system.score,
                    system.wholeness * 100.0,
                )?;
                for gene in &system.genes {
                    writeln!(
                        w,
                        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class=\"num\">{}</td>\
                         <td class=\"num\">{}</td><td>{}</td><td class=\"num\">{:.1}</td>\
                         <td class=\"num\">{:.1e}</td></tr>",
                        escape(&gene.name),
                        escape(&gene.profile),
                        escape(&gene.status),
                        escape(&gene.id),
                        grouped(gene.start),
                        grouped(gene.end),
                        escape(&gene.strand),
                        gene.score,
                        gene.evalue,
                    )?;
                }
                writeln!(w, "</table></div>")?;
            }
        }
    }

    writeln!(w, "</main>\n</body>\n</html>")?;
    w.flush()?;
    Ok(())
}

/// Draw arrays and Cas systems along the sequence as an inline SVG.
fn write_map(
    w: &mut impl Write,
    report: &AnalysisReport,
    seq_id: &str,
    length: usize,
) -> Result<()> {
    if length == 0 {
        return Ok(());
    }
    const WIDTH: f64 = 1000.0;
    const MIN_MARK: f64 = 3.0;
    let x = |position: usize| position as f64 / length as f64 * WIDTH;
    writeln!(
        w,
        "<svg class=\"map\" viewBox=\"0 0 1000 64\" role=\"img\" \
         aria-label=\"Positions of CRISPR arrays and Cas systems on {}\">\n\
         <line x1=\"0\" y1=\"32\" x2=\"1000\" y2=\"32\" stroke=\"var(--line)\" stroke-width=\"2\"/>",
        escape(seq_id)
    )?;
    for system in report
        .cas_clusters
        .iter()
        .filter(|system| system.seq_id == seq_id)
    {
        writeln!(
            w,
            "<rect x=\"{:.2}\" y=\"36\" width=\"{:.2}\" height=\"12\" rx=\"2\" fill=\"var(--cas)\">\
             <title>{} {}–{}</title></rect>",
            x(system.start),
            (x(system.end) - x(system.start)).max(MIN_MARK),
            escape(&system.system),
            grouped(system.start),
            grouped(system.end),
        )?;
    }
    for array in report.crisprs.iter().filter(|array| array.seq_id == seq_id) {
        writeln!(
            w,
            "<rect x=\"{:.2}\" y=\"16\" width=\"{:.2}\" height=\"12\" rx=\"2\" fill=\"var(--crispr)\">\
             <title>{} {}–{} (level {})</title></rect>",
            x(array.start),
            (x(array.end) - x(array.start)).max(MIN_MARK),
            escape(&array.id),
            grouped(array.start),
            grouped(array.end),
            array.evidence_level,
        )?;
    }
    writeln!(
        w,
        "<text x=\"0\" y=\"10\">CRISPR arrays</text><text x=\"0\" y=\"62\">Cas systems</text>\
         <text x=\"1000\" y=\"62\" text-anchor=\"end\">{} bp</text>\n</svg>",
        grouped(length)
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{ReportParameters, SequenceInfo, build_report};
    use crate::{AnalysisOptions, Orientation, Repeat, Spacer};

    #[test]
    fn digits_are_grouped_by_thousands() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(4_641_652), "4,641,652");
    }

    #[test]
    fn page_is_self_contained_and_escapes_untrusted_text() {
        let array = CrisprArray {
            id: "s<1>_CRISPR_1".into(),
            seq_id: "s<1>".into(),
            start: 11,
            end: 22,
            consensus_repeat: "ACGT".into(),
            repeats: vec![
                Repeat {
                    start: 11,
                    end: 14,
                    sequence: "ACGT".into(),
                },
                Repeat {
                    start: 19,
                    end: 22,
                    sequence: "ACGT".into(),
                },
            ],
            spacers: vec![Spacer {
                start: 15,
                end: 18,
                sequence: "TTGG".into(),
            }],
            evidence_level: 1,
            statistics: Default::default(),
            orientation: Orientation::Forward,
            repeat_id: "Unknown".into(),
            crispr_direction: "ND".into(),
            crisprdb_repeat_count: 0,
            known_in_crisprdb: None,
            reclassified_from: None,
        };
        let report = build_report(
            &[SequenceInfo {
                id: "s<1>".into(),
                description: "<script>alert(1)</script>".into(),
                length: 100,
                at_percent: 50.0,
            }],
            vec![array],
            Vec::new(),
            ReportParameters::from_options(&AnalysisOptions::default()),
        );
        let mut output = Vec::new();
        write_html(&report, &mut output).unwrap();
        let page = String::from_utf8(output).unwrap();

        assert!(page.starts_with("<!DOCTYPE html>"));
        assert!(page.trim_end().ends_with("</html>"));
        assert!(!page.contains("<script"));
        assert!(page.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
        assert!(page.contains("id=\"s&lt;1&gt;_CRISPR_1\""));
        assert!(!page.contains("http://") && !page.contains("https://"));
        assert!(page.contains("<span class=\"spacer\">TTGG</span>"));
        assert!(page.contains("<svg class=\"map\""));
    }
}
