// p7_alidisplay.rs - Alignment display formatting
//
// Port of src/p7_alidisplay.c

use crate::hmmer_core::config::*;
use crate::hmmer_core::constants::results::alidisplay::SIMILARITY_GROUPS;
use crate::hmmer_core::hmm::Hmm;
use crate::hmmer_core::trace::{Trace, TraceDomain, TraceStep};

/// Alignment display for one domain hit.
///
/// In the C code, this is packed into a single contiguous memory block.
/// In Rust, we use separate Strings/Options for each field.
#[derive(Debug, Clone)]
pub struct AliDisplay {
    pub reference_line: Option<String>,  // reference annotation line
    pub model_mask_line: Option<String>, // modelmask annotation line
    pub consensus_structure_line: Option<String>, // consensus structure line
    pub model: String,                   // aligned query consensus sequence
    pub match_line: String,              // identity/conservation line
    pub aligned_sequence: String,        // aligned target sequence
    pub nucleotide_sequence: Option<String>, // nucleotide target sequence (translated search)
    pub posterior_prob_line: Option<String>, // posterior probability annotation

    pub alignment_length: usize, // length of alignment strings

    pub hmm_name: String,
    pub hmm_accession: String,
    pub hmm_description: String,
    pub hmm_from: usize,
    pub hmm_to: usize,
    pub hmm_model_length: usize, // total model length

    pub sequence_name: String,
    pub sequence_accession: String,
    pub sequence_description: String,
    pub sequence_from: i64,
    pub sequence_to: i64,
    pub sequence_length: i64, // total sequence length

    pub serialized_size: usize, // for serialized size tracking
}

impl AliDisplay {
    /// Create an empty alignment display.
    pub fn new() -> Self {
        AliDisplay {
            reference_line: None,
            model_mask_line: None,
            consensus_structure_line: None,
            model: String::new(),
            match_line: String::new(),
            aligned_sequence: String::new(),
            nucleotide_sequence: None,
            posterior_prob_line: None,
            alignment_length: 0,
            hmm_name: String::new(),
            hmm_accession: String::new(),
            hmm_description: String::new(),
            hmm_from: 0,
            hmm_to: 0,
            hmm_model_length: 0,
            sequence_name: String::new(),
            sequence_accession: String::new(),
            sequence_description: String::new(),
            sequence_from: 0,
            sequence_to: 0,
            sequence_length: 0,
            serialized_size: 0,
        }
    }

    /// Clone the alignment display.
    pub fn clone_display(&self) -> Self {
        self.clone()
    }

    /// Create an alignment display from a trace, for domain `which` (0-indexed).
    ///
    /// Walks the trace from B to E for the specified domain, building model/sequence
    /// alignment strings, identity line, and optional annotation lines.
    ///
    /// Port of p7_alidisplay_Create() in C HMMER.
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        tr: &Trace,
        domain: &TraceDomain,
        hmm: &Hmm,
        dsq: &[u8],
        sq_name: &str,
        sq_acc: &str,
        sq_desc: &str,
        sq_l: i64,
    ) -> Option<Self> {
        let z1 = domain.trace_start as usize; // B state position in trace
        let z2 = domain.trace_end as usize; // E state position in trace
        let n = count_alignment_columns(tr, z1, z2);
        let has_rf = hmm.reference_annotation.is_some();
        let has_cs = hmm.consensus_structure.is_some();
        let mut build = AlignmentBuildState::new(n, has_rf, has_cs, tr.has_posterior_probs);

        for z in (z1 + 1)..z2 {
            let step = &tr.steps[z];
            build.process_step(step, hmm, dsq);
        }

        let AlignmentBuildState {
            model_str,
            mline_str,
            aseq_str,
            ppline,
            rfline,
            csline,
            hmmfrom,
            hmmto,
            sqfrom,
            sqto,
        } = build;

        Some(AliDisplay {
            reference_line: rfline,
            model_mask_line: None,
            consensus_structure_line: csline,
            model: model_str,
            match_line: mline_str,
            aligned_sequence: aseq_str,
            nucleotide_sequence: None,
            posterior_prob_line: ppline,
            alignment_length: n,
            hmm_name: hmm.name.clone(),
            hmm_accession: hmm.accession.clone().unwrap_or_default(),
            hmm_description: hmm.description.clone().unwrap_or_default(),
            hmm_from: hmmfrom,
            hmm_to: hmmto,
            hmm_model_length: hmm.num_nodes,
            sequence_name: sq_name.to_string(),
            sequence_accession: sq_acc.to_string(),
            sequence_description: sq_desc.to_string(),
            sequence_from: sqfrom,
            sequence_to: sqto,
            sequence_length: sq_l,
            serialized_size: 0,
        })
    }

    /// Return the approximate size of this display in bytes.
    pub fn sizeof_approx(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.model.len()
            + self.match_line.len()
            + self.aligned_sequence.len()
            + self.reference_line.as_ref().map_or(0, |s| s.len())
            + self
                .consensus_structure_line
                .as_ref()
                .map_or(0, |s| s.len())
            + self.posterior_prob_line.as_ref().map_or(0, |s| s.len())
            + self.nucleotide_sequence.as_ref().map_or(0, |s| s.len())
            + self.hmm_name.len()
            + self.sequence_name.len()
    }

    /// Print the alignment display in HMMER standard format.
    pub fn print(&self, show_accessions: bool) -> String {
        let mut out = String::new();

        let qname = if show_accessions && !self.hmm_accession.is_empty() {
            &self.hmm_accession
        } else {
            &self.hmm_name
        };
        let tname = if show_accessions && !self.sequence_accession.is_empty() {
            &self.sequence_accession
        } else {
            &self.sequence_name
        };

        let name_width = std::cmp::max(qname.len(), tname.len());
        let _ali_width = self.alignment_length;

        // Optional annotation lines
        if let Some(rf) = &self.reference_line {
            out.push_str(&format!(
                "  {:width$} {} {}\n",
                "",
                "RF",
                rf,
                width = name_width
            ));
        }
        if let Some(cs) = &self.consensus_structure_line {
            out.push_str(&format!(
                "  {:width$} {} {}\n",
                "",
                "CS",
                cs,
                width = name_width
            ));
        }

        // Query line
        out.push_str(&format!(
            "  {:width$} {:>5} {} {:<5}\n",
            qname,
            self.hmm_from,
            self.model,
            self.hmm_to,
            width = name_width
        ));

        // Match line
        out.push_str(&format!(
            "  {:width$}       {}\n",
            "",
            self.match_line,
            width = name_width
        ));

        // Target line
        out.push_str(&format!(
            "  {:width$} {:>5} {} {:<5}\n",
            tname,
            self.sequence_from,
            self.aligned_sequence,
            self.sequence_to,
            width = name_width
        ));

        // Posterior probability line
        if let Some(pp) = &self.posterior_prob_line {
            out.push_str(&format!(
                "  {:width$}       {}\n",
                "",
                pp,
                width = name_width
            ));
        }

        out
    }

    /// Compare two alignment displays for equality.
    pub fn compare(&self, other: &AliDisplay) -> bool {
        self.model == other.model
            && self.match_line == other.match_line
            && self.aligned_sequence == other.aligned_sequence
            && self.hmm_from == other.hmm_from
            && self.hmm_to == other.hmm_to
            && self.sequence_from == other.sequence_from
            && self.sequence_to == other.sequence_to
    }
}

impl Default for AliDisplay {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert posterior probability to a display character.
/// 0.0-0.05 → '0', 0.05-0.15 → '1', ..., 0.95-1.0 → '*'
fn pp_to_char(pp: f32) -> char {
    if pp >= 0.95 {
        '*'
    } else {
        let d = (pp * 10.0).floor() as u8;
        (b'0' + d.min(9)) as char
    }
}

fn count_alignment_columns(tr: &Trace, z1: usize, z2: usize) -> usize {
    let mut n = 0usize;
    for z in (z1 + 1)..z2 {
        match tr.steps[z].state {
            TraceStateType::Match | TraceStateType::Delete | TraceStateType::Insert => n += 1,
            _ => {}
        }
    }
    n
}

#[derive(Debug)]
struct AlignmentBuildState {
    model_str: String,
    mline_str: String,
    aseq_str: String,
    ppline: Option<String>,
    rfline: Option<String>,
    csline: Option<String>,
    hmmfrom: usize,
    hmmto: usize,
    sqfrom: i64,
    sqto: i64,
}

impl AlignmentBuildState {
    fn new(n: usize, has_rf: bool, has_cs: bool, has_pp: bool) -> Self {
        Self {
            model_str: String::with_capacity(n),
            mline_str: String::with_capacity(n),
            aseq_str: String::with_capacity(n),
            ppline: has_pp.then(|| String::with_capacity(n)),
            rfline: has_rf.then(|| String::with_capacity(n)),
            csline: has_cs.then(|| String::with_capacity(n)),
            hmmfrom: 0,
            hmmto: 0,
            sqfrom: 0,
            sqto: 0,
        }
    }

    fn process_step(&mut self, step: &TraceStep, hmm: &Hmm, dsq: &[u8]) {
        match step.state {
            TraceStateType::Match => self.process_match(step, hmm, dsq),
            TraceStateType::Delete => self.process_delete(step, hmm),
            TraceStateType::Insert => self.process_insert(step, hmm, dsq),
            _ => {}
        }
    }

    fn process_match(&mut self, step: &TraceStep, hmm: &Hmm, dsq: &[u8]) {
        let k = step.node_index as usize;
        let i = step.sequence_position;

        if self.hmmfrom == 0 {
            self.hmmfrom = k;
        }
        self.hmmto = k;
        if self.sqfrom == 0 {
            self.sqfrom = i as i64;
        }
        self.sqto = i as i64;

        let model_c = consensus_char(hmm.consensus.as_deref(), k);
        self.model_str.push(model_c);

        let res_c = residue_char(dsq, i, &hmm.alphabet.symbols).to_ascii_uppercase();
        self.aseq_str.push(res_c);

        let mc = model_c.to_ascii_uppercase();
        if mc == res_c {
            self.mline_str.push(mc);
        } else if is_similar(mc, res_c) {
            self.mline_str.push('+');
        } else {
            self.mline_str.push(' ');
        }

        if let Some(rf) = &mut self.rfline {
            rf.push(annotation_char(hmm.reference_annotation.as_deref(), k));
        }
        if let Some(cs) = &mut self.csline {
            cs.push(annotation_char(hmm.consensus_structure.as_deref(), k));
        }
        if let Some(pp_str) = &mut self.ppline {
            pp_str.push(pp_to_char(step.posterior_prob));
        }
    }

    fn process_delete(&mut self, step: &TraceStep, hmm: &Hmm) {
        let k = step.node_index as usize;

        if self.hmmfrom == 0 {
            self.hmmfrom = k;
        }
        self.hmmto = k;

        self.model_str
            .push(consensus_char(hmm.consensus.as_deref(), k));
        self.aseq_str.push('-');
        self.mline_str.push(' ');

        if let Some(rf) = &mut self.rfline {
            rf.push(annotation_char(hmm.reference_annotation.as_deref(), k));
        }
        if let Some(cs) = &mut self.csline {
            cs.push(annotation_char(hmm.consensus_structure.as_deref(), k));
        }
        if let Some(pp_str) = &mut self.ppline {
            pp_str.push('.');
        }
    }

    fn process_insert(&mut self, step: &TraceStep, hmm: &Hmm, dsq: &[u8]) {
        let i = step.sequence_position;

        if self.sqfrom == 0 {
            self.sqfrom = i as i64;
        }
        self.sqto = i as i64;

        self.model_str.push('.');
        self.aseq_str
            .push(residue_char(dsq, i, &hmm.alphabet.symbols).to_ascii_lowercase());
        self.mline_str.push(' ');

        if let Some(rf) = &mut self.rfline {
            rf.push('.');
        }
        if let Some(cs) = &mut self.csline {
            cs.push('.');
        }
        if let Some(pp_str) = &mut self.ppline {
            pp_str.push(pp_to_char(step.posterior_prob));
        }
    }
}

fn consensus_char(consensus: Option<&[u8]>, k: usize) -> char {
    consensus
        .and_then(|cons| cons.get(k).copied())
        .map(char::from)
        .unwrap_or('?')
}

fn annotation_char(annotation: Option<&[u8]>, k: usize) -> char {
    annotation
        .and_then(|ann| ann.get(k).copied())
        .map(char::from)
        .unwrap_or('.')
}

fn residue_char(dsq: &[u8], i: i32, symbols: &[u8]) -> char {
    let xi = dsq[i as usize - 1] as usize;
    symbols.get(xi).copied().map(char::from).unwrap_or('?')
}

/// Check if two amino acid residues are similar (BLOSUM62 positive score).
fn is_similar(a: char, b: char) -> bool {
    let a = a as u8;
    let b = b as u8;
    for group in SIMILARITY_GROUPS {
        if group.contains(&a) && group.contains(&b) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create() {
        let ad = AliDisplay::new();
        assert_eq!(ad.alignment_length, 0);
        assert!(ad.reference_line.is_none());
    }

    #[test]
    fn test_clone() {
        let mut ad = AliDisplay::new();
        ad.model = "ACDEF".to_string();
        ad.hmm_from = 1;
        ad.hmm_to = 5;
        let ad2 = ad.clone_display();
        assert!(ad.compare(&ad2));
    }
}
