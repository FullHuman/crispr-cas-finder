//! HMMER3 ASCII model reading and writing.
//!
//! The reader handles the versioned annotation layouts used by HMMER3/a-f.
//! The writer always emits the current HMMER3/f layout.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::hmmer_core::alphabet::{Alphabet, AlphabetKind};
use crate::hmmer_core::config::{EV_PARAM_UNSET, EvParam, HMM_NUM_TRANSITIONS, NUM_EV_PARAMS};
use crate::hmmer_core::errors::HmmerError;
use crate::hmmer_core::hmm::{Cutoffs, EvParams, Hmm};

const HMMER3F_TAG: &str = "HMMER3/f";
const HMMER_ZERO_PROB_SENTINEL: f32 = 99_999.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AsciiFormat {
    ThreeA,
    ThreeB,
    ThreeC,
    ThreeD,
    ThreeE,
    ThreeF,
}

impl AsciiFormat {
    fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "HMMER3/a" => Some(Self::ThreeA),
            "HMMER3/b" => Some(Self::ThreeB),
            "HMMER3/c" => Some(Self::ThreeC),
            "HMMER3/d" => Some(Self::ThreeD),
            "HMMER3/e" => Some(Self::ThreeE),
            HMMER3F_TAG => Some(Self::ThreeF),
            _ => None,
        }
    }

    fn annotation_count(self) -> usize {
        match self {
            Self::ThreeA | Self::ThreeB | Self::ThreeC | Self::ThreeD => 3,
            Self::ThreeE => 4,
            Self::ThreeF => 5,
        }
    }
}

/// Reader for one or more HMMER3 ASCII models in a file.
#[derive(Debug)]
pub struct HmmFile {
    pub filename: String,
    reader: Option<BufReader<std::fs::File>>,
}

#[derive(Default)]
struct Header {
    name: Option<String>,
    accession: Option<String>,
    description: Option<String>,
    model_length: Option<usize>,
    max_length: Option<usize>,
    alphabet_kind: Option<AlphabetKind>,
    reference_annotation: bool,
    model_mask: bool,
    consensus: bool,
    consensus_structure: bool,
    map: bool,
    command_log: Vec<String>,
    num_sequences: Option<u32>,
    effective_num_sequences: Option<f32>,
    checksum: Option<u32>,
    ev_params: [Option<f32>; NUM_EV_PARAMS],
    cutoffs: Cutoffs,
}

impl HmmFile {
    /// Open an HMMER3 ASCII file for reading.
    ///
    /// If the direct path does not exist, `search_path` is interpreted as an
    /// OS-native list of directories (like `PATH`).
    pub fn open(filename: &str, search_path: Option<&str>) -> Result<Self, HmmerError> {
        let path = Path::new(filename);
        let file = if path.exists() {
            std::fs::File::open(path)?
        } else if let Some(search_path) = search_path {
            let mut found = None;
            for directory in std::env::split_paths(search_path) {
                let candidate = directory.join(filename);
                if candidate.exists() {
                    found = Some(std::fs::File::open(&candidate)?);
                    break;
                }
            }
            found.ok_or_else(|| HmmerError::NotFound(format!("HMM file {filename} not found")))?
        } else {
            return Err(HmmerError::NotFound(format!(
                "HMM file {filename} not found"
            )));
        };

        Ok(Self {
            filename: filename.to_string(),
            reader: Some(BufReader::new(file)),
        })
    }

    /// Read the next model from the file.
    pub fn read(&mut self) -> Result<(Alphabet, Hmm), HmmerError> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| HmmerError::Internal("HMM file reader is unavailable".into()))?;
        let format = read_format_header(reader)?;
        Self::read_ascii(reader, format)
    }

    /// Parse a model after its HMMER3 format line has been consumed.
    fn read_ascii<R: BufRead>(
        reader: &mut R,
        format: AsciiFormat,
    ) -> Result<(Alphabet, Hmm), HmmerError> {
        let (header, symbol_header) = read_header(reader, format)?;
        let model_length = header
            .model_length
            .ok_or_else(|| HmmerError::Format("missing LENG field".into()))?;
        if model_length == 0 {
            return Err(HmmerError::Format("LENG must be greater than zero".into()));
        }
        let alphabet_kind = header
            .alphabet_kind
            .ok_or_else(|| HmmerError::Format("missing ALPH field".into()))?;
        let alphabet = alphabet_for(alphabet_kind);
        validate_symbol_header(&symbol_header, &alphabet)?;
        validate_transition_header(&read_required_line(reader, "transition header")?)?;

        let mut hmm = Hmm::new(model_length, &alphabet);
        hmm.name = header
            .name
            .clone()
            .ok_or_else(|| HmmerError::Format("missing NAME field".into()))?;
        hmm.accession = header.accession.clone();
        hmm.description = header.description.clone();
        hmm.max_length = header.max_length;
        hmm.command_log = (!header.command_log.is_empty()).then(|| header.command_log.join("\n"));
        hmm.num_sequences = header.num_sequences;
        hmm.effective_num_seq_float = header.effective_num_sequences;
        hmm.effective_num_seq = header.effective_num_sequences.map_or(-1.0, f64::from);
        hmm.checksum = header.checksum;
        hmm.cutoffs = header.cutoffs;
        hmm.ev_params = build_ev_params(header.ev_params)?;
        hmm.map = header.map.then(|| vec![0; model_length + 1]);
        hmm.consensus = header.consensus.then(|| vec![b' '; model_length + 2]);
        hmm.reference_annotation = header
            .reference_annotation
            .then(|| vec![b' '; model_length + 2]);
        hmm.model_mask = header.model_mask.then(|| vec![b' '; model_length + 2]);
        hmm.consensus_structure = header
            .consensus_structure
            .then(|| vec![b' '; model_length + 2]);

        let mut line = read_required_line(reader, "node 0 insert emissions or COMPO")?;
        if line.split_whitespace().next() == Some("COMPO") {
            let tokens: Vec<_> = line.split_whitespace().skip(1).collect();
            hmm.model_composition = Some(parse_probability_row(
                &tokens,
                alphabet.canonical_size,
                "COMPO",
            )?);
            line = read_required_line(reader, "node 0 insert emissions")?;
        }
        parse_probability_row_into(
            &line.split_whitespace().collect::<Vec<_>>(),
            hmm.insert_emissions_mut(0),
            "node 0 insert emissions",
        )?;

        let line = read_required_line(reader, "node 0 transitions")?;
        parse_probability_row_into(
            &line.split_whitespace().collect::<Vec<_>>(),
            hmm.transitions_mut(0),
            "node 0 transitions",
        )?;

        for node in 1..=model_length {
            parse_match_row(reader, &mut hmm, &header, format, node)?;

            let line = read_required_line(reader, &format!("node {node} insert emissions"))?;
            parse_probability_row_into(
                &line.split_whitespace().collect::<Vec<_>>(),
                hmm.insert_emissions_mut(node),
                &format!("node {node} insert emissions"),
            )?;

            let line = read_required_line(reader, &format!("node {node} transitions"))?;
            parse_probability_row_into(
                &line.split_whitespace().collect::<Vec<_>>(),
                hmm.transitions_mut(node),
                &format!("node {node} transitions"),
            )?;
        }

        if matches!(
            format,
            AsciiFormat::ThreeA | AsciiFormat::ThreeB | AsciiFormat::ThreeC | AsciiFormat::ThreeD
        ) {
            hmm.consensus = Some(derive_consensus(&hmm, &alphabet));
        }

        loop {
            let line = read_required_line(reader, "model terminator")?;
            if line.trim().is_empty() {
                continue;
            }
            if line.trim() != "//" {
                return Err(HmmerError::Format(format!(
                    "expected model terminator `//`, found `{}`",
                    line.trim()
                )));
            }
            break;
        }

        Ok((alphabet, hmm))
    }

    /// Write a complete model in HMMER3/f ASCII format.
    pub fn write_ascii<W: Write>(
        writer: &mut W,
        hmm: &Hmm,
        alphabet: &Alphabet,
    ) -> Result<(), std::io::Error> {
        validate_model_for_writing(hmm, alphabet)?;

        writeln!(writer, "HMMER3/f [hmmer-rust]")?;
        writeln!(writer, "NAME  {}", hmm.name)?;
        if let Some(accession) = &hmm.accession {
            writeln!(writer, "ACC   {accession}")?;
        }
        if let Some(description) = &hmm.description {
            writeln!(writer, "DESC  {description}")?;
        }
        writeln!(writer, "LENG  {}", hmm.num_nodes)?;
        if let Some(max_length) = hmm.max_length {
            writeln!(writer, "MAXL  {max_length}")?;
        }
        writeln!(writer, "ALPH  {}", alphabet_name(alphabet.kind))?;
        writeln!(
            writer,
            "RF    {}",
            yes_no(hmm.reference_annotation.is_some())
        )?;
        writeln!(writer, "MM    {}", yes_no(hmm.model_mask.is_some()))?;
        writeln!(writer, "CONS  {}", yes_no(hmm.consensus.is_some()))?;
        writeln!(
            writer,
            "CS    {}",
            yes_no(hmm.consensus_structure.is_some())
        )?;
        writeln!(writer, "MAP   {}", yes_no(hmm.map.is_some()))?;
        if let Some(command_log) = &hmm.command_log {
            for command in command_log.lines() {
                writeln!(writer, "COM   {command}")?;
            }
        }
        if let Some(num_sequences) = hmm.num_sequences {
            writeln!(writer, "NSEQ  {num_sequences}")?;
        }
        if let Some(effective_num_sequences) = hmm.effective_num_seq_float {
            writeln!(writer, "EFFN  {effective_num_sequences:.6}")?;
        }
        if let Some(checksum) = hmm.checksum {
            writeln!(writer, "CKSUM {checksum}")?;
        }
        write_cutoff(writer, "GA", hmm.cutoffs.gathering)?;
        write_cutoff(writer, "TC", hmm.cutoffs.trusted)?;
        write_cutoff(writer, "NC", hmm.cutoffs.noise)?;
        if let Some(params) = hmm.ev_params {
            writeln!(
                writer,
                "STATS LOCAL MSV      {:.6} {:.6}",
                params.msv_mu, params.msv_lambda
            )?;
            writeln!(
                writer,
                "STATS LOCAL VITERBI  {:.6} {:.6}",
                params.viterbi_mu, params.viterbi_lambda
            )?;
            writeln!(
                writer,
                "STATS LOCAL FORWARD  {:.6} {:.6}",
                params.forward_tau, params.forward_lambda
            )?;
        }

        write!(writer, "HMM         ")?;
        for symbol in &alphabet.symbols[..alphabet.canonical_size] {
            write!(writer, " {:>8}", char::from(*symbol))?;
        }
        writeln!(writer)?;
        writeln!(
            writer,
            "            m->m     m->i     m->d     i->m     i->i     d->m     d->d"
        )?;

        if let Some(composition) = &hmm.model_composition {
            write!(writer, "  COMPO")?;
            write_probability_row(writer, composition)?;
        }
        write!(writer, "       ")?;
        write_probability_row(writer, hmm.insert_emissions(0))?;
        write!(writer, "       ")?;
        write_probability_row(writer, hmm.transitions(0))?;

        for node in 1..=hmm.num_nodes {
            write!(writer, "{node:7}")?;
            for probability in hmm.match_emissions(node) {
                write_probability(writer, *probability)?;
            }
            let map = hmm
                .map
                .as_ref()
                .and_then(|values| values.get(node))
                .map_or_else(|| "-".to_string(), i32::to_string);
            write!(
                writer,
                " {map:>6} {:>4} {:>4} {:>4} {:>4}",
                annotation_at(hmm.consensus.as_deref(), node),
                annotation_at(hmm.reference_annotation.as_deref(), node),
                annotation_at(hmm.model_mask.as_deref(), node),
                annotation_at(hmm.consensus_structure.as_deref(), node),
            )?;
            writeln!(writer)?;

            write!(writer, "       ")?;
            write_probability_row(writer, hmm.insert_emissions(node))?;
            write!(writer, "       ")?;
            write_probability_row(writer, hmm.transitions(node))?;
        }
        writeln!(writer, "//")?;
        Ok(())
    }
}

/// Parse a single HMMER3 ASCII model from a buffered reader.
pub fn read_hmm<R: BufRead>(reader: &mut R) -> Result<(Alphabet, Hmm), HmmerError> {
    let format = read_format_header(reader)?;
    HmmFile::read_ascii(reader, format)
}

fn read_format_header<R: BufRead>(reader: &mut R) -> Result<AsciiFormat, HmmerError> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(HmmerError::Eof);
        }
        if !line.trim().is_empty() {
            break;
        }
    }
    let tag = line.split_whitespace().next().unwrap_or_default();
    AsciiFormat::from_tag(tag).ok_or_else(|| {
        HmmerError::Format(format!(
            "unsupported HMM format `{tag}`; expected HMMER3 ASCII"
        ))
    })
}

fn read_header<R: BufRead>(
    reader: &mut R,
    format: AsciiFormat,
) -> Result<(Header, String), HmmerError> {
    let mut header = Header::default();
    loop {
        let line = read_required_line(reader, "HMM header")?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut fields = trimmed.splitn(2, char::is_whitespace);
        let tag = fields.next().unwrap_or_default();
        if tag == "HMM" {
            return Ok((header, line));
        }
        let value = fields.next().map(str::trim).unwrap_or_default();
        match tag {
            "NAME" => header.name = Some(required_text(tag, value)?.to_string()),
            "ACC" => header.accession = Some(required_text(tag, value)?.to_string()),
            "DESC" => header.description = Some(required_text(tag, value)?.to_string()),
            "LENG" => header.model_length = Some(parse_value(tag, value)?),
            "MAXL" => header.max_length = Some(parse_value(tag, value)?),
            "ALPH" => {
                header.alphabet_kind = Some(match value {
                    "amino" => AlphabetKind::Amino,
                    "DNA" => AlphabetKind::Dna,
                    "RNA" => AlphabetKind::Rna,
                    _ => {
                        return Err(HmmerError::Format(format!("invalid ALPH value `{value}`")));
                    }
                });
            }
            "RF" => header.reference_annotation = parse_yes_no(tag, value)?,
            "MM" => header.model_mask = parse_yes_no(tag, value)?,
            "CONS" => header.consensus = parse_yes_no(tag, value)?,
            "CS" => header.consensus_structure = parse_yes_no(tag, value)?,
            "MAP" => header.map = parse_yes_no(tag, value)?,
            "COM" => header.command_log.push(value.to_string()),
            "NSEQ" => header.num_sequences = Some(parse_value(tag, value)?),
            "EFFN" => {
                let parsed = parse_finite_f32(tag, value)?;
                if parsed < 0.0 {
                    return Err(HmmerError::Format("EFFN cannot be negative".into()));
                }
                header.effective_num_sequences = Some(parsed);
            }
            "CKSUM" => header.checksum = Some(parse_value(tag, value)?),
            "GA" => header.cutoffs.gathering = Some(parse_cutoff(tag, value)?),
            "TC" => header.cutoffs.trusted = Some(parse_cutoff(tag, value)?),
            "NC" => header.cutoffs.noise = Some(parse_cutoff(tag, value)?),
            "STATS" => parse_stats(format, value, &mut header.ev_params)?,
            _ => {}
        }
    }
}

fn parse_match_row<R: BufRead>(
    reader: &mut R,
    hmm: &mut Hmm,
    header: &Header,
    format: AsciiFormat,
    node: usize,
) -> Result<(), HmmerError> {
    let context = format!("node {node} match emissions");
    let line = read_required_line(reader, &context)?;
    let tokens: Vec<_> = line.split_whitespace().collect();
    let expected = 1 + hmm.alphabet.canonical_size + format.annotation_count();
    if tokens.len() != expected {
        return Err(HmmerError::Format(format!(
            "{context}: expected {expected} columns, found {}",
            tokens.len()
        )));
    }
    let parsed_node: usize = parse_value("model node", tokens[0])?;
    if parsed_node != node {
        return Err(HmmerError::Format(format!(
            "expected model node {node}, found {parsed_node}"
        )));
    }
    let probability_end = 1 + hmm.alphabet.canonical_size;
    parse_probability_row_into(
        &tokens[1..probability_end],
        hmm.match_emissions_mut(node),
        &context,
    )?;

    let annotations = &tokens[probability_end..];
    if header.map {
        let map = hmm.map.as_mut().expect("MAP allocation follows header");
        map[node] = parse_value("MAP annotation", annotations[0])?;
    }
    match format {
        AsciiFormat::ThreeA | AsciiFormat::ThreeB | AsciiFormat::ThreeC | AsciiFormat::ThreeD => {
            set_annotation(
                hmm.reference_annotation.as_mut(),
                header.reference_annotation,
                node,
                annotations[1],
                "RF",
            )?;
            set_annotation(
                hmm.consensus_structure.as_mut(),
                header.consensus_structure,
                node,
                annotations[2],
                "CS",
            )?;
        }
        AsciiFormat::ThreeE => {
            set_annotation(
                hmm.consensus.as_mut(),
                header.consensus,
                node,
                annotations[1],
                "CONS",
            )?;
            set_annotation(
                hmm.reference_annotation.as_mut(),
                header.reference_annotation,
                node,
                annotations[2],
                "RF",
            )?;
            set_annotation(
                hmm.consensus_structure.as_mut(),
                header.consensus_structure,
                node,
                annotations[3],
                "CS",
            )?;
        }
        AsciiFormat::ThreeF => {
            set_annotation(
                hmm.consensus.as_mut(),
                header.consensus,
                node,
                annotations[1],
                "CONS",
            )?;
            set_annotation(
                hmm.reference_annotation.as_mut(),
                header.reference_annotation,
                node,
                annotations[2],
                "RF",
            )?;
            set_annotation(
                hmm.model_mask.as_mut(),
                header.model_mask,
                node,
                annotations[3],
                "MM",
            )?;
            set_annotation(
                hmm.consensus_structure.as_mut(),
                header.consensus_structure,
                node,
                annotations[4],
                "CS",
            )?;
        }
    }
    Ok(())
}

fn read_required_line<R: BufRead>(reader: &mut R, context: &str) -> Result<String, HmmerError> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Err(HmmerError::Format(format!(
            "premature EOF while reading {context}"
        )));
    }
    Ok(line)
}

fn validate_symbol_header(line: &str, alphabet: &Alphabet) -> Result<(), HmmerError> {
    let tokens: Vec<_> = line.split_whitespace().collect();
    if tokens.len() != alphabet.canonical_size + 1 || tokens.first() != Some(&"HMM") {
        return Err(HmmerError::Format(format!(
            "HMM symbol header must contain {} canonical symbols",
            alphabet.canonical_size
        )));
    }
    for (token, expected) in tokens[1..]
        .iter()
        .zip(&alphabet.symbols[..alphabet.canonical_size])
    {
        if token.as_bytes() != [*expected] {
            return Err(HmmerError::Format(format!(
                "unexpected symbol `{token}` in HMM header"
            )));
        }
    }
    Ok(())
}

fn validate_transition_header(line: &str) -> Result<(), HmmerError> {
    const EXPECTED: [&str; HMM_NUM_TRANSITIONS] =
        ["m->m", "m->i", "m->d", "i->m", "i->i", "d->m", "d->d"];
    let tokens: Vec<_> = line.split_whitespace().collect();
    if tokens != EXPECTED {
        return Err(HmmerError::Format(
            "invalid HMM transition header".to_string(),
        ));
    }
    Ok(())
}

fn parse_probability_row_into(
    tokens: &[&str],
    destination: &mut [f32],
    context: &str,
) -> Result<(), HmmerError> {
    let parsed = parse_probability_row(tokens, destination.len(), context)?;
    destination.copy_from_slice(&parsed);
    Ok(())
}

fn parse_probability_row(
    tokens: &[&str],
    expected: usize,
    context: &str,
) -> Result<Vec<f32>, HmmerError> {
    if tokens.len() != expected {
        return Err(HmmerError::Format(format!(
            "{context}: expected {expected} probabilities, found {}",
            tokens.len()
        )));
    }
    tokens
        .iter()
        .map(|token| parse_probability(token, context))
        .collect()
}

fn parse_probability(token: &str, context: &str) -> Result<f32, HmmerError> {
    if token == "*" {
        return Ok(0.0);
    }
    let score = token.parse::<f32>().map_err(|_| {
        HmmerError::Format(format!("{context}: invalid probability score `{token}`"))
    })?;
    if !score.is_finite() || score < 0.0 {
        return Err(HmmerError::Format(format!(
            "{context}: invalid probability score `{token}`"
        )));
    }
    Ok(if score >= HMMER_ZERO_PROB_SENTINEL {
        0.0
    } else {
        (-score).exp()
    })
}

fn required_text<'a>(tag: &str, value: &'a str) -> Result<&'a str, HmmerError> {
    if value.is_empty() {
        Err(HmmerError::Format(format!("missing value for {tag}")))
    } else {
        Ok(value)
    }
}

fn parse_value<T>(tag: &str, value: &str) -> Result<T, HmmerError>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| HmmerError::Format(format!("invalid {tag} value `{value}`")))
}

fn parse_finite_f32(tag: &str, value: &str) -> Result<f32, HmmerError> {
    let parsed: f32 = parse_value(tag, value.trim_end_matches(';'))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(HmmerError::Format(format!("invalid {tag} value `{value}`")))
    }
}

fn parse_yes_no(tag: &str, value: &str) -> Result<bool, HmmerError> {
    match value {
        "yes" => Ok(true),
        "no" => Ok(false),
        _ => Err(HmmerError::Format(format!(
            "invalid {tag} value `{value}`; expected yes or no"
        ))),
    }
}

fn parse_cutoff(tag: &str, value: &str) -> Result<(f32, f32), HmmerError> {
    let tokens: Vec<_> = value.split_whitespace().collect();
    if !(1..=2).contains(&tokens.len()) {
        return Err(HmmerError::Format(format!(
            "{tag} requires one or two score thresholds"
        )));
    }
    let sequence = parse_finite_f32(tag, tokens[0])?;
    let domain = if tokens.len() == 2 {
        parse_finite_f32(tag, tokens[1])?
    } else {
        sequence
    };
    Ok((sequence, domain))
}

fn parse_stats(
    format: AsciiFormat,
    value: &str,
    params: &mut [Option<f32>; NUM_EV_PARAMS],
) -> Result<(), HmmerError> {
    let tokens: Vec<_> = value.split_whitespace().collect();
    if tokens.first() != Some(&"LOCAL") {
        return Err(HmmerError::Format(format!("invalid STATS line `{value}`")));
    }

    if format == AsciiFormat::ThreeA {
        if tokens.len() != 3 {
            return Err(HmmerError::Format(format!(
                "invalid HMMER3/a STATS line `{value}`"
            )));
        }
        let parsed = parse_finite_f32("STATS", tokens[2])?;
        match tokens[1] {
            "VLAMBDA" => {
                params[EvParam::MsvLambda.idx()] = Some(parsed);
                params[EvParam::ViterbiLambda.idx()] = Some(parsed);
                params[EvParam::ForwardLambda.idx()] = Some(parsed);
            }
            "VMU" => {
                params[EvParam::MsvMu.idx()] = Some(parsed);
                params[EvParam::ViterbiMu.idx()] = Some(parsed);
            }
            "FTAU" => params[EvParam::ForwardTau.idx()] = Some(parsed),
            _ => {
                return Err(HmmerError::Format(format!(
                    "unknown HMMER3/a STATS stage `{}`",
                    tokens[1]
                )));
            }
        }
        return Ok(());
    }

    if tokens.len() != 4 {
        return Err(HmmerError::Format(format!("invalid STATS line `{value}`")));
    }
    let first = parse_finite_f32("STATS", tokens[2])?;
    let second = parse_finite_f32("STATS", tokens[3])?;
    match tokens[1] {
        "MSV" => {
            params[EvParam::MsvMu.idx()] = Some(first);
            params[EvParam::MsvLambda.idx()] = Some(second);
        }
        "VITERBI" => {
            params[EvParam::ViterbiMu.idx()] = Some(first);
            params[EvParam::ViterbiLambda.idx()] = Some(second);
        }
        "FORWARD" => {
            params[EvParam::ForwardTau.idx()] = Some(first);
            params[EvParam::ForwardLambda.idx()] = Some(second);
        }
        _ => {
            return Err(HmmerError::Format(format!(
                "unknown STATS stage `{}`",
                tokens[1]
            )));
        }
    }
    Ok(())
}

fn build_ev_params(params: [Option<f32>; NUM_EV_PARAMS]) -> Result<Option<EvParams>, HmmerError> {
    if params.iter().all(Option::is_none) {
        return Ok(None);
    }
    if params.iter().any(Option::is_none) {
        return Err(HmmerError::Format(
            "STATS must define MSV, VITERBI, and FORWARD".into(),
        ));
    }
    let mut values = [EV_PARAM_UNSET; NUM_EV_PARAMS];
    for (destination, source) in values.iter_mut().zip(params) {
        *destination = source.expect("checked above");
    }
    Ok(Some(EvParams::from_array(&values)))
}

fn set_annotation(
    destination: Option<&mut Vec<u8>>,
    enabled: bool,
    node: usize,
    token: &str,
    tag: &str,
) -> Result<(), HmmerError> {
    if !enabled {
        return Ok(());
    }
    if token.len() != 1 || !token.is_ascii() {
        return Err(HmmerError::Format(format!(
            "invalid {tag} annotation `{token}` at node {node}"
        )));
    }
    destination.expect("annotation allocation follows header")[node] = token.as_bytes()[0];
    Ok(())
}

fn derive_consensus(hmm: &Hmm, alphabet: &Alphabet) -> Vec<u8> {
    let mut consensus = vec![b' '; hmm.num_nodes + 2];
    for (node, destination) in consensus
        .iter_mut()
        .enumerate()
        .take(hmm.num_nodes + 1)
        .skip(1)
    {
        let emissions = hmm.match_emissions(node);
        let (residue, probability) = emissions
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("canonical alphabets are nonempty");
        let symbol = alphabet.symbols[residue];
        *destination = if probability > 0.5 {
            symbol
        } else {
            symbol.to_ascii_lowercase()
        };
    }
    consensus
}

fn alphabet_for(kind: AlphabetKind) -> Alphabet {
    match kind {
        AlphabetKind::Amino => Alphabet::amino(),
        AlphabetKind::Dna => Alphabet::dna(),
        AlphabetKind::Rna => Alphabet::rna(),
    }
}

fn alphabet_name(kind: AlphabetKind) -> &'static str {
    match kind {
        AlphabetKind::Amino => "amino",
        AlphabetKind::Dna => "DNA",
        AlphabetKind::Rna => "RNA",
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn validate_model_for_writing(hmm: &Hmm, alphabet: &Alphabet) -> Result<(), std::io::Error> {
    if hmm.num_nodes == 0 {
        return Err(invalid_input("cannot write a zero-length HMM"));
    }
    if hmm.name.trim().is_empty() {
        return Err(invalid_input("cannot write an HMM without a NAME"));
    }
    if hmm.alphabet.kind != alphabet.kind || hmm.alphabet.canonical_size != alphabet.canonical_size
    {
        return Err(invalid_input("HMM and writer alphabets do not match"));
    }
    if let Some(composition) = &hmm.model_composition
        && composition.len() != alphabet.canonical_size
    {
        return Err(invalid_input(
            "model composition length does not match the alphabet",
        ));
    }
    Ok(())
}

fn invalid_input(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
}

fn write_cutoff<W: Write>(
    writer: &mut W,
    tag: &str,
    cutoff: Option<(f32, f32)>,
) -> Result<(), std::io::Error> {
    if let Some((sequence, domain)) = cutoff {
        writeln!(writer, "{tag:<5} {sequence:.2} {domain:.2}")?;
    }
    Ok(())
}

fn write_probability_row<W: Write>(
    writer: &mut W,
    probabilities: &[f32],
) -> Result<(), std::io::Error> {
    for probability in probabilities {
        write_probability(writer, *probability)?;
    }
    writeln!(writer)
}

fn write_probability<W: Write>(writer: &mut W, probability: f32) -> Result<(), std::io::Error> {
    if probability == 0.0 {
        write!(writer, " {:>8}", "*")
    } else if probability.is_finite() && probability > 0.0 && probability <= 1.0 {
        write!(writer, " {:8.5}", -probability.ln())
    } else {
        Err(invalid_input(
            "HMM probabilities must be finite and in the range 0..=1",
        ))
    }
}

fn annotation_at(annotation: Option<&[u8]>, node: usize) -> char {
    annotation
        .and_then(|values| values.get(node))
        .copied()
        .filter(u8::is_ascii_graphic)
        .map(char::from)
        .unwrap_or('-')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn rejects_non_hmmer3_format_headers() {
        for format in ["HMMER2.0", "not-an-hmm"] {
            let mut reader = Cursor::new(format.as_bytes());
            let error = read_hmm(&mut reader).expect_err("unsupported format must fail");
            assert!(error.to_string().contains("expected HMMER3 ASCII"));
        }
    }

    #[test]
    fn rejects_malformed_probability() {
        let error = parse_probability("not-a-number", "test row").expect_err("must fail");
        assert!(error.to_string().contains("not-a-number"));
    }
}
