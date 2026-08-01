// p7_hmmfile.rs - HMM file reading and writing
//
// Port of src/p7_hmmfile.c

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use hmmer_core::alphabet::{Alphabet, AlphabetKind};
use hmmer_core::errors::HmmerError;
use hmmer_core::hmm::{EvParams, Hmm};

/// HMM file format codes
pub const P7_HMMFILE_20: i32 = 0; // HMMER2.0
pub const P7_HMMFILE_3A: i32 = 1; // 3/a format
pub const P7_HMMFILE_3B: i32 = 2; // 3/b format
pub const P7_HMMFILE_3C: i32 = 3; // 3/c format
pub const P7_HMMFILE_3D: i32 = 4; // 3/d format
pub const P7_HMMFILE_3E: i32 = 5; // 3/e format
pub const P7_HMMFILE_3F: i32 = 6; // 3/f format (current)

/// Binary file magic numbers
pub const V3F_MAGIC: u32 = 0xe8ededba;
pub const V3E_MAGIC: u32 = 0xe8ededb9;
pub const V3D_MAGIC: u32 = 0xe8ededb8;
pub const V3C_MAGIC: u32 = 0xe8ededb7;
pub const V3B_MAGIC: u32 = 0xe8ededb6;
pub const V3A_MAGIC: u32 = 0xe8ededb5;

/// HMM file reader.
#[derive(Debug)]
pub struct HmmFile {
    pub filename: String,
    pub format: i32,
    pub do_gzip: bool,
    pub do_stdin: bool,
    pub newly_opened: bool,
    pub is_pressed: bool,
    reader: Option<BufReader<std::fs::File>>,
}

impl HmmFile {
    /// Open an HMM file for reading.
    pub fn open(filename: &str, env: Option<&str>) -> Result<Self, HmmerError> {
        // Try to open the file directly
        let path = Path::new(filename);
        let file = if path.exists() {
            std::fs::File::open(path)
                .map_err(|e| HmmerError::NotFound(format!("Failed to open {}: {}", filename, e)))?
        } else if let Some(envpath) = env {
            // Search in environment path
            let mut found = None;
            for dir in envpath.split(':') {
                let fullpath = Path::new(dir).join(filename);
                if fullpath.exists() {
                    found = Some(std::fs::File::open(&fullpath).map_err(|e| {
                        HmmerError::NotFound(format!("Failed to open {:?}: {}", fullpath, e))
                    })?);
                    break;
                }
            }
            found.ok_or_else(|| HmmerError::NotFound(format!("HMM file {} not found", filename)))?
        } else {
            return Err(HmmerError::NotFound(format!(
                "HMM file {} not found",
                filename
            )));
        };

        Ok(HmmFile {
            filename: filename.to_string(),
            format: P7_HMMFILE_3F,
            do_gzip: false,
            do_stdin: false,
            newly_opened: true,
            is_pressed: false,
            reader: Some(BufReader::new(file)),
        })
    }

    /// Read the next HMM from the file.
    pub fn read(&mut self) -> Result<(Alphabet, Hmm), HmmerError> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| HmmerError::Internal("No file reader".into()))?;

        // Peek at the first line to determine format
        let mut first_line = String::new();
        let bytes = reader
            .read_line(&mut first_line)
            .map_err(|e| HmmerError::Internal(format!("Read error: {}", e)))?;

        if bytes == 0 {
            return Err(HmmerError::Eof);
        }

        let first_line = first_line.trim();
        if first_line.starts_with("HMMER3/f") {
            self.format = P7_HMMFILE_3F;
        } else if first_line.starts_with("HMMER3/e") {
            self.format = P7_HMMFILE_3E;
        } else if first_line.starts_with("HMMER3/d") {
            self.format = P7_HMMFILE_3D;
        } else if first_line.starts_with("HMMER3/c") {
            self.format = P7_HMMFILE_3C;
        } else if first_line.starts_with("HMMER3/b") {
            self.format = P7_HMMFILE_3B;
        } else if first_line.starts_with("HMMER3/a") {
            self.format = P7_HMMFILE_3A;
        } else if first_line.starts_with("HMMER2.0") {
            return Err(HmmerError::Format(format!(
                "Unsupported HMM file format: {}",
                first_line
            )));
        } else {
            return Err(HmmerError::Format(format!(
                "Unrecognized HMM file format: {}",
                first_line
            )));
        }

        self.newly_opened = false;
        Self::read_3f(reader)
    }

    /// Parse a HMMER3/f format HMM.
    pub(crate) fn read_3f<R: BufRead>(reader: &mut R) -> Result<(Alphabet, Hmm), HmmerError> {
        use hmmer_core::config::*;

        let mut name = String::new();
        let mut accession = None;
        let mut description = None;
        let mut model_length: usize = 0;
        let mut alphabet_kind = AlphabetKind::Amino;
        let mut ev_params_array = [EV_PARAM_UNSET; NUM_EV_PARAMS];
        let mut flags: u32 = 0;
        let mut num_sequences: i32 = -1;
        let mut effective_num_sequences: f32 = -1.0;
        let mut checksum: u32 = 0;
        let mut map_flag = false;
        let mut max_length: i32 = -1;

        // Parse header fields using one reusable line allocation.
        let mut header_line = String::new();
        loop {
            header_line.clear();
            let bytes = reader
                .read_line(&mut header_line)
                .map_err(|e| HmmerError::Internal(format!("Read error: {}", e)))?;
            if bytes == 0 {
                return Err(HmmerError::Format("Premature EOF in HMM header".into()));
            }

            let trimmed = header_line.trim();
            if trimmed.starts_with("HMM ") || trimmed.starts_with("HMM\t") {
                header_line.clear();
                let _ = reader.read_line(&mut header_line);
                break;
            }

            let mut parts = trimmed.splitn(2, char::is_whitespace);
            let Some(tag) = parts.next() else {
                continue;
            };
            let Some(val) = parts.next() else {
                continue;
            };
            let val = val.trim();

            match tag {
                "NAME" => name = val.to_string(),
                "ACC" => {
                    accession = Some(val.to_string());
                    flags |= HMM_FLAG_ACC;
                }
                "DESC" => {
                    description = Some(val.to_string());
                    flags |= HMM_FLAG_DESC;
                }
                "LENG" => model_length = val.parse().unwrap_or(0),
                "MAXL" => max_length = val.parse().unwrap_or(-1),
                "ALPH" => {
                    alphabet_kind = match val {
                        "amino" => AlphabetKind::Amino,
                        "DNA" => AlphabetKind::Dna,
                        "RNA" => AlphabetKind::Rna,
                        _ => AlphabetKind::Amino,
                    };
                }
                "RF" if val == "yes" => {
                    flags |= HMM_FLAG_RF;
                }
                "MM" if val == "yes" => {
                    flags |= HMM_FLAG_MMASK;
                }
                "CONS" if val == "yes" => {
                    flags |= HMM_FLAG_CONS;
                }
                "CS" if val == "yes" => {
                    flags |= HMM_FLAG_CS;
                }
                "MAP" if val == "yes" => {
                    map_flag = true;
                    flags |= HMM_FLAG_MAP;
                }
                "NSEQ" => {
                    num_sequences = val.parse().unwrap_or(-1);
                }
                "EFFN" => {
                    effective_num_sequences = val.parse().unwrap_or(-1.0);
                }
                "CKSUM" => {
                    checksum = val.parse().unwrap_or(0);
                    flags |= HMM_FLAG_CHKSUM;
                }
                "STATS" => {
                    // "LOCAL MSV -9.9014 0.71847" or similar
                    let stat_tokens: Vec<&str> = val.split_whitespace().collect();
                    if stat_tokens.len() >= 3 && stat_tokens[0] == "LOCAL" {
                        let first_param: f32 = stat_tokens[2].parse().unwrap_or(0.0);
                        let second_param: f32 = if stat_tokens.len() > 3 {
                            stat_tokens[3].parse().unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        match stat_tokens[1] {
                            "MSV" => {
                                ev_params_array[EvParam::MsvMu.idx()] = first_param;
                                ev_params_array[EvParam::MsvLambda.idx()] = second_param;
                            }
                            "VITERBI" => {
                                ev_params_array[EvParam::ViterbiMu.idx()] = first_param;
                                ev_params_array[EvParam::ViterbiLambda.idx()] = second_param;
                            }
                            "FORWARD" => {
                                ev_params_array[EvParam::ForwardTau.idx()] = first_param;
                                ev_params_array[EvParam::ForwardLambda.idx()] = second_param;
                            }
                            _ => {}
                        }
                        flags |= HMM_FLAG_STATS;
                    }
                }
                "GA" => {
                    // "GA    25.00 25.00" - gathering thresholds
                    flags |= HMM_FLAG_GA;
                }
                "TC" => {
                    flags |= HMM_FLAG_TC;
                }
                "NC" => {
                    flags |= HMM_FLAG_NC;
                }
                _ => {} // Skip unknown tags
            }
        }

        if model_length == 0 {
            return Err(HmmerError::Format("Zero-length model".into()));
        }

        let alphabet = match alphabet_kind {
            AlphabetKind::Amino => Alphabet::amino(),
            AlphabetKind::Dna => Alphabet::dna(),
            AlphabetKind::Rna => Alphabet::rna(),
        };
        let mut hmm = Hmm::new(model_length, &alphabet);
        hmm.name = name;
        hmm.accession = accession;
        hmm.description = description;
        hmm.ev_params = if flags & HMM_FLAG_STATS != 0 {
            Some(EvParams::from_array(&ev_params_array))
        } else {
            None
        };
        hmm.num_sequences = if num_sequences >= 0 {
            Some(num_sequences as u32)
        } else {
            None
        };
        hmm.effective_num_seq_float = if effective_num_sequences >= 0.0 {
            Some(effective_num_sequences)
        } else {
            None
        };
        hmm.checksum = if flags & HMM_FLAG_CHKSUM != 0 {
            Some(checksum)
        } else {
            None
        };
        hmm.max_length = if max_length > 0 {
            Some(max_length as usize)
        } else {
            None
        };
        if map_flag {
            hmm.map = Some(vec![0; model_length + 1]);
        }

        // Parse model body
        // After "HMM" and transition header lines, we have:
        //   COMPO line (optional)
        //   Insert emissions for node 0
        //   Transitions for node 0
        //   Then for each node=1..M: match emiss, insert emiss, transitions

        let mut line = String::new();

        // Read first line - might be COMPO or insert emissions for node 0
        line.clear();
        let _ = reader.read_line(&mut line);
        if line.trim().starts_with("COMPO") {
            // Parse composition
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() > alphabet.canonical_size {
                let mut composition_values = vec![0.0f32; alphabet.canonical_size];
                for residue_index in 0..alphabet.canonical_size.min(MAX_CANONICAL_ALPHABET) {
                    if let Ok(value) = tokens[residue_index + 1].parse::<f32>() {
                        composition_values[residue_index] = if value >= 99999.0 {
                            0.0
                        } else {
                            (-value).exp()
                        };
                    }
                }
                hmm.model_composition = Some(composition_values);
                flags |= HMM_FLAG_COMPO;
            }
            // Read insert emissions for node 0
            line.clear();
            let _ = reader.read_line(&mut line);
        }
        // Current line = insert emissions for node 0 (parse but we mostly ignore node 0 inserts)
        for (tok, ins_val) in line
            .split_whitespace()
            .zip(hmm.insert_emissions_mut(0).iter_mut())
        {
            if let Ok(val) = tok.parse::<f32>() {
                *ins_val = if val >= 99999.0 { 0.0 } else { (-val).exp() };
            }
        }

        // Read transition line for node 0
        line.clear();
        let _ = reader.read_line(&mut line);
        let mut transition_tokens = line.split_whitespace();
        if let (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f), Some(g)) = (
            transition_tokens.next(),
            transition_tokens.next(),
            transition_tokens.next(),
            transition_tokens.next(),
            transition_tokens.next(),
            transition_tokens.next(),
            transition_tokens.next(),
        ) {
            for (tok, t_val) in [a, b, c, d, e, f, g]
                .into_iter()
                .zip(hmm.transitions_mut(0)[..7].iter_mut())
            {
                *t_val = Self::parse_hmm_prob(tok);
            }
        }

        // Parse nodes 1..M
        for node_index in 1..=model_length {
            // Match emission line: k e1 e2 ... eK [MAP CS RF MM CONS]
            line.clear();
            let _ = reader.read_line(&mut line);
            let mut tokens = line.split_whitespace();
            if tokens.clone().count() > alphabet.canonical_size {
                tokens.next(); // node number
                for (tok, mat_val) in tokens
                    .by_ref()
                    .take(alphabet.canonical_size)
                    .zip(hmm.match_emissions_mut(node_index).iter_mut())
                {
                    if let Ok(val) = tok.parse::<f32>() {
                        *mat_val = if val >= 99999.0 { 0.0 } else { (-val).exp() };
                    }
                }

                if map_flag
                    && let Some(tok) = tokens.next()
                    && let Some(map) = &mut hmm.map
                {
                    map[node_index] = tok.parse().unwrap_or(0);
                }
                if flags & HMM_FLAG_CS != 0 {
                    if hmm.consensus_structure.is_none() {
                        hmm.consensus_structure = Some(vec![b' '; model_length + 2]);
                    }
                    if let Some(c) = tokens.next().and_then(|tok| tok.bytes().next())
                        && let Some(cs) = &mut hmm.consensus_structure
                    {
                        cs[node_index] = c;
                    }
                }
                if flags & HMM_FLAG_RF != 0 {
                    if hmm.reference_annotation.is_none() {
                        hmm.reference_annotation = Some(vec![b' '; model_length + 2]);
                    }
                    if let Some(c) = tokens.next().and_then(|tok| tok.bytes().next())
                        && let Some(rf) = &mut hmm.reference_annotation
                    {
                        rf[node_index] = c;
                    }
                }
                if flags & HMM_FLAG_MMASK != 0 {
                    tokens.next();
                }
                if flags & HMM_FLAG_CONS != 0 {
                    if hmm.consensus.is_none() {
                        hmm.consensus = Some(vec![b' '; model_length + 2]);
                    }
                    if let Some(c) = tokens.next().and_then(|tok| tok.bytes().next())
                        && let Some(cons) = &mut hmm.consensus
                    {
                        cons[node_index] = c;
                    }
                }
            }

            // Insert emission line
            line.clear();
            let _ = reader.read_line(&mut line);
            for (tok, ins_val) in line
                .split_whitespace()
                .zip(hmm.insert_emissions_mut(node_index).iter_mut())
            {
                if let Ok(val) = tok.parse::<f32>() {
                    *ins_val = if val >= 99999.0 { 0.0 } else { (-val).exp() };
                }
            }

            // Transition line (7 values: MM MI MD IM II DM DD)
            line.clear();
            let _ = reader.read_line(&mut line);
            let mut transition_tokens = line.split_whitespace();
            if let (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f), Some(g)) = (
                transition_tokens.next(),
                transition_tokens.next(),
                transition_tokens.next(),
                transition_tokens.next(),
                transition_tokens.next(),
                transition_tokens.next(),
                transition_tokens.next(),
            ) {
                for (tok, t_val) in [a, b, c, d, e, f, g]
                    .into_iter()
                    .zip(hmm.transitions_mut(node_index)[..7].iter_mut())
                {
                    *t_val = Self::parse_hmm_prob(tok);
                }
            }
        }

        // Read to end of model (// terminator)
        loop {
            line.clear();
            let bytes = reader
                .read_line(&mut line)
                .map_err(|e| HmmerError::Internal(format!("Read error: {}", e)))?;
            if bytes == 0 || line.trim() == "//" {
                break;
            }
        }

        Ok((alphabet, hmm))
    }

    /// Parse a probability value from HMMER3/f format.
    /// '*' means zero probability, values are stored as -ln(p).
    fn parse_hmm_prob(s: &str) -> f32 {
        if s == "*" {
            0.0
        } else if let Ok(v) = s.parse::<f32>() {
            if v >= 99999.0 { 0.0 } else { (-v).exp() }
        } else {
            0.0
        }
    }

    /// Write an HMM to a file in HMMER3/f format.
    pub fn write_ascii<W: Write>(
        writer: &mut W,
        hmm: &Hmm,
        alphabet: &Alphabet,
    ) -> Result<(), std::io::Error> {
        writeln!(writer, "HMMER3/f [hmmer-rust | port]")?;
        writeln!(writer, "NAME  {}", hmm.name)?;
        if let Some(ref accession) = hmm.accession {
            writeln!(writer, "ACC   {}", accession)?;
        }
        if let Some(ref description) = hmm.description {
            writeln!(writer, "DESC  {}", description)?;
        }
        writeln!(writer, "LENG  {}", hmm.num_nodes)?;
        let alphabet_name = match alphabet.kind {
            AlphabetKind::Amino => "amino",
            AlphabetKind::Dna => "DNA",
            AlphabetKind::Rna => "RNA",
        };
        writeln!(writer, "ALPH  {}", alphabet_name)?;
        writeln!(writer, "//")?;
        Ok(())
    }
}

/// Parse a single HMM from any buffered reader (file, in-memory bytes, etc.).
///
/// This reads the format header line (e.g. "HMMER3/f ...") and the full model body.
pub fn read_hmm<R: BufRead>(reader: &mut R) -> Result<(Alphabet, Hmm), HmmerError> {
    let mut first_line = String::new();
    let bytes = reader
        .read_line(&mut first_line)
        .map_err(|e| HmmerError::Internal(format!("Read error: {}", e)))?;
    if bytes == 0 {
        return Err(HmmerError::Eof);
    }
    let first_line = first_line.trim();
    if !first_line.starts_with("HMMER3") && !first_line.starts_with("HMMER2") {
        return Err(HmmerError::Format(format!(
            "Unrecognized HMM file format: {}",
            first_line
        )));
    }
    HmmFile::read_3f(reader)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_numbers() {
        assert_eq!(V3F_MAGIC, 0xe8ededba);
        assert_eq!(V3A_MAGIC, 0xe8ededb5);
    }
}
