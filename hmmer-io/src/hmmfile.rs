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
        let mut acc = None;
        let mut desc = None;
        let mut m: usize = 0;
        let mut abc_type = AlphabetKind::Amino;
        let mut evparam = [EV_PARAM_UNSET; NUM_EV_PARAMS];
        let mut flags: u32 = 0;
        let mut nseq: i32 = -1;
        let mut eff_nseq: f32 = -1.0;
        let mut checksum: u32 = 0;
        let mut map_flag = false;
        let _compo_vals: Option<Vec<f32>> = None;
        let mut max_length: i32 = -1;

        // Parse header fields
        loop {
            let mut line = String::new();
            let bytes = reader
                .read_line(&mut line)
                .map_err(|e| HmmerError::Internal(format!("Read error: {}", e)))?;
            if bytes == 0 {
                return Err(HmmerError::Format("Premature EOF in HMM header".into()));
            }

            let trimmed = line.trim().to_string();
            if trimmed.starts_with("HMM ") || trimmed.starts_with("HMM\t") {
                // End of header; start of model body
                // Skip the transition header line
                let mut _skip = String::new();
                let _ = reader.read_line(&mut _skip);
                break;
            }

            let parts: Vec<&str> = trimmed.splitn(2, char::is_whitespace).collect();
            if parts.len() < 2 {
                continue;
            }
            let tag = parts[0];
            let val = parts[1].trim();

            match tag {
                "NAME" => name = val.to_string(),
                "ACC" => {
                    acc = Some(val.to_string());
                    flags |= HMM_FLAG_ACC;
                }
                "DESC" => {
                    desc = Some(val.to_string());
                    flags |= HMM_FLAG_DESC;
                }
                "LENG" => m = val.parse().unwrap_or(0),
                "MAXL" => max_length = val.parse().unwrap_or(-1),
                "ALPH" => {
                    abc_type = match val {
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
                    nseq = val.parse().unwrap_or(-1);
                }
                "EFFN" => {
                    eff_nseq = val.parse().unwrap_or(-1.0);
                }
                "CKSUM" => {
                    checksum = val.parse().unwrap_or(0);
                    flags |= HMM_FLAG_CHKSUM;
                }
                "STATS" => {
                    // "LOCAL MSV -9.9014 0.71847" or similar
                    let stoks: Vec<&str> = val.split_whitespace().collect();
                    if stoks.len() >= 3 && stoks[0] == "LOCAL" {
                        let p1: f32 = stoks[2].parse().unwrap_or(0.0);
                        let p2: f32 = if stoks.len() > 3 {
                            stoks[3].parse().unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        match stoks[1] {
                            "MSV" => {
                                evparam[EvParam::MsvMu.idx()] = p1;
                                evparam[EvParam::MsvLambda.idx()] = p2;
                            }
                            "VITERBI" => {
                                evparam[EvParam::ViterbiMu.idx()] = p1;
                                evparam[EvParam::ViterbiLambda.idx()] = p2;
                            }
                            "FORWARD" => {
                                evparam[EvParam::ForwardTau.idx()] = p1;
                                evparam[EvParam::ForwardLambda.idx()] = p2;
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

        if m == 0 {
            return Err(HmmerError::Format("Zero-length model".into()));
        }

        let abc = match abc_type {
            AlphabetKind::Amino => Alphabet::amino(),
            AlphabetKind::Dna => Alphabet::dna(),
            AlphabetKind::Rna => Alphabet::rna(),
        };
        let mut hmm = Hmm::new(m, &abc);
        hmm.name = name;
        hmm.accession = acc;
        hmm.description = desc;
        hmm.ev_params = if flags & HMM_FLAG_STATS != 0 {
            Some(EvParams::from_array(&evparam))
        } else {
            None
        };
        hmm.num_sequences = if nseq >= 0 { Some(nseq as u32) } else { None };
        hmm.effective_num_seq_float = if eff_nseq >= 0.0 {
            Some(eff_nseq)
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
            hmm.map = Some(vec![0; m + 1]);
        }

        // Parse model body
        // After "HMM" and transition header lines, we have:
        //   COMPO line (optional)
        //   Insert emissions for node 0
        //   Transitions for node 0
        //   Then for each k=1..M: match emiss, insert emiss, transitions

        let mut line = String::new();

        // Read first line - might be COMPO or insert emissions for node 0
        line.clear();
        let _ = reader.read_line(&mut line);
        if line.trim().starts_with("COMPO") {
            // Parse composition
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.len() > abc.canonical_size {
                let mut compo_vec = vec![0.0f32; abc.canonical_size];
                for a in 0..abc.canonical_size.min(MAX_CANONICAL_ALPHABET) {
                    if let Ok(val) = toks[a + 1].parse::<f32>() {
                        compo_vec[a] = val; // stored as -ln(p)
                    }
                }
                hmm.model_composition = Some(compo_vec);
                flags |= HMM_FLAG_COMPO;
            }
            // Read insert emissions for node 0
            line.clear();
            let _ = reader.read_line(&mut line);
        }
        // Current line = insert emissions for node 0 (parse but we mostly ignore node 0 inserts)
        let toks: Vec<&str> = line.split_whitespace().collect();
        for (tok, ins_val) in toks.iter().zip(hmm.insert_emissions_mut(0).iter_mut()) {
            if let Ok(val) = tok.parse::<f32>() {
                *ins_val = if val >= 99999.0 { 0.0 } else { (-val).exp() };
            }
        }

        // Read transition line for node 0
        line.clear();
        let _ = reader.read_line(&mut line);
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() >= 7 {
            for (&tok, t_val) in toks.iter().zip(hmm.transitions_mut(0)[..7].iter_mut()) {
                *t_val = Self::parse_hmm_prob(tok);
            }
        }

        // Parse nodes 1..M
        for k in 1..=m {
            // Match emission line: k e1 e2 ... eK [MAP CS RF MM CONS]
            line.clear();
            let _ = reader.read_line(&mut line);
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.len() > abc.canonical_size {
                let mat = hmm.match_emissions_mut(k);
                for a in 0..abc.canonical_size {
                    if let Ok(val) = toks[a + 1].parse::<f32>() {
                        mat[a] = if val >= 99999.0 { 0.0 } else { (-val).exp() };
                    }
                }
                // Parse optional annotation fields after emissions
                let extra_start = abc.canonical_size + 1;
                if map_flag
                    && let Some(map) = &mut hmm.map
                    && let Some(tok) = toks.get(extra_start)
                {
                    map[k] = tok.parse().unwrap_or(0);
                }
                // Parse consensus character if CONS flag is set
                if flags & HMM_FLAG_CONS != 0 {
                    // Consensus is at position extra_start + (1 if MAP) + (1 if CS) + (1 if RF) + (1 if MM)
                    // Actually the order is: MAP CS RF MM CONS
                    let mut idx = extra_start;
                    if map_flag {
                        idx += 1;
                    }
                    // CS
                    if flags & HMM_FLAG_CS != 0 {
                        if hmm.consensus_structure.is_none() {
                            hmm.consensus_structure = Some(vec![b' '; m + 2]);
                        }
                        if let Some(ref mut cs) = hmm.consensus_structure
                            && let Some(tok) = toks.get(idx)
                            && let Some(c) = tok.bytes().next()
                        {
                            cs[k] = c;
                        }
                        idx += 1;
                    }
                    // RF
                    if flags & HMM_FLAG_RF != 0 {
                        if hmm.reference_annotation.is_none() {
                            hmm.reference_annotation = Some(vec![b' '; m + 2]);
                        }
                        if let Some(ref mut rf) = hmm.reference_annotation
                            && let Some(tok) = toks.get(idx)
                            && let Some(c) = tok.bytes().next()
                        {
                            rf[k] = c;
                        }
                        idx += 1;
                    }
                    // MM
                    if flags & HMM_FLAG_MMASK != 0 {
                        idx += 1;
                    }
                    // CONS
                    if hmm.consensus.is_none() {
                        hmm.consensus = Some(vec![b' '; m + 2]);
                    }
                    if let Some(ref mut cons) = hmm.consensus
                        && let Some(tok) = toks.get(idx)
                        && let Some(c) = tok.bytes().next()
                    {
                        cons[k] = c;
                    }
                }
            }

            // Insert emission line
            line.clear();
            let _ = reader.read_line(&mut line);
            let toks: Vec<&str> = line.split_whitespace().collect();
            for (tok, ins_val) in toks.iter().zip(hmm.insert_emissions_mut(k).iter_mut()) {
                if let Ok(val) = tok.parse::<f32>() {
                    *ins_val = if val >= 99999.0 { 0.0 } else { (-val).exp() };
                }
            }

            // Transition line (7 values: MM MI MD IM II DM DD)
            line.clear();
            let _ = reader.read_line(&mut line);
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.len() >= 7 {
                for (&tok, t_val) in toks.iter().zip(hmm.transitions_mut(k)[..7].iter_mut()) {
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

        Ok((abc, hmm))
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
        w: &mut W,
        hmm: &Hmm,
        abc: &Alphabet,
    ) -> Result<(), std::io::Error> {
        writeln!(w, "HMMER3/f [hmmer-rust | port]")?;
        writeln!(w, "NAME  {}", hmm.name)?;
        if let Some(ref acc) = hmm.accession {
            writeln!(w, "ACC   {}", acc)?;
        }
        if let Some(ref desc) = hmm.description {
            writeln!(w, "DESC  {}", desc)?;
        }
        writeln!(w, "LENG  {}", hmm.num_nodes)?;
        let alph = match abc.kind {
            AlphabetKind::Amino => "amino",
            AlphabetKind::Dna => "DNA",
            AlphabetKind::Rna => "RNA",
        };
        writeln!(w, "ALPH  {}", alph)?;
        writeln!(w, "//")?;
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
