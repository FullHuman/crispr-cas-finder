use anyhow::{Context, Result};
use bio::io::fasta::Reader as FastaReader;
use crate::cas_types::{
    Cluster, DetectedSystem, GeneDefinition, GeneStatus, HmmerHit,
    ModelRegistry, RepliconTopology, SearchResults, SequenceIndex, SystemHit,
    SystemModel, cluster_hits, select_best_solution,
};
use hmmer_core::{
    alphabet::Alphabet,
    background::BackgroundModel,
    config::SearchMode,
    modelconfig,
    pipeline::search::{
        CapacityHints, FilterPolicy, FilterReason, SearchOutcome, SearchPlan, SearchQuery,
    },
    profile::Profile,
    sequence::DigitalSequence,
};
use hmmer_io::HmmFile;
use log::info;
use orphos_core::{
    OrphosAnalyzer,
    config::{OrphosConfig, OutputFormat},
    output::write_results,
};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use crate::types::CasFinderConfig;

/// Run Orphos gene prediction and then detect Cas systems using
/// hmmer-core for HMM search (pure Rust, no external subprocess).
pub fn run_casfinder(
    input: &Path,
    basename: &str,
    outdir: &Path,
    cfg: &CasFinderConfig,
) -> Result<SearchResults> {
    // Prepare annotation directory
    let annotation_dir = outdir.join(format!("orphos_{}", basename));
    fs::create_dir_all(&annotation_dir).context("Creating annotation output dir")?;

    // Predict genes with Orphos
    info!("Running Orphos gene prediction on {}", input.display());
    let config = OrphosConfig {
        metagenomic: cfg.metagenome,
        closed_ends: true,
        quiet: cfg.quiet,
        translation_table: Some(cfg.genetic_code as u8),
        ..OrphosConfig::default()
    };
    let mut analyzer = OrphosAnalyzer::new(config);
    let results = analyzer
        .analyze_fasta_file(input.to_str().unwrap())
        .context("Orphos gene prediction failed")?;

    // Write GFF3
    let gff = annotation_dir.join(format!("{}.gff", basename));
    let mut gff_file = fs::File::create(&gff).context("Creating .gff output")?;
    for r in &results {
        write_results(&mut gff_file, r, OutputFormat::Gff)
            .context("Writing GFF from orphos results")?;
    }

    // Write FAA
    let faa = annotation_dir.join(format!("{}.faa", basename));
    write_faa_from_genes(input, &results, cfg.genetic_code, &faa)
        .context("Writing .faa from orphos gene predictions")?;

    let proteome = faa;
    let metadata = fs::metadata(&proteome)
        .with_context(|| format!("Cannot stat proteome file: {:?}", proteome))?;
    if metadata.len() == 0 {
        info!("Proteome file is empty, skipping CasFinder");
        anyhow::bail!("No CDS found");
    }

    // ---- Load proteins as digital sequences via hmmer-core ----
    let abc = Alphabet::amino();
    let seqs = hmmer_io::sqfile_open_digital(&abc, proteome.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("Failed to load proteins: {}", e))?;

    info!("Loaded {} protein sequences", seqs.len());

    let seq_names: Vec<&str> = seqs.iter().map(|s| s.name.as_str()).collect();
    let seq_index = SequenceIndex::from_ids(&seq_names);

    // ---- Build model registry from CAS XML definitions ----
    let models_dir = resolve_models_dir(cfg)?;
    let profiles_dir = resolve_profiles_dir(cfg)?;

    let (registry, model_fqns) = build_model_registry_from_dir(&models_dir)?;

    let mut needed_profiles: HashSet<String> = HashSet::new();
    for fqn in &model_fqns {
        if let Some(model) = registry.get(fqn) {
            for name in model.all_profile_names() {
                needed_profiles.insert(name.to_string());
            }
        }
    }

    if model_fqns.is_empty() {
        anyhow::bail!("No CAS model definitions found in {:?}", models_dir);
    }
    info!(
        "Need {} HMM profiles for {} models",
        needed_profiles.len(),
        model_fqns.len()
    );

    // ---- HMM search using hmmer-core pipeline ----
    let all_hits = run_hmm_search(&profiles_dir, &needed_profiles, &seqs, &abc)?;

    info!("HMM search found hits for {} profiles", all_hits.len());

    // ---- Cluster hits and evaluate systems ----
    let mut detected_systems = Vec::new();

    for fqn in &model_fqns {
        let model = match registry.get(fqn) {
            Some(m) => m,
            None => continue,
        };

        let system_hits = assign_hits_to_model(model, &all_hits, &seq_index);
        if system_hits.is_empty() {
            continue;
        }

        let mut hits_for_clustering = system_hits;
        let clusters = cluster_hits(
            &mut hits_for_clustering,
            model.inter_gene_max_space,
            seqs.len(),
            RepliconTopology::Circular,
        );

        for c in &clusters {
            if c.is_empty() {
                continue;
            }
            if let Some(system) = evaluate_cluster(c, model) {
                detected_systems.push(system);
            }
        }
    }

    if detected_systems.len() > 1 {
        detected_systems = select_best_solution(detected_systems);
    }

    // ---- Filter low-confidence systems ----
    const MIN_BEST_HIT_SCORE: f64 = 25.0;
    let before = detected_systems.len();
    detected_systems.retain(|sys| {
        let best = sys
            .hits
            .iter()
            .map(|h| h.hit.score)
            .fold(f64::NEG_INFINITY, f64::max);
        if best < MIN_BEST_HIT_SCORE {
            info!(
                "Filtering low-confidence system {} (best hit score {:.1} < {})",
                sys.model_fqn, best, MIN_BEST_HIT_SCORE
            );
            return false;
        }
        true
    });
    if before != detected_systems.len() {
        info!(
            "Filtered {} low-confidence systems ({} -> {})",
            before - detected_systems.len(),
            before,
            detected_systems.len()
        );
    }

    info!("CasFinder found {} systems", detected_systems.len());

    Ok(SearchResults {
        systems: detected_systems,
        rejected: Vec::new(),
        skipped_replicons: Vec::new(),
    })
}

/// Resolve the directory containing CAS model definitions (XML files).
fn resolve_models_dir(cfg: &CasFinderConfig) -> Result<PathBuf> {
    if let Some(ref dir) = cfg.cas_models_dir {
        return Ok(dir.clone());
    }
    let def_suffix = format!("DEF-{}", cfg.definition);
    let candidates = [
        PathBuf::from(format!("CasFinder-2.0.3/{}-2.0.3", def_suffix)),
        PathBuf::from(format!(
            "../CRISPRCasFinder/CasFinder-2.0.3/{}-2.0.3",
            def_suffix
        )),
    ];
    for p in &candidates {
        if p.is_dir() {
            return Ok(p.clone());
        }
    }
    anyhow::bail!(
        "Cannot find CAS model definitions directory. \
         Tried: {:?}. Use --cas-models-dir to specify the path.",
        candidates
    )
}

/// Resolve the directory containing CAS HMM profiles.
fn resolve_profiles_dir(cfg: &CasFinderConfig) -> Result<PathBuf> {
    if let Some(ref dir) = cfg.cas_profiles_dir {
        return Ok(dir.clone());
    }
    let candidates = [
        PathBuf::from("CasFinder-2.0.3/CASprofiles-2.0.3"),
        PathBuf::from("../CRISPRCasFinder/CasFinder-2.0.3/CASprofiles-2.0.3"),
    ];
    for p in &candidates {
        if p.is_dir() {
            return Ok(p.clone());
        }
    }
    anyhow::bail!(
        "Cannot find CAS HMM profiles directory. \
         Tried: {:?}. Use --cas-profiles-dir to specify the path.",
        candidates
    )
}

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

/// Build a ModelRegistry from CAS XML definitions on disk.
fn build_model_registry_from_dir(models_dir: &Path) -> Result<(ModelRegistry, Vec<String>)> {
    let mut registry = ModelRegistry::new();
    let mut fqns = Vec::new();
    let family = "CASFinder".to_string();

    for entry in fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "xml") {
            let model_name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let content = fs::read_to_string(&path)?;
            let model = parse_cas_model_xml(&content, &model_name, &family)
                .map_err(|e| anyhow::anyhow!("Failed to parse model {}: {}", model_name, e))?;
            let fqn = model.fqn.clone();
            registry.add(model);
            fqns.push(fqn);
        }
    }
    Ok((registry, fqns))
}

/// Parse a CAS model definition from XML content.
fn parse_cas_model_xml(
    content: &str,
    model_name: &str,
    family: &str,
) -> std::result::Result<SystemModel, String> {
    let converted = convert_cas_xml(content);

    let fqn = format!("{}/{}", family, model_name);
    let mut inter_gene_max_space: u32 = 20;
    let mut min_mandatory_genes_required: u32 = 0;
    let mut min_genes_required: u32 = 0;
    let mut multi_loci = false;
    let mut genes: Vec<GeneDefinition> = Vec::new();

    for line in converted.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("<model ") || trimmed.starts_with("<model>") {
            if let Some(val) = extract_xml_attr(trimmed, "inter_gene_max_space") {
                inter_gene_max_space = val.parse().unwrap_or(20);
            }
            if let Some(val) = extract_xml_attr(trimmed, "min_mandatory_genes_required") {
                min_mandatory_genes_required = val.parse().unwrap_or(0);
            }
            if let Some(val) = extract_xml_attr(trimmed, "min_genes_required") {
                min_genes_required = val.parse().unwrap_or(0);
            }
            if let Some(val) = extract_xml_attr(trimmed, "multi_loci") {
                multi_loci = val == "True" || val == "true" || val == "1";
            }
        } else if trimmed.starts_with("<gene ") {
            let name = extract_xml_attr(trimmed, "name").unwrap_or_default();
            let presence = extract_xml_attr(trimmed, "presence").unwrap_or_default();
            let status = match presence.as_str() {
                "mandatory" => GeneStatus::Mandatory,
                "accessory" => GeneStatus::Accessory,
                "forbidden" => GeneStatus::Forbidden,
                _ => GeneStatus::Mandatory,
            };
            let loner = extract_xml_attr(trimmed, "loner")
                .is_some_and(|v| v == "True" || v == "true" || v == "1");
            let multi_system = extract_xml_attr(trimmed, "multi_system")
                .is_some_and(|v| v == "True" || v == "true" || v == "1");

            genes.push(GeneDefinition {
                name,
                status,
                loner,
                multi_system,
                exchangeables: Vec::new(),
                inter_gene_max_space: None,
                multi_model: false,
            });
        }
    }

    Ok(SystemModel {
        fqn,
        family: family.to_string(),
        name: model_name.to_string(),
        version: None,
        genes,
        inter_gene_max_space,
        min_mandatory_genes_required,
        min_genes_required,
        multi_loci,
    })
}

/// Convert CASFinder XML (<system>) to <model> tags and strip # comments.
fn convert_cas_xml(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        let converted = line
            .replace("<system ", "<model ")
            .replace("<system>", "<model>")
            .replace("</system>", "</model>");
        out.push_str(&converted);
        out.push('\n');
    }
    out
}

fn extract_xml_attr(tag: &str, attr_name: &str) -> Option<String> {
    let pattern = format!("{}=\"", attr_name);
    if let Some(start) = tag.find(&pattern) {
        let value_start = start + pattern.len();
        if let Some(end) = tag[value_start..].find('"') {
            return Some(tag[value_start..value_start + end].to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// HMM search via hmmer-core
// ---------------------------------------------------------------------------

/// Run the full HMMER3 pipeline (MSV -> Viterbi -> Forward) for each needed
/// profile against all target protein sequences.
fn run_hmm_search(
    profiles_dir: &Path,
    needed_profiles: &HashSet<String>,
    seqs: &[DigitalSequence],
    abc: &Alphabet,
) -> Result<HashMap<String, Vec<HmmerHit>>> {
    let bg = BackgroundModel::new(abc);
    let avg_len = if seqs.is_empty() {
        400
    } else {
        seqs.iter().map(|s| s.len()).sum::<usize>() / seqs.len()
    };

    let coverage_threshold = 0.4;
    let mut all_hits: HashMap<String, Vec<HmmerHit>> = HashMap::new();

    for profile_name in needed_profiles {
        let hmm_path = profiles_dir.join(format!("{}.hmm", profile_name));
        if !hmm_path.exists() {
            continue;
        }

        let mut hfp = HmmFile::open(hmm_path.to_str().unwrap(), None)
            .map_err(|e| anyhow::anyhow!("Failed to open {}: {}", profile_name, e))?;
        let (_abc, hmm) = hfp
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", profile_name, e))?;

        let num_nodes = hmm.num_nodes;
        let mut gm = Profile::new(num_nodes, abc);
        modelconfig::profile_config(&hmm, &bg, &mut gm, avg_len, SearchMode::Local);

        let query = SearchQuery::from_configured_profile(gm, bg.clone())
            .map_err(|e| anyhow::anyhow!("Query config failed for {}: {}", profile_name, e))?;
        let plan = SearchPlan::builder(query)
            .filters(FilterPolicy::default())
            .build();
        let mut worker = plan
            .spawn_worker(CapacityHints {
                target_length: avg_len,
            })
            .map_err(|e| anyhow::anyhow!("Spawn worker failed for {}: {}", profile_name, e))?;

        let mut profile_hits = Vec::new();

        for seq in seqs {
            let report = worker
                .search(seq)
                .map_err(|e| anyhow::anyhow!("Search error {}: {}", profile_name, e))?;

            match &report.outcome {
                SearchOutcome::Hit(hit) => {
                    for domain in &hit.domains {
                        if !domain.is_included {
                            continue;
                        }
                        let prof_cov = (domain.hmm_to - domain.hmm_from + 1) as f64
                            / num_nodes as f64;
                        if prof_cov < coverage_threshold {
                            continue;
                        }
                        let seq_cov = (domain.alignment_end - domain.alignment_start + 1) as f64
                            / seq.len() as f64;
                        profile_hits.push(HmmerHit {
                            id: hit.name.clone(),
                            gene_name: profile_name.clone(),
                            seq_len: seq.len() as u32,
                            i_evalue: domain.log_pvalue.exp(),
                            score: domain.bitscore as f64,
                            profile_coverage: prof_cov,
                            seq_coverage: seq_cov,
                            begin_match: domain.alignment_start as u32,
                            end_match: domain.alignment_end as u32,
                        });
                    }
                }
                SearchOutcome::Filtered(FilterReason::NoDomains) => {
                    // Domain detection failed but the sequence passed all
                    // pre-filters (MSV, Viterbi, Forward).  Compute a
                    // sequence-level bitscore from the Forward raw score and
                    // emit a synthetic hit covering the whole protein.
                    if let (Some(fwd_raw), Some(null_sc)) =
                        (report.trace.forward_raw_score, report.trace.null_score)
                    {
                        let bitscore = (fwd_raw - null_sc) / std::f32::consts::LN_2;
                        // Rough profile coverage: min(seq_len, model_len) / model_len
                        let prof_cov =
                            seq.len().min(num_nodes) as f64 / num_nodes as f64;
                        if prof_cov >= coverage_threshold {
                            profile_hits.push(HmmerHit {
                                id: seq.name.clone(),
                                gene_name: profile_name.clone(),
                                seq_len: seq.len() as u32,
                                i_evalue: 0.0, // not available
                                score: bitscore as f64,
                                profile_coverage: prof_cov,
                                seq_coverage: 1.0,
                                begin_match: 1,
                                end_match: seq.len() as u32,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        if !profile_hits.is_empty() {
            all_hits.insert(profile_name.clone(), profile_hits);
        }
    }

    Ok(all_hits)
}

// ---------------------------------------------------------------------------
// System evaluation (clustering + scoring)
// ---------------------------------------------------------------------------

/// Assign HMMER hits to a model's genes, creating SystemHits.
fn assign_hits_to_model(
    model: &SystemModel,
    hmmer_hits: &HashMap<String, Vec<HmmerHit>>,
    seq_index: &SequenceIndex,
) -> Vec<SystemHit> {
    let mut system_hits = Vec::new();

    for gene in &model.genes {
        let gene_names_to_check: Vec<(&str, bool)> =
            std::iter::once((gene.name.as_str(), false))
                .chain(gene.exchangeables.iter().map(|e| (e.as_str(), true)))
                .collect();

        for (gene_name, is_exchangeable) in gene_names_to_check {
            if let Some(hits) = hmmer_hits.get(gene_name) {
                for hit in hits {
                    if let Some(position) = seq_index.position(&hit.id) {
                        system_hits.push(SystemHit {
                            hit: hit.clone(),
                            position,
                            gene_ref: gene.name.clone(),
                            gene_status: gene.status,
                            model_fqn: model.fqn.clone(),
                            is_exchangeable,
                            locus_num: 1,
                            counterpart: String::new(),
                            used_in: Vec::new(),
                        });
                    }
                }
            }
        }
    }

    system_hits
}

/// Evaluate a cluster against a model to produce a DetectedSystem (if valid).
fn evaluate_cluster(
    c: &Cluster,
    model: &SystemModel,
) -> Option<DetectedSystem> {
    let mandatory_names: HashSet<&str> =
        model.mandatory_genes().map(|g| g.name.as_str()).collect();
    let accessory_names: HashSet<&str> =
        model.accessory_genes().map(|g| g.name.as_str()).collect();
    let forbidden_names: HashSet<&str> =
        model.forbidden_genes().map(|g| g.name.as_str()).collect();

    let found_genes: HashSet<&str> = c.hits.iter().map(|h| h.gene_ref.as_str()).collect();

    let mandatory_found: Vec<String> = mandatory_names
        .iter()
        .filter(|g| found_genes.contains(*g))
        .map(|g| g.to_string())
        .collect();
    let accessory_found: Vec<String> = accessory_names
        .iter()
        .filter(|g| found_genes.contains(*g))
        .map(|g| g.to_string())
        .collect();
    let forbidden_found: Vec<String> = forbidden_names
        .iter()
        .filter(|g| found_genes.contains(*g))
        .map(|g| g.to_string())
        .collect();

    if !forbidden_found.is_empty() {
        return None;
    }

    let min_mandatory = model.min_mandatory_genes_required as usize;
    let min_genes = model.min_genes_required as usize;
    let total_found = mandatory_found.len() + accessory_found.len();

    if mandatory_found.len() < min_mandatory || total_found < min_genes {
        return None;
    }

    let score: f64 = c.hits.iter().map(|h| h.hit.score).sum();
    let max_nb_genes = model.mandatory_count() + model.accessory_count();
    let wholeness = if max_nb_genes > 0 {
        total_found as f64 / max_nb_genes as f64
    } else {
        1.0
    };

    Some(DetectedSystem {
        id: String::new(),
        replicon: String::new(),
        model_fqn: model.fqn.clone(),
        score,
        wholeness,
        loci_count: 1,
        occurrence: 0,
        hits: c.hits.clone(),
        state: "single_locus".to_string(),
        mandatory_found,
        accessory_found,
        forbidden_found,
    })
}

// ---------------------------------------------------------------------------
// Gene prediction helpers
// ---------------------------------------------------------------------------

/// Translate predicted CDS directly from the genome and write a FASTA protein file.
fn write_faa_from_genes(
    fasta_path: &Path,
    results: &[orphos_core::results::OrphosResults],
    genetic_code: usize,
    faa_path: &Path,
) -> Result<()> {
    let reader = FastaReader::from_file(fasta_path)
        .with_context(|| format!("Cannot read genome FASTA: {:?}", fasta_path))?;
    let mut sequences: Vec<(String, Vec<u8>)> = Vec::new();
    for result in reader.records() {
        let record = result.context("Reading FASTA record")?;
        sequences.push((record.id().to_string(), record.seq().to_vec()));
    }

    let table = codon_table(genetic_code);
    let mut out = fs::File::create(faa_path).context("Creating .faa file")?;
    let mut gene_count = 0usize;

    for (seq_idx, r) in results.iter().enumerate() {
        let genomic_seq = if seq_idx < sequences.len() {
            &sequences[seq_idx].1
        } else {
            continue;
        };

        for gene in &r.genes {
            let begin = gene.coordinates.begin.saturating_sub(1);
            let end = gene.coordinates.end.saturating_sub(1);

            if end >= genomic_seq.len() {
                continue;
            }

            let cds_nt = &genomic_seq[begin..=end];
            let cds_nt = match format!("{}", gene.coordinates.strand).as_str() {
                "-" => revcomp_dna(cds_nt),
                _ => cds_nt.to_vec(),
            };

            let protein = translate_dna(&cds_nt, &table);
            if protein.is_empty() {
                continue;
            }

            gene_count += 1;
            let strand_int = match format!("{}", gene.coordinates.strand).as_str() {
                "-" => -1,
                _ => 1,
            };
            let gene_id = format!(
                "{}_{} # {} # {} # {} # ID={}",
                r.sequence_info.header,
                gene_count,
                gene.coordinates.begin,
                gene.coordinates.end,
                strand_int,
                gene_count
            );
            writeln!(out, ">{}", gene_id)?;
            for chunk in protein.as_bytes().chunks(60) {
                out.write_all(chunk)?;
                out.write_all(b"\n")?;
            }
        }
    }
    Ok(())
}

fn revcomp_dna(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b {
            b'A' | b'a' => b'T',
            b'T' | b't' => b'A',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            other => other,
        })
        .collect()
}

fn translate_dna(seq: &[u8], table: &HashMap<[u8; 3], u8>) -> String {
    let mut protein = String::with_capacity(seq.len() / 3);
    for codon in seq.chunks(3) {
        if codon.len() < 3 {
            break;
        }
        let key = [
            codon[0].to_ascii_uppercase(),
            codon[1].to_ascii_uppercase(),
            codon[2].to_ascii_uppercase(),
        ];
        let aa = table.get(&key).copied().unwrap_or(b'X');
        if aa == b'*' {
            break;
        }
        protein.push(aa as char);
    }
    protein
}

fn codon_table(genetic_code: usize) -> HashMap<[u8; 3], u8> {
    let codons = b"TTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGG";
    let second = b"TTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGG";
    let third = b"TCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAG";
    let standard = b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";
    let table11 = b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";

    let aa_table: &[u8] = match genetic_code {
        11 => table11,
        _ => standard,
    };

    let mut map = HashMap::new();
    for i in 0..64 {
        map.insert([codons[i], second[i], third[i]], aa_table[i]);
    }
    map
}
