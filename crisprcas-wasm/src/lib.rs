use crisprcas_core::{
    DetectionParams,
    cas_types::{
        Cluster, DetectedSystem, GeneDefinition, GeneStatus, HmmerHit, HmmerOptions,
        ModelRegistry, RepliconTopology, SequenceIndex, SystemHit, SystemModel,
        cluster_hits, select_best_solution,
    },
    casparser::{CasCluster, CasGene},
    detect_crisprs,
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
use hmmer_io::read_hmm;
use orphos_core::{OrphosAnalyzer, config::OrphosConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use wasm_bindgen::prelude::*;
use web_sys::console;

macro_rules! log {
    ($($t:tt)*) => (console::log_1(&format!($($t)*).into()))
}

pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[derive(Serialize, Deserialize, Default)]
pub struct WasmFinderOptions {
    pub min_repeat_length: Option<usize>,
    pub max_repeat_length: Option<usize>,
    pub min_spacer_length: Option<usize>,
    pub max_spacer_length: Option<usize>,
    pub no_mismatch: Option<bool>,
    pub min_spacer_count: Option<usize>,
    pub min_evidence_level: Option<usize>,
}

/// A single model definition passed from JavaScript.
#[derive(Deserialize)]
pub struct ModelDefinition {
    pub name: String,
    pub family: String,
    /// XML content of the model definition.
    pub content: String,
}

/// CAS detection options.
#[derive(Deserialize, Default)]
pub struct CasOptions {
    pub genetic_code: Option<usize>,
    pub metagenome: Option<bool>,
}

/// Combined result of CRISPR + CAS analysis.
#[derive(Serialize)]
pub struct FullAnalysisResult {
    pub crisprs: Vec<crisprcas_core::CrisprArray>,
    pub cas_clusters: Vec<CasCluster>,
}

fn parse_fasta_sequences(fasta_content: &str) -> Vec<(String, Vec<u8>)> {
    let mut seqs: Vec<(String, Vec<u8>)> = Vec::new();
    let mut current_id: Option<String> = None;

    for line in fasta_content.lines() {
        if let Some(header) = line.strip_prefix('>') {
            let id = header
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_string();
            if !id.is_empty() {
                current_id = Some(id.clone());
                seqs.push((id, Vec::new()));
            } else {
                current_id = None;
            }
            continue;
        }

        if current_id.is_some() {
            let clean = line.trim().as_bytes();
            if !clean.is_empty() {
                if let Some((_, seq)) = seqs.last_mut() {
                    seq.extend_from_slice(clean);
                }
            }
        }
    }

    seqs
}

#[wasm_bindgen]
pub fn find_repeats(fasta_content: &str, options_js: JsValue) -> Result<JsValue, JsValue> {
    let options: WasmFinderOptions = serde_wasm_bindgen::from_value(options_js).unwrap_or_default();
    let params = build_detection_params(&options);

    let sequences = parse_fasta_sequences(fasta_content);
    let seq_refs: Vec<(&str, &[u8])> = sequences
        .iter()
        .map(|(id, seq)| (id.as_str(), seq.as_slice()))
        .collect();

    let arrays = detect_crisprs(&seq_refs, &params);
    serde_wasm_bindgen::to_value(&arrays)
        .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
}

// ---------------------------------------------------------------------------
// Step-by-step CAS detection API (avoids long blocking + 21 MB postMessage)
// ---------------------------------------------------------------------------

/// Opaque handle holding intermediate CAS detection state.
struct CasContext {
    genes: Vec<GeneRecord>,
    targets: Vec<DigitalSequence>,
    abc: Alphabet,
    registry: ModelRegistry,
    needed_profiles: HashSet<String>,
    hmmer_options: HmmerOptions,
    all_hits: HashMap<String, Vec<HmmerHit>>,
    models_raw: Vec<ModelDefinition>,
}

static CAS_CTX: Mutex<Option<CasContext>> = Mutex::new(None);

/// Step 1 — predict genes, translate proteins, build model registry.
/// Returns the list of needed profile names as a JS array of strings.
#[wasm_bindgen]
pub fn cas_prepare(
    fasta_content: &str,
    models_js: JsValue,
    cas_options_js: JsValue,
) -> Result<JsValue, JsValue> {
    let models: Vec<ModelDefinition> = serde_wasm_bindgen::from_value(models_js)
        .map_err(|e| JsValue::from_str(&format!("invalid models: {e}")))?;
    let cas_opts: CasOptions =
        serde_wasm_bindgen::from_value(cas_options_js).unwrap_or_default();

    let genetic_code = cas_opts.genetic_code.unwrap_or(11);
    let metagenome = cas_opts.metagenome.unwrap_or(false);

    // Gene prediction
    let config = OrphosConfig {
        metagenomic: metagenome,
        closed_ends: true,
        quiet: true,
        translation_table: Some(genetic_code as u8),
        ..OrphosConfig::default()
    };
    let mut analyzer = OrphosAnalyzer::new(config);
    let sequences = parse_fasta_sequences(fasta_content);

    let mut all_genes: Vec<GeneRecord> = Vec::new();
    let table = codon_table(genetic_code);

    for (seq_id, seq_bytes) in &sequences {
        let results = analyzer
            .analyze_sequence_bytes(seq_bytes, seq_id.clone(), None)
            .map_err(|e| JsValue::from_str(&format!("Gene prediction failed: {e}")))?;

        for gene in &results.genes {
            let begin = gene.coordinates.begin.saturating_sub(1);
            let end = gene.coordinates.end.saturating_sub(1);
            if end >= seq_bytes.len() {
                continue;
            }
            let cds_nt = &seq_bytes[begin..=end];
            let strand_str = format!("{}", gene.coordinates.strand);
            let cds_nt = if strand_str == "-" {
                revcomp_dna(cds_nt)
            } else {
                cds_nt.to_vec()
            };
            let protein = translate_dna(&cds_nt, &table);
            if protein.is_empty() {
                continue;
            }
            let gene_count = all_genes.len() + 1;
            let gene_id = format!("{}_{}", seq_id, gene_count);
            all_genes.push(GeneRecord {
                id: gene_id,
                protein,
                start: gene.coordinates.begin,
                end: gene.coordinates.end,
                strand: if strand_str == "-" {
                    "-".to_string()
                } else {
                    "+".to_string()
                },
            });
        }
    }

    // Build targets as DigitalSequence for hmmer-core
    let abc = Alphabet::amino();
    let faa_content = build_faa_content(&all_genes);
    log!("[CAS] Gene prediction done: {} genes found", all_genes.len());
    if let Some(g) = all_genes.first() {
        log!("[CAS] First gene: id={}, start={}, end={}, strand={}, protein_len={}", g.id, g.start, g.end, g.strand, g.protein.len());
    }
    let targets: Vec<DigitalSequence> = {
        let mut seqs = Vec::new();
        let mut current_name = String::new();
        let mut current_desc = String::new();
        let mut current_seq = Vec::new();
        for line in faa_content.lines() {
            if let Some(header) = line.strip_prefix('>') {
                if !current_name.is_empty() && !current_seq.is_empty() {
                    seqs.push(DigitalSequence::from_bytes(
                        &current_name, &current_desc, &current_seq, &abc,
                    ));
                }
                let parts: Vec<&str> = header.splitn(2, char::is_whitespace).collect();
                current_name = parts[0].to_string();
                current_desc = parts.get(1).unwrap_or(&"").to_string();
                current_seq = Vec::new();
            } else {
                current_seq.extend_from_slice(line.trim().as_bytes());
            }
        }
        if !current_name.is_empty() && !current_seq.is_empty() {
            seqs.push(DigitalSequence::from_bytes(
                &current_name, &current_desc, &current_seq, &abc,
            ));
        }
        seqs
    };

    // Build model registry
    let registry = build_model_registry(&models)
        .map_err(|e| JsValue::from_str(&e))?;

    // Determine needed profiles
    let mut needed_profiles: HashSet<String> = HashSet::new();
    for model_def in &models {
        let fqn = format!("{}/{}", model_def.family, model_def.name);
        if let Some(model) = registry.get(&fqn) {
            for name in model.all_profile_names() {
                needed_profiles.insert(name.to_string());
            }
        }
    }

    let needed_list: Vec<String> = needed_profiles.iter().cloned().collect();
    let gene_count = all_genes.len();
    log!("[CAS] Models registered: {}", models.len());
    log!("[CAS] Needed profiles: {} profiles", needed_profiles.len());
    log!("[CAS] Built {} digital sequence targets", targets.len());

    // Store context
    *CAS_CTX.lock().unwrap() = Some(CasContext {
        genes: all_genes,
        targets,
        abc,
        registry,
        needed_profiles,
        hmmer_options: HmmerOptions {
            coverage_profile: 0.4,
            ..Default::default()
        },
        all_hits: HashMap::new(),
        models_raw: models,
    });

    // Return { needed_profiles: [...], gene_count: N }
    let info = serde_json::json!({
        "needed_profiles": needed_list,
        "gene_count": gene_count,
    });
    serde_wasm_bindgen::to_value(&info)
        .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
}

/// Step 2 — search one HMM profile against the target proteins.
/// `profile_name` and `profile_data` are the name and text content of one .hmm file.
/// Returns the number of hits found for this profile.
#[wasm_bindgen]
pub fn cas_search_profile(profile_name: &str, profile_data: &str) -> Result<u32, JsValue> {
    let mut guard = CAS_CTX.lock().unwrap();
    let ctx = guard
        .as_mut()
        .ok_or_else(|| JsValue::from_str("cas_prepare not called"))?;

    if !ctx.needed_profiles.contains(profile_name) {
        return Ok(0);
    }

    let mut reader = std::io::BufReader::new(profile_data.as_bytes());
    let (_abc, hmm) = read_hmm(&mut reader)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse HMM {profile_name}: {e}")))?;

    let bg = BackgroundModel::new(&ctx.abc);
    let avg_len = if ctx.targets.is_empty() { 400 } else {
        ctx.targets.iter().map(|s| s.len()).sum::<usize>() / ctx.targets.len()
    };
    let num_nodes = hmm.num_nodes;
    let mut gm = Profile::new(num_nodes, &ctx.abc);
    modelconfig::profile_config(&hmm, &bg, &mut gm, avg_len, SearchMode::Local);

    let query = SearchQuery::from_configured_profile(gm, bg)
        .map_err(|e| JsValue::from_str(&format!("Query config failed for {profile_name}: {e}")))?;
    let plan = SearchPlan::builder(query)
        .filters(FilterPolicy::default())
        .build();
    let mut worker = plan
        .spawn_worker(CapacityHints { target_length: avg_len })
        .map_err(|e| JsValue::from_str(&format!("Spawn worker failed for {profile_name}: {e}")))?;

    let coverage_threshold = ctx.hmmer_options.coverage_profile;
    let mut profile_hits = Vec::new();

    for seq in &ctx.targets {
        let report = worker.search(seq)
            .map_err(|e| JsValue::from_str(&format!("Search error for {profile_name}: {e}")))?;

        match &report.outcome {
            SearchOutcome::Hit(hit) => {
                for domain in &hit.domains {
                    if !domain.is_included { continue; }
                    let prof_cov = (domain.hmm_to - domain.hmm_from + 1) as f64 / num_nodes as f64;
                    if prof_cov < coverage_threshold { continue; }
                    let seq_cov = (domain.alignment_end - domain.alignment_start + 1) as f64 / seq.len() as f64;
                    profile_hits.push(HmmerHit {
                        id: hit.name.clone(),
                        gene_name: profile_name.to_string(),
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
                if let (Some(fwd_raw), Some(null_sc)) =
                    (report.trace.forward_raw_score, report.trace.null_score)
                {
                    let bitscore = (fwd_raw - null_sc) / std::f32::consts::LN_2;
                    let prof_cov = seq.len().min(num_nodes) as f64 / num_nodes as f64;
                    if prof_cov >= coverage_threshold {
                        profile_hits.push(HmmerHit {
                            id: seq.name.clone(),
                            gene_name: profile_name.to_string(),
                            seq_len: seq.len() as u32,
                            i_evalue: 0.0,
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

    let hit_count = profile_hits.len() as u32;
    if !profile_hits.is_empty() {
        ctx.all_hits.insert(profile_name.to_string(), profile_hits);
    }
    Ok(hit_count)
}

/// Step 2b — search ALL HMM profiles in parallel using rayon.
/// `profiles_js` is an array of { name: string, data: string }.
/// Returns total number of hits found across all profiles.
#[wasm_bindgen]
pub fn cas_search_all_profiles(profiles_js: JsValue) -> Result<u32, JsValue> {
    #[derive(Deserialize)]
    struct ProfileEntry {
        name: String,
        data: String,
    }

    let profiles: Vec<ProfileEntry> = serde_wasm_bindgen::from_value(profiles_js)
        .map_err(|e| JsValue::from_str(&format!("invalid profiles: {e}")))?;

    // Extract read-only data from context
    let (targets, hmmer_options, needed_profiles, abc) = {
        let guard = CAS_CTX.lock().unwrap();
        let ctx = guard
            .as_ref()
            .ok_or_else(|| JsValue::from_str("cas_prepare not called"))?;
        (ctx.targets.clone(), ctx.hmmer_options.clone(), ctx.needed_profiles.clone(), ctx.abc.clone())
    };

    let avg_len = if targets.is_empty() { 400 } else {
        targets.iter().map(|s| s.len()).sum::<usize>() / targets.len()
    };
    let coverage_threshold = hmmer_options.coverage_profile;

    // Parallel search — each thread parses its HMM and runs hmmer-core pipeline
    let results: Vec<(String, Vec<HmmerHit>)> = profiles
        .par_iter()
        .filter(|p| needed_profiles.contains(&p.name))
        .filter_map(|p| {
            let mut reader = std::io::BufReader::new(p.data.as_bytes());
            let (_abc, hmm) = read_hmm(&mut reader).ok()?;
            let num_nodes = hmm.num_nodes;

            let bg = BackgroundModel::new(&abc);
            let mut gm = Profile::new(num_nodes, &abc);
            modelconfig::profile_config(&hmm, &bg, &mut gm, avg_len, SearchMode::Local);

            let query = SearchQuery::from_configured_profile(gm, bg).ok()?;
            let plan = SearchPlan::builder(query)
                .filters(FilterPolicy::default())
                .build();
            let mut worker = plan
                .spawn_worker(CapacityHints { target_length: avg_len })
                .ok()?;

            let mut profile_hits = Vec::new();
            for seq in &targets {
                if let Ok(report) = worker.search(seq) {
                    match &report.outcome {
                        SearchOutcome::Hit(hit) => {
                            for domain in &hit.domains {
                                if !domain.is_included { continue; }
                                let prof_cov = (domain.hmm_to - domain.hmm_from + 1) as f64 / num_nodes as f64;
                                if prof_cov < coverage_threshold { continue; }
                                let seq_cov = (domain.alignment_end - domain.alignment_start + 1) as f64 / seq.len() as f64;
                                profile_hits.push(HmmerHit {
                                    id: hit.name.clone(),
                                    gene_name: p.name.clone(),
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
                            if let (Some(fwd_raw), Some(null_sc)) =
                                (report.trace.forward_raw_score, report.trace.null_score)
                            {
                                let bitscore = (fwd_raw - null_sc) / std::f32::consts::LN_2;
                                let prof_cov = seq.len().min(num_nodes) as f64 / num_nodes as f64;
                                if prof_cov >= coverage_threshold {
                                    profile_hits.push(HmmerHit {
                                        id: seq.name.clone(),
                                        gene_name: p.name.clone(),
                                        seq_len: seq.len() as u32,
                                        i_evalue: 0.0,
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
            }

            if profile_hits.is_empty() {
                None
            } else {
                Some((p.name.clone(), profile_hits))
            }
        })
        .collect();

    // Merge results into context
    let total_hits: u32 = results.iter().map(|(_, h)| h.len() as u32).sum();
    log!("[CAS] HMM search complete: {} total hits across {} profiles", total_hits, results.len());
    for (name, hits) in &results {
        log!("[CAS]   Profile {}: {} hits", name, hits.len());
        for h in hits {
            log!("[CAS]     hit id={} score={:.1} evalue={:.2e} prof_cov={:.3} seq_cov={:.3}", h.id, h.score, h.i_evalue, h.profile_coverage, h.seq_coverage);
        }
    }
    {
        let mut guard = CAS_CTX.lock().unwrap();
        let ctx = guard
            .as_mut()
            .ok_or_else(|| JsValue::from_str("cas_prepare not called"))?;
        for (name, hits) in results {
            ctx.all_hits.insert(name, hits);
        }
    }

    Ok(total_hits)
}

/// Step 3 — cluster hits, evaluate systems, and return CAS results.
/// Consumes the context.
#[wasm_bindgen]
pub fn cas_finalize() -> Result<JsValue, JsValue> {
    let ctx = CAS_CTX.lock().unwrap().take()
        .ok_or_else(|| JsValue::from_str("cas_prepare not called"))?;

    let gene_map: HashMap<&str, &GeneRecord> =
        ctx.genes.iter().map(|g| (g.id.as_str(), g)).collect();

    let protein_ids: Vec<&str> = ctx.genes.iter().map(|g| g.id.as_str()).collect();
    let seq_index = SequenceIndex::from_ids(&protein_ids);

    let mut detected_systems = Vec::new();
    log!("[CAS] cas_finalize: {} genes, {} profiles with hits", ctx.genes.len(), ctx.all_hits.len());
    log!("[CAS] all_hits keys: {:?}", ctx.all_hits.keys().collect::<Vec<_>>());

    for model_def in &ctx.models_raw {
        let fqn = format!("{}/{}", model_def.family, model_def.name);
        let model = match ctx.registry.get(&fqn) {
            Some(m) => m,
            None => continue,
        };

        let system_hits = assign_hits_to_model(model, &ctx.all_hits, &seq_index);
        if system_hits.is_empty() {
            continue;
        }
        log!("[CAS] Model {}: {} system_hits assigned", fqn, system_hits.len());

        let mut hits_for_clustering = system_hits;
        let clusters = cluster_hits(
            &mut hits_for_clustering,
            model.inter_gene_max_space,
            ctx.genes.len(),
            RepliconTopology::Circular,
        );
        log!("[CAS] Model {}: {} clusters formed (inter_gene_max_space={})", fqn, clusters.len(), model.inter_gene_max_space);

        for (ci, c) in clusters.iter().enumerate() {
            if c.is_empty() {
                continue;
            }
            log!("[CAS] Model {} cluster {}: {} hits, positions {:?}", fqn, ci, c.hits.len(), c.hits.iter().map(|h| h.position).collect::<Vec<_>>());
            if let Some(system) = evaluate_cluster(c, model) {
                log!("[CAS] Model {} cluster {}: ACCEPTED (score={:.1}, mandatory={:?}, accessory={:?})", fqn, ci, system.score, system.mandatory_found, system.accessory_found);
                detected_systems.push(system);
            } else {
                // Log why it was rejected
                let mandatory_names: std::collections::HashSet<&str> = model.mandatory_genes().map(|g| g.name.as_str()).collect();
                let accessory_names: std::collections::HashSet<&str> = model.accessory_genes().map(|g| g.name.as_str()).collect();
                let forbidden_names: std::collections::HashSet<&str> = model.forbidden_genes().map(|g| g.name.as_str()).collect();
                let found_genes: std::collections::HashSet<&str> = c.hits.iter().map(|h| h.gene_ref.as_str()).collect();
                let mand_found: Vec<&&str> = mandatory_names.iter().filter(|g| found_genes.contains(**g)).collect();
                let acc_found: Vec<&&str> = accessory_names.iter().filter(|g| found_genes.contains(**g)).collect();
                let forb_found: Vec<&&str> = forbidden_names.iter().filter(|g| found_genes.contains(**g)).collect();
                log!("[CAS] Model {} cluster {}: REJECTED (min_mandatory={}, min_genes={}, mandatory_found={:?}, accessory_found={:?}, forbidden_found={:?})",
                    fqn, ci, model.min_mandatory_genes_required, model.min_genes_required, mand_found, acc_found, forb_found);
            }
        }
    }

    if detected_systems.len() > 1 {
        detected_systems = select_best_solution(detected_systems);
    }
    log!("[CAS] After select_best_solution: {} systems", detected_systems.len());

    const MIN_BEST_HIT_SCORE: f64 = 25.0;
    detected_systems.retain(|sys| {
        let max_score = sys.hits.iter().map(|h| h.hit.score).fold(f64::NEG_INFINITY, f64::max);
        let keep = max_score >= MIN_BEST_HIT_SCORE;
        if !keep {
            log!("[CAS] Filtered out system {} (max_score={:.1} < {})", sys.model_fqn, max_score, MIN_BEST_HIT_SCORE);
        }
        keep
    });
    log!("[CAS] Final systems after score filter: {}", detected_systems.len());

    let cas_clusters: Vec<CasCluster> = detected_systems
        .into_iter()
        .map(|sys| {
            let mut genes = Vec::new();
            let mut min_start = usize::MAX;
            let mut max_end = 0usize;

            for hit in &sys.hits {
                let (start, end, strand) = gene_map
                    .get(hit.hit.id.as_str())
                    .map(|g| (g.start, g.end, g.strand.clone()))
                    .unwrap_or((
                        hit.hit.begin_match as usize,
                        hit.hit.end_match as usize,
                        ".".to_string(),
                    ));

                if start < min_start {
                    min_start = start;
                }
                if end > max_end {
                    max_end = end;
                }

                genes.push(CasGene {
                    id: hit.hit.id.clone(),
                    name: hit.gene_ref.clone(),
                    start,
                    end,
                    strand,
                });
            }

            CasCluster {
                system: sys.model_fqn,
                genes,
                start: if min_start == usize::MAX { 0 } else { min_start },
                end: max_end,
            }
        })
        .collect();

    serde_wasm_bindgen::to_value(&cas_clusters)
        .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
}

fn build_detection_params(options: &WasmFinderOptions) -> DetectionParams {
    let mut params = DetectionParams::default();
    if let Some(v) = options.min_repeat_length {
        params.min_repeat_length = v;
    }
    if let Some(v) = options.max_repeat_length {
        params.max_repeat_length = v;
    }
    if let Some(v) = options.min_spacer_length {
        params.min_spacer_length = v;
    }
    if let Some(v) = options.max_spacer_length {
        params.max_spacer_length = v;
    }
    if let Some(v) = options.no_mismatch {
        params.no_mismatch = v;
    }
    if let Some(v) = options.min_spacer_count {
        params.min_spacer_count = v;
    }
    params.min_evidence_level = options.min_evidence_level.unwrap_or(4);
    params
}

struct GeneRecord {
    id: String,
    protein: String,
    start: usize,
    end: usize,
    strand: String,
}

fn build_faa_content(genes: &[GeneRecord]) -> String {
    let mut faa = String::new();
    for gene in genes {
        let strand_int: i32 = if gene.strand == "-" { -1 } else { 1 };
        faa.push_str(&format!(
            ">{} # {} # {} # {} # ID={}\n",
            gene.id, gene.start, gene.end, strand_int, gene.id
        ));
        for chunk in gene.protein.as_bytes().chunks(60) {
            faa.push_str(std::str::from_utf8(chunk).unwrap_or(""));
            faa.push('\n');
        }
    }
    faa
}

/// Build a ModelRegistry from in-memory XML model definitions.
fn build_model_registry(models: &[ModelDefinition]) -> Result<ModelRegistry, String> {
    let mut registry = ModelRegistry::new();
    for model_def in models {
        let model = parse_cas_model_xml(&model_def.content, &model_def.name, &model_def.family)?;
        registry.add(model);
    }
    Ok(registry)
}

/// Parse a CAS model definition from XML content.
/// Handles the CRISPRCasFinder format which uses <system> tags instead of <model>.
fn parse_cas_model_xml(
    content: &str,
    model_name: &str,
    family: &str,
) -> Result<SystemModel, String> {
    // Convert <system> to <model> and strip # comments
    let converted = convert_cas_xml(content);

    let fqn = format!("{}/{}", family, model_name);
    let mut inter_gene_max_space: u32 = 20;
    let mut min_mandatory_genes_required: u32 = 0;
    let mut min_genes_required: u32 = 0;
    let mut multi_loci = false;
    let mut genes: Vec<GeneDefinition> = Vec::new();

    // Simple XML parsing for model definitions
    // The format is: <model inter_gene_max_space="5" ...> <gene name="..." presence="..."/> </model>
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
    let mandatory_names: HashSet<&str> = model.mandatory_genes().map(|g| g.name.as_str()).collect();
    let accessory_names: HashSet<&str> = model.accessory_genes().map(|g| g.name.as_str()).collect();
    let forbidden_names: HashSet<&str> = model.forbidden_genes().map(|g| g.name.as_str()).collect();

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

    // Reject if forbidden genes present
    if !forbidden_found.is_empty() {
        return None;
    }

    // Check minimum requirements
    let min_mandatory = model.min_mandatory_genes_required as usize;
    let min_genes = model.min_genes_required as usize;
    let total_found = mandatory_found.len() + accessory_found.len();

    if mandatory_found.len() < min_mandatory || total_found < min_genes {
        return None;
    }

    // Simple scoring: sum of hit scores
    let score: f64 = c.hits.iter().map(|h| h.hit.score).sum();
    let max_nb_genes = model.mandatory_count() + model.accessory_count();
    let wholeness = if max_nb_genes > 0 {
        (mandatory_found.len() + accessory_found.len()) as f64 / max_nb_genes as f64
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
