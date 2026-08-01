use crispr_cas_finder_core::{
    DetectionParams,
    cas_pipeline::{
        GeneRecord, ModelDefinition, assign_hits_to_model, build_faa_content, build_model_registry,
        codon_table, evaluate_cluster, gene_coordinates_from_records, reverse_complement_dna,
        translate_dna,
    },
    cas_types::{
        HmmerHit, HmmerOptions, ModelRegistry, RepliconTopology, SequenceIndex, cluster_hits,
        select_best_solution,
    },
    casparser::{CasCluster, from_search_results_with_gene_map},
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

#[allow(unused_unsafe)]
fn js_log(message: &str) {
    unsafe {
        console::log_1(&message.into());
    }
}

macro_rules! log {
    ($($t:tt)*) => (js_log(&format!($($t)*)))
}

pub use wasm_bindgen_rayon::init_thread_pool;

/// Match HMMER's default per-domain inclusion threshold used by the CLI.
const MAX_DOMAIN_INCLUSION_EVALUE: f64 = 0.01;

fn included_domain_evalue(log_pvalue: f64, target_count: usize) -> Option<f64> {
    let inclusion_evalue = log_pvalue.exp() * target_count.max(1) as f64;
    (inclusion_evalue <= MAX_DOMAIN_INCLUSION_EVALUE).then_some(inclusion_evalue)
}

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

/// CAS detection options.
#[derive(Deserialize, Default)]
pub struct CasOptions {
    pub genetic_code: Option<usize>,
    pub metagenome: Option<bool>,
}

fn parse_fasta_sequences(fasta_content: &str) -> Vec<(String, Vec<u8>)> {
    let mut parsed_sequences: Vec<(String, Vec<u8>)> = Vec::new();
    let mut current_id: Option<String> = None;

    for line in fasta_content.lines() {
        if let Some(header) = line.strip_prefix('>') {
            let id = header.split_whitespace().next().unwrap_or("").to_string();
            if !id.is_empty() {
                current_id = Some(id.clone());
                parsed_sequences.push((id, Vec::new()));
            } else {
                current_id = None;
            }
            continue;
        }

        if current_id.is_some() {
            let clean = line.trim().as_bytes();
            if !clean.is_empty()
                && let Some((_, sequence_bytes)) = parsed_sequences.last_mut()
            {
                sequence_bytes.extend_from_slice(clean);
            }
        }
    }

    parsed_sequences
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
    amino_alphabet: Alphabet,
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
    let cas_opts: CasOptions = serde_wasm_bindgen::from_value(cas_options_js).unwrap_or_default();

    let genetic_code = cas_opts.genetic_code.unwrap_or(11);
    let metagenome = cas_opts.metagenome.unwrap_or(false);

    let config = OrphosConfig {
        metagenomic: metagenome,
        closed_ends: true,
        quiet: true,
        translation_table: Some(genetic_code as u8),
        ..OrphosConfig::default()
    };
    let analyzer = OrphosAnalyzer::new(config);
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
                reverse_complement_dna(cds_nt)
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

    let amino_alphabet = Alphabet::amino();
    let faa_content = build_faa_content(&all_genes)
        .map_err(|error| JsValue::from_str(&format!("failed to build protein FASTA: {error}")))?;
    log!(
        "[CAS] Gene prediction done: {} genes found",
        all_genes.len()
    );
    if let Some(g) = all_genes.first() {
        log!(
            "[CAS] First gene: id={}, start={}, end={}, strand={}, protein_len={}",
            g.id,
            g.start,
            g.end,
            g.strand,
            g.protein.len()
        );
    }
    let targets: Vec<DigitalSequence> = {
        let mut digital_sequences = Vec::new();
        let mut current_name = String::new();
        let mut current_desc = String::new();
        let mut current_seq = Vec::new();
        for line in faa_content.lines() {
            if let Some(header) = line.strip_prefix('>') {
                if !current_name.is_empty() && !current_seq.is_empty() {
                    digital_sequences.push(DigitalSequence::from_bytes(
                        &current_name,
                        &current_desc,
                        &current_seq,
                        &amino_alphabet,
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
            digital_sequences.push(DigitalSequence::from_bytes(
                &current_name,
                &current_desc,
                &current_seq,
                &amino_alphabet,
            ));
        }
        digital_sequences
    };

    let registry = build_model_registry(&models).map_err(|e| JsValue::from_str(&e))?;

    let mut needed_profiles: HashSet<String> = HashSet::new();
    for model_def in &models {
        let model_fully_qualified_name = format!("{}/{}", model_def.family, model_def.name);
        if let Some(model) = registry.get(&model_fully_qualified_name) {
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

    *CAS_CTX.lock().unwrap() = Some(CasContext {
        genes: all_genes,
        targets,
        amino_alphabet,
        registry,
        needed_profiles,
        hmmer_options: HmmerOptions {
            coverage_profile: 0.4,
        },
        all_hits: HashMap::new(),
        models_raw: models,
    });

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

    let bg = BackgroundModel::new(&ctx.amino_alphabet);
    let avg_len = if ctx.targets.is_empty() {
        400
    } else {
        ctx.targets.iter().map(|s| s.len()).sum::<usize>() / ctx.targets.len()
    };
    let num_nodes = hmm.num_nodes;
    let mut gm = Profile::new(num_nodes, &ctx.amino_alphabet);
    modelconfig::profile_config(&hmm, &bg, &mut gm, avg_len, SearchMode::Local);

    let query = SearchQuery::from_configured_profile(gm, bg)
        .map_err(|e| JsValue::from_str(&format!("Query config failed for {profile_name}: {e}")))?;
    let plan = SearchPlan::builder(query)
        .filters(FilterPolicy::default())
        .build()
        .map_err(|e| {
            JsValue::from_str(&format!(
                "Search plan config failed for {profile_name}: {e}"
            ))
        })?;
    let mut worker = plan
        .spawn_worker(CapacityHints {
            target_length: avg_len,
        })
        .map_err(|e| JsValue::from_str(&format!("Spawn worker failed for {profile_name}: {e}")))?;

    let coverage_threshold = ctx.hmmer_options.coverage_profile;
    let target_count = ctx.targets.len();
    let mut profile_hits = Vec::new();

    for seq in &ctx.targets {
        let report = worker
            .search(seq)
            .map_err(|e| JsValue::from_str(&format!("Search error for {profile_name}: {e}")))?;

        match &report.outcome {
            SearchOutcome::Hit(hit) => {
                for domain in &hit.domains {
                    let Some(inclusion_evalue) =
                        included_domain_evalue(domain.log_pvalue, target_count)
                    else {
                        continue;
                    };
                    let prof_cov = (domain.hmm_to - domain.hmm_from + 1) as f64 / num_nodes as f64;
                    if prof_cov < coverage_threshold {
                        continue;
                    }
                    let seq_cov = (domain.alignment_end - domain.alignment_start + 1) as f64
                        / seq.len() as f64;
                    profile_hits.push(HmmerHit {
                        id: hit.name.clone(),
                        gene_name: profile_name.to_string(),
                        seq_len: seq.len() as u32,
                        i_evalue: inclusion_evalue,
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

    let (targets, hmmer_options, needed_profiles, amino_alphabet) = {
        let guard = CAS_CTX.lock().unwrap();
        let ctx = guard
            .as_ref()
            .ok_or_else(|| JsValue::from_str("cas_prepare not called"))?;
        (
            ctx.targets.clone(),
            ctx.hmmer_options.clone(),
            ctx.needed_profiles.clone(),
            ctx.amino_alphabet.clone(),
        )
    };

    let avg_len = if targets.is_empty() {
        400
    } else {
        targets.iter().map(|s| s.len()).sum::<usize>() / targets.len()
    };
    let coverage_threshold = hmmer_options.coverage_profile;
    let target_count = targets.len();

    let results: Vec<(String, Vec<HmmerHit>)> = profiles
        .par_iter()
        .filter(|p| needed_profiles.contains(&p.name))
        .filter_map(|p| {
            let mut reader = std::io::BufReader::new(p.data.as_bytes());
            let (_abc, hmm) = read_hmm(&mut reader).ok()?;
            let num_nodes = hmm.num_nodes;

            let bg = BackgroundModel::new(&amino_alphabet);
            let mut gm = Profile::new(num_nodes, &amino_alphabet);
            modelconfig::profile_config(&hmm, &bg, &mut gm, avg_len, SearchMode::Local);

            let query = SearchQuery::from_configured_profile(gm, bg).ok()?;
            let plan = SearchPlan::builder(query)
                .filters(FilterPolicy::default())
                .build()
                .ok()?;
            let mut worker = plan
                .spawn_worker(CapacityHints {
                    target_length: avg_len,
                })
                .ok()?;

            let mut profile_hits = Vec::new();
            for seq in &targets {
                if let Ok(report) = worker.search(seq) {
                    match &report.outcome {
                        SearchOutcome::Hit(hit) => {
                            for domain in &hit.domains {
                                let Some(inclusion_evalue) =
                                    included_domain_evalue(domain.log_pvalue, target_count)
                                else {
                                    continue;
                                };
                                let prof_cov =
                                    (domain.hmm_to - domain.hmm_from + 1) as f64 / num_nodes as f64;
                                if prof_cov < coverage_threshold {
                                    continue;
                                }
                                let seq_cov = (domain.alignment_end - domain.alignment_start + 1)
                                    as f64
                                    / seq.len() as f64;
                                profile_hits.push(HmmerHit {
                                    id: hit.name.clone(),
                                    gene_name: p.name.clone(),
                                    seq_len: seq.len() as u32,
                                    i_evalue: inclusion_evalue,
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

    let total_hits: u32 = results.iter().map(|(_, h)| h.len() as u32).sum();
    log!(
        "[CAS] HMM search complete: {} total hits across {} profiles",
        total_hits,
        results.len()
    );
    for (name, hits) in &results {
        log!("[CAS]   Profile {}: {} hits", name, hits.len());
        for h in hits {
            log!(
                "[CAS]     hit id={} score={:.1} evalue={:.2e} prof_cov={:.3} seq_cov={:.3}",
                h.id,
                h.score,
                h.i_evalue,
                h.profile_coverage,
                h.seq_coverage
            );
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
    let ctx = CAS_CTX
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| JsValue::from_str("cas_prepare not called"))?;

    let gene_map = gene_coordinates_from_records(&ctx.genes);

    let protein_ids: Vec<&str> = ctx.genes.iter().map(|g| g.id.as_str()).collect();
    let seq_index = SequenceIndex::from_ids(&protein_ids);

    let mut detected_systems = Vec::new();
    log!(
        "[CAS] cas_finalize: {} genes, {} profiles with hits",
        ctx.genes.len(),
        ctx.all_hits.len()
    );
    log!(
        "[CAS] all_hits keys: {:?}",
        ctx.all_hits.keys().collect::<Vec<_>>()
    );

    for model_def in &ctx.models_raw {
        let model_fully_qualified_name = format!("{}/{}", model_def.family, model_def.name);
        let model = match ctx.registry.get(&model_fully_qualified_name) {
            Some(m) => m,
            None => continue,
        };

        let system_hits = assign_hits_to_model(model, &ctx.all_hits, &seq_index);
        if system_hits.is_empty() {
            continue;
        }
        log!(
            "[CAS] Model {}: {} system_hits assigned",
            model_fully_qualified_name,
            system_hits.len()
        );

        let mut hits_for_clustering = system_hits;
        let clusters = cluster_hits(
            &mut hits_for_clustering,
            model.inter_gene_max_space,
            ctx.genes.len(),
            RepliconTopology::Circular,
        );
        log!(
            "[CAS] Model {}: {} clusters formed (inter_gene_max_space={})",
            model_fully_qualified_name,
            clusters.len(),
            model.inter_gene_max_space
        );

        for (ci, c) in clusters.iter().enumerate() {
            if c.is_empty() {
                continue;
            }
            log!(
                "[CAS] Model {} cluster {}: {} hits, positions {:?}",
                model_fully_qualified_name,
                ci,
                c.hits.len(),
                c.hits.iter().map(|h| h.position).collect::<Vec<_>>()
            );
            if let Some(system) = evaluate_cluster(c, model) {
                log!(
                    "[CAS] Model {} cluster {}: ACCEPTED (score={:.1}, mandatory={:?}, accessory={:?})",
                    model_fully_qualified_name,
                    ci,
                    system.score,
                    system.mandatory_found,
                    system.accessory_found
                );
                detected_systems.push(system);
            } else {
                let mandatory_names: std::collections::HashSet<&str> =
                    model.mandatory_genes().map(|g| g.name.as_str()).collect();
                let accessory_names: std::collections::HashSet<&str> =
                    model.accessory_genes().map(|g| g.name.as_str()).collect();
                let forbidden_names: std::collections::HashSet<&str> =
                    model.forbidden_genes().map(|g| g.name.as_str()).collect();
                let found_genes: std::collections::HashSet<&str> =
                    c.hits.iter().map(|h| h.gene_ref.as_str()).collect();
                let mand_found: Vec<&&str> = mandatory_names
                    .iter()
                    .filter(|g| found_genes.contains(**g))
                    .collect();
                let acc_found: Vec<&&str> = accessory_names
                    .iter()
                    .filter(|g| found_genes.contains(**g))
                    .collect();
                let forb_found: Vec<&&str> = forbidden_names
                    .iter()
                    .filter(|g| found_genes.contains(**g))
                    .collect();
                log!(
                    "[CAS] Model {} cluster {}: REJECTED (min_mandatory={}, min_genes={}, mandatory_found={:?}, accessory_found={:?}, forbidden_found={:?})",
                    model_fully_qualified_name,
                    ci,
                    model.min_mandatory_genes_required,
                    model.min_genes_required,
                    mand_found,
                    acc_found,
                    forb_found
                );
            }
        }
    }

    if detected_systems.len() > 1 {
        detected_systems = select_best_solution(detected_systems);
    }
    log!(
        "[CAS] After select_best_solution: {} systems",
        detected_systems.len()
    );

    const MIN_BEST_HIT_SCORE: f64 = 25.0;
    detected_systems.retain(|sys| {
        let max_score = sys
            .hits
            .iter()
            .map(|h| h.hit.score)
            .fold(f64::NEG_INFINITY, f64::max);
        let keep = max_score >= MIN_BEST_HIT_SCORE;
        if !keep {
            log!(
                "[CAS] Filtered out system {} (max_score={:.1} < {})",
                sys.model_fully_qualified_name,
                max_score,
                MIN_BEST_HIT_SCORE
            );
        }
        keep
    });
    log!(
        "[CAS] Final systems after score filter: {}",
        detected_systems.len()
    );

    let cas_clusters: Vec<CasCluster> = from_search_results_with_gene_map(
        &crispr_cas_finder_core::cas_types::SearchResults {
            systems: detected_systems,
            rejected: Vec::new(),
            skipped_replicons: Vec::new(),
        },
        &gene_map,
    );

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

#[cfg(test)]
mod tests {
    use super::*;
    use crispr_cas_finder_core::cas_types::{GeneDefinition, GeneStatus, SystemModel};

    fn test_hit(id: &str, gene_name: &str, score: f64) -> HmmerHit {
        HmmerHit {
            id: id.to_string(),
            gene_name: gene_name.to_string(),
            seq_len: 100,
            i_evalue: 0.0,
            score,
            profile_coverage: 0.9,
            seq_coverage: 0.8,
            begin_match: 1,
            end_match: 100,
        }
    }

    fn test_gene(name: &str, status: GeneStatus, system_ref: Option<&str>) -> GeneDefinition {
        GeneDefinition {
            name: name.to_string(),
            status,
            loner: false,
            multi_system: false,
            system_ref: system_ref.map(str::to_string),
            exchangeables: Vec::new(),
            inter_gene_max_space: None,
            multi_model: false,
        }
    }

    #[test]
    fn domain_inclusion_uses_database_sized_evalue() {
        let target_count = 4_318;
        let accepted_log_pvalue = (0.005 / target_count as f64).ln();
        let rejected_log_pvalue = (0.02 / target_count as f64).ln();

        let accepted = included_domain_evalue(accepted_log_pvalue, target_count).unwrap();
        assert!((accepted - 0.005).abs() < 1e-12);
        assert!(included_domain_evalue(rejected_log_pvalue, target_count).is_none());
        assert!(included_domain_evalue(f64::NAN, target_count).is_none());
    }

    #[test]
    fn assign_hits_prefers_local_gene_over_inherited_duplicate() {
        let model = SystemModel {
            fully_qualified_name: "CASFinder/CAS-TypeIE".to_string(),
            family: "CASFinder".to_string(),
            name: "CAS-TypeIE".to_string(),
            version: None,
            genes: vec![
                test_gene("Cas1_0_IE", GeneStatus::Mandatory, None),
                test_gene("Cas1_0_I-II-III", GeneStatus::Accessory, Some("CAS")),
            ],
            inter_gene_max_space: 5,
            min_mandatory_genes_required: 1,
            min_genes_required: 1,
            multi_loci: false,
        };

        let hmmer_hits = HashMap::from([
            (
                "Cas1_0_IE".to_string(),
                vec![test_hit("orf_1", "Cas1_0_IE", 90.0)],
            ),
            (
                "Cas1_0_I-II-III".to_string(),
                vec![test_hit("orf_1", "Cas1_0_I-II-III", 140.0)],
            ),
        ]);
        let seq_index = SequenceIndex::from_ids(&["orf_1"]);

        let hits = assign_hits_to_model(&model, &hmmer_hits, &seq_index);

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hit.id, "orf_1");
        assert_eq!(hits[0].gene_ref, "Cas1_0_IE");
    }

    #[test]
    fn assign_hits_keeps_best_scoring_hit_for_same_gene_and_orf() {
        let model = SystemModel {
            fully_qualified_name: "CASFinder/CAS-TypeIE".to_string(),
            family: "CASFinder".to_string(),
            name: "CAS-TypeIE".to_string(),
            version: None,
            genes: vec![test_gene("Cas3_0_I", GeneStatus::Accessory, Some("CAS"))],
            inter_gene_max_space: 5,
            min_mandatory_genes_required: 0,
            min_genes_required: 1,
            multi_loci: false,
        };

        let hmmer_hits = HashMap::from([(
            "Cas3_0_I".to_string(),
            vec![
                test_hit("orf_1", "Cas3_0_I", 55.0),
                test_hit("orf_1", "Cas3_0_I", 65.0),
            ],
        )]);
        let seq_index = SequenceIndex::from_ids(&["orf_1"]);

        let hits = assign_hits_to_model(&model, &hmmer_hits, &seq_index);

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hit.score, 65.0);
    }
}
