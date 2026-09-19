use crispr_cas_finder_core::hmmer_core::{
    alphabet::Alphabet,
    background::BackgroundModel,
    config::SearchMode,
    modelconfig,
    pipeline::search::{CapacityHints, FilterPolicy, SearchOutcome, SearchPlan, SearchQuery},
    profile::Profile,
    sequence::DigitalSequence,
};
use crispr_cas_finder_core::hmmer_io::read_hmm;
use crispr_cas_finder_core::{
    DetectionParams,
    cas_pipeline::{
        GeneRecord, ModelDefinition, build_model_registry, codon_table,
        gene_coordinates_from_records, reverse_complement_dna, translate_dna,
        validate_genetic_code,
    },
    cas_types::{HmmerHit, HmmerOptions, ModelRegistry, RepliconTopology, select_best_solution},
    casfinder::{RepliconProteins, evaluate_detected_systems},
    casparser::{CasCluster, from_search_results_with_gene_map},
    detect_crisprs,
};
use orphos_core::{OrphosAnalyzer, config::OrphosConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard};
use wasm_bindgen::prelude::*;
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

fn parse_fasta_sequences(fasta_content: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
    let reader = bio::io::fasta::Reader::new(fasta_content.as_bytes());
    let mut sequences = Vec::new();
    let mut ids = HashSet::new();
    for record in reader.records() {
        let record = record.map_err(|e| format!("invalid FASTA: {e}"))?;
        record.check().map_err(|e| format!("invalid FASTA: {e}"))?;
        if record.seq().is_empty() {
            return Err(format!("empty FASTA sequence: {}", record.id()));
        }
        if !ids.insert(record.id().to_string()) {
            return Err(format!("duplicate FASTA identifier: {}", record.id()));
        }
        sequences.push((record.id().to_string(), record.seq().to_vec()));
    }
    if sequences.is_empty() {
        return Err("FASTA input contains no sequences".to_string());
    }
    Ok(sequences)
}

fn parse_options<T: serde::de::DeserializeOwned + Default>(value: JsValue) -> Result<T, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(T::default());
    }
    serde_wasm_bindgen::from_value(value)
        .map_err(|e| JsValue::from_str(&format!("invalid options: {e}")))
}

#[wasm_bindgen]
pub fn find_repeats(fasta_content: &str, options_js: JsValue) -> Result<JsValue, JsValue> {
    let options: WasmFinderOptions = parse_options(options_js)?;
    let params = build_detection_params(&options).map_err(|e| JsValue::from_str(&e))?;

    let sequences = parse_fasta_sequences(fasta_content).map_err(|e| JsValue::from_str(&e))?;
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
    replicons: Vec<RepliconProteins>,
    targets: Vec<DigitalSequence>,
    amino_alphabet: Alphabet,
    registry: ModelRegistry,
    needed_profiles: HashSet<String>,
    hmmer_options: HmmerOptions,
    all_hits: HashMap<String, Vec<HmmerHit>>,
    models_raw: Vec<ModelDefinition>,
}

static CAS_CTX: Mutex<Option<CasContext>> = Mutex::new(None);

fn lock_context() -> Result<MutexGuard<'static, Option<CasContext>>, JsValue> {
    CAS_CTX
        .lock()
        .map_err(|_| JsValue::from_str("CAS context lock poisoned; restart the worker"))
}

/// Discard an unfinished analysis, for example after a profile download/search fails.
#[wasm_bindgen]
pub fn cas_abort() -> Result<(), JsValue> {
    *lock_context()? = None;
    Ok(())
}

/// Step 1 — predict genes, translate proteins, build model registry.
/// Returns the list of needed profile names as a JS array of strings.
#[wasm_bindgen]
pub fn cas_prepare(
    fasta_content: &str,
    models_js: JsValue,
    cas_options_js: JsValue,
) -> Result<JsValue, JsValue> {
    let mut context = lock_context()?;
    if context.is_some() {
        return Err(JsValue::from_str(
            "CAS analysis already active; finalize or abort it first",
        ));
    }
    let models: Vec<ModelDefinition> = serde_wasm_bindgen::from_value(models_js)
        .map_err(|e| JsValue::from_str(&format!("invalid models: {e}")))?;
    let cas_opts: CasOptions = parse_options(cas_options_js)?;

    let genetic_code = cas_opts.genetic_code.unwrap_or(11);
    let metagenome = cas_opts.metagenome.unwrap_or(false);
    let translation_table =
        validate_genetic_code(genetic_code).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let config = OrphosConfig {
        metagenomic: metagenome,
        closed_ends: true,
        quiet: true,
        translation_table: Some(translation_table),
        ..OrphosConfig::default()
    };
    let analyzer = OrphosAnalyzer::new(config);
    let sequences = parse_fasta_sequences(fasta_content).map_err(|e| JsValue::from_str(&e))?;

    let mut all_genes: Vec<GeneRecord> = Vec::new();
    let mut replicons = Vec::new();
    let table = codon_table(genetic_code).map_err(|e| JsValue::from_str(&e.to_string()))?;

    for (seq_id, seq_bytes) in &sequences {
        let mut protein_ids = Vec::new();
        let results = analyzer
            .analyze_sequence_bytes(seq_bytes, seq_id.clone(), None)
            .map_err(|e| JsValue::from_str(&format!("Gene prediction failed: {e}")))?;

        for gene in &results.genes {
            let begin = gene.coordinates.begin.saturating_sub(1);
            let end = gene.coordinates.end.saturating_sub(1);
            if begin > end || end >= seq_bytes.len() {
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
            protein_ids.push(gene_id.clone());
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
        replicons.push(RepliconProteins {
            name: seq_id.clone(),
            protein_ids,
        });
    }

    let amino_alphabet = Alphabet::amino();
    let targets = all_genes
        .iter()
        .map(|gene| {
            DigitalSequence::from_bytes(&gene.id, "", gene.protein.as_bytes(), &amino_alphabet)
        })
        .collect();

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
    *context = Some(CasContext {
        replicons,
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

    // Serialize a struct as an ordinary JS object. A serde_json::Value map
    // becomes a JS Map, whose fields are invisible to worker.js property reads.
    #[derive(Serialize)]
    struct PreparationInfo {
        needed_profiles: Vec<String>,
        gene_count: usize,
    }
    let info = PreparationInfo {
        needed_profiles: needed_list,
        gene_count,
    };
    serde_wasm_bindgen::to_value(&info)
        .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
}

/// Step 2 — search one HMM profile against the target proteins.
/// `profile_name` and `profile_data` are the name and text content of one .hmm file.
/// Returns the number of hits found for this profile.
#[wasm_bindgen]
pub fn cas_search_profile(profile_name: &str, profile_data: &str) -> Result<u32, JsValue> {
    let mut guard = lock_context()?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| JsValue::from_str("cas_prepare not called"))?;
    if !ctx.needed_profiles.contains(profile_name) {
        return Ok(0);
    }
    let hits =
        search_profile(ctx, profile_name, profile_data).map_err(|e| JsValue::from_str(&e))?;
    let count = hits.len() as u32;
    // Replace even an empty result, so re-searching cannot leave stale hits.
    ctx.all_hits.insert(profile_name.to_string(), hits);
    Ok(count)
}

fn search_profile(
    ctx: &CasContext,
    profile_name: &str,
    profile_data: &str,
) -> Result<Vec<HmmerHit>, String> {
    let mut reader = std::io::BufReader::new(profile_data.as_bytes());
    let (alphabet, hmm) =
        read_hmm(&mut reader).map_err(|e| format!("Failed to parse HMM {profile_name}: {e}"))?;
    if alphabet.kind != ctx.amino_alphabet.kind || hmm.num_nodes == 0 {
        return Err(format!(
            "HMM {profile_name} must be a nonempty amino-acid model"
        ));
    }
    let avg_len = if ctx.targets.is_empty() {
        400
    } else {
        ctx.targets.iter().map(|s| s.len()).sum::<usize>() / ctx.targets.len()
    };
    let bg = BackgroundModel::new(&ctx.amino_alphabet);
    let mut gm = Profile::new(hmm.num_nodes, &ctx.amino_alphabet);
    modelconfig::profile_config(&hmm, &bg, &mut gm, avg_len, SearchMode::Local);
    let query = SearchQuery::from_configured_profile(gm, bg)
        .map_err(|e| format!("Query config failed for {profile_name}: {e}"))?;
    let plan = SearchPlan::builder(query)
        .filters(FilterPolicy::default())
        .build()
        .map_err(|e| format!("Search plan failed for {profile_name}: {e}"))?;
    let mut worker = plan
        .spawn_worker(CapacityHints {
            target_length: avg_len,
        })
        .map_err(|e| format!("Spawn worker failed for {profile_name}: {e}"))?;
    let mut hits = Vec::new();
    for seq in &ctx.targets {
        let report = worker
            .search(seq)
            .map_err(|e| format!("Search error for {profile_name}: {e}"))?;
        if let SearchOutcome::Hit(hit) = report.outcome {
            for domain in hit.domains {
                let Some(i_evalue) = included_domain_evalue(domain.log_pvalue, ctx.targets.len())
                else {
                    continue;
                };
                let Some(profile_span) = domain
                    .hmm_to
                    .checked_sub(domain.hmm_from)
                    .and_then(|v| v.checked_add(1))
                else {
                    continue;
                };
                let Some(sequence_span) = domain
                    .alignment_end
                    .checked_sub(domain.alignment_start)
                    .and_then(|v| v.checked_add(1))
                else {
                    continue;
                };
                let profile_coverage = profile_span as f64 / hmm.num_nodes as f64;
                if profile_coverage < ctx.hmmer_options.coverage_profile || seq.is_empty() {
                    continue;
                }
                hits.push(HmmerHit {
                    id: hit.name.clone(),
                    gene_name: profile_name.to_string(),
                    seq_len: seq.len() as u32,
                    i_evalue,
                    score: domain.bitscore as f64,
                    profile_coverage,
                    seq_coverage: sequence_span as f64 / seq.len() as f64,
                    begin_match: domain.alignment_start as u32,
                    end_match: domain.alignment_end as u32,
                });
            }
        }
    }
    Ok(hits)
}

/// Search all required profiles in parallel. Any parse/search error fails the
/// request; results are committed only after every profile succeeds.
#[wasm_bindgen]
pub fn cas_search_all_profiles(profiles_js: JsValue) -> Result<u32, JsValue> {
    #[derive(Deserialize)]
    struct ProfileEntry {
        name: String,
        data: String,
    }
    let profiles: Vec<ProfileEntry> = serde_wasm_bindgen::from_value(profiles_js)
        .map_err(|e| JsValue::from_str(&format!("invalid profiles: {e}")))?;
    let mut guard = lock_context()?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| JsValue::from_str("cas_prepare not called"))?;
    let supplied: HashSet<_> = profiles.iter().map(|p| p.name.as_str()).collect();
    for name in &ctx.needed_profiles {
        if !supplied.contains(name.as_str()) {
            return Err(JsValue::from_str(&format!(
                "Required HMM profile is missing: {name}"
            )));
        }
    }
    let results: Result<Vec<_>, String> = profiles
        .par_iter()
        .filter(|p| ctx.needed_profiles.contains(&p.name))
        .map(|p| search_profile(ctx, &p.name, &p.data).map(|hits| (p.name.clone(), hits)))
        .collect();
    let results = results.map_err(|e| JsValue::from_str(&e))?;
    let total = results.iter().map(|(_, hits)| hits.len() as u32).sum();
    for (name, hits) in results {
        ctx.all_hits.insert(name, hits);
    }
    Ok(total)
}

/// Step 3 — cluster hits, evaluate systems, and return CAS results.
/// Consumes the context.
#[wasm_bindgen]
pub fn cas_finalize() -> Result<JsValue, JsValue> {
    let ctx = lock_context()?
        .take()
        .ok_or_else(|| JsValue::from_str("cas_prepare not called"))?;
    let gene_map = gene_coordinates_from_records(&ctx.genes);
    let model_names: Vec<_> = ctx
        .models_raw
        .iter()
        .map(|m| format!("{}/{}", m.family, m.name))
        .collect();
    let mut detected_systems = evaluate_detected_systems(
        &ctx.registry,
        &model_names,
        &ctx.all_hits,
        &ctx.replicons,
        RepliconTopology::Circular,
    );
    if detected_systems.len() > 1 {
        detected_systems = select_best_solution(detected_systems);
    }
    let minimum_score = crispr_cas_finder_core::CasFinderConfig::default().min_best_hit_score;
    detected_systems.retain(|system| system.hits.iter().any(|hit| hit.hit.score >= minimum_score));

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

fn build_detection_params(options: &WasmFinderOptions) -> Result<DetectionParams, String> {
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
    params.validate()?;
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crispr_cas_finder_core::cas_types::{GeneDefinition, GeneStatus, SystemModel};
    use crispr_cas_finder_core::{cas_pipeline::assign_hits_to_model, cas_types::SequenceIndex};

    #[test]
    fn fasta_parser_rejects_missing_empty_and_duplicate_records() {
        for invalid in ["", "ACGT", ">\nACGT", ">a\n", ">a\nACGT\n>a\nACGT"] {
            assert!(parse_fasta_sequences(invalid).is_err(), "{invalid:?}");
        }
        let sequences =
            parse_fasta_sequences(">a description\r\nACGT\r\nTGCA\r\n>b\nAAAA").unwrap();
        assert_eq!(
            sequences,
            vec![
                ("a".into(), b"ACGTTGCA".to_vec()),
                ("b".into(), b"AAAA".to_vec())
            ]
        );
    }

    #[test]
    fn options_reject_invalid_ranges_and_preserve_defaults() {
        let defaults = build_detection_params(&WasmFinderOptions::default()).unwrap();
        assert_eq!(defaults.min_evidence_level, 4);
        for options in [
            WasmFinderOptions {
                min_repeat_length: Some(0),
                ..Default::default()
            },
            WasmFinderOptions {
                max_spacer_length: Some(1),
                ..Default::default()
            },
            WasmFinderOptions {
                min_evidence_level: Some(5),
                ..Default::default()
            },
            WasmFinderOptions {
                min_spacer_count: Some(usize::MAX),
                ..Default::default()
            },
        ] {
            assert!(build_detection_params(&options).is_err());
        }
    }

    #[test]
    fn profile_errors_are_not_silently_skipped() {
        let ctx = CasContext {
            genes: vec![],
            replicons: vec![],
            targets: vec![],
            amino_alphabet: Alphabet::amino(),
            registry: ModelRegistry::new(),
            needed_profiles: HashSet::new(),
            hmmer_options: HmmerOptions::default(),
            all_hits: HashMap::new(),
            models_raw: vec![],
        };
        let error = search_profile(&ctx, "broken", "not an HMM").unwrap_err();
        assert!(error.contains("Failed to parse HMM broken"));
    }

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
