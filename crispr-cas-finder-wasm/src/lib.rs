use crispr_cas_finder_core::hmmer_core::{alphabet::Alphabet, sequence::DigitalSequence};
use crispr_cas_finder_core::hmmer_io::read_hmm;
use crispr_cas_finder_core::{
    CasFinderConfig, CrisprArray, DetectionParams, SequenceRecord,
    annotation::{AUTO_METAGENOME_BELOW, GenePredictionOptions, Proteome, predict_proteome},
    cas_pipeline::{ModelDefinition, build_model_registry, validate_genetic_code},
    cas_types::{HmmerHit, ModelRegistry, RepliconTopology},
    casfinder::{finalize_systems, legacy_hit_filtering, search_hmm_profile},
    casparser::CasCluster,
    detect_crisprs,
    macsy::HitWeights,
    report::{CasParameters, DEFAULT_VICINITY, ReportParameters, SequenceInfo, build_report},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard};
use wasm_bindgen::prelude::*;
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

/// CAS detection options.
#[derive(Deserialize, Default)]
pub struct CasOptions {
    pub genetic_code: Option<usize>,
    pub metagenome: Option<bool>,
    /// Sequences shorter than this use metagenomic gene prediction; 0 disables.
    pub auto_metagenome_below: Option<usize>,
    /// Name of the model set the supplied models belong to.
    pub definition: Option<String>,
    /// Search Cas systems only on sequences carrying a CRISPR array.
    pub only_with_crispr: Option<bool>,
    /// Sequence IDs to annotate; all sequences when absent.
    pub sequence_ids: Option<Vec<String>>,
}

/// Options for relating arrays and Cas systems in the final report.
#[derive(Deserialize, Default)]
pub struct ReportOptions {
    pub vicinity: Option<usize>,
    pub locus_distance: Option<usize>,
}

fn parse_fasta_sequences(fasta_content: &str) -> Result<Vec<SequenceRecord>, String> {
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
        sequences.push(SequenceRecord {
            id: record.id().to_string(),
            description: record.desc().unwrap_or_default().to_string(),
            sequence: record.seq().to_vec(),
        });
    }
    if sequences.is_empty() {
        return Err("FASTA input contains no sequences".to_string());
    }
    Ok(sequences)
}

/// Serialize as plain JS objects and arrays. The default serializer turns
/// maps into JS `Map`s, which `postMessage` consumers and JSON cannot read.
fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
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
        .map(|record| (record.id.as_str(), record.sequence.as_slice()))
        .collect();

    let arrays = detect_crisprs(&seq_refs, &params);
    to_js(&arrays)
}

// ---------------------------------------------------------------------------
// Step-by-step CAS detection API (avoids long blocking + 21 MB postMessage)
// ---------------------------------------------------------------------------

/// Opaque handle holding intermediate CAS detection state.
struct CasContext {
    proteome: Proteome,
    targets: Vec<DigitalSequence>,
    amino_alphabet: Alphabet,
    registry: ModelRegistry,
    needed_profiles: HashSet<String>,
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
    validate_genetic_code(genetic_code).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let prediction_options = GenePredictionOptions {
        genetic_code,
        metagenome: cas_opts.metagenome.unwrap_or(false),
        auto_metagenome_below: cas_opts
            .auto_metagenome_below
            .unwrap_or(AUTO_METAGENOME_BELOW),
    };
    let sequences: Vec<(String, Vec<u8>)> = parse_fasta_sequences(fasta_content)
        .map_err(|e| JsValue::from_str(&e))?
        .into_iter()
        .map(|record| (record.id, record.sequence))
        .collect();
    let include: Option<HashSet<String>> =
        cas_opts.sequence_ids.map(|ids| ids.into_iter().collect());
    let proteome = predict_proteome(&sequences, &prediction_options, include.as_ref())
        .map_err(|e| JsValue::from_str(&format!("Gene prediction failed: {e}")))?;

    let amino_alphabet = Alphabet::amino();
    let targets = proteome
        .genes
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
    let gene_count = proteome.genes.len();
    *context = Some(CasContext {
        proteome,
        targets,
        amino_alphabet,
        registry,
        needed_profiles,
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
    to_js(&info)
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
    search_hmm_profile(
        profile_name,
        &alphabet,
        &hmm,
        &ctx.targets,
        &ctx.amino_alphabet,
        &legacy_hit_filtering(),
    )
    .map_err(|e| e.to_string())
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
    let model_names: Vec<_> = ctx
        .models_raw
        .iter()
        .map(|m| format!("{}/{}", m.family, m.name))
        .collect();
    let (_, cas_clusters): (_, Vec<CasCluster>) = finalize_systems(
        &ctx.registry,
        &model_names,
        &ctx.all_hits,
        &ctx.proteome,
        RepliconTopology::Circular,
        CasFinderConfig::default().min_best_hit_score,
        &HitWeights::default(),
    );
    to_js(&cas_clusters)
}

/// Step 4 — assemble the same report the command-line tool writes.
///
/// `crisprs` is the result of `find_repeats` and `cas_clusters` the result
/// of `cas_finalize`. Pass `null` as `cas_options` when Cas detection did
/// not run.
#[wasm_bindgen]
pub fn build_analysis_report(
    fasta_content: &str,
    crisprs_js: JsValue,
    cas_clusters_js: JsValue,
    options_js: JsValue,
    cas_options_js: JsValue,
    report_options_js: JsValue,
) -> Result<JsValue, JsValue> {
    let crisprs: Vec<CrisprArray> = serde_wasm_bindgen::from_value(crisprs_js)
        .map_err(|e| JsValue::from_str(&format!("invalid CRISPR arrays: {e}")))?;
    let cas_clusters: Vec<CasCluster> = parse_options(cas_clusters_js)?;
    let options: WasmFinderOptions = parse_options(options_js)?;
    let detection = build_detection_params(&options).map_err(|e| JsValue::from_str(&e))?;
    let cas_ran = !(cas_options_js.is_null() || cas_options_js.is_undefined());
    let cas_options: CasOptions = parse_options(cas_options_js)?;
    let report_options: ReportOptions = parse_options(report_options_js)?;
    let defaults = CasFinderConfig::default();

    let parameters = ReportParameters {
        detection,
        cas: cas_ran.then(|| CasParameters {
            definition: cas_options.definition.unwrap_or(defaults.definition),
            genetic_code: cas_options.genetic_code.unwrap_or(defaults.genetic_code),
            metagenome: cas_options.metagenome.unwrap_or(defaults.metagenome),
            auto_metagenome_below: cas_options
                .auto_metagenome_below
                .unwrap_or(AUTO_METAGENOME_BELOW),
            user_annotation: false,
            topology: RepliconTopology::Circular,
            min_best_hit_score: defaults.min_best_hit_score,
            only_with_crispr: cas_options.only_with_crispr.unwrap_or(false),
        }),
        vicinity: report_options.vicinity.unwrap_or(DEFAULT_VICINITY),
        locus_distance: report_options.locus_distance,
    };
    let sequences: Vec<SequenceInfo> = parse_fasta_sequences(fasta_content)
        .map_err(|e| JsValue::from_str(&e))?
        .iter()
        .map(SequenceInfo::from_record)
        .collect();
    to_js(&build_report(&sequences, crisprs, cas_clusters, parameters))
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
                SequenceRecord {
                    id: "a".into(),
                    description: "description".into(),
                    sequence: b"ACGTTGCA".to_vec(),
                },
                SequenceRecord {
                    id: "b".into(),
                    description: String::new(),
                    sequence: b"AAAA".to_vec(),
                },
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
            proteome: Proteome::default(),
            targets: vec![],
            amino_alphabet: Alphabet::amino(),
            registry: ModelRegistry::new(),
            needed_profiles: HashSet::new(),
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
            max_nb_genes: None,
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
            max_nb_genes: None,
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
