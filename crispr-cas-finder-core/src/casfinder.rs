pub use crate::annotation::RepliconProteins;
use crate::annotation::{
    GenePredictionOptions, Proteome, predict_proteome, proteome_from_annotation,
};
use crate::cas_pipeline::{
    ModelDefinition, assign_hits_to_model, build_model_registry, evaluate_cluster,
    gene_coordinates_from_records, validate_genetic_code,
};
use crate::cas_types::{
    DetectedSystem, HmmerHit, ModelRegistry, RepliconTopology, SearchResults, SequenceIndex,
    cluster_hits, select_best_solution,
};
use crate::casparser::{CasCluster, from_search_results_with_gene_map};
use crate::hmmer_core::{
    alphabet::Alphabet,
    background::BackgroundModel,
    config::SearchMode,
    hmm::Hmm,
    modelconfig,
    pipeline::search::{CapacityHints, FilterPolicy, SearchOutcome, SearchPlan, SearchQuery},
    profile::Profile,
    sequence::DigitalSequence,
};
use crate::hmmer_io::HmmFile;
use crate::types::CasFinderConfig;
use anyhow::{Context, Result};
use bio::io::fasta::Reader as FastaReader;
use log::info;
use rayon::{ThreadPoolBuilder, prelude::*};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// CAS profiles must cover at least 40% of the HMM to count as gene hits.
pub const MIN_PROFILE_COVERAGE: f64 = 0.4;
/// Match HMMER's default per-domain inclusion threshold.
pub const MAX_DOMAIN_INCLUSION_EVALUE: f64 = 0.01;

/// Outcome of Cas detection: the annotated proteins and the systems found.
#[derive(Debug, Default)]
pub struct CasAnalysis {
    pub proteome: Proteome,
    pub systems: Vec<DetectedSystem>,
    pub clusters: Vec<CasCluster>,
}

/// Read a genome FASTA and detect Cas systems in it.
pub fn run_casfinder(input: &Path, config: &CasFinderConfig) -> Result<CasAnalysis> {
    validate_config(config)?;
    let reader = FastaReader::from_file(input)
        .with_context(|| format!("Cannot read genome FASTA: {}", input.display()))?;
    let mut sequences: Vec<(String, Vec<u8>)> = Vec::new();
    for record in reader.records() {
        let record = record.context("Reading FASTA record")?;
        sequences.push((record.id().to_string(), record.seq().to_vec()));
    }
    detect_cas_systems(&sequences, config, None)
}

fn validate_config(config: &CasFinderConfig) -> Result<()> {
    validate_genetic_code(config.genetic_code)?;
    if !config.min_best_hit_score.is_finite() {
        anyhow::bail!("Minimum CAS hit score must be finite");
    }
    if config.annotation_gff.is_some() != config.proteome_faa.is_some() {
        anyhow::bail!("An annotation GFF and a protein FASTA must be supplied together");
    }
    Ok(())
}

/// Detect Cas systems in in-memory sequences.
///
/// Proteins are predicted per sequence unless the configuration supplies an
/// annotation. `include` restricts the search to the listed sequence IDs,
/// which is how "only where a CRISPR array was found" is implemented.
pub fn detect_cas_systems(
    sequences: &[(String, Vec<u8>)],
    config: &CasFinderConfig,
    include: Option<&HashSet<String>>,
) -> Result<CasAnalysis> {
    validate_config(config)?;
    let proteome = load_proteome(sequences, config, include)?;
    if proteome.is_empty() {
        info!("No protein-coding gene to search, skipping CasFinder");
        return Ok(CasAnalysis {
            proteome,
            ..CasAnalysis::default()
        });
    }

    let amino_alphabet = Alphabet::amino();
    let protein_sequences: Vec<DigitalSequence> = proteome
        .genes
        .iter()
        .map(|gene| {
            DigitalSequence::from_bytes(&gene.id, "", gene.protein.as_bytes(), &amino_alphabet)
        })
        .collect();
    info!("Loaded {} protein sequences", protein_sequences.len());

    let models_dir = resolve_models_dir(config)?;
    let profiles_dir = resolve_profiles_dir(config)?;
    let (registry, model_fully_qualified_names) = build_model_registry_from_dir(&models_dir)?;
    if model_fully_qualified_names.is_empty() {
        anyhow::bail!("No CAS model definitions found in {}", models_dir.display());
    }
    let needed_profiles = collect_needed_profiles(&registry, &model_fully_qualified_names);
    if needed_profiles.is_empty() {
        anyhow::bail!(
            "CAS model definitions in {} do not reference any HMM profiles",
            models_dir.display()
        );
    }
    info!(
        "Need {} HMM profiles for {} models",
        needed_profiles.len(),
        model_fully_qualified_names.len()
    );

    let all_hits = run_hmm_search(
        &profiles_dir,
        &needed_profiles,
        &protein_sequences,
        &amino_alphabet,
        config.workers,
    )?;
    info!("HMM search found hits for {} profiles", all_hits.len());

    let (systems, clusters) = finalize_systems(
        &registry,
        &model_fully_qualified_names,
        &all_hits,
        &proteome,
        config.replicon_topology,
        config.min_best_hit_score,
    );
    info!("CasFinder found {} systems", systems.len());

    Ok(CasAnalysis {
        proteome,
        systems,
        clusters,
    })
}

fn load_proteome(
    sequences: &[(String, Vec<u8>)],
    config: &CasFinderConfig,
    include: Option<&HashSet<String>>,
) -> Result<Proteome> {
    if let (Some(gff_path), Some(faa_path)) = (&config.annotation_gff, &config.proteome_faa) {
        let gff = fs::read_to_string(gff_path)
            .with_context(|| format!("Cannot read annotation GFF: {}", gff_path.display()))?;
        let faa = fs::read_to_string(faa_path)
            .with_context(|| format!("Cannot read protein FASTA: {}", faa_path.display()))?;
        let mut proteome = proteome_from_annotation(&gff, &faa)?;
        if let Some(include) = include {
            proteome.genes.retain(|gene| include.contains(&gene.seq_id));
            proteome
                .replicons
                .retain(|replicon| include.contains(&replicon.name));
        }
        return Ok(proteome);
    }
    info!("Predicting genes for {} sequence(s)", sequences.len());
    predict_proteome(
        sequences,
        &GenePredictionOptions {
            genetic_code: config.genetic_code,
            metagenome: config.metagenome,
            auto_metagenome_below: config.auto_metagenome_below,
        },
        include,
    )
}

/// Cluster hits, pick the best non-overlapping systems, drop low-confidence
/// ones and attach genomic coordinates. Shared by native and WASM callers.
pub fn finalize_systems(
    registry: &ModelRegistry,
    model_fully_qualified_names: &[String],
    all_hits: &HashMap<String, Vec<HmmerHit>>,
    proteome: &Proteome,
    replicon_topology: RepliconTopology,
    min_best_hit_score: f64,
) -> (Vec<DetectedSystem>, Vec<CasCluster>) {
    let mut systems = evaluate_detected_systems(
        registry,
        model_fully_qualified_names,
        all_hits,
        &proteome.replicons,
        replicon_topology,
    );
    if systems.len() > 1 {
        systems = select_best_solution(systems);
    }
    filter_low_confidence_systems(&mut systems, min_best_hit_score);

    let gene_map = gene_coordinates_from_records(&proteome.genes);
    let clusters = from_search_results_with_gene_map(
        &SearchResults {
            systems: systems.clone(),
            rejected: Vec::new(),
            skipped_replicons: Vec::new(),
        },
        &gene_map,
    );
    (systems, clusters)
}

fn path_as_utf8<'a>(path: &'a Path, description: &str) -> Result<&'a str> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("{description} path is not valid UTF-8: {}", path.display()))
}

fn collect_needed_profiles(
    registry: &ModelRegistry,
    model_fully_qualified_names: &[String],
) -> HashSet<String> {
    model_fully_qualified_names
        .iter()
        .filter_map(|name| registry.get(name))
        .flat_map(|model| model.all_profile_names())
        .map(str::to_owned)
        .collect()
}

/// Resolve the directory containing CAS model definitions (XML files).
fn resolve_models_dir(config: &CasFinderConfig) -> Result<PathBuf> {
    if let Some(ref models_dir_path) = config.cas_models_dir {
        return Ok(models_dir_path.clone());
    }
    let definition_suffix = format!("DEF-{}", config.definition);
    let candidate_directories = [
        PathBuf::from(format!("CasFinder-2.0.3/{}-2.0.3", definition_suffix)),
        PathBuf::from(format!(
            "../CRISPRCasFinder/CasFinder-2.0.3/{}-2.0.3",
            definition_suffix
        )),
    ];
    for candidate_directory in &candidate_directories {
        if candidate_directory.is_dir() {
            return Ok(candidate_directory.clone());
        }
    }
    anyhow::bail!(
        "Cannot find CAS model definitions directory. \
         Tried: {:?}. Use --cas-models-dir to specify the path.",
        candidate_directories
    )
}

/// Resolve the directory containing CAS HMM profiles.
fn resolve_profiles_dir(config: &CasFinderConfig) -> Result<PathBuf> {
    if let Some(ref profiles_dir_path) = config.cas_profiles_dir {
        return Ok(profiles_dir_path.clone());
    }
    let candidate_directories = [
        PathBuf::from("CasFinder-2.0.3/CASprofiles-2.0.3"),
        PathBuf::from("../CRISPRCasFinder/CasFinder-2.0.3/CASprofiles-2.0.3"),
    ];
    for candidate_directory in &candidate_directories {
        if candidate_directory.is_dir() {
            return Ok(candidate_directory.clone());
        }
    }
    anyhow::bail!(
        "Cannot find CAS HMM profiles directory. \
         Tried: {:?}. Use --cas-profiles-dir to specify the path.",
        candidate_directories
    )
}

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

/// Build a ModelRegistry from CAS XML definitions on disk.
fn build_model_registry_from_dir(models_dir: &Path) -> Result<(ModelRegistry, Vec<String>)> {
    let family = "CASFinder".to_string();
    let mut models = Vec::new();

    let mut model_paths = fs::read_dir(models_dir)
        .with_context(|| format!("Cannot read CAS model directory: {}", models_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    model_paths.retain(|path| path.extension().is_some_and(|extension| extension == "xml"));
    model_paths.sort_unstable();

    // ModelRegistry is HashMap-backed; this separately retained, sorted list
    // defines deterministic evaluation and tie-breaking order.
    for path in model_paths {
        let model_name = path
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Cannot read CAS model: {}", path.display()))?;
        models.push(ModelDefinition {
            name: model_name,
            family: family.clone(),
            content,
        });
    }

    let model_fully_qualified_names: Vec<String> = models
        .iter()
        .map(|model| format!("{}/{}", model.family, model.name))
        .collect();
    let registry = build_model_registry(&models)
        .map_err(|e| anyhow::anyhow!("Failed to build CAS model registry: {}", e))?;
    Ok((registry, model_fully_qualified_names))
}

// ---------------------------------------------------------------------------
// HMM search via hmmer-core
// ---------------------------------------------------------------------------

/// Run the full HMMER3 pipeline (MSV -> Viterbi -> Forward) for each needed
/// profile against all target protein sequences.
fn run_hmm_search(
    profiles_dir: &Path,
    needed_profiles: &HashSet<String>,
    protein_sequences: &[DigitalSequence],
    alphabet: &Alphabet,
    workers: usize,
) -> Result<HashMap<String, Vec<HmmerHit>>> {
    let mut all_hits: HashMap<String, Vec<HmmerHit>> = HashMap::new();
    let mut profile_names: Vec<String> = needed_profiles.iter().cloned().collect();
    profile_names.sort_unstable();

    let pool = build_hmm_search_pool(workers)?;
    info!(
        "Searching {} HMM profiles across {} protein sequences with {} thread(s)",
        profile_names.len(),
        protein_sequences.len(),
        pool.current_num_threads()
    );
    let search_results = pool.install(|| {
        profile_names
            .par_iter()
            .map(|profile_name| {
                search_profile_hits(profiles_dir, profile_name, protein_sequences, alphabet)
            })
            .collect::<Vec<_>>()
    });

    for result in search_results {
        if let Some((profile_name, profile_hits)) = result? {
            all_hits.insert(profile_name, profile_hits);
        }
    }

    Ok(all_hits)
}

fn search_profile_hits(
    profiles_dir: &Path,
    profile_name: &str,
    protein_sequences: &[DigitalSequence],
    alphabet: &Alphabet,
) -> Result<Option<(String, Vec<HmmerHit>)>> {
    let hmm_path = profiles_dir.join(format!("{}.hmm", profile_name));
    if !hmm_path.is_file() {
        anyhow::bail!(
            "Required HMM profile '{}' is missing: {}",
            profile_name,
            hmm_path.display()
        );
    }

    let hmm_path_string = path_as_utf8(&hmm_path, "HMM profile")?;
    let mut hmm_file = HmmFile::open(hmm_path_string, None)
        .map_err(|e| anyhow::anyhow!("Failed to open {}: {}", profile_name, e))?;
    let (hmm_alphabet, hmm) = hmm_file
        .read()
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", profile_name, e))?;
    let profile_hits = search_hmm_profile(
        profile_name,
        &hmm_alphabet,
        &hmm,
        protein_sequences,
        alphabet,
    )?;
    Ok((!profile_hits.is_empty()).then(|| (profile_name.to_string(), profile_hits)))
}

/// Search one parsed HMM profile against all proteins and return the
/// domains passing the inclusion E-value and profile-coverage thresholds,
/// in protein order. Shared by native and WASM callers.
pub fn search_hmm_profile(
    profile_name: &str,
    hmm_alphabet: &Alphabet,
    hmm: &Hmm,
    protein_sequences: &[DigitalSequence],
    alphabet: &Alphabet,
) -> Result<Vec<HmmerHit>> {
    if hmm_alphabet.kind != alphabet.kind || hmm_alphabet.canonical_size != alphabet.canonical_size
    {
        anyhow::bail!(
            "HMM profile '{}' uses {:?} rather than the expected {:?} alphabet",
            profile_name,
            hmm_alphabet.kind,
            alphabet.kind
        );
    }
    let num_nodes = hmm.num_nodes;
    if num_nodes == 0 {
        anyhow::bail!("HMM profile '{}' contains no model nodes", profile_name);
    }
    let average_sequence_length = if protein_sequences.is_empty() {
        400
    } else {
        protein_sequences.iter().map(|s| s.len()).sum::<usize>() / protein_sequences.len()
    };
    // Searching in length order keeps the worker's matrices growing steadily.
    let mut sequence_search_order: Vec<usize> = (0..protein_sequences.len()).collect();
    sequence_search_order.sort_unstable_by_key(|&index| (protein_sequences[index].len(), index));

    let background_model = BackgroundModel::new(alphabet);
    let mut configured_profile = Profile::new(num_nodes, alphabet);
    modelconfig::profile_config(
        hmm,
        &background_model,
        &mut configured_profile,
        average_sequence_length,
        SearchMode::Local,
    );

    let query = SearchQuery::from_configured_profile(configured_profile, background_model.clone())
        .map_err(|e| anyhow::anyhow!("Query config failed for {}: {}", profile_name, e))?;
    let plan = SearchPlan::builder(query)
        .filters(FilterPolicy::default())
        .build()
        .map_err(|e| anyhow::anyhow!("Search plan config failed for {}: {}", profile_name, e))?;
    let mut worker = plan
        .spawn_worker(CapacityHints {
            target_length: average_sequence_length,
        })
        .map_err(|e| anyhow::anyhow!("Spawn worker failed for {}: {}", profile_name, e))?;

    let mut profile_hits: Vec<(usize, HmmerHit)> = Vec::new();

    for &sequence_index in &sequence_search_order {
        let target_sequence = &protein_sequences[sequence_index];
        if target_sequence.is_empty() {
            continue;
        }
        let report = worker
            .search(target_sequence)
            .map_err(|e| anyhow::anyhow!("Search error {}: {}", profile_name, e))?;

        if let SearchOutcome::Hit(hit) = &report.outcome {
            for domain in &hit.domains {
                let inclusion_evalue =
                    domain.log_pvalue.exp() * protein_sequences.len().max(1) as f64;
                if !inclusion_evalue.is_finite() || inclusion_evalue > MAX_DOMAIN_INCLUSION_EVALUE {
                    continue;
                }
                let Some(profile_span) = inclusive_span(domain.hmm_from, domain.hmm_to) else {
                    continue;
                };
                let profile_coverage = profile_span as f64 / num_nodes as f64;
                if profile_coverage < MIN_PROFILE_COVERAGE {
                    continue;
                }
                let Some(sequence_span) =
                    inclusive_span(domain.alignment_start, domain.alignment_end)
                else {
                    continue;
                };
                let sequence_coverage = sequence_span as f64 / target_sequence.len() as f64;
                profile_hits.push((
                    sequence_index,
                    HmmerHit {
                        id: hit.name.clone(),
                        gene_name: profile_name.to_string(),
                        seq_len: target_sequence.len() as u32,
                        i_evalue: inclusion_evalue,
                        score: domain.bitscore as f64,
                        profile_coverage,
                        seq_coverage: sequence_coverage,
                        begin_match: domain.alignment_start as u32,
                        end_match: domain.alignment_end as u32,
                    },
                ));
            }
        }
    }

    profile_hits.sort_by_key(|(sequence_index, _)| *sequence_index);
    Ok(profile_hits.into_iter().map(|(_, hit)| hit).collect())
}

fn inclusive_span(start: usize, end: usize) -> Option<usize> {
    end.checked_sub(start)?.checked_add(1)
}

fn build_hmm_search_pool(workers: usize) -> Result<rayon::ThreadPool> {
    let builder = if workers > 0 {
        ThreadPoolBuilder::new().num_threads(workers)
    } else {
        ThreadPoolBuilder::new()
    };

    builder
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build CasFinder HMM thread pool: {}", e))
}

// ---------------------------------------------------------------------------
// System evaluation (clustering + scoring)
// ---------------------------------------------------------------------------

/// Cluster and evaluate each replicon independently (shared by native and WASM).
pub fn evaluate_detected_systems(
    registry: &ModelRegistry,
    model_fully_qualified_names: &[String],
    all_hits: &HashMap<String, Vec<HmmerHit>>,
    replicons: &[RepliconProteins],
    replicon_topology: RepliconTopology,
) -> Vec<DetectedSystem> {
    let mut detected_systems = Vec::new();

    for replicon in replicons {
        if replicon.protein_ids.is_empty() {
            continue;
        }
        let protein_ids = replicon
            .protein_ids
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let sequence_index = SequenceIndex::from_ids(&protein_ids);

        for model_name in model_fully_qualified_names {
            let Some(model) = registry.get(model_name) else {
                continue;
            };
            let mut system_hits = assign_hits_to_model(model, all_hits, &sequence_index);
            if system_hits.is_empty() {
                continue;
            }

            let clusters = cluster_hits(
                &mut system_hits,
                model.inter_gene_max_space,
                replicon.protein_ids.len(),
                replicon_topology,
            );
            detected_systems.extend(
                clusters
                    .iter()
                    .filter(|cluster| !cluster.is_empty())
                    .filter_map(|cluster| evaluate_cluster(cluster, model))
                    .map(|mut system| {
                        system.replicon.clone_from(&replicon.name);
                        system
                    }),
            );
        }
    }

    detected_systems
}

fn filter_low_confidence_systems(
    detected_systems: &mut Vec<DetectedSystem>,
    minimum_best_hit_score: f64,
) {
    let before = detected_systems.len();
    detected_systems.retain(|system| {
        let best_score = system
            .hits
            .iter()
            .map(|hit| hit.hit.score)
            .fold(f64::NEG_INFINITY, f64::max);
        if best_score < minimum_best_hit_score {
            info!(
                "Filtering low-confidence system {} (best hit score {:.1} < {})",
                system.model_fully_qualified_name, best_score, minimum_best_hit_score
            );
            false
        } else {
            true
        }
    });

    if before != detected_systems.len() {
        info!(
            "Filtered {} low-confidence systems ({} -> {})",
            before - detected_systems.len(),
            before,
            detected_systems.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cas_types::{GeneDefinition, GeneStatus, SystemModel};
    use crate::hmmer_core::{hmm::EvParams, rng::XorShift64, test_helpers::hmm_sample};
    use std::fs::File;

    #[test]
    fn unsupported_genetic_code_fails_before_reading_input() {
        let config = CasFinderConfig {
            genetic_code: 267,
            ..CasFinderConfig::default()
        };
        let error = run_casfinder(Path::new("missing.fa"), &config).unwrap_err();
        assert!(error.to_string().contains("only genetic code 11"));
    }

    #[test]
    fn annotation_gff_and_protein_fasta_must_come_together() {
        let config = CasFinderConfig {
            annotation_gff: Some(PathBuf::from("genes.gff")),
            ..CasFinderConfig::default()
        };
        let error = detect_cas_systems(&[], &config, None).unwrap_err();
        assert!(error.to_string().contains("supplied together"));
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

    fn write_test_hmm(directory: &Path, profile_name: &str, alphabet: &Alphabet) {
        let mut random = XorShift64::new(42);
        let mut hmm = hmm_sample(&mut random, 8, alphabet);
        hmm.name = profile_name.to_string();
        hmm.ev_params = Some(EvParams {
            msv_mu: 0.0,
            msv_lambda: 0.7,
            viterbi_mu: 0.0,
            viterbi_lambda: 0.7,
            forward_tau: 0.0,
            forward_lambda: 0.7,
        });
        let path = directory.join(format!("{profile_name}.hmm"));
        let mut file = File::create(path).expect("create test HMM");
        HmmFile::write_ascii(&mut file, &hmm, alphabet).expect("write test HMM");
    }

    #[test]
    fn empty_proteome_returns_an_empty_analysis() {
        let sequences = vec![("short".to_string(), vec![b'A'; 120])];

        let analysis = detect_cas_systems(&sequences, &CasFinderConfig::default(), None)
            .expect("empty proteome is a valid result");

        assert!(analysis.systems.is_empty());
        assert!(analysis.clusters.is_empty());
        assert_eq!(analysis.proteome.replicons.len(), 1);
    }

    #[test]
    fn excluded_sequences_are_not_annotated() {
        let sequences = vec![("short".to_string(), vec![b'A'; 120])];
        let include = HashSet::new();

        let analysis =
            detect_cas_systems(&sequences, &CasFinderConfig::default(), Some(&include)).unwrap();

        assert!(analysis.proteome.replicons.is_empty());
    }

    #[test]
    fn non_finite_minimum_score_is_rejected_before_reading_input() {
        let config = CasFinderConfig {
            min_best_hit_score: f64::NAN,
            ..CasFinderConfig::default()
        };

        let error = run_casfinder(Path::new("missing.fa"), &config)
            .expect_err("NaN threshold must be rejected");

        assert!(error.to_string().contains("must be finite"));
    }

    #[test]
    fn model_registry_uses_sorted_file_order() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let model = concat!(
            "<system inter_gene_max_space=\"5\" min_mandatory_genes_required=\"1\" ",
            "min_genes_required=\"1\">\n",
            "<gene name=\"Cas1\" presence=\"mandatory\"/>\n",
            "</system>\n"
        );
        fs::write(temporary.path().join("zeta.xml"), model).expect("write zeta model");
        fs::write(temporary.path().join("alpha.xml"), model).expect("write alpha model");
        fs::write(temporary.path().join("ignored.txt"), model).expect("write ignored file");

        let (registry, names) =
            build_model_registry_from_dir(temporary.path()).expect("load model registry");

        assert_eq!(names, ["CASFinder/alpha", "CASFinder/zeta"]);
        assert!(registry.get("CASFinder/alpha").is_some());
        assert!(registry.get("CASFinder/zeta").is_some());
    }

    #[test]
    fn all_bundled_cas_models_are_valid_definitions() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("core crate is in the workspace root");
        let data_root = workspace_root
            .join("crispr-cas-finder-cli")
            .join("data")
            .join("CasFinder-2.0.3");
        let definition_directories = ["DEF-Class-2.0.3", "DEF-Typing-2.0.3", "DEF-SubTyping-2.0.3"];

        let mut model_count = 0;
        for directory in definition_directories {
            let (_, names) = build_model_registry_from_dir(&data_root.join(directory))
                .unwrap_or_else(|error| panic!("failed to parse {directory}: {error:#}"));
            model_count += names.len();
        }

        assert_eq!(model_count, 32);
    }

    #[test]
    fn hmm_search_loads_a_profile_once_for_an_empty_target_set() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let alphabet = Alphabet::amino();
        write_test_hmm(temporary.path(), "Cas1", &alphabet);
        let profiles = HashSet::from(["Cas1".to_string()]);

        let hits = run_hmm_search(temporary.path(), &profiles, &[], &alphabet, 1)
            .expect("load and configure HMM profile");

        assert!(hits.is_empty());
    }

    #[test]
    fn hmm_search_rejects_missing_required_profile() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let profiles = HashSet::from(["missing".to_string()]);

        let error = run_hmm_search(temporary.path(), &profiles, &[], &Alphabet::amino(), 1)
            .expect_err("missing profile must fail the search");

        assert!(error.to_string().contains("Required HMM profile 'missing'"));
    }

    #[test]
    fn hmm_search_rejects_profile_with_wrong_alphabet() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        write_test_hmm(temporary.path(), "Cas1", &Alphabet::dna());
        let profiles = HashSet::from(["Cas1".to_string()]);

        let error = run_hmm_search(temporary.path(), &profiles, &[], &Alphabet::amino(), 1)
            .expect_err("DNA profile must not be searched as protein");

        assert!(error.to_string().contains("expected Amino alphabet"));
    }

    #[test]
    fn configured_topology_controls_origin_spanning_systems() {
        let model = SystemModel {
            fully_qualified_name: "CASFinder/origin".to_string(),
            family: "CASFinder".to_string(),
            name: "origin".to_string(),
            version: None,
            genes: vec![
                test_gene("left", GeneStatus::Mandatory, None),
                test_gene("right", GeneStatus::Mandatory, None),
            ],
            inter_gene_max_space: 0,
            min_mandatory_genes_required: 2,
            min_genes_required: 2,
            multi_loci: false,
        };
        let mut registry = ModelRegistry::new();
        registry.add(model);
        let model_names = vec!["CASFinder/origin".to_string()];
        let all_hits = HashMap::from([
            ("left".to_string(), vec![test_hit("orf_1", "left", 50.0)]),
            ("right".to_string(), vec![test_hit("orf_10", "right", 50.0)]),
        ]);
        let replicons = vec![RepliconProteins {
            name: "chromosome".to_string(),
            protein_ids: (1..=10).map(|index| format!("orf_{index}")).collect(),
        }];

        let circular = evaluate_detected_systems(
            &registry,
            &model_names,
            &all_hits,
            &replicons,
            RepliconTopology::Circular,
        );
        let linear = evaluate_detected_systems(
            &registry,
            &model_names,
            &all_hits,
            &replicons,
            RepliconTopology::Linear,
        );

        assert_eq!(circular.len(), 1);
        assert_eq!(circular[0].replicon, "chromosome");
        assert!(linear.is_empty());

        let mut filtered = circular;
        filter_low_confidence_systems(&mut filtered, 50.1);
        assert!(filtered.is_empty());
    }

    #[test]
    fn hits_on_different_replicons_cannot_form_a_system() {
        let model = SystemModel {
            fully_qualified_name: "CASFinder/split".to_string(),
            family: "CASFinder".to_string(),
            name: "split".to_string(),
            version: None,
            genes: vec![
                test_gene("left", GeneStatus::Mandatory, None),
                test_gene("right", GeneStatus::Mandatory, None),
            ],
            inter_gene_max_space: 10,
            min_mandatory_genes_required: 2,
            min_genes_required: 2,
            multi_loci: false,
        };
        let mut registry = ModelRegistry::new();
        registry.add(model);
        let all_hits = HashMap::from([
            ("left".to_string(), vec![test_hit("a_1", "left", 50.0)]),
            ("right".to_string(), vec![test_hit("b_1", "right", 50.0)]),
        ]);
        let replicons = vec![
            RepliconProteins {
                name: "a".to_string(),
                protein_ids: vec!["a_1".to_string()],
            },
            RepliconProteins {
                name: "b".to_string(),
                protein_ids: vec!["b_1".to_string()],
            },
        ];

        let systems = evaluate_detected_systems(
            &registry,
            &["CASFinder/split".to_string()],
            &all_hits,
            &replicons,
            RepliconTopology::Circular,
        );

        assert!(systems.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_paths_return_an_error_instead_of_panicking() {
        use std::os::unix::ffi::OsStrExt;

        let path = Path::new(std::ffi::OsStr::from_bytes(b"profile-\xff.hmm"));
        let error = path_as_utf8(path, "HMM profile").expect_err("path is not UTF-8");

        assert!(error.to_string().contains("not valid UTF-8"));
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
