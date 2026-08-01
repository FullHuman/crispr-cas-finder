use crate::cas_pipeline::{
    ModelDefinition, assign_hits_to_model, build_model_registry, codon_table, evaluate_cluster,
    reverse_complement_dna, translate_dna,
};
use crate::cas_types::{
    HmmerHit, ModelRegistry, RepliconTopology, SearchResults, SequenceIndex, cluster_hits,
    select_best_solution,
};
use crate::types::CasFinderConfig;
use anyhow::{Context, Result};
use bio::io::fasta::Reader as FastaReader;
use hmmer_core::rayon::{ThreadPoolBuilder, prelude::*};
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

/// Run Orphos gene prediction and then detect Cas systems using
/// hmmer-core for HMM search (pure Rust, no external subprocess).
pub fn run_casfinder(
    input: &Path,
    basename: &str,
    outdir: &Path,
    config: &CasFinderConfig,
) -> Result<SearchResults> {
    // Prepare annotation directory
    let annotation_dir = outdir.join(format!("orphos_{}", basename));
    fs::create_dir_all(&annotation_dir).context("Creating annotation output dir")?;

    // Predict genes with Orphos
    info!("Running Orphos gene prediction on {}", input.display());
    let orphos_config = OrphosConfig {
        metagenomic: config.metagenome,
        closed_ends: true,
        quiet: config.quiet,
        translation_table: Some(config.genetic_code as u8),
        ..OrphosConfig::default()
    };
    let analyzer = OrphosAnalyzer::new(orphos_config);
    let results = analyzer
        .analyze_fasta_file(input.to_str().unwrap())
        .context("Orphos gene prediction failed")?;

    // Write GFF3
    let gff_path = annotation_dir.join(format!("{}.gff", basename));
    let mut gff_file = fs::File::create(&gff_path).context("Creating .gff output")?;
    for gene_prediction_result in &results {
        write_results(&mut gff_file, gene_prediction_result, OutputFormat::Gff)
            .context("Writing GFF from orphos results")?;
    }

    // Write FAA
    let faa_path = annotation_dir.join(format!("{}.faa", basename));
    write_faa_from_genes(input, &results, config.genetic_code, &faa_path)
        .context("Writing .faa from orphos gene predictions")?;

    let proteome = faa_path;
    let metadata = fs::metadata(&proteome)
        .with_context(|| format!("Cannot stat proteome file: {:?}", proteome))?;
    if metadata.len() == 0 {
        info!("Proteome file is empty, skipping CasFinder");
        anyhow::bail!("No CDS found");
    }

    // ---- Load proteins as digital sequences via hmmer-core ----
    let amino_alphabet = Alphabet::amino();
    let protein_sequences =
        hmmer_io::read_fasta_digital_sequences(&amino_alphabet, proteome.to_str().unwrap())
            .map_err(|e| anyhow::anyhow!("Failed to load proteins: {}", e))?;

    info!("Loaded {} protein sequences", protein_sequences.len());

    let seq_names: Vec<&str> = protein_sequences.iter().map(|s| s.name.as_str()).collect();
    let seq_index = SequenceIndex::from_ids(&seq_names);

    // ---- Build model registry from CAS XML definitions ----
    let models_dir = resolve_models_dir(config)?;
    let profiles_dir = resolve_profiles_dir(config)?;

    let (registry, model_fully_qualified_names) = build_model_registry_from_dir(&models_dir)?;

    let mut needed_profiles: HashSet<String> = HashSet::new();
    for model_fully_qualified_name in &model_fully_qualified_names {
        if let Some(model) = registry.get(model_fully_qualified_name) {
            for name in model.all_profile_names() {
                needed_profiles.insert(name.to_string());
            }
        }
    }

    if model_fully_qualified_names.is_empty() {
        anyhow::bail!("No CAS model definitions found in {:?}", models_dir);
    }
    info!(
        "Need {} HMM profiles for {} models",
        needed_profiles.len(),
        model_fully_qualified_names.len()
    );

    // ---- HMM search using hmmer-core pipeline ----
    let all_hits = run_hmm_search(
        &profiles_dir,
        &needed_profiles,
        &protein_sequences,
        &amino_alphabet,
        config.workers,
    )?;

    info!("HMM search found hits for {} profiles", all_hits.len());

    // ---- Cluster hits and evaluate systems ----
    let mut detected_systems = Vec::new();

    for model_fully_qualified_name in &model_fully_qualified_names {
        let model = match registry.get(model_fully_qualified_name) {
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
            protein_sequences.len(),
            RepliconTopology::Circular,
        );

        for cluster in &clusters {
            if cluster.is_empty() {
                continue;
            }
            if let Some(system) = evaluate_cluster(cluster, model) {
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
                sys.model_fully_qualified_name, best, MIN_BEST_HIT_SCORE
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

    for entry in fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "xml") {
            let model_name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let content = fs::read_to_string(&path)?;
            models.push(ModelDefinition {
                name: model_name,
                family: family.clone(),
                content,
            });
        }
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
    let average_sequence_length = if protein_sequences.is_empty() {
        400
    } else {
        protein_sequences.iter().map(|s| s.len()).sum::<usize>() / protein_sequences.len()
    };

    let coverage_threshold = 0.4;
    let mut all_hits: HashMap<String, Vec<HmmerHit>> = HashMap::new();
    let mut profile_names: Vec<String> = needed_profiles.iter().cloned().collect();
    profile_names.sort_unstable();
    let mut sequence_search_order: Vec<usize> = (0..protein_sequences.len()).collect();
    sequence_search_order.sort_unstable_by_key(|&index| (protein_sequences[index].len(), index));

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
                search_profile_hits(
                    profiles_dir,
                    profile_name,
                    protein_sequences,
                    &sequence_search_order,
                    alphabet,
                    average_sequence_length,
                    coverage_threshold,
                )
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
    sequence_search_order: &[usize],
    alphabet: &Alphabet,
    average_sequence_length: usize,
    coverage_threshold: f64,
) -> Result<Option<(String, Vec<HmmerHit>)>> {
    let hmm_path = profiles_dir.join(format!("{}.hmm", profile_name));
    if !hmm_path.exists() {
        return Ok(None);
    }

    let background_model = BackgroundModel::new(alphabet);
    let mut hmm_file = HmmFile::open(hmm_path.to_str().unwrap(), None)
        .map_err(|e| anyhow::anyhow!("Failed to open {}: {}", profile_name, e))?;
    let (_hmm_alphabet, hmm) = hmm_file
        .read()
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", profile_name, e))?;

    let num_nodes = hmm.num_nodes;
    let mut configured_profile = Profile::new(num_nodes, alphabet);
    modelconfig::profile_config(
        &hmm,
        &background_model,
        &mut configured_profile,
        average_sequence_length,
        SearchMode::Local,
    );

    let query = SearchQuery::from_configured_profile(configured_profile, background_model.clone())
        .map_err(|e| anyhow::anyhow!("Query config failed for {}: {}", profile_name, e))?;
    let plan = SearchPlan::builder(query)
        .filters(FilterPolicy::default())
        .build();
    let mut worker = plan
        .spawn_worker(CapacityHints {
            target_length: average_sequence_length,
        })
        .map_err(|e| anyhow::anyhow!("Spawn worker failed for {}: {}", profile_name, e))?;

    let mut profile_hits: Vec<(usize, HmmerHit)> = Vec::new();

    for &sequence_index in sequence_search_order {
        let target_sequence = &protein_sequences[sequence_index];
        let report = worker
            .search(target_sequence)
            .map_err(|e| anyhow::anyhow!("Search error {}: {}", profile_name, e))?;

        match &report.outcome {
            SearchOutcome::Hit(hit) => {
                for domain in &hit.domains {
                    if !domain.is_included {
                        continue;
                    }
                    let profile_coverage =
                        (domain.hmm_to - domain.hmm_from + 1) as f64 / num_nodes as f64;
                    if profile_coverage < coverage_threshold {
                        continue;
                    }
                    let sequence_coverage = (domain.alignment_end - domain.alignment_start + 1)
                        as f64
                        / target_sequence.len() as f64;
                    profile_hits.push((
                        sequence_index,
                        HmmerHit {
                            id: hit.name.clone(),
                            gene_name: profile_name.to_string(),
                            seq_len: target_sequence.len() as u32,
                            i_evalue: domain.log_pvalue.exp(),
                            score: domain.bitscore as f64,
                            profile_coverage,
                            seq_coverage: sequence_coverage,
                            begin_match: domain.alignment_start as u32,
                            end_match: domain.alignment_end as u32,
                        },
                    ));
                }
            }
            SearchOutcome::Filtered(FilterReason::NoDomains) => {
                if let (Some(fwd_raw), Some(null_sc)) =
                    (report.trace.forward_raw_score, report.trace.null_score)
                {
                    let bitscore = (fwd_raw - null_sc) / std::f32::consts::LN_2;
                    let profile_coverage =
                        target_sequence.len().min(num_nodes) as f64 / num_nodes as f64;
                    if profile_coverage >= coverage_threshold {
                        profile_hits.push((
                            sequence_index,
                            HmmerHit {
                                id: target_sequence.name.clone(),
                                gene_name: profile_name.to_string(),
                                seq_len: target_sequence.len() as u32,
                                i_evalue: 0.0,
                                score: bitscore as f64,
                                profile_coverage,
                                seq_coverage: 1.0,
                                begin_match: 1,
                                end_match: target_sequence.len() as u32,
                            },
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    profile_hits.sort_by_key(|(sequence_index, _)| *sequence_index);
    let profile_hits: Vec<HmmerHit> = profile_hits.into_iter().map(|(_, hit)| hit).collect();

    if profile_hits.is_empty() {
        Ok(None)
    } else {
        Ok(Some((profile_name.to_string(), profile_hits)))
    }
}

fn build_hmm_search_pool(workers: usize) -> Result<hmmer_core::rayon::ThreadPool> {
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

    let translation_table = codon_table(genetic_code);
    let mut output_file = fs::File::create(faa_path).context("Creating .faa file")?;
    let mut gene_count = 0usize;

    for (sequence_index, orphos_result) in results.iter().enumerate() {
        let genomic_sequence = if sequence_index < sequences.len() {
            &sequences[sequence_index].1
        } else {
            continue;
        };

        for gene in &orphos_result.genes {
            let begin = gene.coordinates.begin.saturating_sub(1);
            let end = gene.coordinates.end.saturating_sub(1);

            if end >= genomic_sequence.len() {
                continue;
            }

            let coding_dna_nucleotides = &genomic_sequence[begin..=end];
            let coding_dna_nucleotides = match format!("{}", gene.coordinates.strand).as_str() {
                "-" => reverse_complement_dna(coding_dna_nucleotides),
                _ => coding_dna_nucleotides.to_vec(),
            };

            let protein = translate_dna(&coding_dna_nucleotides, &translation_table);
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
                orphos_result.sequence_info.header,
                gene_count,
                gene.coordinates.begin,
                gene.coordinates.end,
                strand_int,
                gene_count
            );
            writeln!(output_file, ">{}", gene_id)?;
            for chunk in protein.as_bytes().chunks(60) {
                output_file.write_all(chunk)?;
                output_file.write_all(b"\n")?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cas_types::{GeneDefinition, GeneStatus, SystemModel};

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
