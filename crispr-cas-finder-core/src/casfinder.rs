use crate::cas_pipeline::{
    ModelDefinition, assign_hits_to_model, build_model_registry, codon_table, evaluate_cluster,
    revcomp_dna, translate_dna,
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
    let analyzer = OrphosAnalyzer::new(config);
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
    let all_hits = run_hmm_search(&profiles_dir, &needed_profiles, &seqs, &abc, cfg.workers)?;

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

    let fqns: Vec<String> = models
        .iter()
        .map(|model| format!("{}/{}", model.family, model.name))
        .collect();
    let registry = build_model_registry(&models)
        .map_err(|e| anyhow::anyhow!("Failed to build CAS model registry: {}", e))?;
    Ok((registry, fqns))
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
    workers: usize,
) -> Result<HashMap<String, Vec<HmmerHit>>> {
    let avg_len = if seqs.is_empty() {
        400
    } else {
        seqs.iter().map(|s| s.len()).sum::<usize>() / seqs.len()
    };

    let coverage_threshold = 0.4;
    let mut all_hits: HashMap<String, Vec<HmmerHit>> = HashMap::new();
    let mut profile_names: Vec<String> = needed_profiles.iter().cloned().collect();
    profile_names.sort_unstable();
    let mut sequence_search_order: Vec<usize> = (0..seqs.len()).collect();
    sequence_search_order.sort_unstable_by_key(|&index| (seqs[index].len(), index));

    let pool = build_hmm_search_pool(workers)?;
    info!(
        "Searching {} HMM profiles across {} protein sequences with {} thread(s)",
        profile_names.len(),
        seqs.len(),
        pool.current_num_threads()
    );
    let search_results = pool.install(|| {
        profile_names
            .par_iter()
            .map(|profile_name| {
                search_profile_hits(
                    profiles_dir,
                    profile_name,
                    seqs,
                    &sequence_search_order,
                    abc,
                    avg_len,
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
    seqs: &[DigitalSequence],
    sequence_search_order: &[usize],
    abc: &Alphabet,
    avg_len: usize,
    coverage_threshold: f64,
) -> Result<Option<(String, Vec<HmmerHit>)>> {
    let hmm_path = profiles_dir.join(format!("{}.hmm", profile_name));
    if !hmm_path.exists() {
        return Ok(None);
    }

    let bg = BackgroundModel::new(abc);
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

    let mut profile_hits: Vec<(usize, HmmerHit)> = Vec::new();

    for &seq_index in sequence_search_order {
        let seq = &seqs[seq_index];
        let report = worker
            .search(seq)
            .map_err(|e| anyhow::anyhow!("Search error {}: {}", profile_name, e))?;

        match &report.outcome {
            SearchOutcome::Hit(hit) => {
                for domain in &hit.domains {
                    if !domain.is_included {
                        continue;
                    }
                    let prof_cov = (domain.hmm_to - domain.hmm_from + 1) as f64 / num_nodes as f64;
                    if prof_cov < coverage_threshold {
                        continue;
                    }
                    let seq_cov = (domain.alignment_end - domain.alignment_start + 1) as f64
                        / seq.len() as f64;
                    profile_hits.push((
                        seq_index,
                        HmmerHit {
                            id: hit.name.clone(),
                            gene_name: profile_name.to_string(),
                            seq_len: seq.len() as u32,
                            i_evalue: domain.log_pvalue.exp(),
                            score: domain.bitscore as f64,
                            profile_coverage: prof_cov,
                            seq_coverage: seq_cov,
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
                    let prof_cov = seq.len().min(num_nodes) as f64 / num_nodes as f64;
                    if prof_cov >= coverage_threshold {
                        profile_hits.push((
                            seq_index,
                            HmmerHit {
                                id: seq.name.clone(),
                                gene_name: profile_name.to_string(),
                                seq_len: seq.len() as u32,
                                i_evalue: 0.0,
                                score: bitscore as f64,
                                profile_coverage: prof_cov,
                                seq_coverage: 1.0,
                                begin_match: 1,
                                end_match: seq.len() as u32,
                            },
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    profile_hits.sort_by_key(|(seq_index, _)| *seq_index);
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
            fqn: "CASFinder/CAS-TypeIE".to_string(),
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
            fqn: "CASFinder/CAS-TypeIE".to_string(),
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
