use std::ffi::OsString;
use std::fs::{File, create_dir_all};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use crispr_cas_finder_core::{
    CasFinderConfig, CrisprArray, DetectionParams, FullAnalysisResult,
    casfinder::run_casfinder,
    casparser::from_search_results,
    detect_crisprs_in_fasta_path,
    io::{write_gff, write_json},
};
use log::{error, info};

#[derive(Parser, Debug)]
#[command(name = "crispr-cas-finder", author, version = env!("CARGO_PKG_VERSION"), about = "Find CRISPR arrays and Cas proteins in genomes")]
#[command(args_conflicts_with_subcommands = true)]
pub struct Cli {
    #[arg(short = 'i', long = "in", value_name = "FILE")]
    input: PathBuf,
    #[arg(long, alias = "out", value_name = "DIR")]
    outdir: Option<String>,
    #[arg(short = 'q', long, action = clap::ArgAction::SetTrue)]
    quiet: bool,
    #[arg(long = "fast", alias = "faster", action = clap::ArgAction::SetTrue)]
    fast: bool,
    #[arg(long, alias = "minSeqSize", value_name = "INT", default_value_t = 0)]
    min_sequence_length: usize,
    #[arg(long, alias = "mismDRs", value_name = "FLOAT", default_value_t = 20.0)]
    repeat_mismatch_percent: f64,
    #[arg(long, alias = "truncDR", value_name = "FLOAT", default_value_t = 33.3)]
    truncated_repeat_mismatch_percent: f64,
    #[arg(long, alias = "minDR", value_name = "INT", default_value_t = 23)]
    min_repeat_length: usize,
    #[arg(long, alias = "maxDR", value_name = "INT", default_value_t = 55)]
    max_repeat_length: usize,
    #[arg(long, alias = "minSP", value_name = "INT", default_value_t = 25)]
    min_spacer_length: usize,
    #[arg(long, alias = "maxSP", value_name = "INT", default_value_t = 60)]
    max_spacer_length: usize,
    #[arg(long, alias = "noMism", short = 'n', action = clap::ArgAction::SetTrue)]
    no_mismatch: bool,
    #[arg(long, alias = "percSPmin", value_name = "FLOAT", default_value_t = 0.6)]
    min_spacer_to_repeat_ratio: f64,
    #[arg(long, alias = "percSPmax", value_name = "FLOAT", default_value_t = 2.5)]
    max_spacer_to_repeat_ratio: f64,
    #[arg(long, alias = "spSim", value_name = "FLOAT", default_value_t = 60.0)]
    spacer_similarity_threshold: f64,
    #[arg(long, value_name = "INT", default_value_t = 100)]
    flank: usize,
    #[arg(long, alias = "levelMin", value_name = "INT", default_value_t = 1)]
    min_evidence_level: usize,
    #[arg(long, alias = "forceDetection", action = clap::ArgAction::SetTrue)]
    force_detection: bool,
    #[arg(
        long,
        alias = "fosterDRLength",
        value_name = "INT",
        default_value_t = 30
    )]
    foster_repeat_length: usize,
    #[arg(long, alias = "fosterDRBegin", value_name = "STR", default_value = "G")]
    foster_repeat_begin: String,
    #[arg(long, alias = "fosterDREnd", value_name = "STR", default_value = "AA.")]
    foster_repeat_end: String,
    #[arg(long, alias = "minNbSpacers", value_name = "INT", default_value_t = 1)]
    min_spacer_count: usize,
    #[arg(long, alias = "betterDetectTrunc", action = clap::ArgAction::SetTrue)]
    better_detect_truncated: bool,
    #[arg(
        long,
        alias = "PercMismTrunc",
        value_name = "FLOAT",
        default_value_t = 4.0
    )]
    truncated_mismatch_percent: f64,
    #[arg(long = "cas", action = clap::ArgAction::SetTrue)]
    launch_cas_finder: bool,
    #[arg(long, value_name = "INT", default_value_t = 600)]
    vicinity: usize,
    #[arg(
        long,
        alias = "cpuMacSyFinder",
        value_name = "INT",
        default_value_t = 0
    )]
    workers: usize,
    #[arg(long, value_name = "STR", default_value = "SubTyping")]
    definition: String,
    #[arg(
        long = "clustering-threshold",
        alias = "cluster",
        value_name = "INT",
        default_value_t = 0
    )]
    clustering_threshold: usize,
    #[arg(long, alias = "geneticCode", value_name = "INT", default_value_t = 11)]
    genetic_code: usize,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    metagenome: bool,
    #[arg(long = "archa-cas", alias = "ArchaCas", action = clap::ArgAction::SetTrue)]
    archa_cas: bool,
    #[arg(long = "cas-models-dir", value_name = "DIR")]
    cas_models_dir: Option<String>,
    #[arg(long = "cas-profiles-dir", value_name = "DIR")]
    cas_profiles_dir: Option<String>,
}

impl Cli {
    /// Resolve the default CasFinder data root.
    ///
    /// Search order:
    ///   1. Directory that contains the running binary (installed layout).
    ///   2. CARGO_MANIFEST_DIR at compile time (development layout).
    fn default_cas_finder_data_root() -> Option<PathBuf> {
        // 1. Next to the binary (production / installed)
        if let Ok(exe) = std::env::current_exe()
            && let Some(exe_dir) = exe.parent()
        {
            let candidate = exe_dir.join("data").join("CasFinder-2.0.3");
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
        // 2. Compile-time manifest directory (cargo run / dev builds)
        let development_data_root =
            PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data/CasFinder-2.0.3"));
        if development_data_root.is_dir() {
            return Some(development_data_root);
        }
        None
    }

    fn detection_params(&self) -> DetectionParams {
        DetectionParams {
            min_repeat_length: self.min_repeat_length,
            max_repeat_length: self.max_repeat_length,
            min_spacer_length: self.min_spacer_length,
            max_spacer_length: self.max_spacer_length,
            repeat_mismatch_percent: self.repeat_mismatch_percent,
            truncated_repeat_mismatch_percent: self.truncated_repeat_mismatch_percent,
            no_mismatch: self.no_mismatch,
            min_spacer_to_repeat_ratio: self.min_spacer_to_repeat_ratio,
            max_spacer_to_repeat_ratio: self.max_spacer_to_repeat_ratio,
            spacer_similarity_threshold: self.spacer_similarity_threshold,
            min_spacer_count: self.min_spacer_count,
            min_evidence_level: self.min_evidence_level,
            min_sequence_length: self.min_sequence_length,
            force_detection: self.force_detection,
            flank: self.flank,
            foster_repeat_length: self.foster_repeat_length,
            foster_repeat_begin: self.foster_repeat_begin.clone(),
            foster_repeat_end: self.foster_repeat_end.clone(),
            better_detect_truncated: self.better_detect_truncated,
            truncated_mismatch_percent: self.truncated_mismatch_percent,
        }
    }

    fn casfinder_config(&self) -> CasFinderConfig {
        let data_root = Self::default_cas_finder_data_root();

        let cas_models_dir = self.cas_models_dir.as_ref().map(PathBuf::from).or_else(|| {
            data_root
                .as_ref()
                .map(|root| root.join(format!("DEF-{}-2.0.3", self.definition)))
        });

        let cas_profiles_dir = self
            .cas_profiles_dir
            .as_ref()
            .map(PathBuf::from)
            .or_else(|| {
                data_root
                    .as_ref()
                    .map(|root| root.join("CASprofiles-2.0.3"))
            });

        CasFinderConfig {
            genetic_code: self.genetic_code,
            metagenome: self.metagenome,
            workers: if self.fast { 0 } else { self.workers },
            definition: self.definition.clone(),
            vicinity: self.vicinity,
            clustering_threshold: self.clustering_threshold,
            quiet: self.quiet,
            fast: self.fast,
            cas_models_dir,
            cas_profiles_dir,
        }
    }

    fn should_launch_cas(&self) -> bool {
        self.launch_cas_finder || self.archa_cas
    }
}

pub fn run_from_args<I, T>(args: I) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = Cli::try_parse_from(args)?;
    run(cli)
}

pub fn run(cli: Cli) -> Result<()> {
    info!(
        "Starting crispr-cas-finder version {}",
        env!("CARGO_PKG_VERSION")
    );

    let params = cli.detection_params();
    let input_path = &cli.input;
    info!(
        "Scanning {:?} for CRISPR direct repeats (min_repeat_length={}, max_repeat_length={}, mism={})",
        input_path,
        params.min_repeat_length,
        params.max_repeat_length,
        if params.no_mismatch { 0 } else { 1 }
    );

    let arrays: Vec<CrisprArray> = detect_crisprs_in_fasta_path(input_path, &params)
        .with_context(|| format!("Failed to detect CRISPRs from FASTA: {:?}", input_path))?;

    info!("Detected {} CRISPR arrays", arrays.len());
    for (array_index, array) in arrays.iter().enumerate() {
        info!(
            "  Array {}: {} ({}..{}), DR={} (len={}), {} spacers, evidence={}, orientation={}",
            array_index + 1,
            array.seq_id,
            array.start,
            array.end,
            array.consensus_repeat,
            array.consensus_repeat.len(),
            array.spacers.len(),
            array.evidence_level,
            array.orientation
        );
    }

    let basename = input_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    let outdir = cli
        .outdir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_output_directory(&basename));
    create_dir_all(&outdir)?;

    let gff_path = outdir.join(format!("{}.gff", basename));
    info!("Writing GFF to {:?}", gff_path);
    let mut gff_file = File::create(&gff_path)
        .with_context(|| format!("Failed to create GFF file: {:?}", gff_path))?;
    write_gff(&arrays, &mut gff_file)?;

    let json_path = outdir.join(format!("{}.json", basename));
    info!("Writing JSON to {:?}", json_path);
    let mut json_file = File::create(&json_path)
        .with_context(|| format!("Failed to create JSON file: {:?}", json_path))?;
    write_json(&arrays, &mut json_file)?;

    info!("GFF and JSON outputs created in {:?}", outdir);

    if cli.should_launch_cas() {
        let casfinder_config = cli.casfinder_config();
        match run_casfinder(input_path, &basename, &outdir, &casfinder_config) {
            Ok(search_results) => {
                info!("CasFinder found {} systems", search_results.systems.len());
                let faa_path = outdir.join(format!("orphos_{}/{}.faa", basename, basename));
                let faa_content = std::fs::read_to_string(&faa_path).ok();
                let cas_clusters = from_search_results(&search_results, faa_content.as_deref());
                let report = FullAnalysisResult {
                    crisprs: arrays.clone(),
                    cas_clusters,
                };
                let report_path = outdir.join("report.json");
                let report_file =
                    File::create(&report_path).context("Creating merged report JSON")?;
                serde_json::to_writer_pretty(report_file, &report)
                    .context("Writing merged report JSON")?;
                info!("Merged report written to {:?}", report_path);
            }
            Err(err) => {
                error!("CasFinder failed: {}", err);
            }
        }
    }

    Ok(())
}

fn default_output_directory(basename: &str) -> PathBuf {
    let base_name = format!("Result_{}", basename);
    let mut candidate = PathBuf::from(&base_name);
    let mut suffix = 2usize;

    while candidate.exists() {
        candidate = PathBuf::from(format!("{}_{}", base_name, suffix));
        suffix += 1;
    }

    candidate
}
