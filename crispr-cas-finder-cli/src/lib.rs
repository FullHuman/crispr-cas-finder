use std::ffi::OsString;
use std::fs::{File, create_dir_all};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use crispr_cas_finder_core::{
    CasFinderConfig, CrisprArray, DetectionParams, FullAnalysisResult, RepliconTopology,
    casfinder::run_casfinder,
    casparser::from_search_results,
    detect_crisprs_in_fasta_path,
    io::{write_gff, write_json},
};
use log::info;

const MAX_OUTPUT_DIRECTORY_SUFFIX: usize = 999;

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum TopologyArgument {
    #[default]
    Circular,
    Linear,
}

impl From<TopologyArgument> for RepliconTopology {
    fn from(value: TopologyArgument) -> Self {
        match value {
            TopologyArgument::Circular => Self::Circular,
            TopologyArgument::Linear => Self::Linear,
        }
    }
}

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
    #[arg(help = "Unsupported compatibility setting; only the default is accepted")]
    foster_repeat_length: usize,
    #[arg(long, alias = "fosterDRBegin", value_name = "STR", default_value = "G")]
    #[arg(help = "Unsupported compatibility setting; only the default is accepted")]
    foster_repeat_begin: String,
    #[arg(long, alias = "fosterDREnd", value_name = "STR", default_value = "AA.")]
    #[arg(help = "Unsupported compatibility setting; only the default is accepted")]
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
    #[arg(help = "Unsupported compatibility setting; only the default is accepted")]
    truncated_mismatch_percent: f64,
    #[arg(long = "cas", action = clap::ArgAction::SetTrue)]
    launch_cas_finder: bool,
    #[arg(
        long,
        alias = "cpuMacSyFinder",
        value_name = "INT",
        default_value_t = 0
    )]
    workers: usize,
    #[arg(long, value_name = "STR", default_value = "SubTyping")]
    definition: String,
    #[arg(long, alias = "geneticCode", value_name = "INT", default_value_t = 11)]
    genetic_code: usize,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    metagenome: bool,
    #[arg(long, value_enum, default_value_t = TopologyArgument::Circular)]
    topology: TopologyArgument,
    #[arg(long = "min-cas-score", value_name = "FLOAT", default_value_t = 25.0)]
    min_cas_score: f64,
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

    fn casfinder_config(&self) -> Result<CasFinderConfig> {
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

        let cas_models_dir = require_cas_data_directory(
            cas_models_dir,
            "Cas model definitions",
            "--cas-models-dir",
        )?;
        let cas_profiles_dir =
            require_cas_data_directory(cas_profiles_dir, "Cas HMM profiles", "--cas-profiles-dir")?;

        Ok(CasFinderConfig {
            genetic_code: self.genetic_code,
            metagenome: self.metagenome,
            workers: self.workers,
            definition: self.definition.clone(),
            quiet: self.quiet,
            replicon_topology: self.topology.into(),
            min_best_hit_score: self.min_cas_score,
            cas_models_dir: Some(cas_models_dir),
            cas_profiles_dir: Some(cas_profiles_dir),
        })
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

    let casfinder_config = cli
        .launch_cas_finder
        .then(|| cli.casfinder_config())
        .transpose()?;
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
        .map_or_else(|| default_output_directory(&basename), Ok)?;
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

    if let Some(casfinder_config) = casfinder_config {
        let search_results = run_casfinder(input_path, &basename, &outdir, &casfinder_config)
            .context("CasFinder pipeline failed")?;
        info!("CasFinder found {} systems", search_results.systems.len());
        let faa_path = outdir
            .join(format!("orphos_{basename}"))
            .join(format!("{basename}.faa"));
        let faa_content = std::fs::read_to_string(&faa_path).ok();
        let cas_clusters = from_search_results(&search_results, faa_content.as_deref());
        let report = FullAnalysisResult {
            crisprs: arrays.clone(),
            cas_clusters,
        };
        let report_path = outdir.join("report.json");
        let report_file = File::create(&report_path).context("Creating merged report JSON")?;
        serde_json::to_writer_pretty(report_file, &report).context("Writing merged report JSON")?;
        info!("Merged report written to {:?}", report_path);
    }

    Ok(())
}

fn require_cas_data_directory(
    path: Option<PathBuf>,
    description: &str,
    argument: &str,
) -> Result<PathBuf> {
    let path = path.with_context(|| {
        format!("{description} directory was not found; pass {argument} explicitly")
    })?;
    if !path.is_dir() {
        bail!(
            "{description} directory does not exist or is not a directory: {}; pass {argument} with a valid directory",
            path.display()
        );
    }
    Ok(path)
}

fn default_output_directory(basename: &str) -> Result<PathBuf> {
    available_output_directory(Path::new(""), basename)
}

fn available_output_directory(parent: &Path, basename: &str) -> Result<PathBuf> {
    let base_name = format!("Result_{}", basename);
    let candidate = parent.join(&base_name);
    if !candidate.exists() {
        return Ok(candidate);
    }

    for suffix in 2..=MAX_OUTPUT_DIRECTORY_SUFFIX {
        let candidate = parent.join(format!("{base_name}_{suffix}"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }

    bail!(
        "No available output directory for {base_name}; suffixes 2 through {MAX_OUTPUT_DIRECTORY_SUFFIX} are already in use"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removed_flags_are_rejected() {
        for flag in ["--fast", "--faster", "--archa-cas", "--ArchaCas"] {
            let result = Cli::try_parse_from(["crispr-cas-finder", "--in", "input.fa", flag]);
            assert!(result.is_err(), "{flag} should no longer be accepted");
        }
        for flag in ["--vicinity", "--clustering-threshold", "--cluster"] {
            let result =
                Cli::try_parse_from(["crispr-cas-finder", "--in", "input.fa", flag, "100"]);
            assert!(result.is_err(), "{flag} should no longer be accepted");
        }
    }

    #[test]
    fn nondefault_unsupported_detection_options_are_rejected() {
        for (flag, value) in [
            ("--foster-repeat-length", "31"),
            ("--foster-repeat-begin", "A"),
            ("--foster-repeat-end", "TT"),
            ("--truncated-mismatch-percent", "5"),
        ] {
            let cli = Cli::try_parse_from(["crispr-cas-finder", "--in", "input.fa", flag, value])
                .unwrap();
            assert!(
                cli.detection_params()
                    .validate()
                    .unwrap_err()
                    .contains("unsupported compatibility")
            );
        }
    }

    #[test]
    fn casfinder_options_are_preserved_in_config() {
        let cli = Cli::try_parse_from([
            "crispr-cas-finder",
            "--in",
            "input.fa",
            "--cas",
            "--workers",
            "3",
            "--topology",
            "linear",
            "--min-cas-score",
            "30.5",
        ])
        .expect("parse CLI");

        let config = cli.casfinder_config().expect("resolve bundled Cas data");
        assert_eq!(config.workers, 3);
        assert_eq!(config.replicon_topology, RepliconTopology::Linear);
        assert_eq!(config.min_best_hit_score, 30.5);
    }

    #[test]
    fn missing_explicit_cas_data_directory_fails_early() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let missing = temporary.path().join("missing");
        let cli = Cli::try_parse_from([
            OsString::from("crispr-cas-finder"),
            OsString::from("--in"),
            OsString::from("input.fa"),
            OsString::from("--cas"),
            OsString::from("--cas-models-dir"),
            missing.as_os_str().to_owned(),
            OsString::from("--cas-profiles-dir"),
            missing.as_os_str().to_owned(),
        ])
        .expect("parse CLI");

        let error = cli
            .casfinder_config()
            .expect_err("missing data directory must fail");
        assert!(error.to_string().contains("Cas model definitions"));
        assert!(error.to_string().contains("--cas-models-dir"));
    }

    #[test]
    fn output_directory_uses_first_available_suffix() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        std::fs::create_dir(temporary.path().join("Result_genome")).expect("base output");
        std::fs::create_dir(temporary.path().join("Result_genome_2")).expect("second output");

        let selected =
            available_output_directory(temporary.path(), "genome").expect("select output");
        assert_eq!(selected, temporary.path().join("Result_genome_3"));
    }

    #[test]
    fn empty_proteome_completes_with_an_empty_cas_report() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let input = temporary.path().join("short.fa");
        let output = temporary.path().join("output");
        std::fs::write(&input, format!(">short\n{}\n", "A".repeat(120))).expect("write FASTA");
        let cli = Cli::try_parse_from([
            OsString::from("crispr-cas-finder"),
            OsString::from("--in"),
            input.as_os_str().to_owned(),
            OsString::from("--outdir"),
            output.as_os_str().to_owned(),
            OsString::from("--cas"),
            OsString::from("--cas-models-dir"),
            temporary.path().as_os_str().to_owned(),
            OsString::from("--cas-profiles-dir"),
            temporary.path().as_os_str().to_owned(),
        ])
        .expect("parse CLI");

        run(cli).expect("an input with no predicted CDS is valid");

        let report = std::fs::read_to_string(output.join("report.json")).expect("merged report");
        let report: FullAnalysisResult = serde_json::from_str(&report).expect("parse report");
        assert!(report.cas_clusters.is_empty());
    }
}
