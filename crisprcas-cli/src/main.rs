use std::fs::{File, create_dir_all};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use bio::io::fasta::Reader as FastaReader;
use clap::{ArgAction, Parser};
use crisprcas_core::{
    CasFinderConfig, CrisprArray, DetectionParams,
    casfinder::run_casfinder,
    casparser::from_search_results,
    detect_crisprs,
    io::{write_gff, write_json},
};
use log::{error, info};

#[derive(Parser, Debug)]
#[command(name = "CRISPRCasFinder", author, version = env!("CARGO_PKG_VERSION"), about = "Find CRISPR arrays and Cas proteins in genomes")]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    #[arg(short = 'i', long = "in", value_name = "FILE")]
    input: PathBuf,
    #[arg(long, alias = "out", value_name = "DIR")]
    outdir: Option<String>,
    #[arg(long = "keep-all", alias = "keepAll", action = ArgAction::SetTrue)]
    keep_all: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    log: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    html: bool,
    #[arg(long = "copy-css", alias = "copyCSS", value_name = "CSS_FILE")]
    css_file: Option<String>,
    #[arg(short = 'q', long, action = ArgAction::SetTrue)]
    quiet: bool,
    #[arg(long = "fast", alias = "faster", action = ArgAction::SetTrue)]
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
    #[arg(long, alias = "noMism", short = 'n', action = ArgAction::SetTrue)]
    no_mismatch: bool,
    #[arg(long, alias = "percSPmin", value_name = "FLOAT", default_value_t = 0.6)]
    min_spacer_to_repeat_ratio: f64,
    #[arg(long, alias = "percSPmax", value_name = "FLOAT", default_value_t = 2.5)]
    max_spacer_to_repeat_ratio: f64,
    #[arg(long, alias = "spSim", value_name = "FLOAT", default_value_t = 60.0)]
    spacer_similarity_threshold: f64,
    #[arg(long = "crispr-db", alias = "DBcrispr", value_name = "FILE")]
    crispr_db: Option<String>,
    #[arg(long, value_name = "FILE")]
    repeats_file: Option<String>,
    #[arg(long = "repeat-direction", alias = "DIRrepeat", value_name = "FILE")]
    dir_repeat: Option<String>,
    #[arg(long, value_name = "INT", default_value_t = 100)]
    flank: usize,
    #[arg(long, alias = "levelMin", value_name = "INT", default_value_t = 1)]
    min_evidence_level: usize,
    #[arg(long = "classify-small-arrays", alias = "classifySmallArrays", action = ArgAction::SetTrue)]
    classify_small: bool,
    #[arg(long, alias = "forceDetection", action = ArgAction::SetTrue)]
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
    #[arg(
        long = "matching-repeats",
        alias = "MatchingRepeats",
        value_name = "FILE"
    )]
    matching_repeats: Option<String>,
    #[arg(long, alias = "minNbSpacers", value_name = "INT", default_value_t = 1)]
    min_spacer_count: usize,
    #[arg(long, alias = "betterDetectTrunc", action = ArgAction::SetTrue)]
    better_detect_truncated: bool,
    #[arg(
        long,
        alias = "PercMismTrunc",
        value_name = "FLOAT",
        default_value_t = 4.0
    )]
    truncated_mismatch_percent: f64,
    #[arg(long = "cas", action = ArgAction::SetTrue)]
    launch_cas_finder: bool,
    #[arg(long = "full-report", alias = "ccvRep", action = ArgAction::SetTrue)]
    write_full_report: bool,
    #[arg(long, value_name = "INT", default_value_t = 600)]
    vicinity: usize,
    #[arg(
        long,
        alias = "cpuMacSyFinder",
        value_name = "INT",
        default_value_t = 1
    )]
    workers: usize,
    #[arg(long, action = ArgAction::SetTrue)]
    rcfowce: bool,
    #[arg(long, value_name = "STR", default_value = "SubTyping")]
    definition: String,
    #[arg(long = "annotation-gff", alias = "gffAnnot", value_name = "FILE")]
    user_gff: Option<String>,
    #[arg(long, alias = "faa", value_name = "FILE")]
    proteome: Option<String>,
    #[arg(
        long = "clustering-threshold",
        alias = "cluster",
        value_name = "INT",
        default_value_t = 0
    )]
    clustering_threshold: usize,
    #[arg(long = "summary", alias = "getSummaryCasfinder", action = ArgAction::SetTrue)]
    get_summary: bool,
    #[arg(long, alias = "geneticCode", value_name = "INT", default_value_t = 11)]
    genetic_code: usize,
    #[arg(long, action = ArgAction::SetTrue)]
    metagenome: bool,
    #[arg(long = "archa-cas", alias = "ArchaCas", action = ArgAction::SetTrue)]
    archa_cas: bool,
    #[arg(long = "cas-models-dir", value_name = "DIR")]
    cas_models_dir: Option<String>,
    #[arg(long = "cas-profiles-dir", value_name = "DIR")]
    cas_profiles_dir: Option<String>,
}

impl Cli {
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
        CasFinderConfig {
            genetic_code: self.genetic_code,
            metagenome: self.metagenome,
            workers: if self.fast { 0 } else { self.workers },
            definition: self.definition.clone(),
            vicinity: self.vicinity,
            clustering_threshold: self.clustering_threshold,
            quiet: self.quiet,
            fast: self.fast,
            cas_models_dir: self.cas_models_dir.as_ref().map(std::path::PathBuf::from),
            cas_profiles_dir: self.cas_profiles_dir.as_ref().map(std::path::PathBuf::from),
        }
    }

    fn should_launch_cas(&self) -> bool {
        self.launch_cas_finder || self.archa_cas
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    info!(
        "Starting CRISPRCasFinder version {}",
        env!("CARGO_PKG_VERSION")
    );

    // Check external dependencies early
    // (No external tools required — Orphos and system detection are pure Rust)

    // ── Read input FASTA ─────────────────────────────────────────────────────
    let params = cli.detection_params();
    let input_path = &cli.input;
    info!(
        "Scanning {:?} for CRISPR direct repeats (min_repeat_length={}, max_repeat_length={}, mism={})",
        input_path,
        params.min_repeat_length,
        params.max_repeat_length,
        if params.no_mismatch { 0 } else { 1 }
    );

    let fasta_reader = FastaReader::from_file(input_path)
        .with_context(|| format!("Failed to open input FASTA: {:?}", input_path))?;
    let mut sequences: Vec<(String, Vec<u8>)> = Vec::new();
    for result in fasta_reader.records() {
        let record = result.context("Failed to read FASTA record")?;
        sequences.push((record.id().to_string(), record.seq().to_vec()));
    }

    // ── Run detection (pure in-memory) ───────────────────────────────────────
    let seq_refs: Vec<(&str, &[u8])> = sequences
        .iter()
        .map(|(id, seq)| (id.as_str(), seq.as_slice()))
        .collect();
    let arrays: Vec<CrisprArray> = detect_crisprs(&seq_refs, &params);

    info!("Detected {} CRISPR arrays", arrays.len());
    for (i, a) in arrays.iter().enumerate() {
        info!(
            "  Array {}: {} ({}..{}), DR={} (len={}), {} spacers, evidence={}, orientation={}",
            i + 1,
            a.seq_id,
            a.start,
            a.end,
            a.consensus_repeat,
            a.consensus_repeat.len(),
            a.spacers.len(),
            a.evidence_level,
            a.orientation
        );
    }

    // ── Write outputs ────────────────────────────────────────────────────────
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let basename = input_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    let outdir = cli
        .outdir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("Result_{}_{}", basename, timestamp)));
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

    // ── CasFinder (optional) ─────────────────────────────────────────────────
    if cli.should_launch_cas() {
        let cas_cfg = cli.casfinder_config();
        match run_casfinder(input_path, &basename, &outdir, &cas_cfg) {
            Ok(search_results) => {
                info!("CasFinder found {} systems", search_results.systems.len());
                let faa_path = outdir.join(format!("orphos_{}/{}.faa", basename, basename));
                let faa_content = std::fs::read_to_string(&faa_path).ok();
                let cas_clusters = from_search_results(&search_results, faa_content.as_deref());
                let report = serde_json::json!({
                    "crisprs": &arrays,
                    "cas_clusters": cas_clusters,
                });
                let report_path = outdir.join("report.json");
                let f = File::create(&report_path).context("Creating merged report JSON")?;
                serde_json::to_writer_pretty(f, &report).context("Writing merged report JSON")?;
                info!("Merged report written to {:?}", report_path);
            }
            Err(err) => error!("CasFinder failed: {}", err),
        }
    }

    Ok(())
}
