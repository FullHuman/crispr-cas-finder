use std::ffi::OsString;
use std::fs::{File, create_dir_all};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use crispr_cas_finder_core::{
    AnalysisOptions, CasFinderConfig, DetectionParams, RepliconTopology, analyze, annotation, io,
    parse_fasta,
    repeat_db::{KnownCrisprs, RepeatDatabase},
};
use log::info;

mod bundled_data;

#[derive(Debug)]
struct PreparedCasConfig {
    config: CasFinderConfig,
    // Owns the extracted model files until the analysis completes.
    _bundled_data: Option<tempfile::TempDir>,
}

fn parse_genetic_code(value: &str) -> Result<usize> {
    let code = value.parse()?;
    crispr_cas_finder_core::cas_pipeline::validate_genetic_code(code)?;
    Ok(code)
}

const MAX_OUTPUT_DIRECTORY_SUFFIX: usize = 999;

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum TopologyArgument {
    #[default]
    Circular,
    Linear,
}

/// Cas model sets shipped with CasFinder. `General` is upstream's name for
/// the class-level definitions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum DefinitionArgument {
    #[default]
    #[value(name = "SubTyping", alias = "S")]
    SubTyping,
    #[value(name = "Typing", alias = "T")]
    Typing,
    #[value(name = "General", alias = "G", alias = "Class")]
    General,
}

impl DefinitionArgument {
    /// Directory stem of the model set inside a CasFinder distribution.
    fn directory_stem(self) -> &'static str {
        match self {
            Self::SubTyping => "SubTyping",
            Self::Typing => "Typing",
            Self::General => "Class",
        }
    }
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
    #[arg(
        help = "Favour repeats matching the --foster-repeat-* settings and use --foster-repeat-length as the minimum repeat length"
    )]
    force_detection: bool,
    #[arg(
        long,
        alias = "fosterDRLength",
        value_name = "INT",
        default_value_t = 30
    )]
    #[arg(help = "Repeat length favoured by --force-detection")]
    foster_repeat_length: usize,
    #[arg(long, alias = "fosterDRBegin", value_name = "STR", default_value = "G")]
    #[arg(help = "IUPAC pattern ('.' = any base) a favoured repeat starts with")]
    foster_repeat_begin: String,
    #[arg(long, alias = "fosterDREnd", value_name = "STR", default_value = "AA.")]
    #[arg(help = "IUPAC pattern ('.' = any base) a favoured repeat ends with")]
    foster_repeat_end: String,
    #[arg(long, alias = "minNbSpacers", value_name = "INT", default_value_t = 1)]
    min_spacer_count: usize,
    #[arg(long, alias = "betterDetectTrunc", action = clap::ArgAction::SetTrue)]
    #[arg(help = "Look for a truncated terminal repeat on each side of every array")]
    better_detect_truncated: bool,
    #[arg(
        long,
        alias = "PercMismTrunc",
        value_name = "FLOAT",
        default_value_t = 4.0
    )]
    #[arg(
        help = "Percentage of mismatches allowed in the matched half of a truncated terminal repeat"
    )]
    truncated_mismatch_percent: f64,
    #[arg(long, aliases = ["classifySmallArrays", "classifySmall", "cSA"], action = clap::ArgAction::SetTrue)]
    #[arg(
        help = "Raise level-1 arrays to level 4 when they share their repeat with a level-4 array"
    )]
    classify_small_arrays: bool,
    #[arg(long, aliases = ["MatchingRepeats", "Mrpts"], value_name = "FILE")]
    #[arg(help = "FASTA of repeats to search for instead of discovering repeats")]
    matching_repeats: Option<PathBuf>,
    #[arg(
        long,
        alias = "rpts",
        value_name = "FILE",
        requires = "repeat_directions"
    )]
    #[arg(help = "Repeat list (sequence;id;count;orientation) replacing the embedded one")]
    repeats: Option<PathBuf>,
    #[arg(long, aliases = ["DIRrepeat", "drpt"], value_name = "FILE", requires = "repeats")]
    #[arg(help = "Repeat directions (id<TAB>direction) matching --repeats")]
    repeat_directions: Option<PathBuf>,
    #[arg(long, aliases = ["DBcrispr", "dbc"], value_name = "FILE")]
    #[arg(help = "CRISPRdb export used to flag arrays that are already recorded")]
    crisprdb: Option<PathBuf>,
    #[arg(long = "cas", action = clap::ArgAction::SetTrue)]
    launch_cas_finder: bool,
    #[arg(
        long,
        alias = "cpuMacSyFinder",
        value_name = "INT",
        default_value_t = 0
    )]
    workers: usize,
    #[arg(long, value_enum, ignore_case = true, default_value_t = DefinitionArgument::SubTyping)]
    #[arg(help = "Cas model set: SubTyping, Typing, or General (class level)")]
    definition: DefinitionArgument,
    #[arg(long, alias = "geneticCode", value_name = "INT", default_value_t = 11,
        value_parser = parse_genetic_code, help = "Genetic code (only 11 is currently supported)")]
    genetic_code: usize,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    #[arg(help = "Predict genes in metagenomic mode for every sequence")]
    metagenome: bool,
    #[arg(long, value_name = "INT", default_value_t = annotation::AUTO_METAGENOME_BELOW)]
    #[arg(help = "Sequences shorter than this use metagenomic gene prediction; 0 disables")]
    auto_metagenome_below: usize,
    #[arg(
        long = "gff",
        alias = "gffAnnot",
        value_name = "FILE",
        requires = "faa"
    )]
    #[arg(help = "GFF3 annotation to use instead of predicting genes (needs --faa)")]
    gff: Option<PathBuf>,
    #[arg(
        long = "faa",
        alias = "proteome",
        value_name = "FILE",
        requires = "gff"
    )]
    #[arg(help = "Protein FASTA matching --gff")]
    faa: Option<PathBuf>,
    #[arg(long, alias = "rcfowce", action = clap::ArgAction::SetTrue)]
    #[arg(help = "Search Cas systems only on sequences with at least one CRISPR array")]
    cas_only_with_crispr: bool,
    #[arg(long, value_name = "INT", default_value_t = 600)]
    #[arg(help = "Report arrays within this many bp of a Cas system")]
    vicinity: usize,
    #[arg(long, alias = "cluster", value_name = "INT")]
    #[arg(help = "Chain arrays and Cas systems closer than this many bp into loci")]
    locus_distance: Option<usize>,
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
    fn detection_params(&self) -> Result<DetectionParams> {
        let query_repeats = match &self.matching_repeats {
            Some(path) => {
                let file = File::open(path)
                    .with_context(|| format!("Failed to open repeats FASTA: {:?}", path))?;
                let repeats: Vec<String> = parse_fasta(file)
                    .with_context(|| format!("Failed to read repeats FASTA: {:?}", path))?
                    .into_iter()
                    .map(|record| String::from_utf8_lossy(&record.sequence).into_owned())
                    .collect();
                if repeats.is_empty() {
                    bail!("Repeats FASTA contains no sequence: {:?}", path);
                }
                repeats
            }
            None => Vec::new(),
        };
        Ok(DetectionParams {
            classify_small_arrays: self.classify_small_arrays,
            query_repeats,
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
        })
    }

    fn repeat_database(&self) -> Result<Option<RepeatDatabase>> {
        let (Some(repeats), Some(directions)) = (&self.repeats, &self.repeat_directions) else {
            return Ok(None);
        };
        let repeat_list = std::fs::read_to_string(repeats)
            .with_context(|| format!("Failed to read repeat list: {:?}", repeats))?;
        let direction_list = std::fs::read_to_string(directions)
            .with_context(|| format!("Failed to read repeat directions: {:?}", directions))?;
        let database = RepeatDatabase::parse(&repeat_list, &direction_list);
        if database.is_empty() {
            bail!("Repeat list contains no repeat: {:?}", repeats);
        }
        Ok(Some(database))
    }

    fn known_crisprs(&self) -> Result<Option<KnownCrisprs>> {
        let Some(path) = &self.crisprdb else {
            return Ok(None);
        };
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read CRISPRdb export: {:?}", path))?;
        let known = KnownCrisprs::parse(&content);
        if known.is_empty() {
            bail!("CRISPRdb export contains no array: {:?}", path);
        }
        Ok(Some(known))
    }

    fn casfinder_config(&self) -> Result<PreparedCasConfig> {
        let bundled_data = (self.cas_models_dir.is_none() || self.cas_profiles_dir.is_none())
            .then(bundled_data::extract)
            .transpose()?;
        let data_root = bundled_data.as_ref().map(|directory| directory.path());

        let cas_models_dir = self.cas_models_dir.as_ref().map(PathBuf::from).or_else(|| {
            data_root
                .as_ref()
                .map(|root| root.join(format!("DEF-{}-2.0.3", self.definition.directory_stem())))
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

        Ok(PreparedCasConfig {
            config: CasFinderConfig {
                genetic_code: self.genetic_code,
                metagenome: self.metagenome,
                auto_metagenome_below: self.auto_metagenome_below,
                annotation_gff: self.gff.clone(),
                proteome_faa: self.faa.clone(),
                workers: self.workers,
                definition: self.definition.directory_stem().to_string(),
                quiet: self.quiet,
                replicon_topology: self.topology.into(),
                min_best_hit_score: self.min_cas_score,
                cas_models_dir: Some(cas_models_dir),
                cas_profiles_dir: Some(cas_profiles_dir),
            },
            _bundled_data: bundled_data,
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
    let options = AnalysisOptions {
        detection: cli.detection_params()?,
        cas: casfinder_config
            .as_ref()
            .map(|prepared| prepared.config.clone()),
        cas_only_with_crispr: cli.cas_only_with_crispr,
        vicinity: cli.vicinity,
        locus_distance: cli.locus_distance,
        repeat_database: cli.repeat_database()?,
        known_crisprs: cli.known_crisprs()?,
    };
    options.detection.validate().map_err(anyhow::Error::msg)?;

    let input_path = &cli.input;
    // Open separately so callers can inspect the underlying I/O error kind.
    let input_file = File::open(input_path)
        .with_context(|| format!("Failed to open FASTA: {:?}", input_path))?;
    let sequences = parse_fasta(input_file)
        .with_context(|| format!("Failed to read FASTA: {:?}", input_path))?;
    info!(
        "Analysing {} sequence(s) from {:?}",
        sequences.len(),
        input_path
    );

    let analysis = analyze(&sequences, &options)
        .with_context(|| format!("Analysis failed for {:?}", input_path))?;
    let report = &analysis.report;
    info!(
        "Detected {} CRISPR arrays and {} Cas systems",
        report.crisprs.len(),
        report.cas_clusters.len()
    );
    for array in &report.crisprs {
        info!(
            "  {}: {}..{}, DR={} (len={}), {} spacers, evidence={}, orientation={}",
            array.id,
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

    write_output(&outdir, "report.json", |file| {
        io::write_report_json(report, file)
    })?;
    write_output(&outdir, "sequences.tsv", |file| {
        io::write_sequences_tsv(report, file)
    })?;
    write_output(&outdir, "crisprs.tsv", |file| {
        io::write_crisprs_tsv(report, file)
    })?;
    write_output(&outdir, "features.gff", |file| io::write_gff(report, file))?;
    write_output(&outdir, "arrays.fna", |file| {
        io::write_arrays_fasta(report, file)
    })?;
    write_output(&outdir, "repeats.fna", |file| {
        io::write_repeats_fasta(report, file)
    })?;
    write_output(&outdir, "spacers.fna", |file| {
        io::write_spacers_fasta(report, file)
    })?;
    if let Some(proteome) = &analysis.proteome {
        write_output(&outdir, "cas_systems.tsv", |file| {
            io::write_cas_tsv(report, file)
        })?;
        write_output(&outdir, "associations.tsv", |file| {
            io::write_associations_tsv(report, file)
        })?;
        write_output(&outdir, "cas_proteins.faa", |file| {
            io::write_cas_proteins_fasta(report, proteome, file)
        })?;
        let annotation_dir = outdir.join("annotation");
        create_dir_all(&annotation_dir)?;
        write_output(&annotation_dir, "proteins.faa", |file| {
            io::write_proteome_fasta(proteome, file)
        })?;
        std::fs::write(
            annotation_dir.join("genes.gff"),
            annotation::gff_content(proteome),
        )
        .context("Writing annotation/genes.gff")?;
    }
    if report.parameters.locus_distance.is_some() {
        write_output(&outdir, "loci.tsv", |file| io::write_loci_tsv(report, file))?;
    }
    info!("Results written to {:?}", outdir);

    Ok(())
}

fn write_output(
    directory: &Path,
    name: &str,
    write: impl FnOnce(&mut File) -> Result<()>,
) -> Result<()> {
    let path = directory.join(name);
    let mut file =
        File::create(&path).with_context(|| format!("Failed to create {}", path.display()))?;
    write(&mut file).with_context(|| format!("Failed to write {}", path.display()))
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
    }

    #[test]
    fn association_and_annotation_options_are_parsed() {
        let cli = Cli::try_parse_from([
            "crispr-cas-finder",
            "--in",
            "input.fa",
            "--vicinity",
            "1000",
            "--cluster",
            "20000",
            "--rcfowce",
            "--gff",
            "genes.gff",
            "--faa",
            "proteins.faa",
            "--definition",
            "general",
        ])
        .unwrap();
        assert_eq!(cli.vicinity, 1000);
        assert_eq!(cli.locus_distance, Some(20000));
        assert!(cli.cas_only_with_crispr);
        assert_eq!(cli.gff.as_deref(), Some(Path::new("genes.gff")));
        assert_eq!(cli.definition.directory_stem(), "Class");
    }

    #[test]
    fn annotation_gff_requires_a_protein_fasta() {
        let result = Cli::try_parse_from([
            "crispr-cas-finder",
            "--in",
            "input.fa",
            "--gff",
            "genes.gff",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn foster_and_truncation_options_reach_detection_params() {
        let cli = Cli::try_parse_from([
            "crispr-cas-finder",
            "--in",
            "input.fa",
            "--force-detection",
            "--foster-repeat-length",
            "31",
            "--foster-repeat-begin",
            "A",
            "--foster-repeat-end",
            "TT",
            "--better-detect-truncated",
            "--truncated-mismatch-percent",
            "5",
        ])
        .unwrap();
        let params = cli.detection_params().unwrap();
        params.validate().unwrap();
        assert!(params.force_detection && params.better_detect_truncated);
        assert_eq!(params.foster_repeat_length, 31);
        assert_eq!(params.foster_repeat_begin, "A");
        assert_eq!(params.foster_repeat_end, "TT");
        assert_eq!(params.truncated_mismatch_percent, 5.0);
    }

    #[test]
    fn invalid_foster_patterns_are_rejected() {
        let cli = Cli::try_parse_from([
            "crispr-cas-finder",
            "--in",
            "input.fa",
            "--foster-repeat-begin",
            "[AG]",
        ])
        .unwrap();
        assert!(
            cli.detection_params()
                .unwrap()
                .validate()
                .unwrap_err()
                .contains("IUPAC")
        );
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

        let prepared = cli.casfinder_config().expect("resolve bundled Cas data");
        let config = &prepared.config;
        assert_eq!(config.workers, 3);
        assert_eq!(config.replicon_topology, RepliconTopology::Linear);
        assert_eq!(config.min_best_hit_score, 30.5);
        assert!(
            config
                .cas_models_dir
                .as_ref()
                .unwrap()
                .join("CAS-TypeIE.xml")
                .is_file()
        );
        assert!(
            config
                .cas_profiles_dir
                .as_ref()
                .unwrap()
                .join("Cas5_0_IE.hmm")
                .is_file()
        );
    }

    #[test]
    fn unsupported_genetic_codes_are_rejected_before_analysis() {
        for code in ["1", "4", "25", "267"] {
            let error = Cli::try_parse_from([
                "crispr-cas-finder",
                "--in",
                "input.fa",
                "--genetic-code",
                code,
            ])
            .unwrap_err();
            assert!(error.to_string().contains("only genetic code 11"));
        }
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
        let report: crispr_cas_finder_core::AnalysisReport =
            serde_json::from_str(&report).expect("parse report");
        assert!(report.cas_clusters.is_empty());
    }
}
