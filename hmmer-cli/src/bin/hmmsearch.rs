// hmmsearch - Search sequence database with profile HMM
//
// Port of src/hmmsearch.c

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use rayon::prelude::*;

use hmmer_core::alidisplay::AliDisplay;
use hmmer_core::background::BackgroundModel;
use hmmer_core::config::SearchMode;
use hmmer_core::hmm::Hmm;
use hmmer_core::modelconfig;
use hmmer_core::pipeline::{
    CapacityHints, FilterPolicy, SearchMetrics, SearchOutcome, SearchPlan, SearchQuery, Thresholds,
};
use hmmer_core::profile::Profile;
use hmmer_core::sequence::DigitalSequence;
use hmmer_core::tophits::TopHits;
use hmmer_io::hmmfile::HmmFile;
use hmmer_io::seq_reader::FastaReader;

/// Search profile(s) against a sequence database
#[derive(Parser)]
#[command(name = "hmmsearch")]
struct Args {
    /// Report sequences <= this E-value threshold
    #[arg(short = 'E', default_value_t = 10.0)]
    evalue: f64,

    /// Save per-sequence hits to tabular file
    #[arg(long)]
    tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long)]
    domtblout: Option<PathBuf>,

    /// Turn off all filters (slow)
    #[arg(long = "max")]
    do_max: bool,

    /// Number of parallel CPU threads (0 = all cores)
    #[arg(long, default_value_t = 0)]
    cpu: usize,

    /// HMM profile file
    hmmfile: PathBuf,

    /// Sequence database (FASTA)
    seqdb: PathBuf,
}

struct SearchRunSummary {
    thresholds: Thresholds,
    search_space: f64,
    metrics: SearchMetrics,
}

fn main() {
    let args = Args::parse();

    // Read HMM file
    let hmmfile_str = args
        .hmmfile
        .to_str()
        .expect("HMM file path is not valid UTF-8");
    let seqdb_str = args
        .seqdb
        .to_str()
        .expect("sequence database path is not valid UTF-8");

    let mut hfp = match HmmFile::open(hmmfile_str, None) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening HMM file {}: {}", args.hmmfile.display(), e);
            std::process::exit(1);
        }
    };

    let (abc, hmm) = match hfp.read() {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Error reading HMM: {}", e);
            std::process::exit(1);
        }
    };

    // Configure profile
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    modelconfig::profile_config(&hmm, &bg, &mut gm, 400, SearchMode::Local);
    modelconfig::reconfigure_length(&mut gm, 400);

    // Print banner
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = hmmer_core::hmmer::banner(
        &mut out,
        "hmmsearch",
        "search profile(s) against a sequence database",
    );
    let _ = writeln!(out, "# query HMM file:                  {}", hmmfile_str);
    let _ = writeln!(out, "# target sequence database:        {}", seqdb_str);
    if args.do_max {
        let _ = writeln!(
            out,
            "# max mode:                        on [all heuristic filters off]"
        );
    }
    let _ = writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    );
    let _ = writeln!(out);

    // Read all sequences into memory
    let sequences = read_sequences(&abc, seqdb_str);
    let nseq = sequences.len() as u64;

    // Configure rayon thread pool
    if args.cpu > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.cpu)
            .build_global()
            .ok();
    }

    // Parallel search
    let (mut summary, mut th) = run_parallel_search(&sequences, &hmm, &gm, &bg, &args);

    // Set Z (search space size)
    summary.search_space = nseq as f64;

    // Sort hits and apply thresholds
    th.sort_by_sortkey();
    th.threshold(
        summary.thresholds.evalue,
        summary.thresholds.domain_evalue,
        summary.thresholds.inclusion_evalue,
        summary.thresholds.inclusion_domain_evalue,
        summary.thresholds.use_bit_cutoffs,
    );

    let log_z = summary.search_space.ln();

    // ---- Output results ----
    write_query_header(&mut out, &hmm);
    write_scores_table(&mut out, &th, &summary, log_z);
    write_domain_annotations(&mut out, &th, &summary, log_z);

    if let Some(ref path) = args.tblout {
        write_tblout(path, &th, &hmm, &summary, log_z);
    }
    if let Some(ref path) = args.domtblout {
        write_domtblout(path, &th, &hmm, &summary, log_z);
    }

    write_statistics(&mut out, &hmm, nseq, &summary);
}

fn read_sequences(abc: &hmmer_core::alphabet::Alphabet, seqdb: &str) -> Vec<DigitalSequence> {
    let mut sqfp = match FastaReader::open_digital(abc, seqdb) {
        Ok(f) => f,
        Err(msg) => {
            eprintln!("Error opening sequence file {}: {}", seqdb, msg);
            std::process::exit(1);
        }
    };

    let mut sequences = Vec::new();
    loop {
        match sqfp.read() {
            Ok(Some(sq)) => sequences.push(sq),
            Ok(None) => break,
            Err(msg) => {
                eprintln!("Error reading sequence: {}", msg);
                break;
            }
        }
    }
    sequences
}

/// Run the parallel search pipeline using rayon fold/reduce.
fn run_parallel_search(
    sequences: &[DigitalSequence],
    hmm: &Hmm,
    gm: &Profile,
    bg: &BackgroundModel,
    args: &Args,
) -> (SearchRunSummary, TopHits) {
    let query = SearchQuery::from_configured_profile(gm.clone(), bg.clone())
        .expect("configured profile should compile into a reusable search query");
    let plan = SearchPlan::builder(query)
        .filters(FilterPolicy {
            max_mode: args.do_max,
            ..FilterPolicy::default()
        })
        .build();

    let (metrics, th) = sequences
        .par_iter()
        .fold(
            || {
                let worker = plan
                    .spawn_worker(CapacityHints { target_length: 400 })
                    .expect("search worker allocation should succeed");
                (worker, TopHits::new())
            },
            |(mut worker, mut local_th), sq| {
                match worker.search(sq) {
                    Ok(report) => {
                        if let SearchOutcome::Hit(mut hit) = report.outcome {
                            attach_alignment_displays(&mut hit, hmm, sq);
                            local_th.push_hit(hit);
                        }
                    }
                    Err(msg) => {
                        eprintln!("Warning: pipeline error on {}: {}", sq.name, msg);
                    }
                }
                (worker, local_th)
            },
        )
        .map(|(worker, local_th)| (worker.into_metrics(), local_th))
        .reduce(
            || (SearchMetrics::default(), TopHits::new()),
            |(mut a_metrics, mut a_th), (b_metrics, b_th)| {
                a_metrics.merge(&b_metrics);
                a_th.merge(b_th);
                (a_metrics, a_th)
            },
        );

    let thresholds = Thresholds {
        evalue: args.evalue,
        ..Thresholds::default()
    };
    (
        SearchRunSummary {
            thresholds,
            search_space: 0.0,
            metrics,
        },
        th,
    )
}

fn attach_alignment_displays(hit: &mut hmmer_core::hit::Hit, hmm: &Hmm, sq: &DigitalSequence) {
    for dom in &mut hit.domains {
        if dom.alignment_display.is_none() {
            if let Some(ref tr) = dom.trace {
                let domains = tr.compute_domains();
                if let Some(td) = domains.first() {
                    dom.alignment_display = AliDisplay::create(
                        tr,
                        td,
                        hmm,
                        &sq.residues,
                        &sq.name,
                        sq.accession.as_deref().unwrap_or(""),
                        sq.description.as_deref().unwrap_or(""),
                        sq.source_length,
                    )
                    .map(Box::new);
                }
            }
            dom.trace = None;
        }
    }
}

/// Compute E-value from log p-value and log search-space size.
fn evalue(log_pvalue: f64, log_z: f64) -> f64 {
    (log_pvalue + log_z).exp()
}

fn write_query_header(out: &mut impl Write, hmm: &Hmm) {
    let _ = writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.num_nodes);
    if let Some(ref desc) = hmm.description {
        let _ = writeln!(out, "Description: {}", desc);
    }
    let _ = writeln!(out);
}

fn write_scores_table(out: &mut impl Write, th: &TopHits, summary: &SearchRunSummary, log_z: f64) {
    let _ = writeln!(
        out,
        "Scores for complete sequences (score includes all domains):"
    );
    let _ = writeln!(
        out,
        "   --- full sequence ---   --- best 1 domain ---    -#dom-"
    );
    let _ = writeln!(
        out,
        "    E-value  score  bias    E-value  score  bias    exp  N  Sequence        Description"
    );
    let _ = writeln!(
        out,
        "    ------- ------ -----    ------- ------ -----   ---- --  --------        -----------"
    );

    let mut nhits_shown = 0;
    for i in 0..th.len() {
        let Some(hit) = th.get_hit(i) else { continue };

        let ev = evalue(hit.log_pvalue, log_z);
        if ev > summary.thresholds.evalue {
            continue;
        }

        let (best_ev, best_score, best_bias) = match hit.best_domain {
            Some(bd) => {
                let d = &hit.domains[bd];
                (
                    evalue(d.log_pvalue, log_z),
                    d.bitscore,
                    d.domain_bias / std::f32::consts::LN_2,
                )
            }
            None => (ev, hit.score, 0.0),
        };
        let seq_bias = hit.pre_score - hit.score;

        let _ = writeln!(
            out,
            " {:9.2e} {:6.1} {:5.1}   {:9.2e} {:6.1} {:5.1}   {:4.1} {:2}  {:<15} {}",
            ev,
            hit.score,
            seq_bias,
            best_ev,
            best_score,
            best_bias,
            hit.num_expected,
            hit.num_domains,
            hit.name,
            hit.description.as_deref().unwrap_or(""),
        );
        nhits_shown += 1;
    }

    if nhits_shown == 0 {
        let _ = writeln!(
            out,
            "\n   [No hits detected that satisfy reporting thresholds]"
        );
    }
    let _ = writeln!(out);
}

fn write_domain_annotations(
    out: &mut impl Write,
    th: &TopHits,
    summary: &SearchRunSummary,
    log_z: f64,
) {
    let _ = writeln!(out, "Domain annotation for each sequence:");
    for i in 0..th.len() {
        let Some(hit) = th.get_hit(i) else { continue };
        if evalue(hit.log_pvalue, log_z) > summary.thresholds.evalue {
            continue;
        }

        let _ = writeln!(
            out,
            ">> {}  {}",
            hit.name,
            hit.description.as_deref().unwrap_or("")
        );
        let _ = writeln!(
            out,
            "   #    score  bias  c-Evalue  i-Evalue hmmfrom  hmm to    alifrom  ali to    envfrom  env to     acc"
        );
        let _ = writeln!(
            out,
            "   ---   ------ ----- --------- --------- ------- -------    ------- -------    ------- -------    ------"
        );

        for (d, dom) in hit.domains.iter().enumerate() {
            let cev = evalue(dom.log_pvalue, log_z);
            let _ = writeln!(
                out,
                "   {:3} {:>8.1} {:5.1}  {:9.2e}  {:9.2e} {:>7} {:>7} .. {:>7} {:>7} .. {:>7} {:>7}    {:.2}",
                d + 1,
                dom.bitscore,
                dom.domain_bias / std::f32::consts::LN_2,
                cev,
                cev,
                dom.hmm_from,
                dom.hmm_to,
                dom.alignment_start,
                dom.alignment_end,
                dom.envelope_start,
                dom.envelope_end,
                dom.optimal_accuracy_score,
            );
        }
        let _ = writeln!(out);

        for (d, dom) in hit.domains.iter().enumerate() {
            let cev = evalue(dom.log_pvalue, log_z);
            let _ = writeln!(out, "  Alignments for each domain:");
            let _ = writeln!(
                out,
                "  == domain {}  score: {:.1} bits;  conditional E-value: {:.2e}",
                d + 1,
                dom.bitscore,
                cev,
            );
            if let Some(ref ad) = dom.alignment_display {
                let _ = write!(out, "{}", ad.print(false));
            }
            let _ = writeln!(out);
        }
    }
}

fn write_tblout(path: &PathBuf, th: &TopHits, hmm: &Hmm, summary: &SearchRunSummary, log_z: f64) {
    let Ok(mut fp) = std::fs::File::create(path) else {
        eprintln!("Error: could not create {}", path.display());
        return;
    };
    let _ = writeln!(
        fp,
        "#                                                               --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----"
    );
    let _ = writeln!(
        fp,
        "# target name        accession  query name           accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target"
    );
    let _ = writeln!(
        fp,
        "#------------------- ---------- -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- ---  -- --- --- --- --- ---------------------"
    );

    for i in 0..th.len() {
        let Some(hit) = th.get_hit(i) else { continue };
        let ev = evalue(hit.log_pvalue, log_z);
        if ev > summary.thresholds.evalue {
            continue;
        }

        let (best_ev, best_score, best_bias) = match hit.best_domain {
            Some(bd) => (
                evalue(hit.domains[bd].log_pvalue, log_z),
                hit.domains[bd].bitscore,
                hit.domains[bd].domain_bias / std::f32::consts::LN_2,
            ),
            None => (ev, hit.score, 0.0),
        };

        let _ = writeln!(
            fp,
            "{:<20} {:<10} {:<20} {:<10} {:9.2e} {:6.1} {:5.1} {:9.2e} {:6.1} {:5.1}   {:3.1} {:3} {:3}  {:2} {:3} {:3} {:3} {:3} {}",
            hit.name,
            hit.accession.as_deref().unwrap_or("-"),
            hmm.name,
            hmm.accession.as_deref().unwrap_or("-"),
            ev,
            hit.score,
            hit.pre_score - hit.score,
            best_ev,
            best_score,
            best_bias,
            hit.num_expected,
            hit.num_regions,
            hit.num_clustered,
            hit.num_overlaps,
            hit.num_envelopes,
            hit.num_domains,
            hit.num_domains,
            hit.num_domains,
            hit.description.as_deref().unwrap_or(""),
        );
    }
}

fn write_domtblout(
    path: &PathBuf,
    th: &TopHits,
    hmm: &Hmm,
    summary: &SearchRunSummary,
    log_z: f64,
) {
    let Ok(mut fp) = std::fs::File::create(path) else {
        eprintln!("Error: could not create {}", path.display());
        return;
    };
    let _ = writeln!(
        fp,
        "#                                                                            --- full sequence --- -------------- this domain ----------   hmm coord   ali coord   env coord"
    );
    let _ = writeln!(
        fp,
        "# target name        accession   tlen query name           accession   qlen   E-value  score  bias   #  of  c-Evalue  i-Evalue  score  bias  from    to  from    to  from    to  acc description of target"
    );
    let _ = writeln!(
        fp,
        "#------------------- ---------- ----- -------------------- ---------- ----- --------- ------ ----- --- --- --------- --------- ------ ----- ----- ----- ----- ----- ----- ----- ---- ---------------------"
    );

    for i in 0..th.len() {
        let Some(hit) = th.get_hit(i) else { continue };
        let ev = evalue(hit.log_pvalue, log_z);
        if ev > summary.thresholds.evalue {
            continue;
        }

        for (d, dom) in hit.domains.iter().enumerate() {
            let cev = evalue(dom.log_pvalue, log_z);
            let _ = writeln!(
                fp,
                "{:<20} {:<10} {:>5} {:<20} {:<10} {:>5} {:9.2e} {:6.1} {:5.1} {:3} {:3} {:9.2e} {:9.2e} {:6.1} {:5.1} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:.2} {}",
                hit.name,
                hit.accession.as_deref().unwrap_or("-"),
                "-",
                hmm.name,
                hmm.accession.as_deref().unwrap_or("-"),
                hmm.num_nodes,
                ev,
                hit.score,
                hit.pre_score - hit.score,
                d + 1,
                hit.num_domains,
                cev,
                cev,
                dom.bitscore,
                dom.domain_bias / std::f32::consts::LN_2,
                dom.hmm_from,
                dom.hmm_to,
                dom.alignment_start,
                dom.alignment_end,
                dom.envelope_start,
                dom.envelope_end,
                dom.optimal_accuracy_score,
                hit.description.as_deref().unwrap_or(""),
            );
        }
    }
}

fn write_statistics(out: &mut impl Write, hmm: &Hmm, nseq: u64, summary: &SearchRunSummary) {
    let stats = &summary.metrics.stats;
    let ratio = |count: u64| {
        if nseq > 0 {
            count as f64 / nseq as f64
        } else {
            0.0
        }
    };

    let _ = writeln!(out, "Internal pipeline statistics summary:");
    let _ = writeln!(out, "-------------------------------------");
    let _ = writeln!(
        out,
        "Query model(s):              {:>10}  ({}  nodes)",
        1, hmm.num_nodes
    );
    let _ = writeln!(
        out,
        "Target sequences:            {:>10}  ({} residues searched)",
        nseq, stats.num_residues
    );
    let _ = writeln!(
        out,
        "Passed MSV filter:           {:>10}  ({:.6})",
        stats.sequences_past_msv,
        ratio(stats.sequences_past_msv)
    );
    let _ = writeln!(
        out,
        "Passed bias filter:          {:>10}  ({:.6})",
        stats.sequences_past_bias,
        ratio(stats.sequences_past_bias)
    );
    let _ = writeln!(
        out,
        "Passed Vit filter:           {:>10}  ({:.6})",
        stats.sequences_past_viterbi,
        ratio(stats.sequences_past_viterbi)
    );
    let _ = writeln!(
        out,
        "Passed Fwd filter:           {:>10}  ({:.6})",
        stats.sequences_past_forward,
        ratio(stats.sequences_past_forward)
    );
    let _ = writeln!(
        out,
        "Initial search space (Z):    {:>10.0}",
        summary.search_space
    );
    let _ = writeln!(out, "//");
}
