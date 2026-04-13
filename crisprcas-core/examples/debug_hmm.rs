//! Quick test: search one HMM profile against predicted proteins to debug 0-hits issue.
use hmmer_core::{
    alphabet::Alphabet,
    background::BackgroundModel,
    config::SearchMode,
    modelconfig,
    pipeline::search::{CapacityHints, FilterPolicy, SearchOutcome, SearchPlan, SearchQuery},
    profile::Profile,
    sequence::DigitalSequence,
};
use hmmer_io::HmmFile;
use std::env;
use std::io::BufRead;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <profile.hmm> <proteins.faa>", args[0]);
        std::process::exit(1);
    }

    let hmm_path = &args[1];
    let faa_path = &args[2];

    // Parse proteins
    let abc = Alphabet::amino();
    let faa_content = std::fs::read_to_string(faa_path).expect("read faa");
    let mut targets: Vec<DigitalSequence> = Vec::new();
    let mut current_name = String::new();
    let mut current_desc = String::new();
    let mut current_seq = Vec::new();
    for line in faa_content.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if !current_name.is_empty() && !current_seq.is_empty() {
                targets.push(DigitalSequence::from_bytes(
                    &current_name, &current_desc, &current_seq, &abc,
                ));
            }
            let parts: Vec<&str> = header.splitn(2, char::is_whitespace).collect();
            current_name = parts[0].to_string();
            current_desc = parts.get(1).unwrap_or(&"").to_string();
            current_seq = Vec::new();
        } else {
            current_seq.extend_from_slice(line.trim().as_bytes());
        }
    }
    if !current_name.is_empty() && !current_seq.is_empty() {
        targets.push(DigitalSequence::from_bytes(
            &current_name, &current_desc, &current_seq, &abc,
        ));
    }
    println!("Loaded {} protein targets", targets.len());
    if let Some(t) = targets.first() {
        println!("  First target: name={}, len={}", t.name, t.len());
    }

    // Parse HMM
    let mut hfp = HmmFile::open(hmm_path, None).expect("open hmm");
    let (_habc, hmm) = hfp.read().expect("read hmm");
    println!("HMM: name={}, nodes={}", hmm.name, hmm.num_nodes);

    let avg_len = if targets.is_empty() { 400 } else {
        targets.iter().map(|s| s.len()).sum::<usize>() / targets.len()
    };
    println!("Average target length: {}", avg_len);

    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    modelconfig::profile_config(&hmm, &bg, &mut gm, avg_len, SearchMode::Local);

    let query = SearchQuery::from_configured_profile(gm, bg)
        .expect("query config");
    // First run with default filters to check what happens
    let plan = SearchPlan::builder(query)
        .filters(FilterPolicy::default())
        .build();
    let mut worker = plan
        .spawn_worker(CapacityHints { target_length: avg_len })
        .expect("spawn worker");

    let mut total_hits = 0;
    let mut total_domains = 0;
    let mut total_included = 0;
    let mut passing_coverage = 0;
    let mut filter_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for seq in &targets {
        let is_target = seq.name == "CP014688.1_298";
        if is_target {
            println!("  >>> Target seq CP014688.1_298: len={}", seq.len());
        }
        let report = worker.search(seq).expect("search");
        if is_target {
            println!("  >>> trace: null_score={:?}, msv_raw={:?}, vit_raw={:?}, fwd_raw={:?}, domain_count={}, filter={:?}",
                report.trace.null_score,
                report.trace.msv_raw_score,
                report.trace.viterbi_raw_score,
                report.trace.forward_raw_score,
                report.trace.domain_count,
                report.trace.filter_reason,
            );
        }
        match report.outcome {
            SearchOutcome::Hit(ref hit) => {
                total_hits += 1;
                if is_target {
                    println!("  >>> Got Hit for CP014688.1_298, {} domains", hit.domains.len());
                }
                for domain in &hit.domains {
                    total_domains += 1;
                    // Print ALL domains for target or high-scoring
                    if is_target || domain.bitscore > 20.0 {
                        println!(
                            "  DOM: seq={} score={:.1} pvalue={:.2e} included={} reported={} hmm={}-{}/{} aln={}-{}/{} env={}-{} env_score={:.1} bias={:.1}",
                            hit.name,
                            domain.bitscore,
                            domain.log_pvalue.exp(),
                            domain.is_included,
                            domain.is_reported,
                            domain.hmm_from,
                            domain.hmm_to,
                            hmm.num_nodes,
                            domain.alignment_start,
                            domain.alignment_end,
                            seq.len(),
                            domain.envelope_start,
                            domain.envelope_end,
                            domain.envelope_score,
                            domain.domain_bias,
                        );
                    }
                    if domain.is_included {
                        total_included += 1;
                        let prof_cov = (domain.hmm_to - domain.hmm_from + 1) as f64
                            / hmm.num_nodes as f64;
                        if prof_cov >= 0.4 {
                            passing_coverage += 1;
                            println!(
                                "  HIT: {} score={:.1} pvalue={:.2e} prof_cov={:.3} hmm={}-{} aln={}-{}",
                                hit.name,
                                domain.bitscore,
                                domain.log_pvalue.exp(),
                                prof_cov,
                                domain.hmm_from,
                                domain.hmm_to,
                                domain.alignment_start,
                                domain.alignment_end,
                            );
                        }
                    }
                }
            }
            SearchOutcome::Filtered(reason) => {
                if is_target {
                    println!("  >>> CP014688.1_298 FILTERED: {:?}", reason);
                }
                let key = format!("{:?}", reason);
                *filter_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    println!("\nSummary:");
    println!("  Sequences with any hit: {}", total_hits);
    println!("  Total domains: {}", total_domains);
    println!("  Included domains: {}", total_included);
    println!("  Passing coverage (>=0.4): {}", passing_coverage);
    println!("  Filter reasons:");
    for (reason, count) in &filter_counts {
        println!("    {:?}: {}", reason, count);
    }
}
