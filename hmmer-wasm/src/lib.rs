// hmmer-wasm: WebAssembly bindings via wasm-bindgen.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use hmmer_core::background::BackgroundModel;
use hmmer_core::config::SearchMode;
use hmmer_core::modelconfig;
use hmmer_core::pipeline::{CapacityHints, SearchOutcome, SearchPlan, SearchQuery, Thresholds};
use hmmer_core::profile::Profile;
use hmmer_core::results::tophits::TopHits;
use hmmer_io::HmmFile;
use hmmer_io::seq_reader;

#[derive(Serialize)]
struct WasmHit {
    name: String,
    score: f32,
    evalue: f64,
    ndom: usize,
}

#[derive(Serialize)]
struct WasmResults {
    nhits: usize,
    nseq: usize,
    hits: Vec<WasmHit>,
}

/// Search an HMM against a sequence database.
///
/// Both `hmm_path` and `seq_path` should be accessible from the WASM
/// virtual filesystem (e.g. via Emscripten FS or WASI).
///
/// Returns a JSON-serialized results object.
#[wasm_bindgen]
pub fn hmmsearch(hmm_path: &str, seq_path: &str, e_threshold: f64) -> Result<JsValue, JsValue> {
    let mut hfp = HmmFile::open(hmm_path, None).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let (abc, hmm) = hfp.read().map_err(|e| JsValue::from_str(&e.to_string()))?;

    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    modelconfig::profile_config(&hmm, &bg, &mut gm, 400, SearchMode::Local);

    let seqs = seq_reader::sqfile_open_digital(&abc, seq_path)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let query = SearchQuery::from_configured_profile(gm, bg)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let plan = SearchPlan::builder(query).build();
    let mut worker = plan
        .spawn_worker(CapacityHints { target_length: 400 })
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut th = TopHits::new();
    let thresholds = Thresholds {
        evalue: e_threshold,
        ..Thresholds::default()
    };

    let nseq = seqs.len();
    for sq in &seqs {
        let report = worker
            .search(sq)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        if let SearchOutcome::Hit(hit) = report.outcome {
            th.push_hit(hit);
        }
    }

    th.sort_by_sortkey();
    let z = nseq as f64;
    th.threshold(
        thresholds.evalue,
        thresholds.domain_evalue,
        thresholds.inclusion_evalue,
        thresholds.inclusion_domain_evalue,
        thresholds.use_bit_cutoffs,
    );

    let hits: Vec<WasmHit> = th
        .hit_order
        .iter()
        .filter_map(|&idx| {
            let hit = &th.hits[idx];
            let evalue = hit.log_pvalue.exp() * z;
            if evalue <= e_threshold {
                Some(WasmHit {
                    name: hit.name.clone(),
                    score: hit.score,
                    evalue,
                    ndom: hit.num_domains,
                })
            } else {
                None
            }
        })
        .collect();

    let results = WasmResults {
        nhits: hits.len(),
        nseq,
        hits,
    };
    serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
}
