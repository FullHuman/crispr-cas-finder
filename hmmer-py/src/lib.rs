// hmmer-py: Python bindings via PyO3.

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

use hmmer_core::alphabet::Alphabet as CoreAlphabet;
use hmmer_core::background::BackgroundModel;
use hmmer_core::config::SearchMode;
use hmmer_core::modelconfig;
use hmmer_core::pipeline::{CapacityHints, SearchOutcome, SearchPlan, SearchQuery, Thresholds};
use hmmer_core::profile::Profile;
use hmmer_core::results::tophits::TopHits;
use hmmer_io::HmmFile;
use hmmer_io::seq_reader;

/// Python wrapper for an amino acid alphabet.
#[pyclass]
struct Alphabet {
    inner: CoreAlphabet,
}

#[pymethods]
impl Alphabet {
    #[staticmethod]
    fn amino() -> Self {
        Alphabet {
            inner: CoreAlphabet::amino(),
        }
    }

    #[staticmethod]
    fn dna() -> Self {
        Alphabet {
            inner: CoreAlphabet::dna(),
        }
    }

    #[getter]
    fn k(&self) -> usize {
        self.inner.canonical_size
    }

    fn __repr__(&self) -> String {
        format!(
            "Alphabet(type={:?}, K={})",
            self.inner.kind, self.inner.canonical_size
        )
    }
}

/// A single search hit result.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct HitResult {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    acc: Option<String>,
    #[pyo3(get)]
    desc: Option<String>,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    pre_score: f32,
    #[pyo3(get)]
    evalue: f64,
    #[pyo3(get)]
    ndom: usize,
    #[pyo3(get)]
    is_reported: bool,
    #[pyo3(get)]
    is_included: bool,
}

#[pymethods]
impl HitResult {
    fn __repr__(&self) -> String {
        format!(
            "Hit(name='{}', score={:.1}, E={:.2e}, ndom={})",
            self.name, self.score, self.evalue, self.ndom
        )
    }
}

/// Container for search results.
#[pyclass]
struct SearchResults {
    hits: Vec<HitResult>,
    #[pyo3(get)]
    nseq: usize,
}

#[pymethods]
impl SearchResults {
    #[getter]
    fn nhits(&self) -> usize {
        self.hits.len()
    }

    fn __len__(&self) -> usize {
        self.hits.len()
    }

    fn __getitem__(&self, idx: usize) -> PyResult<HitResult> {
        self.hits
            .get(idx)
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("hit index out of range"))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> HitIterator {
        HitIterator {
            hits: slf.hits.clone(),
            index: 0,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResults(nhits={}, nseq={})",
            self.hits.len(),
            self.nseq
        )
    }
}

#[pyclass]
struct HitIterator {
    hits: Vec<HitResult>,
    index: usize,
}

#[pymethods]
impl HitIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<HitResult> {
        if self.index < self.hits.len() {
            let hit = self.hits[self.index].clone();
            self.index += 1;
            Some(hit)
        } else {
            None
        }
    }
}

/// Perform an hmmsearch: search an HMM against a sequence database.
///
/// Args:
///     hmm_path: Path to the HMM file (.hmm)
///     seq_path: Path to the sequence database (FASTA)
///     e_threshold: E-value reporting threshold (default: 10.0)
///
/// Returns:
///     SearchResults containing all hits above threshold
#[pyfunction]
#[pyo3(signature = (hmm_path, seq_path, e_threshold=10.0))]
fn hmmsearch(hmm_path: &str, seq_path: &str, e_threshold: f64) -> PyResult<SearchResults> {
    // Open and read HMM
    let mut hfp = HmmFile::open(hmm_path, None).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let (abc, hmm) = hfp.read().map_err(|e| PyIOError::new_err(e.to_string()))?;

    // Set up background model and profile
    let bg = BackgroundModel::new(&abc);
    let mut gm = Profile::new(hmm.num_nodes, &abc);
    modelconfig::profile_config(&hmm, &bg, &mut gm, 400, SearchMode::Local);

    // Open sequence file
    let seqs = seq_reader::sqfile_open_digital(&abc, seq_path)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

    // Set up pipeline
    let query = SearchQuery::from_configured_profile(gm, bg)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    let plan = SearchPlan::builder(query).build();
    let mut worker = plan
        .spawn_worker(CapacityHints { target_length: 400 })
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    let mut th = TopHits::new();
    let thresholds = Thresholds {
        evalue: e_threshold,
        ..Thresholds::default()
    };

    // Search each sequence
    let nseq = seqs.len();
    for sq in &seqs {
        let report = worker
            .search(sq)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let SearchOutcome::Hit(hit) = report.outcome {
            th.push_hit(hit);
        }
    }

    // Sort and threshold
    th.sort_by_sortkey();
    let z = nseq as f64;
    th.threshold(
        thresholds.evalue,
        thresholds.domain_evalue,
        thresholds.inclusion_evalue,
        thresholds.inclusion_domain_evalue,
        thresholds.use_bit_cutoffs,
    );

    // Collect results
    let hits: Vec<HitResult> = th
        .hit_order
        .iter()
        .filter_map(|&idx| {
            let hit = &th.hits[idx];
            let evalue = hit.log_pvalue.exp() * z;
            if evalue <= e_threshold {
                Some(HitResult {
                    name: hit.name.clone(),
                    acc: hit.accession.clone(),
                    desc: hit.description.clone(),
                    score: hit.score,
                    pre_score: hit.pre_score,
                    evalue,
                    ndom: hit.num_domains,
                    is_reported: hit.is_reported(),
                    is_included: hit.is_included(),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(SearchResults { hits, nseq })
}

/// Python module for HMMER sequence analysis.
#[pymodule]
fn hmmer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Alphabet>()?;
    m.add_class::<HitResult>()?;
    m.add_class::<SearchResults>()?;
    m.add_function(wrap_pyfunction!(hmmsearch, m)?)?;
    Ok(())
}
