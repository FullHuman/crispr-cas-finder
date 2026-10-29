use crispr_cas_finder_core::{
    AnalysisOptions, CasFinderConfig, CrisprArray as CoreCrisprArray, DetectionParams, Orientation,
    Repeat as CoreRepeat, RepliconTopology, Spacer as CoreSpacer, detect_crisprs_in_fasta_path,
    detect_crisprs_in_fasta_str, parse_fasta,
};
use pyo3::exceptions::{PyFileNotFoundError, PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Python-exposed data classes
// ---------------------------------------------------------------------------

/// A single direct repeat within a CRISPR array.
#[pyclass(frozen, module = "crispr_cas_finder", skip_from_py_object)]
#[derive(Clone)]
pub struct Repeat {
    /// 1-based start position on the sequence.
    #[pyo3(get)]
    pub start: usize,
    /// 1-based end position on the sequence (inclusive).
    #[pyo3(get)]
    pub end: usize,
    /// DNA sequence of the direct repeat, in the case the input has it in.
    #[pyo3(get)]
    pub sequence: String,
}

#[pymethods]
impl Repeat {
    fn __repr__(&self) -> String {
        format!(
            "Repeat(start={}, end={}, sequence={:?})",
            self.start, self.end, self.sequence
        )
    }

    fn __len__(&self) -> usize {
        self.sequence.len()
    }
}

/// A single spacer between two direct repeats.
#[pyclass(frozen, module = "crispr_cas_finder", skip_from_py_object)]
#[derive(Clone)]
pub struct Spacer {
    /// 1-based start position on the sequence.
    #[pyo3(get)]
    pub start: usize,
    /// 1-based end position on the sequence (inclusive).
    #[pyo3(get)]
    pub end: usize,
    /// DNA sequence of the spacer, in the case the input has it in.
    #[pyo3(get)]
    pub sequence: String,
}

#[pymethods]
impl Spacer {
    fn __repr__(&self) -> String {
        format!(
            "Spacer(start={}, end={}, sequence={:?})",
            self.start, self.end, self.sequence
        )
    }

    fn __len__(&self) -> usize {
        self.sequence.len()
    }
}

/// A detected CRISPR array with its direct repeats, spacers, and evidence level.
///
/// Attributes
/// ----------
/// id : str
///     Stable identifier, ``<seq_id>_CRISPR_<n>``.
/// seq_id : str
///     Identifier of the source sequence (FASTA header without the ``>``)
/// start : int
///     1-based start position of the first direct repeat.
/// end : int
///     1-based end position of the last direct repeat (inclusive).
/// consensus_repeat : str
///     Consensus direct repeat sequence for this array, in uppercase.
/// repeats : list[Repeat]
///     All direct repeats in array order.
/// spacers : list[Spacer]
///     All spacers in array order (one fewer than repeats).
/// evidence_level : int
///     Confidence level (1–4). Higher means more confident.
/// statistics : dict
///     Conservation scores and spacer length statistics behind the level.
/// orientation : str
///     Strand orientation: ``"+"`` (forward), ``"-"`` (reverse), or ``"."``
///     (unknown).
/// repeat_id : str
///     Canonical repeat identifier from the CRISPRdb database (e.g. ``"R10"``),
///     or ``"Unknown"`` when no match was found.
/// crispr_direction : str
///     Direction from the CRISPRDirection database (``"+"``, ``"-"``, or
///     ``"ND"``).
#[pyclass(frozen, module = "crispr_cas_finder", skip_from_py_object)]
#[derive(Clone)]
pub struct CrisprArray {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub seq_id: String,
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
    #[pyo3(get)]
    pub consensus_repeat: String,
    #[pyo3(get)]
    pub repeats: Vec<Repeat>,
    #[pyo3(get)]
    pub spacers: Vec<Spacer>,
    #[pyo3(get)]
    pub evidence_level: usize,
    #[pyo3(get)]
    pub orientation: String,
    #[pyo3(get)]
    pub repeat_id: String,
    #[pyo3(get)]
    pub crispr_direction: String,
    statistics: crispr_cas_finder_core::ArrayStatistics,
}

#[pymethods]
impl CrisprArray {
    fn __repr__(&self) -> String {
        format!(
            "CrisprArray(seq_id={:?}, start={}, end={}, repeats={}, spacers={}, evidence_level={}, orientation={:?})",
            self.seq_id,
            self.start,
            self.end,
            self.repeats.len(),
            self.spacers.len(),
            self.evidence_level,
            self.orientation,
        )
    }

    /// Conservation scores and spacer length statistics as a dictionary.
    #[getter]
    fn statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let statistics = &self.statistics;
        let dict = PyDict::new(py);
        dict.set_item("repeat_conservation", statistics.repeat_conservation)?;
        dict.set_item("repeat_identity", statistics.repeat_identity)?;
        dict.set_item("spacer_conservation", statistics.spacer_conservation)?;
        dict.set_item("spacer_identity", statistics.spacer_identity)?;
        dict.set_item(
            "spacer_average_identity",
            statistics.spacer_average_identity,
        )?;
        dict.set_item("mean_spacer_length", statistics.mean_spacer_length)?;
        dict.set_item("spacer_length_std_dev", statistics.spacer_length_std_dev)?;
        dict.set_item("exact_repeat_count", statistics.exact_repeat_count)?;
        dict.set_item("has_duplicated_spacers", statistics.has_duplicated_spacers)?;
        Ok(dict)
    }

    /// Length of the array in base pairs (end − start + 1).
    fn __len__(&self) -> usize {
        self.end.saturating_sub(self.start) + 1
    }

    /// Number of spacers in this array.
    #[getter]
    fn spacer_count(&self) -> usize {
        self.spacers.len()
    }

    /// Number of direct repeats in this array.
    #[getter]
    fn repeat_count(&self) -> usize {
        self.repeats.len()
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn repeat_from_core(r: &CoreRepeat) -> Repeat {
    Repeat {
        start: r.start,
        end: r.end,
        sequence: r.sequence.clone(),
    }
}

fn spacer_from_core(s: &CoreSpacer) -> Spacer {
    Spacer {
        start: s.start,
        end: s.end,
        sequence: s.sequence.clone(),
    }
}

fn crispr_array_from_core(a: &CoreCrisprArray) -> CrisprArray {
    let orientation = match a.orientation {
        Orientation::Forward => "+",
        Orientation::Reverse => "-",
        Orientation::Unknown => ".",
    }
    .to_string();
    CrisprArray {
        id: a.id.clone(),
        statistics: a.statistics.clone(),
        seq_id: a.seq_id.clone(),
        start: a.start,
        end: a.end,
        consensus_repeat: a.consensus_repeat.clone(),
        repeats: a.repeats.iter().map(repeat_from_core).collect(),
        spacers: a.spacers.iter().map(spacer_from_core).collect(),
        evidence_level: a.evidence_level,
        orientation,
        repeat_id: a.repeat_id.clone(),
        crispr_direction: a.crispr_direction.clone(),
    }
}

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Find CRISPR arrays in a FASTA-formatted string.
///
/// Parameters
/// ----------
/// fasta_content : str
///     Multi-sequence FASTA text to scan for CRISPR arrays.
/// min_repeat_length : int, optional
///     Minimum direct repeat length in bp. Default: 23.
/// max_repeat_length : int, optional
///     Maximum direct repeat length in bp. Default: 55.
/// min_spacer_length : int, optional
///     Shortest distance in bp between the two copies of a repeat that can
///     seed an array; the spacers themselves may be as short as 0.6 times the
///     repeat length. Default: 25.
/// max_spacer_length : int, optional
///     Longest distance in bp between the two copies of a repeat that can seed
///     an array; spacers up to 2.5 times the repeat length are kept.
///     Default: 60.
/// no_mismatch : bool, optional
///     If ``True``, require repeats to be identical (no mismatches allowed).
///     Default: ``False``.
/// min_spacer_count : int, optional
///     Minimum number of spacers an array must contain. Default: 1.
/// min_evidence_level : int, optional
///     Discard arrays with an evidence level below this threshold (1–4).
///     Default: 1.
///
/// Returns
/// -------
/// list[CrisprArray]
///     All CRISPR arrays found, in sequence / position order.
///
/// Raises
/// ------
/// ValueError
///     If the FASTA content is malformed or detection fails.
///
/// Examples
/// --------
/// >>> arrays = find_crispr_arrays(open("genome.fasta").read())
/// >>> for array in arrays:
/// ...     print(array.seq_id, array.start, array.end, array.evidence_level)
#[pyfunction]
#[pyo3(signature = (
    fasta_content,
    *,
    min_repeat_length = 23,
    max_repeat_length = 55,
    min_spacer_length = 25,
    max_spacer_length = 60,
    no_mismatch = false,
    min_spacer_count = 1,
    min_evidence_level = 1,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "the Python API exposes detection settings as keyword arguments"
)]
fn find_crispr_arrays(
    fasta_content: &str,
    min_repeat_length: usize,
    max_repeat_length: usize,
    min_spacer_length: usize,
    max_spacer_length: usize,
    no_mismatch: bool,
    min_spacer_count: usize,
    min_evidence_level: usize,
) -> PyResult<Vec<CrisprArray>> {
    let params = DetectionParams {
        min_repeat_length,
        max_repeat_length,
        min_spacer_length,
        max_spacer_length,
        no_mismatch,
        min_spacer_count,
        min_evidence_level,
        ..DetectionParams::default()
    };

    let arrays = detect_crisprs_in_fasta_str(fasta_content, &params)
        .map_err(|e| PyValueError::new_err(format!("CRISPR detection failed: {e}")))?;

    Ok(arrays.iter().map(crispr_array_from_core).collect())
}

/// Find CRISPR arrays in a FASTA file.
///
/// This is a convenience wrapper around :func:`find_crispr_arrays` that reads
/// the file for you.
///
/// Parameters
/// ----------
/// path : str
///     Path to the input FASTA file.
/// min_repeat_length : int, optional
///     Minimum direct repeat length in bp. Default: 23.
/// max_repeat_length : int, optional
///     Maximum direct repeat length in bp. Default: 55.
/// min_spacer_length : int, optional
///     Shortest distance in bp between the two copies of a repeat that can
///     seed an array; the spacers themselves may be as short as 0.6 times the
///     repeat length. Default: 25.
/// max_spacer_length : int, optional
///     Longest distance in bp between the two copies of a repeat that can seed
///     an array; spacers up to 2.5 times the repeat length are kept.
///     Default: 60.
/// no_mismatch : bool, optional
///     If ``True``, require repeats to be identical. Default: ``False``.
/// min_spacer_count : int, optional
///     Minimum spacer count per array. Default: 1.
/// min_evidence_level : int, optional
///     Minimum evidence level (1–4). Default: 1.
///
/// Returns
/// -------
/// list[CrisprArray]
///     All detected CRISPR arrays.
///
/// Raises
/// ------
/// FileNotFoundError
///     If *path* does not exist.
/// IOError
///     If the file cannot be read.
/// ValueError
///     If the file is not valid FASTA or detection fails.
///
/// Examples
/// --------
/// >>> arrays = find_crispr_arrays_in_file("genome.fasta")
/// >>> print(f"Found {len(arrays)} CRISPR arrays")
#[pyfunction]
#[pyo3(signature = (
    path,
    *,
    min_repeat_length = 23,
    max_repeat_length = 55,
    min_spacer_length = 25,
    max_spacer_length = 60,
    no_mismatch = false,
    min_spacer_count = 1,
    min_evidence_level = 1,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "the Python API exposes detection settings as keyword arguments"
)]
fn find_crispr_arrays_in_file(
    path: &str,
    min_repeat_length: usize,
    max_repeat_length: usize,
    min_spacer_length: usize,
    max_spacer_length: usize,
    no_mismatch: bool,
    min_spacer_count: usize,
    min_evidence_level: usize,
) -> PyResult<Vec<CrisprArray>> {
    let params = DetectionParams {
        min_repeat_length,
        max_repeat_length,
        min_spacer_length,
        max_spacer_length,
        no_mismatch,
        min_spacer_count,
        min_evidence_level,
        ..DetectionParams::default()
    };
    let arrays = detect_crisprs_in_fasta_path(&PathBuf::from(path), &params).map_err(|error| {
        if let Some(io) = error.downcast_ref::<std::io::Error>() {
            if io.kind() == std::io::ErrorKind::NotFound {
                return PyFileNotFoundError::new_err(format!("{path:?}: {error}"));
            }
            // rust-bio reports malformed records as synthetic io::Errors (even
            // ErrorKind::Other). File-system failures retain an OS error code.
            if io.raw_os_error().is_some() {
                return PyIOError::new_err(format!("{path:?}: {error}"));
            }
        }
        PyValueError::new_err(format!("CRISPR detection failed: {error}"))
    })?;
    Ok(arrays.iter().map(crispr_array_from_core).collect())
}

// ---------------------------------------------------------------------------
// Full analysis
// ---------------------------------------------------------------------------

/// Run CRISPR detection and, optionally, Cas detection on FASTA text.
///
/// Parameters
/// ----------
/// fasta_content : str
///     Multi-sequence FASTA text.
/// cas : bool, optional
///     Also detect Cas systems. Requires ``cas_models_dir``. Default: ``False``.
/// cas_models_dir : str, optional
///     Directory of CasFinder model definitions (XML files), or a MacSyFinder
///     model package such as CasFinder 3.
/// cas_profiles_dir : str, optional
///     Directory of CasFinder HMM profiles. Not needed with a model package.
/// gff, faa : str, optional
///     GFF3 annotation and matching protein FASTA to use instead of
///     predicting genes. Must be given together.
/// metagenome : bool, optional
///     Predict genes in metagenomic mode for every sequence.
/// auto_metagenome_below : int, optional
///     Sequences shorter than this use metagenomic gene prediction; 0
///     disables the switch. Default: 100000.
/// linear : bool, optional
///     Treat sequences as linear instead of circular. Default: ``False``.
/// cas_only_with_crispr : bool, optional
///     Search Cas systems only on sequences carrying an array.
/// vicinity : int, optional
///     Report arrays within this many bp of a Cas system. Default: 600.
/// locus_distance : int, optional
///     Chain arrays and Cas systems closer than this many bp into loci.
/// workers : int, optional
///     HMM search threads; 0 uses all cores. Default: 0.
/// min_repeat_length, max_repeat_length, min_spacer_length,
/// max_spacer_length, no_mismatch, min_spacer_count, min_evidence_level
///     As for :func:`find_crispr_arrays`.
///
/// Returns
/// -------
/// dict
///     The report, with the same structure as the command-line tool's
///     ``report.json``: ``tool``, ``parameters``, ``sequences``, ``crisprs``,
///     ``cas_clusters``, ``associations``, ``loci`` and ``summary``.
///
/// Raises
/// ------
/// ValueError
///     If the input or an option is invalid, or the analysis fails.
#[pyfunction]
#[pyo3(signature = (
    fasta_content,
    *,
    cas = false,
    cas_models_dir = None,
    cas_profiles_dir = None,
    gff = None,
    faa = None,
    metagenome = false,
    auto_metagenome_below = 100_000,
    linear = false,
    cas_only_with_crispr = false,
    vicinity = 600,
    locus_distance = None,
    workers = 0,
    min_repeat_length = 23,
    max_repeat_length = 55,
    min_spacer_length = 25,
    max_spacer_length = 60,
    no_mismatch = false,
    min_spacer_count = 1,
    min_evidence_level = 1,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "the Python API exposes analysis settings as keyword arguments"
)]
fn analyze<'py>(
    py: Python<'py>,
    fasta_content: &str,
    cas: bool,
    cas_models_dir: Option<PathBuf>,
    cas_profiles_dir: Option<PathBuf>,
    gff: Option<PathBuf>,
    faa: Option<PathBuf>,
    metagenome: bool,
    auto_metagenome_below: usize,
    linear: bool,
    cas_only_with_crispr: bool,
    vicinity: usize,
    locus_distance: Option<usize>,
    workers: usize,
    min_repeat_length: usize,
    max_repeat_length: usize,
    min_spacer_length: usize,
    max_spacer_length: usize,
    no_mismatch: bool,
    min_spacer_count: usize,
    min_evidence_level: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let cas_config = if cas {
        let Some(cas_models_dir) = cas_models_dir else {
            return Err(PyValueError::new_err("cas=True requires cas_models_dir"));
        };
        // A MacSyFinder model package carries its own profiles.
        if cas_profiles_dir.is_none() && !cas_models_dir.join("definitions").is_dir() {
            return Err(PyValueError::new_err(
                "cas=True requires cas_profiles_dir unless cas_models_dir is a model package",
            ));
        }
        Some(CasFinderConfig {
            metagenome,
            auto_metagenome_below,
            annotation_gff: gff,
            proteome_faa: faa,
            workers,
            quiet: true,
            replicon_topology: if linear {
                RepliconTopology::Linear
            } else {
                RepliconTopology::Circular
            },
            cas_models_dir: Some(cas_models_dir),
            cas_profiles_dir,
            ..CasFinderConfig::default()
        })
    } else {
        None
    };
    let options = AnalysisOptions {
        detection: DetectionParams {
            min_repeat_length,
            max_repeat_length,
            min_spacer_length,
            max_spacer_length,
            no_mismatch,
            min_spacer_count,
            min_evidence_level,
            ..DetectionParams::default()
        },
        cas: cas_config,
        cas_only_with_crispr,
        vicinity,
        locus_distance,
        ..AnalysisOptions::default()
    };

    let sequences = parse_fasta(fasta_content.as_bytes())
        .map_err(|e| PyValueError::new_err(format!("Invalid FASTA: {e}")))?;
    // Release the GIL: the analysis is pure Rust and can run for minutes.
    let report_json = py
        .detach(|| {
            crispr_cas_finder_core::analyze(&sequences, &options)
                .and_then(|analysis| Ok(serde_json::to_string(&analysis.report)?))
        })
        .map_err(|e| PyValueError::new_err(format!("Analysis failed: {e:#}")))?;
    py.import("json")?.call_method1("loads", (report_json,))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn crispr_cas_finder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Repeat>()?;
    m.add_class::<Spacer>()?;
    m.add_class::<CrisprArray>()?;
    m.add_function(wrap_pyfunction!(find_crispr_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(find_crispr_arrays_in_file, m)?)?;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
