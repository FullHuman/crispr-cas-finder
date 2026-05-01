use crispr_cas_finder_core::{
    detect_crisprs_in_fasta_str, CrisprArray as CoreCrisprArray, DetectionParams, Orientation,
    Repeat as CoreRepeat, Spacer as CoreSpacer,
};
use pyo3::exceptions::{PyFileNotFoundError, PyIOError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Python-exposed data classes
// ---------------------------------------------------------------------------

/// A single direct repeat within a CRISPR array.
#[pyclass(frozen, module = "crispr_cas_finder")]
#[derive(Clone)]
pub struct Repeat {
    /// 1-based start position on the sequence.
    #[pyo3(get)]
    pub start: usize,
    /// 1-based end position on the sequence (inclusive).
    #[pyo3(get)]
    pub end: usize,
    /// DNA sequence of the direct repeat.
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
#[pyclass(frozen, module = "crispr_cas_finder")]
#[derive(Clone)]
pub struct Spacer {
    /// 1-based start position on the sequence.
    #[pyo3(get)]
    pub start: usize,
    /// 1-based end position on the sequence (inclusive).
    #[pyo3(get)]
    pub end: usize,
    /// DNA sequence of the spacer.
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
/// seq_id : str
///     Identifier of the source sequence (FASTA header without the ``>``)
/// start : int
///     1-based start position of the first direct repeat.
/// end : int
///     1-based end position of the last direct repeat (inclusive).
/// consensus_repeat : str
///     Consensus direct repeat sequence for this array.
/// repeats : list[Repeat]
///     All direct repeats in array order.
/// spacers : list[Spacer]
///     All spacers in array order (one fewer than repeats).
/// evidence_level : int
///     Confidence level (1–4). Higher means more confident.
/// orientation : str
///     Strand orientation: ``"+"`` (forward), ``"-"`` (reverse), or ``"."``
///     (unknown).
/// repeat_id : str
///     Canonical repeat identifier from the CRISPRdb database (e.g. ``"R10"``),
///     or ``"Unknown"`` when no match was found.
/// crispr_direction : str
///     Direction from the CRISPRDirection database (``"+"``, ``"-"``, or
///     ``"ND"``).
#[pyclass(frozen, module = "crispr_cas_finder")]
#[derive(Clone)]
pub struct CrisprArray {
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
///     Minimum spacer length in bp. Default: 25.
/// max_spacer_length : int, optional
///     Maximum spacer length in bp. Default: 60.
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
///     Minimum spacer length in bp. Default: 25.
/// max_spacer_length : int, optional
///     Maximum spacer length in bp. Default: 60.
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
    let file_path = PathBuf::from(path);
    if !file_path.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "No such file or directory: {path:?}"
        )));
    }
    let content = std::fs::read_to_string(&file_path)
        .map_err(|e| PyIOError::new_err(format!("Failed to read {path:?}: {e}")))?;

    find_crispr_arrays(
        &content,
        min_repeat_length,
        max_repeat_length,
        min_spacer_length,
        max_spacer_length,
        no_mismatch,
        min_spacer_count,
        min_evidence_level,
    )
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
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
