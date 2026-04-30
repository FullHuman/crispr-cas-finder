use crispr_cas_finder_core::{DetectionParams, detect_crisprs_in_fasta_str};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (fasta_content, min_repeat_length=23, max_repeat_length=55, min_spacer_length=25, max_spacer_length=60, no_mismatch=false, min_spacer_count=1, min_evidence_level=1))]
fn find_repeats(
    fasta_content: &str,
    min_repeat_length: usize,
    max_repeat_length: usize,
    min_spacer_length: usize,
    max_spacer_length: usize,
    no_mismatch: bool,
    min_spacer_count: usize,
    min_evidence_level: usize,
) -> PyResult<String> {
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
        .map_err(|e| PyValueError::new_err(format!("Analysis failed: {e}")))?;

    serde_json::to_string(&arrays)
        .map_err(|e| PyValueError::new_err(format!("Serialization failed: {e}")))
}

#[pymodule]
fn crispr_cas_finder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_repeats, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
