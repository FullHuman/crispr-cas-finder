use anyhow::{Context, Result};

include!(concat!(env!("OUT_DIR"), "/cas_data.rs"));

/// Keep this directory alive for the entire Cas run. Each invocation has its own
/// private temporary directory, so concurrent analyses never share mutable data.
pub fn extract() -> Result<tempfile::TempDir> {
    let directory = tempfile::Builder::new()
        .prefix("crispr-cas-models-")
        .tempdir()
        .context("Creating temporary directory for embedded Cas models")?;
    for (name, contents) in FILES {
        let path = directory.path().join(name);
        std::fs::create_dir_all(path.parent().unwrap())?;
        std::fs::write(&path, contents)
            .with_context(|| format!("Extracting embedded Cas data: {name}"))?;
    }
    Ok(directory)
}
