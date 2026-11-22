/// Central error type for the HMMER library.
///
/// Replaces the legacy C-style `EslStatus` integer return codes with a proper
/// Rust error enum.
#[derive(Debug, thiserror::Error)]
pub enum HmmerError {
    #[error("file or key not found: {0}")]
    NotFound(String),

    #[error("invalid format: {0}")]
    Format(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("value out of range: {0}")]
    Range(String),

    #[error("traceback failed at {state}: {reason}")]
    TracebackFailed { state: String, reason: String },

    #[error("end of file")]
    Eof,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Internal(String),
}
