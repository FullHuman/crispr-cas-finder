use std::fmt;

/// Central error type for the HMMER library.
///
/// Replaces the legacy C-style `EslStatus` integer return codes with a proper
/// Rust error enum. All library functions return `Result<T, HmmerError>`.
#[derive(Debug, thiserror::Error)]
pub enum HmmerError {
    #[error("file or key not found: {0}")]
    NotFound(String),

    #[error("invalid format: {0}")]
    Format(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("incompatible parameters: {0}")]
    Incompatible(String),

    #[error("data corruption: {0}")]
    Corrupt(String),

    #[error("value out of range: {0}")]
    Range(String),

    #[error("syntax error: {0}")]
    Syntax(String),

    #[error("traceback failed at {state}: {reason}")]
    TracebackFailed { state: String, reason: String },

    #[error("end of file")]
    Eof,

    #[error("no result")]
    NoResult,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Internal(String),
}

/// Convenience type alias used throughout the library.
pub type Result<T> = std::result::Result<T, HmmerError>;

/// Format and create an `HmmerError::Internal` for "can't happen" situations.
pub fn internal_err(args: fmt::Arguments) -> HmmerError {
    HmmerError::Internal(fmt::format(args))
}

#[macro_export]
macro_rules! internal_err {
    ($($arg:tt)*) => {
        $crate::errors::internal_err(format_args!($($arg)*))
    };
}
