pub use search::CapacityHints;
pub use search::FilterPolicy;
pub mod domain;
pub mod domaindef;
pub mod search;

// Re-export key types at the pipeline module level
pub use search::FilterReason;
pub use search::PipelineStats;
pub use search::SearchMetrics;
pub use search::SearchOutcome;
pub use search::SearchPlan;
pub use search::SearchPlanBuilder;
pub use search::SearchQuery;
pub use search::SearchReport;
pub use search::SearchTrace;
pub use search::SearchWorker;
pub use search::Thresholds;
