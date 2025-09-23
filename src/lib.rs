pub mod data;
pub mod compression;
pub mod corpus;
pub mod analysis;
pub mod reporting;
pub mod errors;

// Re-export commonly used types
pub use data::{DataFile, DataType, RedundancyLevel};
pub use compression::{CompressionFormat, CompressionResult, BenchmarkResults};
pub use corpus::{ReferenceCorpus, CorpusBuilder, CorpusMetadata};
pub use analysis::{PerformanceAnalyzer, AnalysisResults, Recommendation};
pub use errors::{BenchmarkError, BenchmarkResult};

/// Prelude module for common imports
pub mod prelude {
    pub use crate::{
        data::{DataFile, DataType, RedundancyLevel, DataSource},
        compression::{CompressionFormat, CompressionResult, BenchmarkResults},
        corpus::{ReferenceCorpus, CorpusBuilder, CorpusMetadata},
        analysis::{PerformanceAnalyzer, AnalysisResults, Recommendation, OptimizationScenario},
        errors::{BenchmarkError, BenchmarkResult},
    };
    
    pub use async_trait::async_trait;
    pub use serde::{Serialize, Deserialize};
    pub use std::collections::HashMap;
    pub use std::time::{Duration, Instant};
    pub use tokio;
}