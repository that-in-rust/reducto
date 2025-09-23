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

use std::path::PathBuf;
use std::time::Duration;

/// Main benchmark result containing the final recommendation
#[derive(Debug)]
pub struct BenchmarkResult {
    pub recommendation: String,
    pub is_recommended: bool,
    pub details: String,
}

/// Main benchmark function that orchestrates the entire process
/// 
/// # Arguments
/// * `data_path` - Optional path to user data directory (generates test data if None)
/// * `timeout_seconds` - Maximum time to run benchmark
/// * `output_path` - Path to save detailed results
/// * `verbose` - Enable verbose output
/// 
/// # Returns
/// * `BenchmarkResult` with final recommendation and details
pub async fn run_benchmark(
    data_path: Option<PathBuf>,
    timeout_seconds: u64,
    output_path: PathBuf,
    verbose: bool,
) -> Result<BenchmarkResult, BenchmarkError> {
    let timeout = Duration::from_secs(timeout_seconds);
    
    if verbose {
        println!("=== Compression Benchmark Starting ===");
        println!("Timeout: {:?}", timeout);
    }
    
    // TODO: Implement the actual benchmark logic in subsequent tasks
    // For now, return a placeholder result
    
    let recommendation = if data_path.is_some() {
        "PLACEHOLDER: Benchmark not yet implemented - will analyze your data"
    } else {
        "PLACEHOLDER: Benchmark not yet implemented - will generate test data"
    };
    
    let result = BenchmarkResult {
        recommendation: recommendation.to_string(),
        is_recommended: false,
        details: "Benchmark implementation in progress".to_string(),
    };
    
    if verbose {
        println!("=== Benchmark Complete ===");
    }
    
    Ok(result)
}

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