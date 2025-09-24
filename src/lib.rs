// Basic modules - will be implemented in subsequent tasks
pub mod errors;
pub mod data_loader;
pub mod compression;
pub mod decision;
pub mod formatter;

use crate::compression::CompressionTester;

// Re-export error types and data loading functionality
pub use errors::BenchmarkError;
pub use data_loader::{LoadedData, DataSource, load_data};

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
    _output_path: PathBuf,
    verbose: bool,
) -> Result<BenchmarkResult, BenchmarkError> {
    let timeout = Duration::from_secs(timeout_seconds);
    
    if verbose {
        println!("=== Compression Benchmark Starting ===");
        println!("Timeout: {:?}", timeout);
    }
    
    // Step 1: Load data (Task 2 implementation)
    if verbose {
        println!("Loading data...");
    }
    
    let loaded_data = load_data(data_path.as_deref())?;
    
    if verbose {
        match &loaded_data.source {
            DataSource::Directory(path) => {
                println!("Loaded {} files from directory: {}", 
                         loaded_data.file_count, path.display());
            }
            DataSource::Generated => {
                println!("Generated test data with {} simulated files", 
                         loaded_data.file_count);
            }
        }
        println!("Total data size: {} bytes ({:.1}MB)", 
                 loaded_data.total_size, 
                 loaded_data.total_size as f64 / (1024.0 * 1024.0));
    }
    
// --- Compression tests ---
    if verbose { println!("Running gzip test..."); }
    let gzip_tester = crate::compression::GzipTester::default();
    let gzip_result = gzip_tester.test(&loaded_data.data)
        .map_err(|e| BenchmarkError::AnalysisFailed { details: e.to_string() })?;

    if verbose { println!("Running Reducto Mode 3 test (mock)..."); }
    let reducto_tester = crate::compression::ReductoMockTester::default();
    let reducto_result = reducto_tester.test(&loaded_data.data)
        .map_err(|e| BenchmarkError::AnalysisFailed { details: e.to_string() })?;

    // --- Decision engine ---
    let rec = crate::decision::compare_results(&reducto_result, &gzip_result);
    let is_recommended = rec.decision == crate::decision::Decision::Recommended;

    // --- Output formatting ---
    let report = crate::formatter::format_console(&reducto_result, &gzip_result, &rec);
    println!("{}", report);
    if let Err(e) = crate::formatter::save_report(&_output_path, &report) {
        eprintln!("Warning: failed to save report: {}", e);
    }

    let result = BenchmarkResult {
        recommendation: if is_recommended {
            "RECOMMENDED: Use Reducto Mode 3".into()
        } else {
            "NOT RECOMMENDED: Stick with gzip".into()
        },
        is_recommended,
        details: rec.reason.clone(),
    };

    if verbose {
        println!("=== Benchmark Complete ===");
    }

    Ok(result)
}

// Prelude module will be added in subsequent tasks when other modules are implemented