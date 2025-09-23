# Design Document

## Overview

**Dead-simple benchmark**: Run `cargo run --bin benchmark [path]` → get YES/NO answer in 60 seconds.

**The entire "architecture":**
1. Load some data (user's or generate 20MB test data)
2. Try Reducto Mode 3 vs gzip
3. Check if Reducto is better + not too slow
4. Print "RECOMMENDED" or "NOT RECOMMENDED"

That's it.

## Architecture

**Simple flow:**
```
main() → load_data() → compress_both() → compare() → print_result()
```

**Files needed:**
- `src/bin/benchmark.rs` - main CLI
- `src/lib.rs` - compress with Reducto + gzip, compare results
- `Cargo.toml` - dependencies

**Dependencies:**
- `clap` - CLI args
- `flate2` - gzip
- Whatever Reducto needs (probably already exists)

## Core Functions

**That's literally it:**

```rust
// Load data (user's directory or generate test data)
fn load_data(path: Option<&Path>) -> Vec<u8> { ... }

// Compress with both formats, time it
fn test_compression(data: &[u8]) -> (CompressResult, CompressResult) { ... }

// Compare results, decide winner
fn make_recommendation(reducto: CompressResult, gzip: CompressResult) -> String { ... }

struct CompressResult {
    ratio: f64,
    time_ms: u64,
    decompressed_ok: bool,
}
```

## Decision Logic

**Simple rules:**
- If Reducto compresses better than gzip AND isn't >10x slower → "RECOMMENDED"  
- Otherwise → "NOT RECOMMENDED"
- If decompression fails → "NOT RECOMMENDED - data corruption"

**Data limits:**
- Max 100MB of input data
- Max 60 seconds total time
- Generate 20MB test data if no path given

## Error Handling

**Keep it simple:**
- File not found → print error, exit
- Compression fails → print error, exit  
- Timeout → print "benchmark incomplete", exit
- Use `anyhow` for everything

## Testing Strategy

### Unit Testing Approach

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_redundancy_analysis_accuracy() {
        // Test with known redundant data patterns
        let data = create_redundant_test_data(50); // 50% redundancy
        let analyzer = DataAnalyzer::new();
        let analysis = analyzer.analyze_redundancy(&data);
        
        assert!((analysis.redundancy_percentage - 50.0).abs() < 5.0);
    }
    
    #[test]
    fn test_corpus_build_time_limit() {
        let builder = CorpusBuilder::new();
        let large_data = create_large_test_data(50_000_000); // 50MB
        
        let start = Instant::now();
        let result = builder.build_optimal_corpus(&large_data, &analysis);
        let elapsed = start.elapsed();
        
        assert!(elapsed <= Duration::from_secs(10));
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_60_second_time_limit() {
        let cli = BenchmarkCli::new(None, Duration::from_secs(60));
        
        let start = Instant::now();
        let result = cli.run().await;
        let elapsed = start.elapsed();
        
        assert!(elapsed <= Duration::from_secs(60));
        assert!(result.is_ok());
    }
}
```

### Integration Testing

```rust
#[cfg(test)]
mod integration_tests {
    #[tokio::test]
    async fn test_end_to_end_with_real_data() {
        // Test with actual source code directory
        let test_data_dir = create_test_source_code_directory();
        let cli = BenchmarkCli::new(Some(test_data_dir), Duration::from_secs(60));
        
        let result = cli.run().await.unwrap();
        
        assert!(result.recommendation.decision == Decision::Recommended || 
                result.recommendation.decision == Decision::NotRecommended);
        assert!(result.total_time <= Duration::from_secs(60));
    }
    
    #[tokio::test]
    async fn test_reliability_verification_catches_corruption() {
        // Test with intentionally corrupted decompression
        let mut mock_tester = MockCompressionTester::new();
        mock_tester.set_corruption_mode(true);
        
        let verifier = ReliabilityVerifier::new();
        let result = verifier.verify_integrity(&original_data, &corrupted_data);
        
        assert!(!result);
    }
}
```

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn compression_ratio_is_consistent(
        data in prop::collection::vec(any::<u8>(), 1000..10000)
    ) {
        let tester = GzipTester::new(6);
        let result1 = tester.compress(&data, None).unwrap();
        let result2 = tester.compress(&data, None).unwrap();
        
        // Compression should be deterministic
        prop_assert_eq!(result1.compressed_size, result2.compressed_size);
        prop_assert_eq!(result1.compression_ratio, result2.compression_ratio);
    }
    
    #[test]
    fn decompression_is_lossless(
        data in prop::collection::vec(any::<u8>(), 100..1000)
    ) {
        let tester = GzipTester::new(6);
        let compressed = tester.compress(&data, None).unwrap();
        let decompressed = tester.decompress(&compressed.compressed_data, None).unwrap();
        
        prop_assert_eq!(data, decompressed);
    }
}
```

## Performance Considerations

### Time Budget Allocation

The 60-second time limit is allocated as follows:

- **Data Loading/Generation**: 5 seconds (8%)
- **Data Analysis**: 10 seconds (17%)
- **Corpus Building**: 10 seconds (17%)
- **Compression Testing**: 25 seconds (42%)
- **Reliability Verification**: 8 seconds (13%)
- **Results Analysis & Output**: 2 seconds (3%)

### Memory Management

```rust
pub struct MemoryManager {
    max_memory_usage: usize, // 500MB limit
    current_usage: AtomicUsize,
}

impl MemoryManager {
    pub fn check_memory_limit(&self, additional: usize) -> Result<(), MemoryError>;
    pub fn track_allocation(&self, size: usize);
    pub fn track_deallocation(&self, size: usize);
}
```

**Key Optimizations:**
- Stream processing for large files to avoid loading everything into memory
- Aggressive cleanup of intermediate results
- Memory-mapped files for corpus access during compression
- Early exit strategies when time/memory limits are approached

### Scalability Constraints

- **Maximum data size**: 100MB (prevents excessive memory usage)
- **Maximum file count**: 10,000 files (prevents filesystem overhead)
- **Corpus size limit**: 30% of input data or 30MB, whichever is smaller
- **Compression block size**: 64KB (balances compression ratio vs speed)

## Security Considerations

### Input Validation

```rust
pub struct InputValidator;

impl InputValidator {
    pub fn validate_path(&self, path: &Path) -> Result<(), ValidationError>;
    pub fn validate_file_size(&self, size: usize) -> Result<(), ValidationError>;
    pub fn validate_file_type(&self, path: &Path) -> Result<(), ValidationError>;
}
```

**Security Measures:**
- Path traversal prevention (no `../` in paths)
- File size limits to prevent DoS attacks
- File type validation (only process safe file types)
- No execution of user-provided code or scripts
- Sandboxed temporary directory for intermediate files

### Resource Limits

- CPU time limits enforced via tokio timeouts
- Memory usage monitoring and limits
- Disk space limits for temporary files
- Network access completely disabled (no external dependencies)

## Deployment and Distribution

### Binary Distribution

```toml
# Cargo.toml
[package]
name = "compression-benchmark"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "benchmark"
path = "src/bin/benchmark.rs"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
clap = { version = "4.0", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"
blake3 = "1.5"
flate2 = "1.0"  # For gzip
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### Installation and Usage

```bash
# Installation
cargo install compression-benchmark

# Usage examples
benchmark                           # Use generated test data
benchmark /path/to/my/project      # Test specific directory
benchmark --help                   # Show all options
```

### Output Format

```
Compression Benchmark Results
============================

Data Analysis: 45% redundancy detected, proceeding with compression test
Corpus built: 1,247 blocks, 42% redundancy detected

Compression Results:
- Reducto Mode 3: 2.3:1 ratio, 1.2s compression time
- Gzip (level 6): 1.8:1 ratio, 0.3s compression time

Reliability: PASS - All data verified with BLAKE3

RECOMMENDED: Use Reducto Mode 3
- 28% smaller files than gzip
- 4x slower compression (acceptable for batch processing)
- Estimated savings: $45/month in storage costs

Next steps: Add Reducto to your project with: cargo add reducto

Results saved to: benchmark_results.txt
Total benchmark time: 47 seconds
```

This design provides a comprehensive foundation for implementing the compression benchmark suite while maintaining the simplicity and speed requirements outlined in the specifications.