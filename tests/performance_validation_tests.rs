//! Performance Validation Tests
//!
//! This module validates all enterprise performance claims with automated measurement
//! and ensures that the optimizations work correctly together.

use reducto_mode_3::prelude::*;
use reducto_mode_3::{SimdHashCalculator, StandardChunker, StandardRollingHasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Performance validation test suite
#[cfg(test)]
mod performance_validation {
    use super::*;

    /// Test that rolling hash meets O(1) performance contract
    #[tokio::test]
    async fn test_rolling_hash_o1_performance_contract() {
        let data_sizes = [8192, 32768, 131072, 524288]; // 8KB to 512KB
        let window_size = 4096;
        let base = HASH_BASE;
        
        let mut times_per_byte = Vec::new();
        
        for &data_size in &data_sizes {
            let data = vec![0xAA; data_size];
            let mut hasher = RollingHasher::new(base, window_size);
            
            // Initialize
            hasher.init(&data[0..window_size]).unwrap();
            
            // Measure rolling hash updates
            let updates = data_size - window_size;
            let start = Instant::now();
            
            for i in window_size..data_size {
                let exiting_byte = data[i - window_size];
                let entering_byte = data[i];
                let _hash = hasher.roll(exiting_byte, entering_byte).unwrap();
            }
            
            let elapsed = start.elapsed();
            let time_per_byte = elapsed.as_nanos() as f64 / updates as f64;
            times_per_byte.push(time_per_byte);
            
            // Each update should be very fast (O(1) requirement)
            assert!(time_per_byte < 1000.0, 
                "Rolling hash update took {:.1} ns/byte, expected < 1000 ns (O(1) requirement)", 
                time_per_byte);
        }
        
        // Verify that time per byte doesn't grow significantly with data size (O(1) property)
        let first_time = times_per_byte[0];
        let last_time = times_per_byte[times_per_byte.len() - 1];
        let growth_factor = last_time / first_time;
        
        assert!(growth_factor < 2.0, 
            "Rolling hash time grew by {:.1}x with data size, violating O(1) contract", 
            growth_factor);
        
        println!("✅ Rolling hash O(1) performance contract validated");
        println!("   Time per byte: {:.1} - {:.1} ns", times_per_byte[0], times_per_byte[times_per_byte.len() - 1]);
    }

    /// Test CDC chunking throughput meets enterprise requirements (50+ MB/s)
    #[tokio::test]
    async fn test_cdc_chunking_throughput_contract() {
        let data_sizes = [1048576, 4194304, 16777216]; // 1MB, 4MB, 16MB
        let config = ChunkConfig::default();
        
        for &data_size in &data_sizes {
            let data = generate_test_data(data_size, 0xBB);
            let mut chunker = FastCDCChunker::new(config).unwrap();
            
            let start = Instant::now();
            let chunks = chunker.chunk_data(&data).unwrap();
            let _final_chunk = chunker.finalize().unwrap();
            let elapsed = start.elapsed();
            
            let throughput_mbps = (data_size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
            
            // Enterprise requirement: 50+ MB/s throughput
            assert!(throughput_mbps >= 50.0, 
                "CDC chunking throughput {:.1} MB/s below enterprise requirement of 50 MB/s for {} MB data", 
                throughput_mbps, data_size / 1048576);
            
            // Verify chunking produced reasonable results
            assert!(!chunks.is_empty(), "Chunking should produce chunks");
            let total_chunk_size: usize = chunks.iter().map(|c| c.size()).sum();
            assert!(total_chunk_size <= data_size, "Total chunk size should not exceed input");
            
            println!("✅ CDC chunking throughput: {:.1} MB/s for {} MB data", 
                     throughput_mbps, data_size / 1048576);
        }
    }

    /// Test compression pipeline meets enterprise performance requirements
    #[tokio::test]
    async fn test_compression_pipeline_performance_contract() {
        let temp_dir = TempDir::new().unwrap();
        let corpus_path = temp_dir.path().join("perf_corpus.bin");
        
        // Create test corpus
        let corpus_size = 4 * 1024 * 1024; // 4MB
        let corpus_data = generate_test_data(corpus_size, 0xCC);
        std::fs::write(&corpus_path, &corpus_data).unwrap();
        
        // Build corpus
        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );
        let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
        let shared_corpus = Arc::new(corpus_manager);
        
        // Test compression performance with different data sizes
        let test_sizes = [1048576, 4194304, 8388608]; // 1MB, 4MB, 8MB
        
        for &test_size in &test_sizes {
            let test_data = generate_test_data(test_size, 0xDD);
            let mut compressor = Compressor::new(Arc::clone(&shared_corpus));
            
            let start = Instant::now();
            let instructions = compressor.compress(&test_data).await.unwrap();
            let elapsed = start.elapsed();
            
            let throughput_mbps = (test_size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
            
            // Enterprise requirement: 30+ MB/s for compression pipeline
            assert!(throughput_mbps >= 30.0, 
                "Compression pipeline throughput {:.1} MB/s below enterprise requirement of 30 MB/s for {} MB data", 
                throughput_mbps, test_size / 1048576);
            
            // Verify compression produced valid results
            assert!(!instructions.is_empty(), "Compression should produce instructions");
            let total_output_size: usize = instructions.iter().map(|i| i.output_size()).sum();
            assert_eq!(total_output_size, test_size, "Output size should match input size");
            
            println!("✅ Compression pipeline throughput: {:.1} MB/s for {} MB data", 
                     throughput_mbps, test_size / 1048576);
        }
    }

    /// Test memory usage stays within enterprise bounds
    #[tokio::test]
    async fn test_memory_usage_bounds_contract() {
        let temp_dir = TempDir::new().unwrap();
        let corpus_path = temp_dir.path().join("memory_corpus.bin");
        
        // Create large corpus to test memory efficiency
        let corpus_size = 16 * 1024 * 1024; // 16MB
        let corpus_data = generate_test_data(corpus_size, 0xEE);
        std::fs::write(&corpus_path, &corpus_data).unwrap();
        
        // Measure memory before corpus building
        let initial_memory = get_approximate_memory_usage();
        
        // Build corpus
        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );
        let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
        
        let post_corpus_memory = get_approximate_memory_usage();
        let corpus_memory_usage = post_corpus_memory.saturating_sub(initial_memory);
        
        // Memory usage should be reasonable (less than 3x corpus size)
        let max_acceptable_memory = corpus_size * 3;
        assert!(corpus_memory_usage <= max_acceptable_memory, 
            "Corpus memory usage {} bytes exceeds acceptable limit {} bytes (3x corpus size)", 
            corpus_memory_usage, max_acceptable_memory);
        
        // Test compression memory usage
        let shared_corpus = Arc::new(corpus_manager);
        let test_data = generate_test_data(8 * 1024 * 1024, 0xFF); // 8MB test data
        let mut compressor = Compressor::new(shared_corpus);
        
        let pre_compression_memory = get_approximate_memory_usage();
        let _instructions = compressor.compress(&test_data).await.unwrap();
        let post_compression_memory = get_approximate_memory_usage();
        
        let compression_memory_usage = post_compression_memory.saturating_sub(pre_compression_memory);
        let max_compression_memory = test_data.len() * 2; // Should not use more than 2x input size
        
        assert!(compression_memory_usage <= max_compression_memory,
            "Compression memory usage {} bytes exceeds limit {} bytes (2x input size)",
            compression_memory_usage, max_compression_memory);
        
        println!("✅ Memory usage within bounds:");
        println!("   Corpus memory: {} MB (limit: {} MB)", 
                 corpus_memory_usage / 1048576, max_acceptable_memory / 1048576);
        println!("   Compression memory: {} MB (limit: {} MB)", 
                 compression_memory_usage / 1048576, max_compression_memory / 1048576);
    }

    /// Test concurrent performance scaling
    #[tokio::test]
    async fn test_concurrent_performance_scaling_contract() {
        let temp_dir = TempDir::new().unwrap();
        let corpus_path = temp_dir.path().join("concurrent_corpus.bin");
        
        // Create shared corpus
        let corpus_size = 8 * 1024 * 1024; // 8MB
        let corpus_data = generate_test_data(corpus_size, 0x77);
        std::fs::write(&corpus_path, &corpus_data).unwrap();
        
        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );
        let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
        let shared_corpus = Arc::new(corpus_manager);
        
        // Test different concurrency levels
        let concurrency_levels = [1, 2, 4, 8];
        let data_size = 2 * 1024 * 1024; // 2MB per task
        
        for &concurrency in &concurrency_levels {
            let mut join_set = tokio::task::JoinSet::new();
            let start_time = Instant::now();
            
            for i in 0..concurrency {
                let corpus = Arc::clone(&shared_corpus);
                let test_data = generate_test_data(data_size, 0x88 + i as u8);
                
                join_set.spawn(async move {
                    let mut compressor = Compressor::new(corpus);
                    let task_start = Instant::now();
                    let result = compressor.compress(&test_data).await.unwrap();
                    let task_elapsed = task_start.elapsed();
                    (i, result.len(), task_elapsed)
                });
            }
            
            // Wait for all tasks to complete
            let mut task_results = Vec::new();
            while let Some(result) = join_set.join_next().await {
                task_results.push(result.unwrap());
            }
            
            let total_elapsed = start_time.elapsed();
            let total_data_processed = data_size * concurrency;
            let concurrent_throughput = (total_data_processed as f64) / (1024.0 * 1024.0) / total_elapsed.as_secs_f64();
            
            // Concurrent efficiency should be reasonable (at least 60% of linear scaling)
            let expected_min_throughput = if concurrency == 1 {
                20.0 // Base expectation for single-threaded
            } else {
                20.0 * concurrency as f64 * 0.6 // 60% efficiency
            };
            
            assert!(concurrent_throughput >= expected_min_throughput,
                "Concurrent throughput {:.1} MB/s below expected minimum {:.1} MB/s for {} threads",
                concurrent_throughput, expected_min_throughput, concurrency);
            
            println!("✅ Concurrent performance ({}threads): {:.1} MB/s", 
                     concurrency, concurrent_throughput);
        }
    }

    /// Test SIMD optimizations provide performance benefits
    #[tokio::test]
    async fn test_simd_optimization_benefits() {
        let calculator = SimdHashCalculator::new(HASH_BASE);
        let data_size = 65536; // 64KB
        let window_size = 4096;
        
        // Create test windows
        let test_data = generate_test_data(data_size, 0xAA);
        let windows: Vec<&[u8]> = (0..8).map(|i| {
            let start = i * window_size;
            &test_data[start..start + window_size]
        }).collect();
        
        // Benchmark scalar implementation
        let scalar_start = Instant::now();
        let _scalar_results = calculator.batch_rolling_hash_scalar(&windows, window_size).unwrap();
        let scalar_time = scalar_start.elapsed();
        
        // Benchmark SIMD implementation
        let simd_start = Instant::now();
        let _simd_results = calculator.batch_rolling_hash(&windows, window_size).unwrap();
        let simd_time = simd_start.elapsed();
        
        // SIMD should be at least as fast as scalar (may not always be faster due to overhead)
        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        
        // Allow for some variance - SIMD might not always be faster for small datasets
        assert!(speedup >= 0.8, 
            "SIMD implementation significantly slower than scalar: {:.2}x", speedup);
        
        println!("✅ SIMD optimization speedup: {:.2}x", speedup);
        
        // Test SIMD chunk comparison
        let chunk1 = &test_data[0..4096];
        let chunk2 = &test_data[0..4096];
        let chunk3 = &test_data[4096..8192];
        
        assert!(calculator.simd_compare_chunks(chunk1, chunk2), "Identical chunks should match");
        assert!(!calculator.simd_compare_chunks(chunk1, chunk3), "Different chunks should not match");
        
        println!("✅ SIMD chunk comparison working correctly");
    }

    /// Test const optimizations compile-time benefits
    #[tokio::test]
    async fn test_const_optimization_benefits() {
        // Test const chunker
        let mut const_chunker = StandardChunker::new();
        let test_data = generate_test_data(32768, 0xBB); // 32KB
        
        let start = Instant::now();
        let chunks = const_chunker.chunk_data(&test_data).unwrap();
        let _final_chunk = const_chunker.finalize().unwrap();
        let elapsed = start.elapsed();
        
        // Const optimized chunker should be efficient
        let throughput = (test_data.len() as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
        assert!(throughput >= 100.0, 
            "Const optimized chunker throughput {:.1} MB/s below expected 100 MB/s", throughput);
        
        // Verify chunking correctness
        assert!(!chunks.is_empty(), "Should produce chunks");
        let total_size: usize = chunks.iter().map(|c| c.size()).sum();
        assert!(total_size <= test_data.len(), "Chunk total should not exceed input");
        
        println!("✅ Const optimized chunker throughput: {:.1} MB/s", throughput);
        
        // Test const rolling hasher
        let mut const_hasher = StandardRollingHasher::new();
        let window_data = &test_data[0..8192];
        
        const_hasher.init(window_data).unwrap();
        let initial_hash = const_hasher.current_hash();
        
        let rolled_hash = const_hasher.roll(test_data[0], test_data[8192]);
        assert_ne!(initial_hash, rolled_hash, "Rolling should change hash");
        
        println!("✅ Const optimized rolling hasher working correctly");
    }

    /// Test large dataset handling within time bounds
    #[tokio::test]
    async fn test_large_dataset_time_bounds_contract() {
        let temp_dir = TempDir::new().unwrap();
        let corpus_path = temp_dir.path().join("large_corpus.bin");
        
        // Create large corpus
        let corpus_size = 32 * 1024 * 1024; // 32MB
        let corpus_data = generate_test_data(corpus_size, 0x99);
        std::fs::write(&corpus_path, &corpus_data).unwrap();
        
        // Build corpus with time limit
        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );
        
        let corpus_start = Instant::now();
        let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
        let corpus_build_time = corpus_start.elapsed();
        
        // Corpus building should complete within reasonable time (2 minutes for 32MB)
        assert!(corpus_build_time <= Duration::from_secs(120),
            "Corpus building took {:.1}s, expected <= 120s for 32MB corpus", 
            corpus_build_time.as_secs_f64());
        
        // Test compression of large dataset
        let large_test_data = generate_test_data(16 * 1024 * 1024, 0xAA); // 16MB
        let mut compressor = Compressor::new(Arc::new(corpus_manager));
        
        let compression_start = Instant::now();
        let _instructions = compressor.compress(&large_test_data).await.unwrap();
        let compression_time = compression_start.elapsed();
        
        // Large dataset compression should complete within time bounds (1 minute for 16MB)
        assert!(compression_time <= Duration::from_secs(60),
            "Large dataset compression took {:.1}s, expected <= 60s for 16MB data",
            compression_time.as_secs_f64());
        
        let throughput = (large_test_data.len() as f64) / (1024.0 * 1024.0) / compression_time.as_secs_f64();
        
        // Should maintain reasonable throughput even for large datasets
        assert!(throughput >= 15.0,
            "Large dataset throughput {:.1} MB/s below minimum 15 MB/s", throughput);
        
        println!("✅ Large dataset handling:");
        println!("   Corpus build time: {:.1}s for 32MB", corpus_build_time.as_secs_f64());
        println!("   Compression time: {:.1}s for 16MB", compression_time.as_secs_f64());
        println!("   Throughput: {:.1} MB/s", throughput);
    }
}

/// Generate test data with specific patterns
fn generate_test_data(size: usize, pattern: u8) -> Vec<u8> {
    let mut data = vec![0u8; size];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = match i % 1024 {
            0..=511 => pattern,
            512..=767 => (i % 256) as u8,
            _ => 0x00,
        };
    }
    data
}

/// Get approximate memory usage (simplified for testing)
fn get_approximate_memory_usage() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    // Fallback: return a reasonable estimate
    16 * 1024 * 1024 // 16MB baseline
}