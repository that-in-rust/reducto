//! Performance Property Tests for Reducto Mode 3
//!
//! This module implements comprehensive property-based testing for performance contracts:
//! - Rolling hash O(1) time complexity validation
//! - Compression throughput requirements
//! - Memory usage bounds verification
//! - Concurrent performance characteristics
//! - Large dataset handling performance

use proptest::prelude::*;
use reducto_mode_3::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::task::JoinSet;

// === Performance Test Strategies ===

/// Generate data sizes for performance testing
fn performance_data_size_strategy() -> impl Strategy<Value = usize> {
    prop_oneof![
        1024usize..=8192,      // Small data (1-8 KB)
        8192..=65536,          // Medium data (8-64 KB)
        65536..=1048576,       // Large data (64KB-1MB)
        1048576..=8388608,     // Very large data (1-8 MB)
    ]
}

/// Generate chunk configurations for performance testing
fn performance_chunk_config_strategy() -> impl Strategy<Value = ChunkConfig> {
    (4096usize..=32768).prop_map(|target_size| {
        ChunkConfig::new(target_size).unwrap_or_else(|_| ChunkConfig::default())
    })
}

/// Generate realistic performance test data
fn performance_test_data(size: usize, pattern: u8) -> Vec<u8> {
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Property: Rolling hash computation is O(1) per byte
    /// Hash update time should be constant regardless of data content or position
    #[test]
    fn prop_rolling_hash_constant_time(
        data_size in 8192usize..=65536,
        pattern in any::<u8>()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let data = performance_test_data(data_size, pattern);
            let window_size = 4096;
            let mut hasher = RollingHasher::new(HASH_BASE, window_size);

            // Initialize hasher
            hasher.init(&data[0..window_size]).unwrap();

            // Measure time for hash updates
            let num_updates = data_size - window_size;
            let start_time = Instant::now();

            for i in window_size..data_size {
                let exiting_byte = data[i - window_size];
                let entering_byte = data[i];
                let _hash = hasher.roll(exiting_byte, entering_byte).unwrap();
            }

            let elapsed = start_time.elapsed();
            let time_per_update_ns = elapsed.as_nanos() as f64 / num_updates as f64;

            // Each hash update should take less than 500 nanoseconds (very generous for O(1))
            prop_assert!(time_per_update_ns < 500.0,
                "Rolling hash update took {:.1} ns per update, expected < 500 ns (O(1) requirement)",
                time_per_update_ns);

            // Verify hash updates are working correctly
            let final_hash = hasher.current_weak_hash().unwrap();
            prop_assert_ne!(final_hash.get(), 0, "Hash should not be zero after processing");
        });
    }

    /// Property: CDC chunking throughput meets performance requirements
    /// Should process at least 50 MB/s for typical data patterns
    #[test]
    fn prop_cdc_chunking_throughput(
        data_size in 1048576usize..=4194304, // 1-4 MB
        config in performance_chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let data = performance_test_data(data_size, 0xAA);
            let mut chunker = FastCDCChunker::new(config).unwrap();

            let start_time = Instant::now();
            let chunks = chunker.chunk_data(&data).unwrap();
            let _final_chunk = chunker.finalize().unwrap();
            let elapsed = start_time.elapsed();

            let throughput_mbps = (data_size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();

            // Should achieve at least 50 MB/s throughput
            prop_assert!(throughput_mbps >= 50.0,
                "CDC chunking throughput {:.1} MB/s below minimum 50 MB/s", throughput_mbps);

            // Verify chunking produced reasonable results
            prop_assert!(!chunks.is_empty(), "Chunking should produce at least one chunk");
            
            let total_chunk_size: usize = chunks.iter().map(|c| c.size()).sum();
            prop_assert!(total_chunk_size <= data_size,
                "Total chunk size should not exceed input size");
        });
    }

    /// Property: Compression performance scales linearly with input size
    /// Processing time should be roughly proportional to input size
    #[test]
    fn prop_compression_linear_scaling(
        base_size in 524288usize..=1048576, // 512KB-1MB base
        scale_factor in 2usize..=4,         // 2x-4x scaling
        config in performance_chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("perf_corpus.bin");

            // Create corpus data
            let corpus_data = performance_test_data(base_size, 0x55);
            std::fs::write(&corpus_path, &corpus_data).unwrap();

            // Build corpus
            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
            let shared_corpus = Arc::new(corpus_manager);

            // Test with base size
            let base_data = performance_test_data(base_size, 0xAA);
            let mut base_compressor = Compressor::new(Arc::clone(&shared_corpus));
            
            let base_start = Instant::now();
            let _base_instructions = base_compressor.compress(&base_data).await.unwrap();
            let base_elapsed = base_start.elapsed();

            // Test with scaled size
            let scaled_size = base_size * scale_factor;
            let scaled_data = performance_test_data(scaled_size, 0xAA);
            let mut scaled_compressor = Compressor::new(Arc::clone(&shared_corpus));
            
            let scaled_start = Instant::now();
            let _scaled_instructions = scaled_compressor.compress(&scaled_data).await.unwrap();
            let scaled_elapsed = scaled_start.elapsed();

            // Calculate scaling efficiency
            let expected_time_ratio = scale_factor as f64;
            let actual_time_ratio = scaled_elapsed.as_secs_f64() / base_elapsed.as_secs_f64();

            // Scaling should be reasonably linear (within 50% of expected)
            let scaling_efficiency = (expected_time_ratio - actual_time_ratio).abs() / expected_time_ratio;
            prop_assert!(scaling_efficiency <= 0.5,
                "Compression scaling efficiency {:.1}% (expected linear scaling within 50%)",
                (1.0 - scaling_efficiency) * 100.0);
        });
    }

    /// Property: Memory usage is bounded regardless of input size
    /// Memory usage should not grow unboundedly with input size
    #[test]
    fn prop_memory_usage_bounded(
        data_size in 2097152usize..=8388608, // 2-8 MB
        config in performance_chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let data = performance_test_data(data_size, 0xBB);
            
            // Measure memory before processing
            let initial_memory = get_approximate_memory_usage();
            
            let mut chunker = FastCDCChunker::new(config).unwrap();
            let chunks = chunker.chunk_data(&data).unwrap();
            let _final_chunk = chunker.finalize().unwrap();
            
            // Measure memory after processing
            let peak_memory = get_approximate_memory_usage();
            let memory_growth = peak_memory.saturating_sub(initial_memory);
            
            // Memory growth should be reasonable (less than 2x input size)
            let max_acceptable_growth = data_size * 2;
            prop_assert!(memory_growth <= max_acceptable_growth,
                "Memory growth {} bytes exceeds acceptable limit {} bytes",
                memory_growth, max_acceptable_growth);

            // Verify processing was successful
            prop_assert!(!chunks.is_empty(), "Processing should produce chunks");
        });
    }

    /// Property: Concurrent processing maintains performance
    /// Multiple concurrent operations should not severely degrade performance
    #[test]
    fn prop_concurrent_performance_scaling(
        data_size in 1048576usize..=2097152, // 1-2 MB per task
        num_tasks in 2usize..=4,             // 2-4 concurrent tasks
        config in performance_chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("concurrent_corpus.bin");

            // Create shared corpus
            let corpus_data = performance_test_data(data_size, 0x77);
            std::fs::write(&corpus_path, &corpus_data).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
            let shared_corpus = Arc::new(corpus_manager);

            // Measure sequential performance
            let sequential_data = performance_test_data(data_size, 0x88);
            let mut sequential_compressor = Compressor::new(Arc::clone(&shared_corpus));
            
            let sequential_start = Instant::now();
            let _sequential_result = sequential_compressor.compress(&sequential_data).await.unwrap();
            let sequential_elapsed = sequential_start.elapsed();
            let sequential_throughput = (data_size as f64) / (1024.0 * 1024.0) / sequential_elapsed.as_secs_f64();

            // Measure concurrent performance
            let mut join_set = JoinSet::new();
            let concurrent_start = Instant::now();

            for i in 0..num_tasks {
                let corpus = Arc::clone(&shared_corpus);
                let task_data = performance_test_data(data_size, 0x88 + i as u8);
                
                join_set.spawn(async move {
                    let mut compressor = Compressor::new(corpus);
                    let start = Instant::now();
                    let result = compressor.compress(&task_data).await.unwrap();
                    let elapsed = start.elapsed();
                    (i, result.len(), elapsed)
                });
            }

            // Wait for all tasks to complete
            let mut task_results = Vec::new();
            while let Some(result) = join_set.join_next().await {
                task_results.push(result.unwrap());
            }
            
            let concurrent_total_elapsed = concurrent_start.elapsed();

            // Calculate concurrent throughput
            let total_concurrent_data = data_size * num_tasks;
            let concurrent_throughput = (total_concurrent_data as f64) / (1024.0 * 1024.0) / concurrent_total_elapsed.as_secs_f64();

            // Concurrent throughput should be at least 60% of sequential throughput
            let efficiency = concurrent_throughput / (sequential_throughput * num_tasks as f64);
            prop_assert!(efficiency >= 0.6,
                "Concurrent efficiency {:.1}% (expected >= 60%)", efficiency * 100.0);

            // Verify all tasks completed successfully
            prop_assert_eq!(task_results.len(), num_tasks, "All concurrent tasks should complete");
        });
    }

    /// Property: Large dataset processing completes within time bounds
    /// Very large datasets should still complete within reasonable time limits
    #[test]
    fn prop_large_dataset_time_bounds(
        data_size in 8388608usize..=16777216, // 8-16 MB
        config in performance_chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let data = performance_test_data(data_size, 0xCC);
            let mut chunker = FastCDCChunker::new(config).unwrap();

            let start_time = Instant::now();
            let chunks = chunker.chunk_data(&data).unwrap();
            let _final_chunk = chunker.finalize().unwrap();
            let elapsed = start_time.elapsed();

            // Should complete within 30 seconds for datasets up to 16MB
            let max_time = Duration::from_secs(30);
            prop_assert!(elapsed <= max_time,
                "Large dataset processing took {:.1}s, expected <= 30s", elapsed.as_secs_f64());

            // Should maintain minimum throughput
            let throughput_mbps = (data_size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
            prop_assert!(throughput_mbps >= 20.0,
                "Large dataset throughput {:.1} MB/s below minimum 20 MB/s", throughput_mbps);

            // Verify processing quality
            prop_assert!(!chunks.is_empty(), "Large dataset should produce chunks");
            
            let total_size: usize = chunks.iter().map(|c| c.size()).sum();
            prop_assert!(total_size <= data_size, "Chunk total should not exceed input");
        });
    }

    /// Property: Hash collision handling performance
    /// Performance should degrade gracefully with hash collisions
    #[test]
    fn prop_hash_collision_performance(
        collision_rate in 0.1f64..=0.5, // 10-50% collision rate
        data_size in 65536usize..=262144, // 64-256 KB
        config in performance_chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Create data designed to cause hash collisions
            let mut collision_data = vec![0u8; data_size];
            let collision_pattern_size = 64;
            let num_collision_blocks = (data_size as f64 * collision_rate) as usize / collision_pattern_size;
            
            // Insert collision-prone patterns
            for i in 0..num_collision_blocks {
                let start = (i * data_size) / num_collision_blocks;
                let end = (start + collision_pattern_size).min(data_size);
                
                // Use patterns likely to cause weak hash collisions
                for j in start..end {
                    collision_data[j] = (j % 4) as u8; // Very repetitive pattern
                }
            }

            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("collision_corpus.bin");
            std::fs::write(&corpus_path, &collision_data).unwrap();

            // Build corpus (this will create the collision-prone index)
            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();

            // Test compression performance with collision-prone data
            let test_data = collision_data.clone();
            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            
            let start_time = Instant::now();
            let instructions = compressor.compress(&test_data).await.unwrap();
            let elapsed = start_time.elapsed();

            // Even with collisions, should maintain reasonable performance
            let throughput_mbps = (data_size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
            prop_assert!(throughput_mbps >= 10.0,
                "Collision-heavy compression throughput {:.1} MB/s below minimum 10 MB/s", 
                throughput_mbps);

            // Should still produce valid results
            prop_assert!(!instructions.is_empty(), "Should produce compression instructions");
            
            let total_output_size: usize = instructions.iter().map(|i| i.output_size()).sum();
            prop_assert_eq!(total_output_size, data_size, "Output size should match input");
        });
    }
}

// === Stress Tests for Performance Validation ===

#[cfg(test)]
mod stress_tests {
    use super::*;

    /// Stress test: Sustained high-throughput processing
    #[tokio::test]
    async fn stress_sustained_throughput() {
        let temp_dir = TempDir::new().unwrap();
        let corpus_path = temp_dir.path().join("stress_corpus.bin");
        
        // Create large corpus
        let corpus_size = 4 * 1024 * 1024; // 4 MB
        let corpus_data = performance_test_data(corpus_size, 0x99);
        std::fs::write(&corpus_path, &corpus_data).unwrap();

        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );
        let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
        let shared_corpus = Arc::new(corpus_manager);

        // Process multiple datasets in sequence
        let num_iterations = 10;
        let data_size = 1024 * 1024; // 1 MB per iteration
        let mut throughputs = Vec::new();

        for i in 0..num_iterations {
            let test_data = performance_test_data(data_size, 0xAA + i as u8);
            let mut compressor = Compressor::new(Arc::clone(&shared_corpus));
            
            let start = Instant::now();
            let _instructions = compressor.compress(&test_data).await.unwrap();
            let elapsed = start.elapsed();
            
            let throughput = (data_size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
            throughputs.push(throughput);
        }

        // Verify sustained performance
        let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let min_throughput = throughputs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_throughput = throughputs.iter().fold(0.0, |a, &b| a.max(b));

        assert!(avg_throughput >= 30.0, 
            "Average sustained throughput {:.1} MB/s below 30 MB/s", avg_throughput);
        assert!(min_throughput >= 20.0,
            "Minimum throughput {:.1} MB/s below 20 MB/s", min_throughput);
        
        // Throughput should be relatively stable (max/min ratio < 3)
        let stability_ratio = max_throughput / min_throughput;
        assert!(stability_ratio < 3.0,
            "Throughput stability ratio {:.1} indicates performance instability", stability_ratio);
    }

    /// Stress test: Memory pressure handling
    #[tokio::test]
    async fn stress_memory_pressure() {
        let temp_dir = TempDir::new().unwrap();
        let corpus_path = temp_dir.path().join("memory_stress_corpus.bin");
        
        // Create corpus that would be large in memory
        let corpus_size = 8 * 1024 * 1024; // 8 MB corpus
        let corpus_data = performance_test_data(corpus_size, 0xDD);
        std::fs::write(&corpus_path, &corpus_data).unwrap();

        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );

        // Measure memory before corpus building
        let initial_memory = get_approximate_memory_usage();
        
        let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
        let shared_corpus = Arc::new(corpus_manager);

        let post_corpus_memory = get_approximate_memory_usage();

        // Process large dataset
        let large_data_size = 16 * 1024 * 1024; // 16 MB
        let large_data = performance_test_data(large_data_size, 0xEE);
        let mut compressor = Compressor::new(shared_corpus);
        
        let start = Instant::now();
        let _instructions = compressor.compress(&large_data).await.unwrap();
        let elapsed = start.elapsed();

        let final_memory = get_approximate_memory_usage();

        // Verify memory usage is reasonable
        let corpus_memory_growth = post_corpus_memory.saturating_sub(initial_memory);
        let total_memory_growth = final_memory.saturating_sub(initial_memory);

        // Corpus should not use more than 2x its size in memory
        assert!(corpus_memory_growth <= corpus_size * 2,
            "Corpus memory usage {} exceeds 2x corpus size {}", 
            corpus_memory_growth, corpus_size * 2);

        // Total memory growth should be reasonable
        assert!(total_memory_growth <= (corpus_size + large_data_size) * 3,
            "Total memory growth {} exceeds 3x data size", total_memory_growth);

        // Should still maintain reasonable performance under memory pressure
        let throughput = (large_data_size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
        assert!(throughput >= 15.0,
            "Memory pressure throughput {:.1} MB/s below 15 MB/s", throughput);
    }

    /// Stress test: Concurrent load with resource contention
    #[tokio::test]
    async fn stress_concurrent_resource_contention() {
        let temp_dir = TempDir::new().unwrap();
        let corpus_path = temp_dir.path().join("contention_corpus.bin");
        
        // Create shared corpus
        let corpus_size = 2 * 1024 * 1024; // 2 MB
        let corpus_data = performance_test_data(corpus_size, 0xFF);
        std::fs::write(&corpus_path, &corpus_data).unwrap();

        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );
        let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
        let shared_corpus = Arc::new(corpus_manager);

        // Launch many concurrent tasks to create resource contention
        let num_tasks = 8;
        let data_size = 512 * 1024; // 512 KB per task
        let mut join_set = JoinSet::new();

        let start_time = Instant::now();

        for i in 0..num_tasks {
            let corpus = Arc::clone(&shared_corpus);
            let task_data = performance_test_data(data_size, i as u8);
            
            join_set.spawn(async move {
                let mut compressor = Compressor::new(corpus);
                let task_start = Instant::now();
                let result = compressor.compress(&task_data).await.unwrap();
                let task_elapsed = task_start.elapsed();
                (i, result.len(), task_elapsed)
            });
        }

        // Collect results
        let mut results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            results.push(result.unwrap());
        }

        let total_elapsed = start_time.elapsed();

        // Verify all tasks completed
        assert_eq!(results.len(), num_tasks, "All concurrent tasks should complete");

        // Calculate performance metrics
        let total_data_processed = data_size * num_tasks;
        let overall_throughput = (total_data_processed as f64) / (1024.0 * 1024.0) / total_elapsed.as_secs_f64();

        // Should maintain reasonable throughput even under contention
        assert!(overall_throughput >= 10.0,
            "Concurrent contention throughput {:.1} MB/s below 10 MB/s", overall_throughput);

        // Individual task times should be reasonable
        let task_times: Vec<_> = results.iter().map(|(_, _, elapsed)| elapsed.as_secs_f64()).collect();
        let max_task_time = task_times.iter().fold(0.0, |a, &b| a.max(b));
        let avg_task_time = task_times.iter().sum::<f64>() / task_times.len() as f64;

        assert!(max_task_time <= 10.0,
            "Maximum task time {:.1}s exceeds 10s limit", max_task_time);
        assert!(avg_task_time <= 5.0,
            "Average task time {:.1}s exceeds 5s limit", avg_task_time);
    }
}

// === Utility Functions ===

/// Get approximate memory usage (simplified for testing)
fn get_approximate_memory_usage() -> usize {
    // This is a simplified approximation
    // In a real implementation, you would use proper memory monitoring
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
    std::mem::size_of::<usize>() * 1024 * 1024 // 1MB baseline
}

/// Benchmark a function and return execution time
fn benchmark_function<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed)
}

#[cfg(test)]
mod performance_unit_tests {
    use super::*;

    #[test]
    fn test_performance_test_data_generation() {
        let data = performance_test_data(1024, 0xAA);
        assert_eq!(data.len(), 1024);
        
        // Verify pattern is applied correctly
        assert_eq!(data[0], 0xAA);
        assert_eq!(data[512], 0x00);
        assert_eq!(data[768], 0xFF);
    }

    #[test]
    fn test_benchmark_function() {
        let (result, duration) = benchmark_function(|| {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(9));
        assert!(duration <= Duration::from_millis(20));
    }

    #[tokio::test]
    async fn test_memory_usage_measurement() {
        let initial = get_approximate_memory_usage();
        
        // Allocate some memory
        let _large_vec = vec![0u8; 1024 * 1024]; // 1 MB
        
        let after_allocation = get_approximate_memory_usage();
        
        // Memory usage should have increased (though exact amount depends on allocator)
        assert!(after_allocation >= initial, 
            "Memory usage should increase after allocation");
    }

    #[tokio::test]
    async fn test_rolling_hash_performance_basic() {
        let data = vec![0xAA; 8192];
        let mut hasher = RollingHasher::new(HASH_BASE, 4096);
        
        hasher.init(&data[0..4096]).unwrap();
        
        let start = Instant::now();
        for i in 4096..data.len() {
            let _hash = hasher.roll(data[i - 4096], data[i]).unwrap();
        }
        let elapsed = start.elapsed();
        
        let time_per_update = elapsed.as_nanos() as f64 / (data.len() - 4096) as f64;
        
        // Should be very fast (less than 1 microsecond per update)
        assert!(time_per_update < 1000.0, 
            "Rolling hash update took {:.1} ns, expected < 1000 ns", time_per_update);
    }
}