//! Comprehensive Stress Tests for Reducto Mode 3
//!
//! This module implements stress tests for memory constraints, large dataset handling,
//! and validation of all performance contracts with automated measurement.
//! These tests push the system to its limits to ensure enterprise-grade reliability.

use proptest::prelude::*;
use reducto_mode_3::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::task::JoinSet;

// === Stress Test Strategies ===

/// Generate large datasets for stress testing
fn large_dataset_strategy() -> impl Strategy<Value = (usize, u8)> {
    (
        8388608usize..=33554432, // 8-32 MB datasets
        0u8..=255u8,             // Pattern byte
    )
}

/// Generate memory pressure scenarios
fn memory_pressure_strategy() -> impl Strategy<Value = (usize, usize, usize)> {
    (
        4usize..=16,              // Number of concurrent operations
        1048576usize..=4194304,   // Size per operation (1-4 MB)
        2usize..=8,               // Memory multiplier for pressure
    )
}

/// Generate concurrent load scenarios
fn concurrent_load_strategy() -> impl Strategy<Value = (usize, usize, u64)> {
    (
        8usize..=32,              // Number of concurrent tasks
        262144usize..=1048576,    // Data size per task (256KB-1MB)
        100u64..=5000u64,         // Max processing time in milliseconds
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Property: System handles very large datasets without memory exhaustion
    /// Memory usage should remain bounded regardless of input size
    #[test]
    fn prop_large_dataset_memory_bounds(
        (data_size, pattern) in large_dataset_strategy(),
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Generate large test dataset
            let mut large_data = vec![pattern; data_size];
            
            // Add some structure to make chunking meaningful
            for i in (0..large_data.len()).step_by(4096) {
                if i + 8 < large_data.len() {
                    large_data[i..i+8].copy_from_slice(&(i as u64).to_le_bytes());
                }
            }

            // Measure initial memory usage
            let initial_memory = get_memory_usage();

            // Process with CDC chunker
            let mut chunker = FastCDCChunker::new(config.clone()).unwrap();
            
            let chunk_start = Instant::now();
            let chunks = chunker.chunk_data(&large_data).unwrap();
            let final_chunk = chunker.finalize().unwrap();
            let chunk_elapsed = chunk_start.elapsed();

            let post_chunk_memory = get_memory_usage();

            // Create corpus from chunks
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("large_corpus.bin");
            std::fs::write(&corpus_path, &large_data[0..data_size/2]).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            
            let corpus_start = Instant::now();
            let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
            let corpus_elapsed = corpus_start.elapsed();

            let post_corpus_memory = get_memory_usage();

            // Test compression
            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            
            let compress_start = Instant::now();
            let instructions = compressor.compress(&large_data[data_size/2..]).await.unwrap();
            let compress_elapsed = compress_start.elapsed();

            let final_memory = get_memory_usage();

            // Verify memory usage is bounded
            let chunk_memory_growth = post_chunk_memory.saturating_sub(initial_memory);
            let corpus_memory_growth = post_corpus_memory.saturating_sub(post_chunk_memory);
            let compress_memory_growth = final_memory.saturating_sub(post_corpus_memory);

            // Memory growth should be reasonable (less than 4x input size)
            let max_acceptable_growth = data_size * 4;
            
            prop_assert!(chunk_memory_growth <= max_acceptable_growth,
                "Chunking memory growth {} exceeds limit {} for {}MB input",
                chunk_memory_growth, max_acceptable_growth, data_size / 1048576);

            prop_assert!(corpus_memory_growth <= max_acceptable_growth,
                "Corpus memory growth {} exceeds limit {} for {}MB input", 
                corpus_memory_growth, max_acceptable_growth, data_size / 1048576);

            prop_assert!(compress_memory_growth <= max_acceptable_growth / 2,
                "Compression memory growth {} exceeds limit {} for {}MB input",
                compress_memory_growth, max_acceptable_growth / 2, data_size / 1048576);

            // Verify performance contracts
            let chunk_throughput = (data_size as f64) / (1024.0 * 1024.0) / chunk_elapsed.as_secs_f64();
            prop_assert!(chunk_throughput >= 20.0,
                "Chunking throughput {:.1} MB/s below minimum 20 MB/s for large dataset", 
                chunk_throughput);

            let compress_throughput = ((data_size / 2) as f64) / (1024.0 * 1024.0) / compress_elapsed.as_secs_f64();
            prop_assert!(compress_throughput >= 10.0,
                "Compression throughput {:.1} MB/s below minimum 10 MB/s for large dataset",
                compress_throughput);

            // Verify correctness
            prop_assert!(!chunks.is_empty(), "Should produce chunks for large dataset");
            prop_assert!(!instructions.is_empty(), "Should produce instructions for large dataset");

            let total_chunk_size: usize = chunks.iter().map(|c| c.size()).sum();
            if let Some(ref final_chunk) = final_chunk {
                let total_with_final = total_chunk_size + final_chunk.size();
                prop_assert_eq!(total_with_final, data_size, "Chunk sizes should sum to input size");
            } else {
                prop_assert_eq!(total_chunk_size, data_size, "Chunk sizes should sum to input size");
            }

            let total_instruction_size: usize = instructions.iter().map(|i| i.output_size()).sum();
            prop_assert_eq!(total_instruction_size, data_size / 2, "Instruction sizes should sum to input size");
        });
    }

    /// Property: System maintains performance under memory pressure
    /// Performance should degrade gracefully under memory constraints
    #[test]
    fn prop_memory_pressure_performance(
        (num_operations, size_per_op, memory_multiplier) in memory_pressure_strategy(),
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            
            // Create memory pressure by allocating large buffers
            let pressure_size = size_per_op * memory_multiplier;
            let _pressure_buffers: Vec<Vec<u8>> = (0..num_operations)
                .map(|i| vec![(i % 256) as u8; pressure_size])
                .collect();

            let initial_memory = get_memory_usage();

            // Create corpus under memory pressure
            let corpus_data = vec![0xAA; size_per_op];
            let corpus_path = temp_dir.path().join("pressure_corpus.bin");
            std::fs::write(&corpus_path, &corpus_data).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            
            let corpus_start = Instant::now();
            let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
            let corpus_elapsed = corpus_start.elapsed();
            let shared_corpus = Arc::new(corpus_manager);

            let post_corpus_memory = get_memory_usage();

            // Perform multiple compression operations under memory pressure
            let mut join_set = JoinSet::new();
            let operations_start = Instant::now();

            for i in 0..num_operations {
                let corpus = Arc::clone(&shared_corpus);
                let test_data = vec![(0xBB + i as u8) % 256; size_per_op];
                
                join_set.spawn(async move {
                    let mut compressor = Compressor::new(corpus);
                    let start = Instant::now();
                    let result = compressor.compress(&test_data).await;
                    let elapsed = start.elapsed();
                    (i, result, elapsed)
                });
            }

            // Collect results
            let mut results = Vec::new();
            while let Some(result) = join_set.join_next().await {
                results.push(result.unwrap());
            }

            let total_elapsed = operations_start.elapsed();
            let final_memory = get_memory_usage();

            // Verify all operations completed successfully
            prop_assert_eq!(results.len(), num_operations, "All operations should complete");
            
            for (i, result, _) in &results {
                prop_assert!(result.is_ok(), "Operation {} should succeed under memory pressure", i);
            }

            // Verify performance degradation is acceptable
            let avg_operation_time = total_elapsed.as_secs_f64() / num_operations as f64;
            prop_assert!(avg_operation_time <= 10.0,
                "Average operation time {:.1}s exceeds 10s limit under memory pressure", 
                avg_operation_time);

            // Verify memory usage is still reasonable
            let memory_growth = final_memory.saturating_sub(initial_memory);
            let expected_max_growth = (size_per_op * num_operations * memory_multiplier) + (size_per_op * 2);
            
            prop_assert!(memory_growth <= expected_max_growth * 2,
                "Memory growth {} exceeds expected limit {} under pressure",
                memory_growth, expected_max_growth * 2);

            // Verify throughput is still acceptable (reduced but not terrible)
            let total_data_processed = size_per_op * num_operations;
            let overall_throughput = (total_data_processed as f64) / (1024.0 * 1024.0) / total_elapsed.as_secs_f64();
            
            prop_assert!(overall_throughput >= 5.0,
                "Overall throughput {:.1} MB/s below minimum 5 MB/s under memory pressure",
                overall_throughput);
        });
    }

    /// Property: System handles high concurrent load gracefully
    /// Concurrent operations should not cause deadlocks or severe performance degradation
    #[test]
    fn prop_concurrent_load_handling(
        (num_tasks, data_size, max_time_ms) in concurrent_load_strategy(),
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) -> Result<(), TestCaseError> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let max_time = Duration::from_millis(max_time_ms);
            
            // Create shared corpus
            let corpus_data = vec![0xCC; data_size * 2];
            let corpus_path = temp_dir.path().join("concurrent_corpus.bin");
            std::fs::write(&corpus_path, &corpus_data).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                StorageBackend::InMemory,
                config.clone()
            ).unwrap();
            let _metadata = corpus_manager.build_corpus(&[corpus_path], config.clone()).await.unwrap();
            let shared_corpus = Arc::new(std::sync::RwLock::new(corpus_manager));

            // Launch concurrent tasks
            let mut join_set = JoinSet::new();
            let start_time = Instant::now();

            for task_id in 0..num_tasks {
                let corpus = Arc::clone(&shared_corpus);
                let task_data = vec![(task_id % 256) as u8; data_size];
                let task_max_time = max_time;
                let task_config = config.clone();
                
                join_set.spawn(async move {
                    let task_start = Instant::now();
                    
                    // Perform multiple operations per task
                    let mut task_results = Vec::new();
                    
                    for op_id in 0..3 {
                        let mut compressor = Compressor::new(task_config.clone(), Arc::clone(&corpus)).unwrap();
                        let mut op_data = task_data.clone();
                        
                        // Modify data slightly for each operation
                        for byte in op_data.iter_mut().take(op_id * 100) {
                            *byte = byte.wrapping_add(op_id as u8);
                        }
                        
                        let op_start = Instant::now();
                        let result = compressor.compress(&op_data);
                        let op_elapsed = op_start.elapsed();
                        
                        task_results.push((op_id, result, op_elapsed));
                        
                        // Check if we're exceeding time limit
                        if task_start.elapsed() > task_max_time {
                            break;
                        }
                    }
                    
                    let task_elapsed = task_start.elapsed();
                    (task_id, task_results, task_elapsed)
                });
            }

            // Collect all results with timeout
            let mut all_results = Vec::new();
            let collection_timeout = max_time + Duration::from_secs(10);
            
            while let Ok(Some(result)) = tokio::time::timeout(collection_timeout, join_set.join_next()).await {
                match result {
                    Ok(task_result) => {
                        all_results.push(task_result);
                    }
                    Err(_) => {
                        // Task panicked - this is not acceptable
                        break;
                    }
                }
            }

            let total_elapsed = start_time.elapsed();

            // Verify most tasks completed successfully
            let completion_rate = all_results.len() as f64 / num_tasks as f64;
            prop_assert!(completion_rate >= 0.8,
                "Completion rate {:.1}% below 80% under concurrent load", 
                completion_rate * 100.0);

            // Verify successful operations
            let mut total_operations = 0;
            let mut successful_operations = 0;
            let mut total_operation_time = Duration::ZERO;

            for (task_id, task_results, task_time) in &all_results {
                for (op_id, result, op_time) in task_results {
                    total_operations += 1;
                    total_operation_time += *op_time;
                    
                    if result.is_ok() {
                        successful_operations += 1;
                    }
                }
                
                // Individual task time should be reasonable
                prop_assert!(task_time <= &(max_time + Duration::from_secs(5)),
                    "Task {} took {:.1}s, exceeds limit {:.1}s", 
                    task_id, task_time.as_secs_f64(), (max_time + Duration::from_secs(5)).as_secs_f64());
            }

            // Verify operation success rate
            if total_operations > 0 {
                let success_rate = successful_operations as f64 / total_operations as f64;
                prop_assert!(success_rate >= 0.9,
                    "Operation success rate {:.1}% below 90% under concurrent load",
                    success_rate * 100.0);
            }

            // Verify overall performance
            let total_data_processed = successful_operations * data_size;
            if total_data_processed > 0 {
                let overall_throughput = (total_data_processed as f64) / (1024.0 * 1024.0) / total_elapsed.as_secs_f64();
                
                prop_assert!(overall_throughput >= 2.0,
                    "Concurrent throughput {:.1} MB/s below minimum 2 MB/s",
                    overall_throughput);
            }

            // Verify no deadlocks occurred (all tasks either completed or timed out gracefully)
            prop_assert!(total_elapsed <= collection_timeout + Duration::from_secs(5),
                "Total execution time suggests possible deadlock");
                
            Ok(())
        })
    }

    /// Property: Performance contracts are validated under stress
    /// All performance claims should hold even under stress conditions
    #[test]
    fn prop_performance_contract_validation(
        stress_level in 1usize..=5,  // 1=light, 5=extreme stress
        data_size in 1048576usize..=8388608, // 1-8 MB
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            
            // Create stress conditions based on stress level
            let num_concurrent_ops = stress_level * 4;
            let memory_pressure_multiplier = stress_level;
            let corpus_size = data_size * stress_level;

            // Generate stress data
            let mut stress_data = vec![0u8; data_size];
            for (i, byte) in stress_data.iter_mut().enumerate() {
                *byte = ((i * stress_level + i / 1024) % 256) as u8;
            }

            // Create memory pressure
            let _pressure_buffers: Vec<Vec<u8>> = (0..memory_pressure_multiplier)
                .map(|i| vec![(i % 256) as u8; data_size])
                .collect();

            // Build corpus under stress
            let corpus_data = vec![0xDD; corpus_size];
            let corpus_path = temp_dir.path().join("stress_corpus.bin");
            std::fs::write(&corpus_path, &corpus_data).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            
            let corpus_build_start = Instant::now();
            let metadata = corpus_manager.build_corpus(&[corpus_path], config.clone()).await.unwrap();
            let corpus_build_time = corpus_build_start.elapsed();
            let shared_corpus = Arc::new(corpus_manager);

            // Validate corpus build performance contract
            let corpus_build_throughput = (corpus_size as f64) / (1024.0 * 1024.0) / corpus_build_time.as_secs_f64();
            let min_corpus_throughput = 10.0 / stress_level as f64; // Reduced expectation under stress
            
            prop_assert!(corpus_build_throughput >= min_corpus_throughput,
                "Corpus build throughput {:.1} MB/s below minimum {:.1} MB/s under stress level {}",
                corpus_build_throughput, min_corpus_throughput, stress_level);

            // Test rolling hash performance contract under stress
            let mut hasher = RollingHasher::new(HASH_BASE, config.target_size);
            hasher.init(&stress_data[0..config.target_size]).unwrap();
            
            let hash_ops = (data_size - config.target_size).min(10000); // Limit for reasonable test time
            let hash_start = Instant::now();
            
            for i in config.target_size..config.target_size + hash_ops {
                let exiting_byte = stress_data[i - config.target_size];
                let entering_byte = stress_data[i];
                let _hash = hasher.roll(exiting_byte, entering_byte).unwrap();
            }
            
            let hash_elapsed = hash_start.elapsed();
            let time_per_hash_ns = hash_elapsed.as_nanos() as f64 / hash_ops as f64;
            let max_time_per_hash = 1000.0 * stress_level as f64; // Allow more time under stress
            
            prop_assert!(time_per_hash_ns <= max_time_per_hash,
                "Rolling hash time {:.1} ns/op exceeds limit {:.1} ns/op under stress level {}",
                time_per_hash_ns, max_time_per_hash, stress_level);

            // Test concurrent compression performance contracts
            let mut join_set = JoinSet::new();
            let concurrent_start = Instant::now();

            for i in 0..num_concurrent_ops {
                let corpus = Arc::clone(&shared_corpus);
                let mut task_data = stress_data.clone();
                
                // Modify data for each task
                for byte in task_data.iter_mut().take(i * 100) {
                    *byte = byte.wrapping_add(i as u8);
                }
                
                join_set.spawn(async move {
                    let mut compressor = Compressor::new(corpus);
                    let start = Instant::now();
                    let result = compressor.compress(&task_data).await;
                    let elapsed = start.elapsed();
                    (i, result, elapsed)
                });
            }

            // Collect concurrent results
            let mut concurrent_results = Vec::new();
            while let Some(result) = join_set.join_next().await {
                concurrent_results.push(result.unwrap());
            }

            let concurrent_total_time = concurrent_start.elapsed();

            // Validate concurrent performance contracts
            let successful_ops = concurrent_results.iter()
                .filter(|(_, result, _)| result.is_ok())
                .count();
            
            let success_rate = successful_ops as f64 / num_concurrent_ops as f64;
            let min_success_rate = 0.8 / (stress_level as f64 * 0.2 + 0.8); // Lower expectation under stress
            
            prop_assert!(success_rate >= min_success_rate,
                "Concurrent success rate {:.1}% below minimum {:.1}% under stress level {}",
                success_rate * 100.0, min_success_rate * 100.0, stress_level);

            // Validate throughput under concurrent stress
            let total_concurrent_data = successful_ops * data_size;
            let concurrent_throughput = (total_concurrent_data as f64) / (1024.0 * 1024.0) / concurrent_total_time.as_secs_f64();
            let min_concurrent_throughput = 5.0 / stress_level as f64;
            
            prop_assert!(concurrent_throughput >= min_concurrent_throughput,
                "Concurrent throughput {:.1} MB/s below minimum {:.1} MB/s under stress level {}",
                concurrent_throughput, min_concurrent_throughput, stress_level);

            // Validate memory usage contracts
            let final_memory = get_memory_usage();
            let max_acceptable_memory = (corpus_size + data_size * num_concurrent_ops) * (2 + stress_level);
            
            prop_assert!(final_memory <= max_acceptable_memory,
                "Memory usage {} exceeds limit {} under stress level {}",
                final_memory, max_acceptable_memory, stress_level);

            // Validate correctness under stress
            prop_assert!(metadata.chunk_count > 0, "Corpus should contain chunks under stress");
            prop_assert!(metadata.total_size > 0, "Corpus should have size under stress");
            
            for (i, result, _) in &concurrent_results {
                if let Ok(instructions) = result {
                    prop_assert!(!instructions.is_empty(), 
                        "Task {} should produce instructions under stress", i);
                    
                    let total_output: usize = instructions.iter().map(|inst| inst.output_size()).sum();
                    prop_assert_eq!(total_output, data_size,
                        "Task {} output size should match input under stress", i);
                }
            }
        });
    }
}

// === Utility Functions ===

/// Get current memory usage (simplified implementation)
fn get_memory_usage() -> usize {
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
    
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("ps")
            .args(&["-o", "rss=", "-p"])
            .arg(std::process::id().to_string())
            .output()
        {
            if let Ok(rss_str) = String::from_utf8(output.stdout) {
                if let Ok(rss_kb) = rss_str.trim().parse::<usize>() {
                    return rss_kb * 1024; // Convert KB to bytes
                }
            }
        }
    }
    
    // Fallback: return a reasonable estimate
    16 * 1024 * 1024 // 16MB baseline
}

/// Create test data with specific patterns for stress testing
fn create_stress_test_data(size: usize, pattern_type: u8) -> Vec<u8> {
    let mut data = vec![0u8; size];
    
    match pattern_type % 4 {
        0 => {
            // Highly repetitive pattern
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = (i % 16) as u8;
            }
        }
        1 => {
            // Random-like pattern
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = ((i * 17 + i / 256) % 256) as u8;
            }
        }
        2 => {
            // Block-structured pattern
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = match i % 1024 {
                    0..=511 => 0xAA,
                    512..=767 => (i / 1024) as u8,
                    _ => 0x00,
                };
            }
        }
        3 => {
            // Sparse pattern (mostly zeros)
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = if i % 64 == 0 { 0xFF } else { 0x00 };
            }
        }
        _ => unreachable!(),
    }
    
    data
}

#[cfg(test)]
mod stress_unit_tests {
    use super::*;

    #[test]
    fn test_memory_usage_measurement() {
        let initial = get_memory_usage();
        
        // Allocate some memory
        let _large_allocation = vec![0u8; 1024 * 1024]; // 1MB
        
        let after_allocation = get_memory_usage();
        
        // Memory usage should have increased
        assert!(after_allocation >= initial, 
            "Memory usage should increase after allocation: {} -> {}", 
            initial, after_allocation);
    }

    #[test]
    fn test_stress_data_generation() {
        for pattern_type in 0..4 {
            let data = create_stress_test_data(1024, pattern_type);
            assert_eq!(data.len(), 1024);
            
            // Verify pattern was applied
            match pattern_type % 4 {
                0 => {
                    // Repetitive pattern
                    assert_eq!(data[0], 0);
                    assert_eq!(data[16], 0);
                    assert_eq!(data[8], 8);
                }
                1 => {
                    // Random-like pattern
                    assert_ne!(data[0], data[1]);
                }
                2 => {
                    // Block-structured
                    assert_eq!(data[0], 0xAA);
                    assert_eq!(data[512], 0xAA);
                    assert_eq!(data[768], 0x00);
                }
                3 => {
                    // Sparse pattern
                    assert_eq!(data[0], 0xFF);
                    assert_eq!(data[64], 0xFF);
                    assert_eq!(data[1], 0x00);
                }
                _ => unreachable!(),
            }
        }
    }

    #[tokio::test]
    async fn test_basic_stress_scenario() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create moderate stress scenario
        let data_size = 1024 * 1024; // 1MB
        let stress_data = create_stress_test_data(data_size, 1);
        
        // Build corpus
        let corpus_path = temp_dir.path().join("stress_corpus.bin");
        std::fs::write(&corpus_path, &stress_data[0..data_size/2]).unwrap();
        
        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );
        
        let start = Instant::now();
        let metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
        let build_time = start.elapsed();
        
        // Verify basic performance
        assert!(build_time.as_secs() <= 10, 
            "Corpus build took {}s, expected <= 10s", build_time.as_secs());
        assert!(metadata.chunk_count > 0, "Should produce chunks");
        
        // Test compression
        let mut compressor = Compressor::new(Arc::new(corpus_manager));
        
        let compress_start = Instant::now();
        let instructions = compressor.compress(&stress_data[data_size/2..]).await.unwrap();
        let compress_time = compress_start.elapsed();
        
        assert!(compress_time.as_secs() <= 5,
            "Compression took {}s, expected <= 5s", compress_time.as_secs());
        assert!(!instructions.is_empty(), "Should produce instructions");
        
        let total_output: usize = instructions.iter().map(|i| i.output_size()).sum();
        assert_eq!(total_output, data_size / 2, "Output size should match input");
    }

    #[tokio::test]
    async fn test_concurrent_stress_basic() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create shared corpus
        let corpus_data = vec![0xAA; 512 * 1024]; // 512KB
        let corpus_path = temp_dir.path().join("concurrent_corpus.bin");
        std::fs::write(&corpus_path, &corpus_data).unwrap();
        
        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );
        let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
        let shared_corpus = Arc::new(corpus_manager);
        
        // Launch concurrent tasks
        let num_tasks = 4;
        let mut join_set = JoinSet::new();
        
        for i in 0..num_tasks {
            let corpus = Arc::clone(&shared_corpus);
            let test_data = vec![(i % 256) as u8; 256 * 1024]; // 256KB per task
            
            join_set.spawn(async move {
                let mut compressor = Compressor::new(corpus);
                let result = compressor.compress(&test_data).await;
                (i, result)
            });
        }
        
        // Collect results
        let mut results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            results.push(result.unwrap());
        }
        
        // Verify all tasks completed successfully
        assert_eq!(results.len(), num_tasks);
        for (i, result) in results {
            assert!(result.is_ok(), "Task {} should succeed", i);
            let instructions = result.unwrap();
            assert!(!instructions.is_empty(), "Task {} should produce instructions", i);
        }
    }
}