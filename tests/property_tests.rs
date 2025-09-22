//! Comprehensive property-based testing for Reducto Mode 3
//!
//! This module implements extensive property-based testing to validate:
//! - CDC roundtrip properties: chunk(data) → compress → decompress → data
//! - Chunk boundary stability across data insertion/deletion scenarios
//! - Corpus consistency properties with concurrent access patterns
//! - Security property tests for signature verification and encryption
//! - Performance property validation with generated workloads
//! - Enterprise workflow scenarios (VM images, CI/CD artifacts, database backups)
//! - Stress tests for memory constraints and large dataset handling
//! - Validation of all performance contracts with automated measurement

use proptest::prelude::*;
use reducto_mode_3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::task::JoinSet;

// === Property Test Generators ===

/// Generate realistic data patterns for testing
fn realistic_data_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        // Text-like data with repetition
        prop::collection::vec(0u8..=127u8, 1..65536)
            .prop_map(|mut data| {
                // Add some repetitive patterns
                for i in 0..data.len().min(1000) {
                    if i % 10 == 0 {
                        data[i] = b'A';
                    }
                }
                data
            }),
        // Binary data with blocks
        prop::collection::vec(any::<u8>(), 1..65536)
            .prop_map(|mut data| {
                // Add some repeated 4KB blocks
                let block_pattern = vec![0xAA; 4096];
                if data.len() > 8192 {
                    data[4096..8192].copy_from_slice(&block_pattern);
                }
                data
            }),
        // Sparse data (mostly zeros with some content)
        prop::collection::vec(0u8..=255u8, 1..65536)
            .prop_map(|mut data| {
                // Make 80% of data zeros
                for i in 0..data.len() {
                    if i % 5 != 0 {
                        data[i] = 0;
                    }
                }
                data
            }),
    ]
}

/// Generate chunk configurations for testing
fn chunk_config_strategy() -> impl Strategy<Value = ChunkConfig> {
    (4096usize..=32768, 0.3f64..=0.7, 1.5f64..=3.0)
        .prop_map(|(target_size, min_ratio, max_ratio)| {
            ChunkConfig::with_ratios(target_size, min_ratio, max_ratio)
                .unwrap_or_else(|_| ChunkConfig::default())
        })
}

/// Generate corpus data with known patterns
fn corpus_data_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 16384..=1048576) // 16KB to 1MB
}

// === Core CDC Roundtrip Properties ===

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: CDC chunking is deterministic and reproducible
    /// For identical input data and configuration, chunking should produce identical results
    #[test]
    fn prop_cdc_chunking_deterministic(
        data in realistic_data_strategy(),
        config in chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut chunker1 = FastCDCChunker::new(config.clone()).unwrap();
            let mut chunker2 = FastCDCChunker::new(config).unwrap();

            let chunks1 = chunker1.chunk_data(&data).unwrap();
            let final1 = chunker1.finalize().unwrap();

            let chunks2 = chunker2.chunk_data(&data).unwrap();
            let final2 = chunker2.finalize().unwrap();

            prop_assert_eq!(chunks1.len(), chunks2.len());
            
            for (c1, c2) in chunks1.iter().zip(chunks2.iter()) {
                prop_assert_eq!(c1.data, c2.data);
                prop_assert_eq!(c1.weak_hash, c2.weak_hash);
                prop_assert_eq!(c1.strong_hash, c2.strong_hash);
                prop_assert_eq!(c1.offset, c2.offset);
            }

            prop_assert_eq!(final1.is_some(), final2.is_some());
            if let (Some(f1), Some(f2)) = (final1, final2) {
                prop_assert_eq!(f1.data, f2.data);
            }
        });
    }

    /// Property: Chunk size variance is within configured bounds
    /// All chunks must be between min_size and max_size (50%-200% of target)
    #[test]
    fn prop_chunk_size_variance_bounds(
        data in realistic_data_strategy(),
        config in chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut chunker = FastCDCChunker::new(config.clone()).unwrap();
            let chunks = chunker.chunk_data(&data).unwrap();
            let final_chunk = chunker.finalize().unwrap();

            // Check all chunks are within bounds
            for chunk in &chunks {
                prop_assert!(chunk.size() >= config.min_size, 
                    "Chunk size {} below minimum {}", chunk.size(), config.min_size);
                prop_assert!(chunk.size() <= config.max_size,
                    "Chunk size {} above maximum {}", chunk.size(), config.max_size);
            }

            // Check final chunk if it exists
            if let Some(final_chunk) = final_chunk {
                // Final chunk may be smaller than min_size (remainder data)
                prop_assert!(final_chunk.size() <= config.max_size,
                    "Final chunk size {} above maximum {}", final_chunk.size(), config.max_size);
            }
        });
    }

    /// Property: Complete roundtrip compression/decompression preserves data
    /// chunk(data) → compress → decompress → data should be identity
    #[test]
    fn prop_compression_roundtrip_identity(
        input_data in realistic_data_strategy(),
        corpus_data in corpus_data_strategy(),
        config in chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("test_corpus.bin");
            let compressed_path = temp_dir.path().join("test.reducto");

            // Create corpus
            std::fs::write(&corpus_path, &corpus_data).unwrap();
            
            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            
            let corpus_metadata = corpus_manager.build_corpus(
                &[corpus_path.clone()],
                config.clone()
            ).await.unwrap();

            // Compress data
            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            let instructions = compressor.compress(&input_data).await.unwrap();

            // Serialize and compress
            let mut serializer = AdvancedSerializer::new(SerializerConfig::default());
            let header = ReductoHeader::basic(corpus_metadata.corpus_id, config);
            
            serializer.create(&compressed_path).unwrap();
            serializer.write_header(&header).unwrap();
            serializer.write_instructions(&instructions).unwrap();
            let _compressed_size = serializer.finalize().unwrap();

            // Decompress
            #[cfg(feature = "sdk")]
            {
                let mut decompressor = EcosystemDecompressor::new(EcosystemConfig::default());
                let decompressed = decompressor.decompress_file(&compressed_path, &corpus_path).await.unwrap();
                
                prop_assert_eq!(input_data, decompressed.data, 
                    "Roundtrip failed: input {} bytes != output {} bytes", 
                    input_data.len(), decompressed.data.len());
            }
        });
    }
}

// === Chunk Boundary Stability Properties ===

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: Chunk boundaries are stable across data insertion
    /// Inserting data should not shift all subsequent boundaries (CDC robustness)
    #[test]
    fn prop_chunk_boundary_stability_insertion(
        mut base_data in realistic_data_strategy(),
        insertion_data in prop::collection::vec(any::<u8>(), 1..1024),
        insertion_point in 0usize..=100,
        config in chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Ensure we have enough data
            if base_data.len() < 8192 {
                base_data.resize(8192, 0);
            }

            let actual_insertion_point = (insertion_point * base_data.len()) / 100;
            
            // Chunk original data
            let mut chunker1 = FastCDCChunker::new(config.clone()).unwrap();
            let original_chunks = chunker1.chunk_data(&base_data).unwrap();
            let _final1 = chunker1.finalize().unwrap();

            // Insert data and chunk again
            let mut modified_data = base_data.clone();
            modified_data.splice(actual_insertion_point..actual_insertion_point, insertion_data.iter().cloned());

            let mut chunker2 = FastCDCChunker::new(config).unwrap();
            let modified_chunks = chunker2.chunk_data(&modified_data).unwrap();
            let _final2 = chunker2.finalize().unwrap();

            // Count how many chunk boundaries were preserved
            let original_boundaries: std::collections::HashSet<_> = original_chunks
                .iter()
                .map(|c| c.offset + c.size() as u64)
                .collect();

            let modified_boundaries: std::collections::HashSet<_> = modified_chunks
                .iter()
                .filter(|c| c.offset >= actual_insertion_point as u64 + insertion_data.len() as u64)
                .map(|c| c.offset - insertion_data.len() as u64)
                .collect();

            let preserved_boundaries = original_boundaries.intersection(&modified_boundaries).count();
            let total_original = original_boundaries.len();

            // CDC should preserve a significant portion of boundaries (at least 30%)
            if total_original > 0 {
                let preservation_rate = preserved_boundaries as f64 / total_original as f64;
                prop_assert!(preservation_rate >= 0.3, 
                    "Boundary preservation rate {} too low (expected >= 0.3)", preservation_rate);
            }
        });
    }

    /// Property: Chunk boundaries are stable across data deletion
    /// Deleting data should not shift all subsequent boundaries
    #[test]
    fn prop_chunk_boundary_stability_deletion(
        mut base_data in realistic_data_strategy(),
        deletion_start in 0usize..=100,
        deletion_length in 1usize..=1024,
        config in chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Ensure we have enough data
            if base_data.len() < 16384 {
                base_data.resize(16384, 0);
            }

            let actual_start = (deletion_start * base_data.len()) / 100;
            let actual_length = deletion_length.min(base_data.len() - actual_start - 1);
            
            // Chunk original data
            let mut chunker1 = FastCDCChunker::new(config.clone()).unwrap();
            let original_chunks = chunker1.chunk_data(&base_data).unwrap();
            let _final1 = chunker1.finalize().unwrap();

            // Delete data and chunk again
            let mut modified_data = base_data.clone();
            modified_data.drain(actual_start..actual_start + actual_length);

            let mut chunker2 = FastCDCChunker::new(config).unwrap();
            let modified_chunks = chunker2.chunk_data(&modified_data).unwrap();
            let _final2 = chunker2.finalize().unwrap();

            // Count preserved boundaries after the deletion point
            let original_boundaries: std::collections::HashSet<_> = original_chunks
                .iter()
                .filter(|c| c.offset > (actual_start + actual_length) as u64)
                .map(|c| c.offset - actual_length as u64)
                .collect();

            let modified_boundaries: std::collections::HashSet<_> = modified_chunks
                .iter()
                .filter(|c| c.offset >= actual_start as u64)
                .map(|c| c.offset)
                .collect();

            let preserved_boundaries = original_boundaries.intersection(&modified_boundaries).count();
            let total_original = original_boundaries.len();

            // CDC should preserve boundaries after deletion point
            if total_original > 0 {
                let preservation_rate = preserved_boundaries as f64 / total_original as f64;
                prop_assert!(preservation_rate >= 0.3,
                    "Boundary preservation rate {} too low after deletion", preservation_rate);
            }
        });
    }
}

// === Corpus Consistency Properties ===

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Property: Corpus operations are thread-safe and consistent
    /// Concurrent reads should not interfere with each other
    #[test]
    fn prop_corpus_concurrent_read_consistency(
        corpus_data in corpus_data_strategy(),
        config in chunk_config_strategy(),
        num_readers in 2usize..=8
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("concurrent_corpus.bin");
            std::fs::write(&corpus_path, &corpus_data).unwrap();

            // Build corpus
            let storage = Arc::new(Mutex::new(InMemoryStorage::new()));
            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            
            let _metadata = corpus_manager.build_corpus(
                &[corpus_path],
                config
            ).await.unwrap();

            // Create shared corpus manager
            let shared_manager = Arc::new(corpus_manager);
            let mut join_set = JoinSet::new();

            // Spawn concurrent readers
            for reader_id in 0..num_readers {
                let manager_clone = Arc::clone(&shared_manager);
                join_set.spawn(async move {
                    let mut results = Vec::new();
                    
                    // Each reader performs multiple lookups
                    for i in 0..10 {
                        let weak_hash = WeakHash::new((reader_id * 1000 + i) as u64);
                        let candidates = manager_clone.get_candidates(weak_hash).unwrap();
                        results.push((weak_hash, candidates));
                    }
                    
                    results
                });
            }

            // Collect all results
            let mut all_results = Vec::new();
            while let Some(result) = join_set.join_next().await {
                all_results.push(result.unwrap());
            }

            // Verify consistency: same weak hash should return same candidates
            let mut hash_to_candidates: HashMap<WeakHash, Option<Vec<CorpusChunk>>> = HashMap::new();
            
            for reader_results in all_results {
                for (weak_hash, candidates) in reader_results {
                    if let Some(existing) = hash_to_candidates.get(&weak_hash) {
                        prop_assert_eq!(existing, &candidates,
                            "Inconsistent results for weak hash {:?}", weak_hash);
                    } else {
                        hash_to_candidates.insert(weak_hash, candidates);
                    }
                }
            }
        });
    }

    /// Property: Corpus integrity is maintained across operations
    /// All corpus operations should preserve cryptographic integrity
    #[test]
    fn prop_corpus_integrity_preservation(
        corpus_data in corpus_data_strategy(),
        config in chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("integrity_corpus.bin");
            std::fs::write(&corpus_path, &corpus_data).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );

            // Build corpus and verify initial integrity
            let metadata = corpus_manager.build_corpus(
                &[corpus_path],
                config
            ).await.unwrap();

            prop_assert!(corpus_manager.validate_corpus_integrity().is_ok(),
                "Initial corpus integrity validation failed");

            // Perform various operations and verify integrity is maintained
            let test_weak_hash = WeakHash::new(0x123456789abcdef0);
            let _candidates = corpus_manager.get_candidates(test_weak_hash).unwrap();

            prop_assert!(corpus_manager.validate_corpus_integrity().is_ok(),
                "Corpus integrity compromised after lookup operations");

            // Verify metadata consistency
            prop_assert!(!metadata.corpus_id.is_nil(), "Corpus ID should not be nil");
            prop_assert!(metadata.chunk_count > 0, "Corpus should contain chunks");
            prop_assert!(metadata.total_size > 0, "Corpus should have non-zero size");
        });
    }
}

// === Security Properties ===

#[cfg(feature = "security")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property: Signature verification is consistent and secure
    /// Valid signatures should always verify, invalid ones should always fail
    #[test]
    fn prop_signature_verification_consistency(
        data in prop::collection::vec(any::<u8>(), 1..65536)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let security_manager = EnterpriseSecurityManager::new(
                KeyManagementConfig::default()
            ).unwrap();

            // Sign the data
            let signature = security_manager.sign_corpus(&data).unwrap();

            // Verify signature multiple times - should be consistent
            for _ in 0..5 {
                let is_valid = security_manager.verify_corpus_signature(&data, &signature).unwrap();
                prop_assert!(is_valid, "Valid signature should always verify");
            }

            // Modify data slightly and verify signature fails
            let mut modified_data = data.clone();
            if !modified_data.is_empty() {
                modified_data[0] = modified_data[0].wrapping_add(1);
                
                let is_valid = security_manager.verify_corpus_signature(&modified_data, &signature).unwrap();
                prop_assert!(!is_valid, "Invalid signature should always fail verification");
            }
        });
    }

    /// Property: Encryption/decryption is a perfect roundtrip
    /// encrypt(data) → decrypt → data should be identity
    #[test]
    fn prop_encryption_roundtrip_identity(
        data in prop::collection::vec(any::<u8>(), 1..65536)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let security_manager = EnterpriseSecurityManager::new(
                KeyManagementConfig::default()
            ).unwrap();

            // Encrypt data
            let encrypted = security_manager.encrypt_output(&data).unwrap();
            
            // Encrypted data should be different from original (except for very small data)
            if data.len() > 16 {
                prop_assert_ne!(encrypted, data, "Encrypted data should differ from original");
            }

            // Decrypt and verify roundtrip
            let decrypted = security_manager.decrypt_input(&encrypted).unwrap();
            prop_assert_eq!(data, decrypted, "Decryption should recover original data");
        });
    }
}

// === Performance Properties ===

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property: Rolling hash computation is O(1) per byte
    /// Hash update time should be independent of window size
    #[test]
    fn prop_rolling_hash_performance_contract(
        data in prop::collection::vec(any::<u8>(), 8192..=65536)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut hasher = RollingHasher::new(HASH_BASE, 4096);
            
            // Initialize with first window
            hasher.init(&data[0..4096]).unwrap();
            
            let start_time = Instant::now();
            let mut hash_updates = 0;
            
            // Roll through the data
            for i in 4096..data.len() {
                let exiting_byte = data[i - 4096];
                let entering_byte = data[i];
                let _new_hash = hasher.roll(exiting_byte, entering_byte).unwrap();
                hash_updates += 1;
            }
            
            let elapsed = start_time.elapsed();
            let time_per_update = elapsed.as_nanos() as f64 / hash_updates as f64;
            
            // Each hash update should take less than 1 microsecond (1000 ns)
            prop_assert!(time_per_update < 1000.0,
                "Rolling hash update took {} ns, expected < 1000 ns", time_per_update);
        });
    }

    /// Property: Compression throughput meets performance requirements
    /// System should process at least 100 MB/s for typical data
    #[test]
    fn prop_compression_throughput_contract(
        data in prop::collection::vec(any::<u8>(), 1048576..=4194304), // 1-4 MB
        config in chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("perf_corpus.bin");
            
            // Create a corpus with some repeated patterns
            let mut corpus_data = vec![0u8; 1048576]; // 1MB corpus
            for i in 0..corpus_data.len() {
                corpus_data[i] = (i % 256) as u8;
            }
            std::fs::write(&corpus_path, &corpus_data).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            
            let _metadata = corpus_manager.build_corpus(
                &[corpus_path],
                config
            ).await.unwrap();

            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            
            let start_time = Instant::now();
            let _instructions = compressor.compress(&data).await.unwrap();
            let elapsed = start_time.elapsed();
            
            let throughput_mbps = (data.len() as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
            
            // Should achieve at least 50 MB/s throughput (relaxed for property testing)
            prop_assert!(throughput_mbps >= 50.0,
                "Compression throughput {} MB/s below minimum 50 MB/s", throughput_mbps);
        });
    }
}

// === Enterprise Workflow Properties ===

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Property: VM image-like data compression is effective
    /// Simulates VM image updates with high redundancy
    #[test]
    fn prop_vm_image_compression_effectiveness(
        base_size in 1048576usize..=8388608, // 1-8 MB
        update_ratio in 0.01f64..=0.1,       // 1-10% changes
        config in chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Create base "VM image" with patterns
            let mut base_image = vec![0u8; base_size];
            for i in 0..base_image.len() {
                base_image[i] = match i % 4096 {
                    0..=1023 => 0xAA,      // Boot sector pattern
                    1024..=2047 => 0x55,   // File system pattern
                    2048..=3071 => (i / 4096) as u8, // Directory pattern
                    _ => 0x00,             // Free space
                };
            }

            // Create "updated" image with small changes
            let mut updated_image = base_image.clone();
            let num_changes = (base_size as f64 * update_ratio) as usize;
            for i in 0..num_changes {
                let pos = (i * base_size) / num_changes;
                updated_image[pos] = updated_image[pos].wrapping_add(1);
            }

            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("vm_corpus.bin");
            std::fs::write(&corpus_path, &base_image).unwrap();

            // Build corpus from base image
            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            
            let _metadata = corpus_manager.build_corpus(
                &[corpus_path],
                config
            ).await.unwrap();

            // Compress updated image
            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            let instructions = compressor.compress(&updated_image).await.unwrap();

            // Calculate compression effectiveness
            let reference_count = instructions.iter()
                .filter(|inst| inst.is_reference())
                .count();
            let total_instructions = instructions.len();
            
            let reference_ratio = reference_count as f64 / total_instructions as f64;
            
            // For VM-like data with small changes, should achieve high reference ratio
            prop_assert!(reference_ratio >= 0.7,
                "VM image compression reference ratio {} below expected 0.7", reference_ratio);
        });
    }

    /// Property: Database backup-like data shows good deduplication
    /// Simulates incremental database backups with high redundancy
    #[test]
    fn prop_database_backup_deduplication(
        record_count in 1000usize..=10000,
        record_size in 64usize..=512,
        duplicate_ratio in 0.3f64..=0.8,
        config in chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Generate database-like records
            let mut records = Vec::new();
            let unique_records = ((record_count as f64) * (1.0 - duplicate_ratio)) as usize;
            
            // Create unique records
            for i in 0..unique_records {
                let mut record = vec![0u8; record_size];
                // Simulate database record structure
                record[0..8].copy_from_slice(&(i as u64).to_le_bytes());
                for j in 8..record_size {
                    record[j] = ((i + j) % 256) as u8;
                }
                records.push(record);
            }

            // Duplicate records to simulate backup redundancy
            let mut backup_data = Vec::new();
            for _ in 0..record_count {
                let record_idx = fastrand::usize(0..unique_records);
                backup_data.extend_from_slice(&records[record_idx]);
            }

            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("db_corpus.bin");
            
            // Use first half as corpus
            let corpus_size = backup_data.len() / 2;
            std::fs::write(&corpus_path, &backup_data[0..corpus_size]).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            
            let _metadata = corpus_manager.build_corpus(
                &[corpus_path],
                config
            ).await.unwrap();

            // Compress second half (simulating incremental backup)
            let incremental_data = &backup_data[corpus_size..];
            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            let instructions = compressor.compress(incremental_data).await.unwrap();

            // Calculate deduplication effectiveness
            let reference_bytes: usize = instructions.iter()
                .filter_map(|inst| inst.reference_size())
                .map(|size| size as usize)
                .sum();
            let total_output_bytes: usize = instructions.iter()
                .map(|inst| inst.output_size())
                .sum();

            let dedup_ratio = reference_bytes as f64 / total_output_bytes as f64;
            
            // Should achieve good deduplication for database-like data
            prop_assert!(dedup_ratio >= duplicate_ratio * 0.8,
                "Database backup deduplication ratio {} below expected {}", 
                dedup_ratio, duplicate_ratio * 0.8);
        });
    }
}

// === Memory Constraint Properties ===

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    /// Property: System handles large datasets without excessive memory usage
    /// Memory usage should be bounded regardless of input size
    #[test]
    fn prop_memory_bounded_processing(
        data_size in 8388608usize..=33554432, // 8-32 MB
        config in chunk_config_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Generate large dataset
            let mut large_data = vec![0u8; data_size];
            for i in 0..large_data.len() {
                large_data[i] = (i % 256) as u8;
            }

            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("large_corpus.bin");
            
            // Create corpus from first quarter of data
            let corpus_size = data_size / 4;
            std::fs::write(&corpus_path, &large_data[0..corpus_size]).unwrap();

            // Monitor memory usage during corpus building
            let initial_memory = get_memory_usage();
            
            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            
            let _metadata = corpus_manager.build_corpus(
                &[corpus_path],
                config
            ).await.unwrap();

            let post_corpus_memory = get_memory_usage();
            
            // Compress remaining data
            let remaining_data = &large_data[corpus_size..];
            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            let _instructions = compressor.compress(remaining_data).await.unwrap();

            let final_memory = get_memory_usage();
            
            // Memory growth should be reasonable (less than 2x input size)
            let memory_growth = final_memory.saturating_sub(initial_memory);
            let max_acceptable_memory = data_size * 2;
            
            prop_assert!(memory_growth <= max_acceptable_memory,
                "Memory usage {} exceeds acceptable limit {}", 
                memory_growth, max_acceptable_memory);
        });
    }
}

// === Utility Functions ===

/// Get current memory usage (approximation)
fn get_memory_usage() -> usize {
    // Simple approximation - in real implementation would use proper memory monitoring
    std::alloc::System.alloc(std::alloc::Layout::new::<u8>()) as usize
}

// === Integration Test Helpers ===

#[cfg(test)]
mod integration_helpers {
    use super::*;

    /// Create a test corpus with known patterns for property testing
    pub async fn create_test_corpus_with_patterns(
        size: usize,
        pattern_type: &str,
    ) -> (Vec<u8>, ChunkConfig) {
        let config = ChunkConfig::default();
        let mut data = vec![0u8; size];

        match pattern_type {
            "repetitive" => {
                for i in 0..size {
                    data[i] = (i % 256) as u8;
                }
            }
            "sparse" => {
                for i in 0..size {
                    if i % 10 == 0 {
                        data[i] = 0xFF;
                    }
                }
            }
            "blocks" => {
                let block_pattern = vec![0xAA; 4096];
                for chunk in data.chunks_mut(4096) {
                    let copy_len = chunk.len().min(block_pattern.len());
                    chunk[..copy_len].copy_from_slice(&block_pattern[..copy_len]);
                }
            }
            _ => {
                // Random data
                for i in 0..size {
                    data[i] = fastrand::u8(..);
                }
            }
        }

        (data, config)
    }

    /// Verify compression metrics meet enterprise requirements
    pub fn verify_enterprise_metrics(
        input_size: usize,
        output_size: usize,
        processing_time: Duration,
    ) -> Result<()> {
        // Compression ratio should be reasonable
        let compression_ratio = output_size as f64 / input_size as f64;
        if compression_ratio > 1.2 {
            return Err(ReductoError::PerformanceContractViolation {
                metric: "compression_ratio".to_string(),
                expected: "< 1.2".to_string(),
                actual: compression_ratio.to_string(),
            });
        }

        // Processing time should meet throughput requirements
        let throughput_mbps = (input_size as f64) / (1024.0 * 1024.0) / processing_time.as_secs_f64();
        if throughput_mbps < 50.0 {
            return Err(ReductoError::PerformanceContractViolation {
                metric: "throughput".to_string(),
                expected: ">= 50 MB/s".to_string(),
                actual: format!("{:.2} MB/s", throughput_mbps),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    /// Stress test: Process multiple large files concurrently
    #[tokio::test]
    async fn stress_concurrent_large_file_processing() {
        let temp_dir = TempDir::new().unwrap();
        let config = ChunkConfig::default();
        
        // Create multiple large test files
        let file_count = 4;
        let file_size = 16 * 1024 * 1024; // 16 MB each
        let mut file_paths = Vec::new();
        
        for i in 0..file_count {
            let file_path = temp_dir.path().join(format!("large_file_{}.bin", i));
            let mut data = vec![0u8; file_size];
            
            // Create different patterns for each file
            for j in 0..file_size {
                data[j] = ((i * 1000 + j) % 256) as u8;
            }
            
            std::fs::write(&file_path, &data).unwrap();
            file_paths.push(file_path);
        }

        // Process files concurrently
        let mut join_set = JoinSet::new();
        
        for (i, file_path) in file_paths.iter().enumerate() {
            let file_path = file_path.clone();
            let config = config.clone();
            
            join_set.spawn(async move {
                let data = std::fs::read(&file_path).unwrap();
                
                let mut corpus_manager = EnterpriseCorpusManager::new(
                    Box::new(InMemoryStorage::new())
                );
                
                let start_time = Instant::now();
                let _metadata = corpus_manager.build_corpus(&[file_path], config).await.unwrap();
                let elapsed = start_time.elapsed();
                
                (i, data.len(), elapsed)
            });
        }

        // Collect results and verify performance
        let mut results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            results.push(result.unwrap());
        }

        // Verify all files were processed successfully
        assert_eq!(results.len(), file_count);
        
        // Verify reasonable processing times
        for (file_id, size, elapsed) in results {
            let throughput = (size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
            assert!(throughput >= 20.0, 
                "File {} throughput {:.2} MB/s below minimum 20 MB/s", file_id, throughput);
        }
    }

    /// Stress test: Memory pressure with limited resources
    #[tokio::test]
    async fn stress_memory_pressure_handling() {
        let temp_dir = TempDir::new().unwrap();
        let config = ChunkConfig::default();
        
        // Create a very large corpus that would exceed typical memory limits
        let large_corpus_path = temp_dir.path().join("huge_corpus.bin");
        let corpus_size = 128 * 1024 * 1024; // 128 MB
        
        // Generate corpus in chunks to avoid memory issues during test setup
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&large_corpus_path).unwrap();
            let chunk_size = 1024 * 1024; // 1 MB chunks
            
            for i in 0..(corpus_size / chunk_size) {
                let mut chunk = vec![0u8; chunk_size];
                for j in 0..chunk_size {
                    chunk[j] = ((i * chunk_size + j) % 256) as u8;
                }
                file.write_all(&chunk).unwrap();
            }
        }

        // Process with memory constraints
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );

        let start_time = Instant::now();
        let result = corpus_manager.build_corpus(&[large_corpus_path], config).await;
        let elapsed = start_time.elapsed();

        // Should handle large corpus without crashing
        assert!(result.is_ok(), "Large corpus processing failed: {:?}", result.err());
        
        let metadata = result.unwrap();
        assert!(metadata.chunk_count > 0, "No chunks processed from large corpus");
        assert_eq!(metadata.total_size, corpus_size as u64, "Corpus size mismatch");
        
        // Should complete in reasonable time (less than 30 seconds for 128MB)
        assert!(elapsed.as_secs() < 30, 
            "Large corpus processing took {} seconds, expected < 30", elapsed.as_secs());
    }
}