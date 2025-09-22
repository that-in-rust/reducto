//! Enterprise Workflow Property Tests
//!
//! This module tests realistic enterprise scenarios with property-based testing:
//! - VM image update workflows with high redundancy
//! - CI/CD artifact compression and distribution
//! - Database backup deduplication scenarios
//! - Large-scale corpus management workflows
//! - Multi-tenant security and isolation

use proptest::prelude::*;
use reducto_mode_3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::task::JoinSet;

// === Enterprise Workflow Test Strategies ===

/// Generate VM image-like data with realistic patterns
fn vm_image_strategy() -> impl Strategy<Value = (Vec<u8>, f64)> {
    (
        1048576usize..=8388608, // 1-8 MB VM images
        0.01f64..=0.15,         // 1-15% change rate for updates
    ).prop_map(|(size, change_rate)| {
        let mut vm_image = vec![0u8; size];
        
        // Simulate VM image structure
        for (i, byte) in vm_image.iter_mut().enumerate() {
            *byte = match i % 4096 {
                // Boot sector pattern
                0..=511 => 0xAA,
                // File system metadata
                512..=1023 => (i / 512) as u8,
                // Application data (more varied)
                1024..=3071 => ((i * 7) % 256) as u8,
                // Free space (mostly zeros)
                _ => if i % 16 == 0 { 0xFF } else { 0x00 },
            };
        }
        
        (vm_image, change_rate)
    })
}

/// Generate CI/CD artifact-like data
fn cicd_artifact_strategy() -> impl Strategy<Value = Vec<Vec<u8>>> {
    (
        2usize..=8,           // 2-8 artifacts in a build
        65536usize..=524288,  // 64KB-512KB per artifact
    ).prop_map(|(num_artifacts, base_size)| {
        let mut artifacts = Vec::new();
        
        // Common shared libraries/dependencies
        let shared_lib = vec![0xBB; base_size / 4];
        
        for i in 0..num_artifacts {
            let mut artifact = vec![0u8; base_size + (i * 1024)];
            
            // Include shared library in each artifact
            artifact[0..shared_lib.len()].copy_from_slice(&shared_lib);
            
            // Add artifact-specific code
            for j in shared_lib.len()..artifact.len() {
                artifact[j] = ((i * 100 + j) % 256) as u8;
            }
            
            artifacts.push(artifact);
        }
        
        artifacts
    })
}

/// Generate database backup-like data with high redundancy
fn database_backup_strategy() -> impl Strategy<Value = (Vec<u8>, Vec<u8>)> {
    (
        1000usize..=5000,     // 1K-5K records
        128usize..=512,       // 128-512 bytes per record
        0.7f64..=0.95,        // 70-95% data overlap between backups
    ).prop_map(|(num_records, record_size, overlap_ratio)| {
        let mut base_backup = Vec::new();
        let mut incremental_backup = Vec::new();
        
        // Generate base records
        let mut base_records = Vec::new();
        for i in 0..num_records {
            let mut record = vec![0u8; record_size];
            
            // Record header (ID, timestamp, etc.)
            record[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            
            // Record data
            for j in 8..record_size {
                record[j] = ((i * 17 + j * 3) % 256) as u8;
            }
            
            base_records.push(record);
        }
        
        // Create base backup
        for record in &base_records {
            base_backup.extend_from_slice(record);
        }
        
        // Create incremental backup with overlap
        let num_overlapping = (num_records as f64 * overlap_ratio) as usize;
        let num_new = num_records - num_overlapping;
        
        // Add overlapping records (unchanged)
        for i in 0..num_overlapping {
            incremental_backup.extend_from_slice(&base_records[i]);
        }
        
        // Add new/modified records
        for i in 0..num_new {
            let mut record = vec![0u8; record_size];
            let record_id = num_records + i;
            
            record[0..8].copy_from_slice(&(record_id as u64).to_le_bytes());
            for j in 8..record_size {
                record[j] = ((record_id * 19 + j * 5) % 256) as u8;
            }
            
            incremental_backup.extend_from_slice(&record);
        }
        
        (base_backup, incremental_backup)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property: VM image updates achieve high compression ratios
    /// VM image updates with small changes should compress very effectively
    #[test]
    fn prop_vm_image_update_compression(
        (base_image, change_rate) in vm_image_strategy(),
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("vm_base.img");

            // Create updated image with small changes
            let mut updated_image = base_image.clone();
            let num_changes = (base_image.len() as f64 * change_rate) as usize;
            
            for i in 0..num_changes {
                let pos = (i * base_image.len()) / num_changes;
                updated_image[pos] = updated_image[pos].wrapping_add(1);
            }

            // Use base image as corpus
            std::fs::write(&corpus_path, &base_image).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();

            // Compress updated image
            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            let instructions = compressor.compress(&updated_image).await.unwrap();

            // Calculate compression effectiveness
            let reference_bytes: usize = instructions.iter()
                .filter_map(|inst| inst.reference_size())
                .map(|size| size as usize)
                .sum();
            let total_output_bytes: usize = instructions.iter()
                .map(|inst| inst.output_size())
                .sum();

            let reference_ratio = reference_bytes as f64 / total_output_bytes as f64;
            let expected_min_ratio = 1.0 - (change_rate * 2.0); // Should reference most unchanged data

            prop_assert!(reference_ratio >= expected_min_ratio,
                "VM image reference ratio {:.2} below expected {:.2} for {:.1}% changes",
                reference_ratio, expected_min_ratio, change_rate * 100.0);

            // Verify compression ratio
            let compression_ratio = total_output_bytes as f64 / updated_image.len() as f64;
            prop_assert!(compression_ratio <= 0.5,
                "VM image compression ratio {:.2} should be <= 0.5 for {:.1}% changes",
                compression_ratio, change_rate * 100.0);
        });
    }

    /// Property: CI/CD artifacts show good deduplication across builds
    /// Multiple artifacts in a build should share common dependencies effectively
    #[test]
    fn prop_cicd_artifact_deduplication(
        artifacts in cicd_artifact_strategy(),
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            
            // Use first artifact as corpus (previous build)
            let corpus_path = temp_dir.path().join("previous_build.bin");
            std::fs::write(&corpus_path, &artifacts[0]).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();
            let shared_corpus = Arc::new(corpus_manager);

            // Compress remaining artifacts (current build)
            let mut total_input_size = 0;
            let mut total_reference_bytes = 0;
            let mut total_output_size = 0;

            for artifact in &artifacts[1..] {
                let mut compressor = Compressor::new(Arc::clone(&shared_corpus));
                let instructions = compressor.compress(artifact).await.unwrap();

                total_input_size += artifact.len();
                
                let reference_bytes: usize = instructions.iter()
                    .filter_map(|inst| inst.reference_size())
                    .map(|size| size as usize)
                    .sum();
                total_reference_bytes += reference_bytes;

                let output_size: usize = instructions.iter()
                    .map(|inst| inst.output_size())
                    .sum();
                total_output_size += output_size;
            }

            // Calculate deduplication effectiveness
            let dedup_ratio = total_reference_bytes as f64 / total_output_size as f64;
            
            // Should achieve good deduplication (at least 40% references)
            prop_assert!(dedup_ratio >= 0.4,
                "CI/CD artifact deduplication ratio {:.2} below expected 0.4", dedup_ratio);

            // Overall compression should be effective
            let compression_ratio = total_output_size as f64 / total_input_size as f64;
            prop_assert!(compression_ratio <= 0.8,
                "CI/CD artifact compression ratio {:.2} should be <= 0.8", compression_ratio);
        });
    }

    /// Property: Database backup incremental compression is highly effective
    /// Incremental backups should achieve very high compression ratios
    #[test]
    fn prop_database_backup_incremental_compression(
        (base_backup, incremental_backup) in database_backup_strategy(),
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("base_backup.db");

            // Use base backup as corpus
            std::fs::write(&corpus_path, &base_backup).unwrap();

            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            let _metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();

            // Compress incremental backup
            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            let instructions = compressor.compress(&incremental_backup).await.unwrap();

            // Calculate compression metrics
            let reference_bytes: usize = instructions.iter()
                .filter_map(|inst| inst.reference_size())
                .map(|size| size as usize)
                .sum();
            let residual_bytes: usize = instructions.iter()
                .filter_map(|inst| match inst {
                    ReductoInstruction::Residual(data) => Some(data.len()),
                    _ => None,
                })
                .sum();
            let total_output = reference_bytes + residual_bytes;

            let reference_ratio = reference_bytes as f64 / total_output as f64;
            let compression_ratio = total_output as f64 / incremental_backup.len() as f64;

            // Database backups should achieve high reference ratios (>60%)
            prop_assert!(reference_ratio >= 0.6,
                "Database backup reference ratio {:.2} below expected 0.6", reference_ratio);

            // Should achieve excellent compression ratios (<0.4)
            prop_assert!(compression_ratio <= 0.4,
                "Database backup compression ratio {:.2} should be <= 0.4", compression_ratio);

            // Verify correctness
            prop_assert_eq!(total_output, incremental_backup.len(),
                "Output size should match input size");
        });
    }

    /// Property: Multi-tenant corpus isolation and performance
    /// Multiple tenants should be isolated while maintaining performance
    #[test]
    fn prop_multi_tenant_isolation(
        num_tenants in 2usize..=5,
        data_size_per_tenant in 262144usize..=1048576, // 256KB-1MB per tenant
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            
            // Create separate corpus for each tenant
            let mut tenant_corpora = HashMap::new();
            
            for tenant_id in 0..num_tenants {
                let corpus_path = temp_dir.path().join(format!("tenant_{}_corpus.bin", tenant_id));
                
                // Create tenant-specific data pattern
                let mut corpus_data = vec![0u8; data_size_per_tenant];
                for (i, byte) in corpus_data.iter_mut().enumerate() {
                    *byte = ((tenant_id * 100 + i) % 256) as u8;
                }
                std::fs::write(&corpus_path, &corpus_data).unwrap();

                let mut corpus_manager = EnterpriseCorpusManager::new(
                    Box::new(InMemoryStorage::new())
                );
                let metadata = corpus_manager.build_corpus(&[corpus_path], config.clone()).await.unwrap();
                
                tenant_corpora.insert(tenant_id, (Arc::new(corpus_manager), metadata));
            }

            // Test compression for each tenant with their own and others' data
            let mut isolation_results = Vec::new();
            
            for tenant_id in 0..num_tenants {
                let (tenant_corpus, _) = &tenant_corpora[&tenant_id];
                
                // Create test data similar to tenant's corpus
                let mut tenant_data = vec![0u8; data_size_per_tenant / 2];
                for (i, byte) in tenant_data.iter_mut().enumerate() {
                    *byte = ((tenant_id * 100 + i) % 256) as u8;
                }

                // Compress with own corpus (should be effective)
                let mut own_compressor = Compressor::new(Arc::clone(tenant_corpus));
                let own_instructions = own_compressor.compress(&tenant_data).await.unwrap();
                
                let own_reference_bytes: usize = own_instructions.iter()
                    .filter_map(|inst| inst.reference_size())
                    .map(|size| size as usize)
                    .sum();
                let own_total_output: usize = own_instructions.iter()
                    .map(|inst| inst.output_size())
                    .sum();
                let own_reference_ratio = own_reference_bytes as f64 / own_total_output as f64;

                // Compress with another tenant's corpus (should be less effective)
                let other_tenant_id = (tenant_id + 1) % num_tenants;
                let (other_corpus, _) = &tenant_corpora[&other_tenant_id];
                
                let mut other_compressor = Compressor::new(Arc::clone(other_corpus));
                let other_instructions = other_compressor.compress(&tenant_data).await.unwrap();
                
                let other_reference_bytes: usize = other_instructions.iter()
                    .filter_map(|inst| inst.reference_size())
                    .map(|size| size as usize)
                    .sum();
                let other_total_output: usize = other_instructions.iter()
                    .map(|inst| inst.output_size())
                    .sum();
                let other_reference_ratio = other_reference_bytes as f64 / other_total_output as f64;

                isolation_results.push((tenant_id, own_reference_ratio, other_reference_ratio));
            }

            // Verify isolation: own corpus should be more effective than others'
            for (tenant_id, own_ratio, other_ratio) in isolation_results {
                prop_assert!(own_ratio > other_ratio,
                    "Tenant {} isolation failed: own ratio {:.2} <= other ratio {:.2}",
                    tenant_id, own_ratio, other_ratio);
                
                // Own corpus should achieve reasonable compression
                prop_assert!(own_ratio >= 0.3,
                    "Tenant {} own corpus ratio {:.2} below expected 0.3",
                    tenant_id, own_ratio);
            }
        });
    }

    /// Property: Large-scale corpus management workflow
    /// Building and managing very large corpora should work efficiently
    #[test]
    fn prop_large_scale_corpus_management(
        num_source_files in 5usize..=15,
        file_size in 1048576usize..=4194304, // 1-4 MB per file
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let mut source_files = Vec::new();

            // Create multiple source files with some overlap
            for i in 0..num_source_files {
                let file_path = temp_dir.path().join(format!("source_{}.bin", i));
                let mut file_data = vec![0u8; file_size];
                
                // Create patterns with some overlap between files
                for (j, byte) in file_data.iter_mut().enumerate() {
                    *byte = match j % 2048 {
                        // Shared pattern across files
                        0..=1023 => ((j / 64) % 256) as u8,
                        // File-specific pattern
                        _ => ((i * 50 + j) % 256) as u8,
                    };
                }
                
                std::fs::write(&file_path, &file_data).unwrap();
                source_files.push(file_path);
            }

            // Build large corpus from all source files
            let start_time = std::time::Instant::now();
            
            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            let metadata = corpus_manager.build_corpus(&source_files, config).await.unwrap();
            
            let build_time = start_time.elapsed();

            // Verify corpus was built successfully
            prop_assert!(metadata.chunk_count > 0, "Corpus should contain chunks");
            prop_assert!(metadata.total_size > 0, "Corpus should have non-zero size");
            
            let expected_total_size = (num_source_files * file_size) as u64;
            prop_assert!(metadata.total_size <= expected_total_size,
                "Corpus size {} should not exceed input size {}", 
                metadata.total_size, expected_total_size);

            // Build time should be reasonable (less than 30 seconds for up to 60MB)
            prop_assert!(build_time.as_secs() <= 30,
                "Corpus build took {}s, expected <= 30s", build_time.as_secs());

            // Test corpus integrity
            corpus_manager.validate_corpus_integrity().unwrap();

            // Test compression with the large corpus
            let test_data = vec![0xAA; file_size / 2];
            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            
            let compress_start = std::time::Instant::now();
            let instructions = compressor.compress(&test_data).await.unwrap();
            let compress_time = compress_start.elapsed();

            // Compression should complete quickly
            prop_assert!(compress_time.as_secs() <= 5,
                "Compression took {}s, expected <= 5s", compress_time.as_secs());

            // Should produce valid instructions
            prop_assert!(!instructions.is_empty(), "Should produce compression instructions");
            
            let total_output: usize = instructions.iter().map(|i| i.output_size()).sum();
            prop_assert_eq!(total_output, test_data.len(), "Output size should match input");
        });
    }
}

// === Enterprise Integration Tests ===

#[cfg(test)]
mod enterprise_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_vm_image_update_workflow() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create base VM image (8MB)
        let base_image_size = 8 * 1024 * 1024;
        let mut base_image = vec![0u8; base_image_size];
        
        // Simulate VM image structure
        for (i, byte) in base_image.iter_mut().enumerate() {
            *byte = match i % 4096 {
                0..=511 => 0xAA,      // Boot sector
                512..=1023 => 0x55,   // File system
                1024..=2047 => (i / 1024) as u8, // Directories
                _ => 0x00,            // Free space
            };
        }

        // Create updated image with 2% changes
        let mut updated_image = base_image.clone();
        let num_changes = base_image_size / 50; // 2% changes
        
        for i in 0..num_changes {
            let pos = (i * base_image_size) / num_changes;
            updated_image[pos] = updated_image[pos].wrapping_add(1);
        }

        // Build corpus from base image
        let corpus_path = temp_dir.path().join("base_vm.img");
        std::fs::write(&corpus_path, &base_image).unwrap();

        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );
        let metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();

        // Compress updated image
        let mut compressor = Compressor::new(Arc::new(corpus_manager));
        let instructions = compressor.compress(&updated_image).await.unwrap();

        // Verify compression effectiveness
        let reference_bytes: usize = instructions.iter()
            .filter_map(|inst| inst.reference_size())
            .map(|size| size as usize)
            .sum();
        let total_output: usize = instructions.iter()
            .map(|inst| inst.output_size())
            .sum();

        let reference_ratio = reference_bytes as f64 / total_output as f64;
        let compression_ratio = total_output as f64 / updated_image.len() as f64;

        assert!(reference_ratio >= 0.95, 
            "VM image should achieve >95% reference ratio, got {:.2}", reference_ratio);
        assert!(compression_ratio <= 0.1,
            "VM image should achieve <10% compression ratio, got {:.2}", compression_ratio);

        println!("VM Image Update Results:");
        println!("  Reference ratio: {:.2}%", reference_ratio * 100.0);
        println!("  Compression ratio: {:.2}%", compression_ratio * 100.0);
        println!("  Corpus chunks: {}", metadata.chunk_count);
    }

    #[tokio::test]
    async fn test_cicd_pipeline_workflow() {
        let temp_dir = TempDir::new().unwrap();
        
        // Simulate CI/CD pipeline with multiple builds
        let num_builds = 3;
        let artifacts_per_build = 4;
        let artifact_size = 256 * 1024; // 256KB per artifact

        // Shared library that appears in all artifacts
        let shared_lib = vec![0xBB; artifact_size / 4];
        
        let mut build_results = Vec::new();

        for build_id in 0..num_builds {
            let build_dir = temp_dir.path().join(format!("build_{}", build_id));
            std::fs::create_dir_all(&build_dir).unwrap();

            // Create artifacts for this build
            let mut build_artifacts = Vec::new();
            for artifact_id in 0..artifacts_per_build {
                let mut artifact = vec![0u8; artifact_size];
                
                // Include shared library
                artifact[0..shared_lib.len()].copy_from_slice(&shared_lib);
                
                // Add build and artifact specific code
                for i in shared_lib.len()..artifact.len() {
                    artifact[i] = ((build_id * 100 + artifact_id * 10 + i) % 256) as u8;
                }
                
                let artifact_path = build_dir.join(format!("artifact_{}.bin", artifact_id));
                std::fs::write(&artifact_path, &artifact).unwrap();
                build_artifacts.push(artifact_path);
            }

            // Use previous build as corpus (if available)
            if build_id > 0 {
                let prev_build_dir = temp_dir.path().join(format!("build_{}", build_id - 1));
                let corpus_files: Vec<_> = std::fs::read_dir(&prev_build_dir).unwrap()
                    .map(|entry| entry.unwrap().path())
                    .collect();

                let config = ChunkConfig::default();
                let mut corpus_manager = EnterpriseCorpusManager::new(
                    Box::new(InMemoryStorage::new())
                );
                let _metadata = corpus_manager.build_corpus(&corpus_files, config).await.unwrap();
                let shared_corpus = Arc::new(corpus_manager);

                // Compress current build artifacts
                let mut total_input_size = 0;
                let mut total_output_size = 0;
                let mut total_reference_bytes = 0;

                for artifact_path in &build_artifacts {
                    let artifact_data = std::fs::read(artifact_path).unwrap();
                    let mut compressor = Compressor::new(Arc::clone(&shared_corpus));
                    let instructions = compressor.compress(&artifact_data).await.unwrap();

                    total_input_size += artifact_data.len();
                    
                    let reference_bytes: usize = instructions.iter()
                        .filter_map(|inst| inst.reference_size())
                        .map(|size| size as usize)
                        .sum();
                    total_reference_bytes += reference_bytes;

                    let output_size: usize = instructions.iter()
                        .map(|inst| inst.output_size())
                        .sum();
                    total_output_size += output_size;
                }

                let reference_ratio = total_reference_bytes as f64 / total_output_size as f64;
                let compression_ratio = total_output_size as f64 / total_input_size as f64;

                build_results.push((build_id, reference_ratio, compression_ratio));
            }
        }

        // Verify CI/CD compression effectiveness
        for (build_id, reference_ratio, compression_ratio) in build_results {
            assert!(reference_ratio >= 0.4,
                "Build {} should achieve >40% reference ratio, got {:.2}", 
                build_id, reference_ratio);
            assert!(compression_ratio <= 0.7,
                "Build {} should achieve <70% compression ratio, got {:.2}", 
                build_id, compression_ratio);

            println!("Build {} Results:", build_id);
            println!("  Reference ratio: {:.2}%", reference_ratio * 100.0);
            println!("  Compression ratio: {:.2}%", compression_ratio * 100.0);
        }
    }

    #[tokio::test]
    async fn test_database_backup_workflow() {
        let temp_dir = TempDir::new().unwrap();
        
        // Simulate database backup scenario
        let num_records = 2000;
        let record_size = 256;
        let overlap_percentage = 85; // 85% of records unchanged

        // Generate base backup
        let mut base_records = Vec::new();
        for i in 0..num_records {
            let mut record = vec![0u8; record_size];
            
            // Record ID
            record[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            
            // Record data (simulating customer data, etc.)
            for j in 8..record_size {
                record[j] = ((i * 17 + j * 3) % 256) as u8;
            }
            
            base_records.push(record);
        }

        let mut base_backup = Vec::new();
        for record in &base_records {
            base_backup.extend_from_slice(record);
        }

        // Generate incremental backup
        let num_unchanged = (num_records * overlap_percentage) / 100;
        let num_changed = num_records - num_unchanged;

        let mut incremental_backup = Vec::new();
        
        // Add unchanged records
        for i in 0..num_unchanged {
            incremental_backup.extend_from_slice(&base_records[i]);
        }
        
        // Add changed/new records
        for i in 0..num_changed {
            let mut record = vec![0u8; record_size];
            let record_id = num_records + i;
            
            record[0..8].copy_from_slice(&(record_id as u64).to_le_bytes());
            for j in 8..record_size {
                record[j] = ((record_id * 19 + j * 5) % 256) as u8;
            }
            
            incremental_backup.extend_from_slice(&record);
        }

        // Build corpus from base backup
        let corpus_path = temp_dir.path().join("base_backup.db");
        std::fs::write(&corpus_path, &base_backup).unwrap();

        let config = ChunkConfig::default();
        let mut corpus_manager = EnterpriseCorpusManager::new(
            Box::new(InMemoryStorage::new())
        );
        let metadata = corpus_manager.build_corpus(&[corpus_path], config).await.unwrap();

        // Compress incremental backup
        let mut compressor = Compressor::new(Arc::new(corpus_manager));
        let instructions = compressor.compress(&incremental_backup).await.unwrap();

        // Analyze results
        let reference_bytes: usize = instructions.iter()
            .filter_map(|inst| inst.reference_size())
            .map(|size| size as usize)
            .sum();
        let total_output: usize = instructions.iter()
            .map(|inst| inst.output_size())
            .sum();

        let reference_ratio = reference_bytes as f64 / total_output as f64;
        let compression_ratio = total_output as f64 / incremental_backup.len() as f64;

        // Database backups should achieve excellent compression
        assert!(reference_ratio >= 0.8,
            "Database backup should achieve >80% reference ratio, got {:.2}", reference_ratio);
        assert!(compression_ratio <= 0.3,
            "Database backup should achieve <30% compression ratio, got {:.2}", compression_ratio);

        println!("Database Backup Results:");
        println!("  Base backup size: {} KB", base_backup.len() / 1024);
        println!("  Incremental backup size: {} KB", incremental_backup.len() / 1024);
        println!("  Reference ratio: {:.2}%", reference_ratio * 100.0);
        println!("  Compression ratio: {:.2}%", compression_ratio * 100.0);
        println!("  Corpus chunks: {}", metadata.chunk_count);
        println!("  Expected overlap: {}%", overlap_percentage);
    }
}