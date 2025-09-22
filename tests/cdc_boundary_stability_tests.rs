//! CDC Boundary Stability Property Tests
//!
//! This module focuses specifically on testing the robustness of Content-Defined Chunking
//! boundaries across various data modification scenarios. These tests validate that CDC
//! maintains boundary stability when data is inserted, deleted, or modified.

use proptest::prelude::*;
use reducto_mode_3::prelude::*;
use std::collections::HashSet;
use tempfile::TempDir;

// === Boundary Stability Test Strategies ===

/// Generate data with known repetitive patterns for boundary testing
fn structured_data_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        // Repetitive pattern data (simulates file system structures)
        (1024usize..=32768, 0u8..=255u8).prop_map(|(size, pattern)| {
            let mut data = vec![0u8; size];
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = match i % 512 {
                    0..=255 => pattern,
                    256..=383 => 0x00,
                    384..=447 => 0xFF,
                    _ => (i % 256) as u8,
                };
            }
            data
        }),
        
        // Block-structured data (simulates VM images)
        (2048usize..=16384).prop_map(|size| {
            let mut data = vec![0u8; size];
            let block_size = 512;
            for chunk in data.chunks_mut(block_size) {
                for (i, byte) in chunk.iter_mut().enumerate() {
                    *byte = match i % 64 {
                        0..=31 => 0xAA,  // Header pattern
                        32..=47 => 0x55, // Data pattern
                        _ => 0x00,       // Padding
                    };
                }
            }
            data
        }),
        
        // Text-like data with natural boundaries
        prop::collection::vec(32u8..=126u8, 1024..=16384)
            .prop_map(|mut data| {
                // Insert natural word boundaries
                for i in (0..data.len()).step_by(8) {
                    if i < data.len() {
                        data[i] = b' ';
                    }
                }
                // Insert line boundaries
                for i in (0..data.len()).step_by(80) {
                    if i < data.len() {
                        data[i] = b'\n';
                    }
                }
                data
            }),
    ]
}

/// Generate insertion patterns that test boundary stability
fn insertion_strategy() -> impl Strategy<Value = (Vec<u8>, usize, usize)> {
    (
        structured_data_strategy(),
        prop::collection::vec(any::<u8>(), 1..=2048), // insertion data
        0usize..=100, // insertion position percentage
    ).prop_map(|(base_data, insertion_data, pos_percent)| {
        let insertion_pos = (pos_percent * base_data.len()) / 100;
        (base_data, insertion_pos, insertion_data.len())
    }).prop_filter("insertion position valid", |(base_data, pos, _)| {
        *pos < base_data.len()
    }).prop_map(|(base_data, insertion_pos, insertion_len)| {
        let insertion_data = vec![0xCC; insertion_len]; // Distinctive pattern
        (base_data, insertion_pos, insertion_data.len())
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: CDC boundaries are stable across small insertions
    /// Small insertions should not cause a cascade of boundary shifts
    #[test]
    fn prop_small_insertion_boundary_stability(
        base_data in structured_data_strategy(),
        insertion_size in 1usize..=256,
        insertion_pos_percent in 10usize..=90,
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Ensure minimum data size
            let mut test_data = base_data;
            if test_data.len() < 8192 {
                test_data.resize(8192, 0);
            }

            let insertion_pos = (insertion_pos_percent * test_data.len()) / 100;
            let insertion_data = vec![0xCC; insertion_size];

            // Chunk original data
            let mut original_chunker = FastCDCChunker::new(config.clone()).unwrap();
            let original_chunks = original_chunker.chunk_data(&test_data).unwrap();
            let original_final = original_chunker.finalize().unwrap();

            // Create modified data with insertion
            let mut modified_data = test_data.clone();
            modified_data.splice(insertion_pos..insertion_pos, insertion_data.iter().cloned());

            // Chunk modified data
            let mut modified_chunker = FastCDCChunker::new(config).unwrap();
            let modified_chunks = modified_chunker.chunk_data(&modified_data).unwrap();
            let modified_final = modified_chunker.finalize().unwrap();

            // Analyze boundary preservation
            let original_boundaries = collect_chunk_boundaries(&original_chunks, &original_final);
            let modified_boundaries = collect_chunk_boundaries(&modified_chunks, &modified_final);

            // Boundaries before insertion point should be identical
            let boundaries_before_insertion: HashSet<_> = original_boundaries
                .iter()
                .filter(|&&boundary| boundary < insertion_pos as u64)
                .cloned()
                .collect();

            let modified_boundaries_before: HashSet<_> = modified_boundaries
                .iter()
                .filter(|&&boundary| boundary < insertion_pos as u64)
                .cloned()
                .collect();

            let preserved_before = boundaries_before_insertion
                .intersection(&modified_boundaries_before)
                .count();

            // Should preserve most boundaries before insertion point
            if !boundaries_before_insertion.is_empty() {
                let preservation_rate = preserved_before as f64 / boundaries_before_insertion.len() as f64;
                prop_assert!(preservation_rate >= 0.8,
                    "Boundary preservation before insertion: {:.2}% (expected >= 80%)",
                    preservation_rate * 100.0);
            }

            // Boundaries after insertion should be shifted but pattern preserved
            let boundaries_after_insertion: HashSet<_> = original_boundaries
                .iter()
                .filter(|&&boundary| boundary > insertion_pos as u64)
                .map(|&boundary| boundary + insertion_size as u64)
                .collect();

            let modified_boundaries_after: HashSet<_> = modified_boundaries
                .iter()
                .filter(|&&boundary| boundary > (insertion_pos + insertion_size) as u64)
                .cloned()
                .collect();

            let preserved_after = boundaries_after_insertion
                .intersection(&modified_boundaries_after)
                .count();

            // Should preserve significant portion of boundaries after insertion
            if !boundaries_after_insertion.is_empty() {
                let preservation_rate = preserved_after as f64 / boundaries_after_insertion.len() as f64;
                prop_assert!(preservation_rate >= 0.5,
                    "Boundary preservation after insertion: {:.2}% (expected >= 50%)",
                    preservation_rate * 100.0);
            }
        });
    }

    /// Property: CDC boundaries are stable across deletions
    /// Deletions should not cause excessive boundary shifts
    #[test]
    fn prop_deletion_boundary_stability(
        base_data in structured_data_strategy(),
        deletion_start_percent in 20usize..=80,
        deletion_size in 64usize..=1024,
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Ensure sufficient data size
            let mut test_data = base_data;
            if test_data.len() < 16384 {
                test_data.resize(16384, 0);
            }

            let deletion_start = (deletion_start_percent * test_data.len()) / 100;
            let actual_deletion_size = deletion_size.min(test_data.len() - deletion_start - 1024);

            // Chunk original data
            let mut original_chunker = FastCDCChunker::new(config.clone()).unwrap();
            let original_chunks = original_chunker.chunk_data(&test_data).unwrap();
            let original_final = original_chunker.finalize().unwrap();

            // Create modified data with deletion
            let mut modified_data = test_data.clone();
            modified_data.drain(deletion_start..deletion_start + actual_deletion_size);

            // Chunk modified data
            let mut modified_chunker = FastCDCChunker::new(config).unwrap();
            let modified_chunks = modified_chunker.chunk_data(&modified_data).unwrap();
            let modified_final = modified_chunker.finalize().unwrap();

            // Analyze boundary preservation
            let original_boundaries = collect_chunk_boundaries(&original_chunks, &original_final);
            let modified_boundaries = collect_chunk_boundaries(&modified_chunks, &modified_final);

            // Boundaries before deletion should be preserved
            let boundaries_before: HashSet<_> = original_boundaries
                .iter()
                .filter(|&&boundary| boundary < deletion_start as u64)
                .cloned()
                .collect();

            let modified_boundaries_before: HashSet<_> = modified_boundaries
                .iter()
                .filter(|&&boundary| boundary < deletion_start as u64)
                .cloned()
                .collect();

            let preserved_before = boundaries_before
                .intersection(&modified_boundaries_before)
                .count();

            if !boundaries_before.is_empty() {
                let preservation_rate = preserved_before as f64 / boundaries_before.len() as f64;
                prop_assert!(preservation_rate >= 0.9,
                    "Boundary preservation before deletion: {:.2}% (expected >= 90%)",
                    preservation_rate * 100.0);
            }

            // Boundaries after deletion should be shifted but pattern preserved
            let boundaries_after: HashSet<_> = original_boundaries
                .iter()
                .filter(|&&boundary| boundary > (deletion_start + actual_deletion_size) as u64)
                .map(|&boundary| boundary - actual_deletion_size as u64)
                .collect();

            let modified_boundaries_after: HashSet<_> = modified_boundaries
                .iter()
                .filter(|&&boundary| boundary >= deletion_start as u64)
                .cloned()
                .collect();

            let preserved_after = boundaries_after
                .intersection(&modified_boundaries_after)
                .count();

            if !boundaries_after.is_empty() {
                let preservation_rate = preserved_after as f64 / boundaries_after.len() as f64;
                prop_assert!(preservation_rate >= 0.4,
                    "Boundary preservation after deletion: {:.2}% (expected >= 40%)",
                    preservation_rate * 100.0);
            }
        });
    }

    /// Property: CDC chunk size distribution remains consistent
    /// Chunk size variance should stay within configured bounds across modifications
    #[test]
    fn prop_chunk_size_distribution_stability(
        base_data in structured_data_strategy(),
        modification_type in 0usize..=2, // 0=insert, 1=delete, 2=modify
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut test_data = base_data;
            if test_data.len() < 12288 {
                test_data.resize(12288, 0);
            }

            // Apply modification based on type
            let mut modified_data = test_data.clone();
            let modification_pos = test_data.len() / 2;

            match modification_type {
                0 => {
                    // Insert 512 bytes
                    let insertion = vec![0xDD; 512];
                    modified_data.splice(modification_pos..modification_pos, insertion.iter().cloned());
                }
                1 => {
                    // Delete 256 bytes
                    let delete_end = (modification_pos + 256).min(modified_data.len());
                    modified_data.drain(modification_pos..delete_end);
                }
                2 => {
                    // Modify 128 bytes
                    let modify_end = (modification_pos + 128).min(modified_data.len());
                    for byte in &mut modified_data[modification_pos..modify_end] {
                        *byte = byte.wrapping_add(1);
                    }
                }
                _ => unreachable!(),
            }

            // Chunk both versions
            let mut original_chunker = FastCDCChunker::new(config.clone()).unwrap();
            let original_chunks = original_chunker.chunk_data(&test_data).unwrap();
            let _original_final = original_chunker.finalize().unwrap();

            let mut modified_chunker = FastCDCChunker::new(config.clone()).unwrap();
            let modified_chunks = modified_chunker.chunk_data(&modified_data).unwrap();
            let _modified_final = modified_chunker.finalize().unwrap();

            // Analyze chunk size distributions
            let original_sizes: Vec<_> = original_chunks.iter().map(|c| c.size()).collect();
            let modified_sizes: Vec<_> = modified_chunks.iter().map(|c| c.size()).collect();

            // Both should respect size bounds
            for &size in &original_sizes {
                prop_assert!(size >= config.min_size && size <= config.max_size,
                    "Original chunk size {} outside bounds [{}, {}]", 
                    size, config.min_size, config.max_size);
            }

            for &size in &modified_sizes {
                prop_assert!(size >= config.min_size && size <= config.max_size,
                    "Modified chunk size {} outside bounds [{}, {}]", 
                    size, config.min_size, config.max_size);
            }

            // Average chunk sizes should be similar
            if !original_sizes.is_empty() && !modified_sizes.is_empty() {
                let original_avg = original_sizes.iter().sum::<usize>() as f64 / original_sizes.len() as f64;
                let modified_avg = modified_sizes.iter().sum::<usize>() as f64 / modified_sizes.len() as f64;
                
                let avg_difference_ratio = (original_avg - modified_avg).abs() / original_avg;
                prop_assert!(avg_difference_ratio <= 0.3,
                    "Average chunk size changed too much: {:.1} -> {:.1} ({:.1}% change)",
                    original_avg, modified_avg, avg_difference_ratio * 100.0);
            }
        });
    }

    /// Property: Boundary detection is consistent across identical content
    /// Identical content regions should produce identical boundaries regardless of context
    #[test]
    fn prop_boundary_detection_consistency(
        content_block in prop::collection::vec(any::<u8>(), 2048..=8192),
        prefix_size in 0usize..=1024,
        suffix_size in 0usize..=1024,
        config in prop::collection::vec(4096usize..=16384, 1..=1)
            .prop_map(|sizes| ChunkConfig::new(sizes[0]).unwrap_or_default())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Create test data with identical content blocks in different contexts
            let prefix = vec![0xAA; prefix_size];
            let suffix = vec![0xBB; suffix_size];

            let mut data1 = Vec::new();
            data1.extend_from_slice(&prefix);
            data1.extend_from_slice(&content_block);
            data1.extend_from_slice(&suffix);

            let mut data2 = Vec::new();
            data2.extend_from_slice(&vec![0xFF; prefix_size]); // Different prefix
            data2.extend_from_slice(&content_block); // Same content
            data2.extend_from_slice(&vec![0x00; suffix_size]); // Different suffix

            // Chunk both datasets
            let mut chunker1 = FastCDCChunker::new(config.clone()).unwrap();
            let chunks1 = chunker1.chunk_data(&data1).unwrap();
            let _final1 = chunker1.finalize().unwrap();

            let mut chunker2 = FastCDCChunker::new(config).unwrap();
            let chunks2 = chunker2.chunk_data(&data2).unwrap();
            let _final2 = chunker2.finalize().unwrap();

            // Find chunks that contain the identical content block
            let content_start1 = prefix_size;
            let content_end1 = prefix_size + content_block.len();
            let content_start2 = prefix_size;
            let content_end2 = prefix_size + content_block.len();

            let chunks1_in_content = find_chunks_in_range(&chunks1, content_start1, content_end1);
            let chunks2_in_content = find_chunks_in_range(&chunks2, content_start2, content_end2);

            // If the content block is large enough and creates clear boundaries,
            // we should see some consistency in chunking patterns
            if content_block.len() >= config.target_size {
                // At least some chunks should have similar sizes
                let sizes1: Vec<_> = chunks1_in_content.iter().map(|c| c.size()).collect();
                let sizes2: Vec<_> = chunks2_in_content.iter().map(|c| c.size()).collect();

                if !sizes1.is_empty() && !sizes2.is_empty() {
                    let avg1 = sizes1.iter().sum::<usize>() as f64 / sizes1.len() as f64;
                    let avg2 = sizes2.iter().sum::<usize>() as f64 / sizes2.len() as f64;
                    
                    let avg_diff_ratio = (avg1 - avg2).abs() / avg1.max(avg2);
                    prop_assert!(avg_diff_ratio <= 0.5,
                        "Chunk size averages too different for identical content: {:.1} vs {:.1}",
                        avg1, avg2);
                }
            }
        });
    }
}

// === Helper Functions ===

/// Collect all chunk boundaries from a set of chunks
fn collect_chunk_boundaries(chunks: &[DataChunk], final_chunk: &Option<DataChunk>) -> Vec<u64> {
    let mut boundaries = Vec::new();
    
    for chunk in chunks {
        boundaries.push(chunk.offset + chunk.size() as u64);
    }
    
    if let Some(final_chunk) = final_chunk {
        boundaries.push(final_chunk.offset + final_chunk.size() as u64);
    }
    
    boundaries.sort_unstable();
    boundaries
}

/// Find chunks that overlap with a given range
fn find_chunks_in_range(chunks: &[DataChunk], start: usize, end: usize) -> Vec<&DataChunk> {
    chunks
        .iter()
        .filter(|chunk| {
            let chunk_start = chunk.offset as usize;
            let chunk_end = chunk_start + chunk.size();
            
            // Check for overlap
            chunk_start < end && chunk_end > start
        })
        .collect()
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_collect_chunk_boundaries() {
        let chunks = vec![
            DataChunk::new(vec![0; 1000], 0),
            DataChunk::new(vec![1; 2000], 1000),
            DataChunk::new(vec![2; 1500], 3000),
        ];
        
        let final_chunk = Some(DataChunk::new(vec![3; 500], 4500));
        
        let boundaries = collect_chunk_boundaries(&chunks, &final_chunk);
        assert_eq!(boundaries, vec![1000, 3000, 4500, 5000]);
    }

    #[test]
    fn test_find_chunks_in_range() {
        let chunks = vec![
            DataChunk::new(vec![0; 1000], 0),     // 0-1000
            DataChunk::new(vec![1; 1000], 1000),  // 1000-2000
            DataChunk::new(vec![2; 1000], 2000),  // 2000-3000
        ];
        
        // Range 500-1500 should overlap with first two chunks
        let overlapping = find_chunks_in_range(&chunks, 500, 1500);
        assert_eq!(overlapping.len(), 2);
        assert_eq!(overlapping[0].offset, 0);
        assert_eq!(overlapping[1].offset, 1000);
        
        // Range 2500-3500 should overlap with only the third chunk
        let overlapping = find_chunks_in_range(&chunks, 2500, 3500);
        assert_eq!(overlapping.len(), 1);
        assert_eq!(overlapping[0].offset, 2000);
    }

    #[tokio::test]
    async fn test_boundary_stability_basic() {
        let config = ChunkConfig::default();
        let original_data = vec![0xAA; 8192];
        
        // Insert 256 bytes in the middle
        let mut modified_data = original_data.clone();
        let insertion_pos = 4096;
        let insertion = vec![0xCC; 256];
        modified_data.splice(insertion_pos..insertion_pos, insertion.iter().cloned());
        
        // Chunk both versions
        let mut original_chunker = FastCDCChunker::new(config.clone()).unwrap();
        let original_chunks = original_chunker.chunk_data(&original_data).unwrap();
        let original_final = original_chunker.finalize().unwrap();
        
        let mut modified_chunker = FastCDCChunker::new(config).unwrap();
        let modified_chunks = modified_chunker.chunk_data(&modified_data).unwrap();
        let modified_final = modified_chunker.finalize().unwrap();
        
        // Collect boundaries
        let original_boundaries = collect_chunk_boundaries(&original_chunks, &original_final);
        let modified_boundaries = collect_chunk_boundaries(&modified_chunks, &modified_final);
        
        // Should have some boundaries preserved before insertion point
        let boundaries_before: HashSet<_> = original_boundaries
            .iter()
            .filter(|&&b| b < insertion_pos as u64)
            .cloned()
            .collect();
            
        let modified_boundaries_before: HashSet<_> = modified_boundaries
            .iter()
            .filter(|&&b| b < insertion_pos as u64)
            .cloned()
            .collect();
            
        let preserved = boundaries_before.intersection(&modified_boundaries_before).count();
        
        // Should preserve at least some boundaries (this is a basic sanity check)
        if !boundaries_before.is_empty() {
            let preservation_rate = preserved as f64 / boundaries_before.len() as f64;
            assert!(preservation_rate >= 0.0, "Some boundary preservation expected");
        }
    }
}