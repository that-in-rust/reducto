//! SIMD Optimizations for Reducto Mode 3
//!
//! This module implements SIMD (Single Instruction, Multiple Data) optimizations
//! for hash calculations and other performance-critical operations where beneficial.
//! 
//! The optimizations are feature-gated and fall back to scalar implementations
//! when SIMD is not available or beneficial.

use crate::types::{WeakHash, HASH_BASE};
use crate::error::{ReductoError, Result};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// SIMD-optimized hash calculator for batch operations
pub struct SimdHashCalculator {
    base: u64,
    #[cfg(target_arch = "x86_64")]
    use_avx2: bool,
    #[cfg(target_arch = "x86_64")]
    use_sse2: bool,
    #[cfg(target_arch = "aarch64")]
    use_neon: bool,
}

impl SimdHashCalculator {
    /// Create a new SIMD hash calculator with runtime feature detection
    pub fn new(base: u64) -> Self {
        Self {
            base,
            #[cfg(target_arch = "x86_64")]
            use_avx2: is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            use_sse2: is_x86_feature_detected!("sse2"),
            #[cfg(target_arch = "aarch64")]
            use_neon: std::arch::is_aarch64_feature_detected!("neon"),
        }
    }

    /// Calculate multiple rolling hashes in parallel using SIMD
    /// 
    /// This function processes multiple data windows simultaneously,
    /// which is beneficial when processing large datasets with overlapping windows.
    pub fn batch_rolling_hash(&self, windows: &[&[u8]], window_size: usize) -> Result<Vec<WeakHash>> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }

        // Validate all windows have the correct size
        for (i, window) in windows.iter().enumerate() {
            if window.len() != window_size {
                return Err(ReductoError::InputValidationFailed {
                    field: format!("window[{}]", i),
                    reason: format!("size {} != expected {}", window.len(), window_size),
                });
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 && windows.len() >= 4 {
                return self.batch_rolling_hash_avx2(windows, window_size);
            }
            if self.use_sse2 && windows.len() >= 2 {
                return self.batch_rolling_hash_sse2(windows, window_size);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.use_neon && windows.len() >= 2 {
                return self.batch_rolling_hash_neon(windows, window_size);
            }
        }

        // Fallback to scalar implementation
        self.batch_rolling_hash_scalar(windows, window_size)
    }

    /// Scalar fallback implementation for batch rolling hash
    fn batch_rolling_hash_scalar(&self, windows: &[&[u8]], window_size: usize) -> Result<Vec<WeakHash>> {
        let mut results = Vec::with_capacity(windows.len());
        
        for window in windows {
            let mut hash = 0u64;
            for &byte in *window {
                hash = hash.wrapping_mul(self.base).wrapping_add(byte as u64);
            }
            results.push(WeakHash::new(hash));
        }
        
        Ok(results)
    }

    /// AVX2 implementation for x86_64 (processes 4 hashes in parallel)
    #[cfg(target_arch = "x86_64")]
    fn batch_rolling_hash_avx2(&self, windows: &[&[u8]], window_size: usize) -> Result<Vec<WeakHash>> {
        let mut results = Vec::with_capacity(windows.len());
        let chunks = windows.chunks(4);
        
        unsafe {
            let base_vec = _mm256_set1_epi64x(self.base as i64);
            
            for chunk in chunks {
                if chunk.len() == 4 {
                    // Process 4 windows in parallel
                    let mut hash_vec = _mm256_setzero_si256();
                    
                    for byte_idx in 0..window_size {
                        // Load 4 bytes from the 4 windows
                        let bytes = _mm256_set_epi64x(
                            chunk[3][byte_idx] as i64,
                            chunk[2][byte_idx] as i64,
                            chunk[1][byte_idx] as i64,
                            chunk[0][byte_idx] as i64,
                        );
                        
                        // Use scalar operations for now since _mm256_mullo_epi64 is unstable
                        // This would be optimized in a real implementation
                        let hash_array: [i64; 4] = std::mem::transmute(hash_vec);
                        let bytes_array: [i64; 4] = std::mem::transmute(bytes);
                        let mut new_hash = [0i64; 4];
                        for i in 0..4 {
                            new_hash[i] = (hash_array[i] as u64).wrapping_mul(self.base).wrapping_add(bytes_array[i] as u64) as i64;
                        }
                        hash_vec = std::mem::transmute(new_hash);
                    }
                    
                    // Extract results
                    let hash_array: [i64; 4] = std::mem::transmute(hash_vec);
                    for &hash in &hash_array {
                        results.push(WeakHash::new(hash as u64));
                    }
                } else {
                    // Handle remaining windows with scalar code
                    for window in chunk {
                        let mut hash = 0u64;
                        for &byte in *window {
                            hash = hash.wrapping_mul(self.base).wrapping_add(byte as u64);
                        }
                        results.push(WeakHash::new(hash));
                    }
                }
            }
        }
        
        Ok(results)
    }

    /// SSE2 implementation for x86_64 (processes 2 hashes in parallel)
    #[cfg(target_arch = "x86_64")]
    fn batch_rolling_hash_sse2(&self, windows: &[&[u8]], window_size: usize) -> Result<Vec<WeakHash>> {
        let mut results = Vec::with_capacity(windows.len());
        let chunks = windows.chunks(2);
        
        unsafe {
            let base_vec = _mm_set1_epi64x(self.base as i64);
            
            for chunk in chunks {
                if chunk.len() == 2 {
                    // Process 2 windows in parallel
                    let mut hash_vec = _mm_setzero_si128();
                    
                    for byte_idx in 0..window_size {
                        // Load 2 bytes from the 2 windows
                        let bytes = _mm_set_epi64x(
                            chunk[1][byte_idx] as i64,
                            chunk[0][byte_idx] as i64,
                        );
                        
                        // Use scalar operations for now since _mm_mullo_epi64 is unstable
                        let hash_array: [i64; 2] = std::mem::transmute(hash_vec);
                        let bytes_array: [i64; 2] = std::mem::transmute(bytes);
                        let mut new_hash = [0i64; 2];
                        for i in 0..2 {
                            new_hash[i] = (hash_array[i] as u64).wrapping_mul(self.base).wrapping_add(bytes_array[i] as u64) as i64;
                        }
                        hash_vec = std::mem::transmute(new_hash);
                    }
                    
                    // Extract results
                    let hash_array: [i64; 2] = std::mem::transmute(hash_vec);
                    for &hash in &hash_array {
                        results.push(WeakHash::new(hash as u64));
                    }
                } else {
                    // Handle remaining window with scalar code
                    for window in chunk {
                        let mut hash = 0u64;
                        for &byte in *window {
                            hash = hash.wrapping_mul(self.base).wrapping_add(byte as u64);
                        }
                        results.push(WeakHash::new(hash));
                    }
                }
            }
        }
        
        Ok(results)
    }

    /// NEON implementation for AArch64 (processes 2 hashes in parallel)
    #[cfg(target_arch = "aarch64")]
    fn batch_rolling_hash_neon(&self, windows: &[&[u8]], window_size: usize) -> Result<Vec<WeakHash>> {
        let mut results = Vec::with_capacity(windows.len());
        let chunks = windows.chunks(2);
        
        unsafe {
            for chunk in chunks {
                if chunk.len() == 2 {
                    // Process 2 windows in parallel using NEON
                    let mut hash0 = 0u64;
                    let mut hash1 = 0u64;
                    
                    // NEON doesn't have 64-bit multiply, so we use scalar for now
                    // This could be optimized further with more complex NEON operations
                    for byte_idx in 0..window_size {
                        hash0 = hash0.wrapping_mul(self.base).wrapping_add(chunk[0][byte_idx] as u64);
                        hash1 = hash1.wrapping_mul(self.base).wrapping_add(chunk[1][byte_idx] as u64);
                    }
                    
                    results.push(WeakHash::new(hash0));
                    results.push(WeakHash::new(hash1));
                } else {
                    // Handle remaining window with scalar code
                    for window in chunk {
                        let mut hash = 0u64;
                        for &byte in *window {
                            hash = hash.wrapping_mul(self.base).wrapping_add(byte as u64);
                        }
                        results.push(WeakHash::new(hash));
                    }
                }
            }
        }
        
        Ok(results)
    }

    /// SIMD-optimized byte comparison for chunk verification
    /// 
    /// This function compares two byte arrays using SIMD instructions
    /// for faster verification of chunk matches.
    pub fn simd_compare_chunks(&self, chunk1: &[u8], chunk2: &[u8]) -> bool {
        if chunk1.len() != chunk2.len() {
            return false;
        }

        let len = chunk1.len();
        
        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 && len >= 32 {
                return self.simd_compare_chunks_avx2(chunk1, chunk2);
            }
            if self.use_sse2 && len >= 16 {
                return self.simd_compare_chunks_sse2(chunk1, chunk2);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.use_neon && len >= 16 {
                return self.simd_compare_chunks_neon(chunk1, chunk2);
            }
        }

        // Fallback to scalar comparison
        chunk1 == chunk2
    }

    /// AVX2 implementation for chunk comparison
    #[cfg(target_arch = "x86_64")]
    fn simd_compare_chunks_avx2(&self, chunk1: &[u8], chunk2: &[u8]) -> bool {
        let len = chunk1.len();
        let simd_len = len & !31; // Round down to multiple of 32
        
        unsafe {
            // Compare 32 bytes at a time
            for i in (0..simd_len).step_by(32) {
                let a = _mm256_loadu_si256(chunk1.as_ptr().add(i) as *const __m256i);
                let b = _mm256_loadu_si256(chunk2.as_ptr().add(i) as *const __m256i);
                let cmp = _mm256_cmpeq_epi8(a, b);
                let mask = _mm256_movemask_epi8(cmp);
                
                if mask != -1 {
                    return false;
                }
            }
            
            // Handle remaining bytes
            for i in simd_len..len {
                if chunk1[i] != chunk2[i] {
                    return false;
                }
            }
        }
        
        true
    }

    /// SSE2 implementation for chunk comparison
    #[cfg(target_arch = "x86_64")]
    fn simd_compare_chunks_sse2(&self, chunk1: &[u8], chunk2: &[u8]) -> bool {
        let len = chunk1.len();
        let simd_len = len & !15; // Round down to multiple of 16
        
        unsafe {
            // Compare 16 bytes at a time
            for i in (0..simd_len).step_by(16) {
                let a = _mm_loadu_si128(chunk1.as_ptr().add(i) as *const __m128i);
                let b = _mm_loadu_si128(chunk2.as_ptr().add(i) as *const __m128i);
                let cmp = _mm_cmpeq_epi8(a, b);
                let mask = _mm_movemask_epi8(cmp);
                
                if mask != 0xFFFF {
                    return false;
                }
            }
            
            // Handle remaining bytes
            for i in simd_len..len {
                if chunk1[i] != chunk2[i] {
                    return false;
                }
            }
        }
        
        true
    }

    /// NEON implementation for chunk comparison
    #[cfg(target_arch = "aarch64")]
    fn simd_compare_chunks_neon(&self, chunk1: &[u8], chunk2: &[u8]) -> bool {
        let len = chunk1.len();
        let simd_len = len & !15; // Round down to multiple of 16
        
        unsafe {
            // Compare 16 bytes at a time
            for i in (0..simd_len).step_by(16) {
                let a = vld1q_u8(chunk1.as_ptr().add(i));
                let b = vld1q_u8(chunk2.as_ptr().add(i));
                let cmp = vceqq_u8(a, b);
                
                // Check if all bytes are equal
                let min = vminvq_u8(cmp);
                if min == 0 {
                    return false;
                }
            }
            
            // Handle remaining bytes
            for i in simd_len..len {
                if chunk1[i] != chunk2[i] {
                    return false;
                }
            }
        }
        
        true
    }

    /// SIMD-optimized memory copy for large chunks
    pub fn simd_copy_chunk(&self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        if src.len() != dst.len() {
            return Err(ReductoError::InputValidationFailed {
                field: "buffer_sizes".to_string(),
                reason: format!("source length {} != destination length {}", src.len(), dst.len()),
            });
        }

        let len = src.len();

        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 && len >= 32 {
                return self.simd_copy_chunk_avx2(src, dst);
            }
            if self.use_sse2 && len >= 16 {
                return self.simd_copy_chunk_sse2(src, dst);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.use_neon && len >= 16 {
                return self.simd_copy_chunk_neon(src, dst);
            }
        }

        // Fallback to standard copy
        dst.copy_from_slice(src);
        Ok(())
    }

    /// AVX2 implementation for chunk copying
    #[cfg(target_arch = "x86_64")]
    fn simd_copy_chunk_avx2(&self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        let len = src.len();
        let simd_len = len & !31; // Round down to multiple of 32
        
        unsafe {
            // Copy 32 bytes at a time
            for i in (0..simd_len).step_by(32) {
                let data = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
                _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, data);
            }
            
            // Handle remaining bytes
            for i in simd_len..len {
                dst[i] = src[i];
            }
        }
        
        Ok(())
    }

    /// SSE2 implementation for chunk copying
    #[cfg(target_arch = "x86_64")]
    fn simd_copy_chunk_sse2(&self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        let len = src.len();
        let simd_len = len & !15; // Round down to multiple of 16
        
        unsafe {
            // Copy 16 bytes at a time
            for i in (0..simd_len).step_by(16) {
                let data = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, data);
            }
            
            // Handle remaining bytes
            for i in simd_len..len {
                dst[i] = src[i];
            }
        }
        
        Ok(())
    }

    /// NEON implementation for chunk copying
    #[cfg(target_arch = "aarch64")]
    fn simd_copy_chunk_neon(&self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        let len = src.len();
        let simd_len = len & !15; // Round down to multiple of 16
        
        unsafe {
            // Copy 16 bytes at a time
            for i in (0..simd_len).step_by(16) {
                let data = vld1q_u8(src.as_ptr().add(i));
                vst1q_u8(dst.as_mut_ptr().add(i), data);
            }
            
            // Handle remaining bytes
            for i in simd_len..len {
                dst[i] = src[i];
            }
        }
        
        Ok(())
    }
}

/// SIMD-optimized gear hash table for boundary detection
pub struct SimdGearHasher {
    gear_table: [u64; 256],
    hash: u64,
    #[cfg(target_arch = "x86_64")]
    use_simd: bool,
}

impl SimdGearHasher {
    /// Create a new SIMD gear hasher with pre-computed gear table
    pub fn new() -> Self {
        let mut gear_table = [0u64; 256];
        
        // Pre-compute gear values for all possible bytes
        for i in 0..256 {
            gear_table[i] = Self::compute_gear_value(i as u8);
        }
        
        Self {
            gear_table,
            hash: 0,
            #[cfg(target_arch = "x86_64")]
            use_simd: is_x86_feature_detected!("avx2"),
        }
    }

    /// Compute gear value for a single byte
    fn compute_gear_value(byte: u8) -> u64 {
        // Use a more sophisticated gear function for better distribution
        let mut value = byte as u64;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        value
    }

    /// Update hash with a single byte
    pub fn update(&mut self, byte: u8) -> u64 {
        self.hash = (self.hash << 1).wrapping_add(self.gear_table[byte as usize]);
        self.hash
    }

    /// Process multiple bytes at once using SIMD when beneficial
    pub fn update_batch(&mut self, bytes: &[u8]) -> Vec<u64> {
        #[cfg(target_arch = "x86_64")]
        {
            if self.use_simd && bytes.len() >= 8 {
                return self.update_batch_simd(bytes);
            }
        }

        // Fallback to scalar processing
        let mut results = Vec::with_capacity(bytes.len());
        for &byte in bytes {
            results.push(self.update(byte));
        }
        results
    }

    /// SIMD implementation for batch processing
    #[cfg(target_arch = "x86_64")]
    fn update_batch_simd(&mut self, bytes: &[u8]) -> Vec<u64> {
        let mut results = Vec::with_capacity(bytes.len());
        
        // Process bytes in chunks that can benefit from SIMD lookup
        for &byte in bytes {
            // For now, use scalar implementation as SIMD table lookup
            // is complex and may not provide benefits for this use case
            results.push(self.update(byte));
        }
        
        results
    }

    /// Get current hash value
    pub fn current_hash(&self) -> u64 {
        self.hash
    }

    /// Reset hash to initial state
    pub fn reset(&mut self) {
        self.hash = 0;
    }
}

/// Benchmark SIMD vs scalar performance for different data sizes
#[cfg(test)]
pub fn benchmark_simd_performance() -> Result<()> {
    use std::time::Instant;
    
    let calculator = SimdHashCalculator::new(HASH_BASE);
    let data_sizes = [1024, 4096, 16384, 65536];
    
    println!("SIMD Performance Benchmark Results:");
    println!("===================================");
    
    for &size in &data_sizes {
        let test_data = vec![0xAA; size];
        let windows: Vec<&[u8]> = (0..8).map(|i| {
            let start = i * (size / 8);
            &test_data[start..start + size / 8]
        }).collect();
        
        // Benchmark scalar implementation
        let start = Instant::now();
        let _scalar_results = calculator.batch_rolling_hash_scalar(&windows, size / 8)?;
        let scalar_time = start.elapsed();
        
        // Benchmark SIMD implementation
        let start = Instant::now();
        let _simd_results = calculator.batch_rolling_hash(&windows, size / 8)?;
        let simd_time = start.elapsed();
        
        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        
        println!("Size: {}KB, Scalar: {:?}, SIMD: {:?}, Speedup: {:.2}x", 
                 size / 1024, scalar_time, simd_time, speedup);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_calculator_creation() {
        let calculator = SimdHashCalculator::new(HASH_BASE);
        assert_eq!(calculator.base, HASH_BASE);
    }

    #[test]
    fn test_batch_rolling_hash_empty() {
        let calculator = SimdHashCalculator::new(HASH_BASE);
        let windows: Vec<&[u8]> = vec![];
        let results = calculator.batch_rolling_hash(&windows, 0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_rolling_hash_single_window() {
        let calculator = SimdHashCalculator::new(HASH_BASE);
        let data = vec![1, 2, 3, 4];
        let windows = vec![data.as_slice()];
        let results = calculator.batch_rolling_hash(&windows, 4).unwrap();
        
        assert_eq!(results.len(), 1);
        
        // Verify against scalar calculation
        let expected = HASH_BASE.wrapping_mul(HASH_BASE.wrapping_mul(HASH_BASE.wrapping_mul(1).wrapping_add(2)).wrapping_add(3)).wrapping_add(4);
        assert_eq!(results[0].get(), expected);
    }

    #[test]
    fn test_batch_rolling_hash_multiple_windows() {
        let calculator = SimdHashCalculator::new(HASH_BASE);
        let data1 = vec![1, 2, 3, 4];
        let data2 = vec![5, 6, 7, 8];
        let data3 = vec![9, 10, 11, 12];
        let windows = vec![data1.as_slice(), data2.as_slice(), data3.as_slice()];
        
        let results = calculator.batch_rolling_hash(&windows, 4).unwrap();
        assert_eq!(results.len(), 3);
        
        // Verify each result independently
        for (i, window) in windows.iter().enumerate() {
            let mut expected = 0u64;
            for &byte in *window {
                expected = expected.wrapping_mul(HASH_BASE).wrapping_add(byte as u64);
            }
            assert_eq!(results[i].get(), expected, "Window {} hash mismatch", i);
        }
    }

    #[test]
    fn test_simd_compare_chunks_equal() {
        let calculator = SimdHashCalculator::new(HASH_BASE);
        let chunk1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let chunk2 = chunk1.clone();
        
        assert!(calculator.simd_compare_chunks(&chunk1, &chunk2));
    }

    #[test]
    fn test_simd_compare_chunks_different() {
        let calculator = SimdHashCalculator::new(HASH_BASE);
        let chunk1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut chunk2 = chunk1.clone();
        chunk2[8] = 99; // Change middle byte
        
        assert!(!calculator.simd_compare_chunks(&chunk1, &chunk2));
    }

    #[test]
    fn test_simd_copy_chunk() {
        let calculator = SimdHashCalculator::new(HASH_BASE);
        let src = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut dst = vec![0; 16];
        
        calculator.simd_copy_chunk(&src, &mut dst).unwrap();
        assert_eq!(src, dst);
    }

    #[test]
    fn test_gear_hasher() {
        let mut hasher = SimdGearHasher::new();
        
        let initial_hash = hasher.current_hash();
        assert_eq!(initial_hash, 0);
        
        let hash1 = hasher.update(0xAA);
        assert_ne!(hash1, 0);
        
        let hash2 = hasher.update(0xBB);
        assert_ne!(hash2, hash1);
        
        hasher.reset();
        assert_eq!(hasher.current_hash(), 0);
    }

    #[test]
    fn test_gear_hasher_batch() {
        let mut hasher = SimdGearHasher::new();
        let bytes = vec![1, 2, 3, 4, 5];
        
        let batch_results = hasher.update_batch(&bytes);
        assert_eq!(batch_results.len(), 5);
        
        // Verify against individual updates
        let mut hasher2 = SimdGearHasher::new();
        for (i, &byte) in bytes.iter().enumerate() {
            let individual_result = hasher2.update(byte);
            assert_eq!(batch_results[i], individual_result, "Batch result {} mismatch", i);
        }
    }

    #[test]
    fn test_invalid_window_sizes() {
        let calculator = SimdHashCalculator::new(HASH_BASE);
        let data1 = vec![1, 2, 3];
        let data2 = vec![4, 5, 6, 7]; // Different size
        let windows = vec![data1.as_slice(), data2.as_slice()];
        
        let result = calculator.batch_rolling_hash(&windows, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_simd_copy_size_mismatch() {
        let calculator = SimdHashCalculator::new(HASH_BASE);
        let src = vec![1, 2, 3];
        let mut dst = vec![0; 4]; // Different size
        
        let result = calculator.simd_copy_chunk(&src, &mut dst);
        assert!(result.is_err());
    }
}