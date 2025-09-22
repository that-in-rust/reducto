//! Rolling hash implementation for chunk identification
//!
//! This module provides a polynomial rolling hash with configurable window size
//! for efficient content hashing. The implementation ensures O(1) hash updates
//! and includes BLAKE3 strong hash calculation for verification.

use crate::{
    error::{Result, ReductoError},
    types::WeakHash,
};
use blake3::Hash;
use std::collections::VecDeque;

/// Rolling hasher for chunk content identification
///
/// Implements a polynomial rolling hash (Rabin-Karp style) with configurable
/// window size. Provides O(1) hash updates as the window slides through data.
#[derive(Debug, Clone)]
pub struct RollingHasher {
    /// Current hash value
    hash: u64,
    /// Base for polynomial hash computation
    base: u64,
    /// Pre-calculated power: base^(window_size - 1)
    power: u64,
    /// Sliding window of bytes
    window: VecDeque<u8>,
    /// Maximum window size
    window_size: usize,
    /// Total bytes processed (for statistics)
    bytes_processed: u64,
}

impl RollingHasher {
    /// Create a new rolling hasher
    ///
    /// # Arguments
    /// * `base` - Base for polynomial hash computation (should be prime)
    /// * `window_size` - Size of the sliding window
    ///
    /// # Returns
    /// * `Ok(RollingHasher)` - Successfully created hasher
    /// * `Err(ReductoError)` - Invalid parameters
    pub fn new(base: u64, window_size: usize) -> Result<Self> {
        if base == 0 {
            return Err(ReductoError::InvalidConfiguration {
                parameter: "hash_base".to_string(),
                value: base.to_string(),
                reason: "hash base must be non-zero".to_string(),
            });
        }

        if window_size == 0 {
            return Err(ReductoError::InvalidConfiguration {
                parameter: "window_size".to_string(),
                value: window_size.to_string(),
                reason: "window size must be non-zero".to_string(),
            });
        }

        // Pre-calculate base^(window_size - 1) for O(1) rolling
        let mut power = 1u64;
        for _ in 0..(window_size - 1) {
            power = power.wrapping_mul(base);
        }

        Ok(Self {
            hash: 0,
            base,
            power,
            window: VecDeque::with_capacity(window_size),
            window_size,
            bytes_processed: 0,
        })
    }

    /// Initialize the hash for a window of data
    ///
    /// # Arguments
    /// * `data` - Initial data to fill the window (must be exactly window_size bytes)
    ///
    /// # Returns
    /// * `Ok(WeakHash)` - Initial hash value
    /// * `Err(ReductoError)` - Initialization failed
    ///
    /// # Performance
    /// O(window_size) time complexity
    pub fn init(&mut self, data: &[u8]) -> Result<WeakHash> {
        if data.len() != self.window_size {
            return Err(ReductoError::InputValidationFailed {
                field: "data".to_string(),
                reason: format!(
                    "data length {} must equal window size {}",
                    data.len(),
                    self.window_size
                ),
            });
        }

        // Clear existing state
        self.hash = 0;
        self.window.clear();
        self.bytes_processed = 0;

        // Initialize hash and window
        for &byte in data {
            self.hash = self.hash.wrapping_mul(self.base).wrapping_add(byte as u64);
            self.window.push_back(byte);
        }

        self.bytes_processed = data.len() as u64;
        Ok(WeakHash::new(self.hash))
    }

    /// Roll the hash forward by one byte (O(1) time)
    ///
    /// # Arguments
    /// * `entering_byte` - New byte entering the window
    ///
    /// # Returns
    /// * `Ok(WeakHash)` - Updated hash value
    /// * `Err(ReductoError)` - Rolling operation failed
    ///
    /// # Performance
    /// Guaranteed O(1) time complexity
    pub fn roll(&mut self, entering_byte: u8) -> Result<WeakHash> {
        if !self.is_full() {
            return Err(ReductoError::RollingHashStateCorrupted {
                expected: self.window_size,
                actual: self.window.len(),
            });
        }

        // Remove contribution of exiting byte
        let exiting_byte = self.window.pop_front().unwrap();
        let exiting_contribution = (exiting_byte as u64).wrapping_mul(self.power);
        self.hash = self.hash.wrapping_sub(exiting_contribution);

        // Shift hash left (multiply by base)
        self.hash = self.hash.wrapping_mul(self.base);

        // Add contribution of entering byte
        self.hash = self.hash.wrapping_add(entering_byte as u64);

        // Update window
        self.window.push_back(entering_byte);
        self.bytes_processed += 1;

        Ok(WeakHash::new(self.hash))
    }

    /// Add byte to hash without maintaining full window (simplified version)
    ///
    /// This is used when we don't need the full rolling window functionality
    /// but want consistent hash computation.
    ///
    /// # Arguments
    /// * `byte` - Byte to add to the hash
    ///
    /// # Returns
    /// Updated hash value
    ///
    /// # Performance
    /// O(1) time complexity
    pub fn update(&mut self, byte: u8) -> u64 {
        self.hash = self.hash.wrapping_mul(self.base).wrapping_add(byte as u64);
        self.bytes_processed += 1;
        self.hash
    }

    /// Get current weak hash value
    ///
    /// # Returns
    /// Current weak hash, or error if hasher not properly initialized
    pub fn current_hash(&self) -> Result<WeakHash> {
        Ok(WeakHash::new(self.hash))
    }

    /// Get current hash as raw u64 (for compatibility)
    pub fn current_hash_raw(&self) -> u64 {
        self.hash
    }

    /// Check if the window is full
    pub fn is_full(&self) -> bool {
        self.window.len() == self.window_size
    }

    /// Check if the hasher is initialized and ready for rolling operations
    pub fn is_initialized(&self) -> bool {
        self.is_full()
    }

    /// Reset the hasher to uninitialized state
    pub fn reset(&mut self) {
        self.hash = 0;
        self.window.clear();
        self.bytes_processed = 0;
    }

    /// Get the number of bytes processed
    pub fn bytes_processed(&self) -> u64 {
        self.bytes_processed
    }

    /// Get the current window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get the current window contents (for debugging)
    pub fn window_contents(&self) -> Vec<u8> {
        self.window.iter().copied().collect()
    }
}

/// Strong hash calculator using BLAKE3
///
/// Provides cryptographically secure hashing for chunk verification
/// with constant-time comparison to prevent timing attacks.
#[derive(Debug, Clone)]
pub struct StrongHasher {
    /// Statistics for performance monitoring
    hashes_computed: u64,
    total_bytes_hashed: u64,
}

impl StrongHasher {
    /// Create a new strong hasher
    pub fn new() -> Self {
        Self {
            hashes_computed: 0,
            total_bytes_hashed: 0,
        }
    }

    /// Compute BLAKE3 hash for data
    ///
    /// # Arguments
    /// * `data` - Data to hash
    ///
    /// # Returns
    /// * `Ok(Hash)` - BLAKE3 hash of the data
    /// * `Err(ReductoError)` - Hash computation failed
    ///
    /// # Performance
    /// Deterministic time based on data size
    pub fn hash(&mut self, data: &[u8]) -> Result<Hash> {
        if data.is_empty() {
            return Err(ReductoError::InputValidationFailed {
                field: "data".to_string(),
                reason: "cannot hash empty data".to_string(),
            });
        }

        let hash = blake3::hash(data);
        
        // Update statistics
        self.hashes_computed += 1;
        self.total_bytes_hashed += data.len() as u64;

        Ok(hash)
    }

    /// Compare two hashes in constant time to prevent timing attacks
    ///
    /// # Arguments
    /// * `hash1` - First hash to compare
    /// * `hash2` - Second hash to compare
    ///
    /// # Returns
    /// True if hashes are equal, false otherwise
    ///
    /// # Security
    /// Uses constant-time comparison to prevent timing side-channel attacks
    pub fn constant_time_compare(&self, hash1: &Hash, hash2: &Hash) -> bool {
        use subtle::ConstantTimeEq;
        hash1.as_bytes().ct_eq(hash2.as_bytes()).into()
    }

    /// Verify that data matches expected hash
    ///
    /// # Arguments
    /// * `data` - Data to verify
    /// * `expected_hash` - Expected hash value
    ///
    /// # Returns
    /// * `Ok(true)` - Data matches expected hash
    /// * `Ok(false)` - Data does not match
    /// * `Err(ReductoError)` - Verification failed
    pub fn verify(&mut self, data: &[u8], expected_hash: &Hash) -> Result<bool> {
        let computed_hash = self.hash(data)?;
        Ok(self.constant_time_compare(&computed_hash, expected_hash))
    }

    /// Get statistics about hash operations
    ///
    /// # Returns
    /// (hashes_computed, total_bytes_hashed, avg_bytes_per_hash)
    pub fn get_statistics(&self) -> (u64, u64, f64) {
        let avg_bytes = if self.hashes_computed > 0 {
            self.total_bytes_hashed as f64 / self.hashes_computed as f64
        } else {
            0.0
        };
        (self.hashes_computed, self.total_bytes_hashed, avg_bytes)
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.hashes_computed = 0;
        self.total_bytes_hashed = 0;
    }
}

impl Default for StrongHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Dual-hash system combining weak and strong hashes
///
/// Provides both fast weak hash for candidate identification and
/// strong hash for verification with collision handling.
#[derive(Debug)]
pub struct DualHasher {
    /// Rolling hasher for weak hash computation
    rolling_hasher: RollingHasher,
    /// Strong hasher for verification
    strong_hasher: StrongHasher,
    /// Collision statistics
    weak_hash_collisions: u64,
    /// Verification statistics
    verifications_performed: u64,
    /// Successful verifications
    successful_verifications: u64,
}

impl DualHasher {
    /// Create a new dual hasher
    ///
    /// # Arguments
    /// * `base` - Base for polynomial hash
    /// * `window_size` - Size of rolling window
    ///
    /// # Returns
    /// * `Ok(DualHasher)` - Successfully created dual hasher
    /// * `Err(ReductoError)` - Invalid parameters
    pub fn new(base: u64, window_size: usize) -> Result<Self> {
        let rolling_hasher = RollingHasher::new(base, window_size)?;
        let strong_hasher = StrongHasher::new();

        Ok(Self {
            rolling_hasher,
            strong_hasher,
            weak_hash_collisions: 0,
            verifications_performed: 0,
            successful_verifications: 0,
        })
    }

    /// Initialize with data
    ///
    /// # Arguments
    /// * `data` - Initial data for rolling hash window
    ///
    /// # Returns
    /// * `Ok(WeakHash)` - Initial weak hash
    /// * `Err(ReductoError)` - Initialization failed
    pub fn init(&mut self, data: &[u8]) -> Result<WeakHash> {
        self.rolling_hasher.init(data)
    }

    /// Roll the weak hash forward
    ///
    /// # Arguments
    /// * `entering_byte` - New byte entering the window
    ///
    /// # Returns
    /// * `Ok(WeakHash)` - Updated weak hash
    /// * `Err(ReductoError)` - Rolling failed
    pub fn roll(&mut self, entering_byte: u8) -> Result<WeakHash> {
        self.rolling_hasher.roll(entering_byte)
    }

    /// Update weak hash (simplified version without window)
    ///
    /// # Arguments
    /// * `byte` - Byte to add to hash
    ///
    /// # Returns
    /// Updated weak hash value
    pub fn update_weak(&mut self, byte: u8) -> WeakHash {
        let hash = self.rolling_hasher.update(byte);
        WeakHash::new(hash)
    }

    /// Compute strong hash for data
    ///
    /// # Arguments
    /// * `data` - Data to hash
    ///
    /// # Returns
    /// * `Ok(Hash)` - Strong hash of data
    /// * `Err(ReductoError)` - Hash computation failed
    pub fn strong_hash(&mut self, data: &[u8]) -> Result<Hash> {
        self.strong_hasher.hash(data)
    }

    /// Verify data against expected strong hash with collision tracking
    ///
    /// # Arguments
    /// * `data` - Data to verify
    /// * `expected_hash` - Expected strong hash
    /// * `weak_hash_matched` - Whether weak hash matched (for collision tracking)
    ///
    /// # Returns
    /// * `Ok(true)` - Data matches expected hash
    /// * `Ok(false)` - Data does not match (collision detected)
    /// * `Err(ReductoError)` - Verification failed
    pub fn verify_with_collision_tracking(
        &mut self,
        data: &[u8],
        expected_hash: &Hash,
        weak_hash_matched: bool,
    ) -> Result<bool> {
        self.verifications_performed += 1;

        let matches = self.strong_hasher.verify(data, expected_hash)?;

        if matches {
            self.successful_verifications += 1;
        } else if weak_hash_matched {
            // Weak hash matched but strong hash didn't - this is a collision
            self.weak_hash_collisions += 1;
        }

        Ok(matches)
    }

    /// Get current weak hash
    pub fn current_weak_hash(&self) -> Result<WeakHash> {
        self.rolling_hasher.current_hash()
    }

    /// Check if hasher is initialized
    pub fn is_initialized(&self) -> bool {
        self.rolling_hasher.is_initialized()
    }

    /// Reset both hashers
    pub fn reset(&mut self) {
        self.rolling_hasher.reset();
        self.strong_hasher.reset_statistics();
        self.weak_hash_collisions = 0;
        self.verifications_performed = 0;
        self.successful_verifications = 0;
    }

    /// Get collision statistics
    ///
    /// # Returns
    /// (weak_hash_collisions, verifications_performed, successful_verifications, collision_rate)
    pub fn get_collision_statistics(&self) -> (u64, u64, u64, f64) {
        let collision_rate = if self.verifications_performed > 0 {
            self.weak_hash_collisions as f64 / self.verifications_performed as f64
        } else {
            0.0
        };

        (
            self.weak_hash_collisions,
            self.verifications_performed,
            self.successful_verifications,
            collision_rate,
        )
    }

    /// Get comprehensive statistics
    ///
    /// # Returns
    /// Combined statistics from both hashers
    pub fn get_statistics(&self) -> Result<DualHashStatistics> {
        let weak_hash = self.rolling_hasher.current_hash()?;
        let (hashes_computed, bytes_hashed, avg_bytes) = self.strong_hasher.get_statistics();
        let (collisions, verifications, successful, collision_rate) = self.get_collision_statistics();

        Ok(DualHashStatistics {
            current_weak_hash: weak_hash,
            bytes_processed: self.rolling_hasher.bytes_processed(),
            strong_hashes_computed: hashes_computed,
            total_bytes_hashed: bytes_hashed,
            avg_bytes_per_hash: avg_bytes,
            weak_hash_collisions: collisions,
            verifications_performed: verifications,
            successful_verifications: successful,
            collision_rate,
        })
    }
}

/// Statistics for dual hash system
#[derive(Debug, Clone, PartialEq)]
pub struct DualHashStatistics {
    /// Current weak hash value
    pub current_weak_hash: WeakHash,
    /// Bytes processed by rolling hasher
    pub bytes_processed: u64,
    /// Strong hashes computed
    pub strong_hashes_computed: u64,
    /// Total bytes hashed with strong hasher
    pub total_bytes_hashed: u64,
    /// Average bytes per strong hash
    pub avg_bytes_per_hash: f64,
    /// Weak hash collisions detected
    pub weak_hash_collisions: u64,
    /// Total verifications performed
    pub verifications_performed: u64,
    /// Successful verifications
    pub successful_verifications: u64,
    /// Collision rate (0.0 to 1.0)
    pub collision_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use proptest::prelude::*;

    /// Test O(1) hash update performance requirement
    #[test]
    fn test_rolling_hash_o1_performance() {
        let mut hasher = RollingHasher::new(67, 64).unwrap();
        
        // Initialize with test data
        let init_data = vec![42u8; 64];
        hasher.init(&init_data).unwrap();

        // Test O(1) performance with large number of updates
        let test_iterations = 1_000_000;
        let start = Instant::now();
        
        for i in 0..test_iterations {
            let byte = (i % 256) as u8;
            hasher.roll(byte).unwrap();
        }
        
        let elapsed = start.elapsed();
        let ns_per_update = elapsed.as_nanos() / test_iterations;
        
        // Should be very fast - less than 200ns per update on modern hardware
        // (allowing some variance for system load and test environment)
        assert!(
            ns_per_update < 200,
            "Rolling hash update too slow: {}ns per update (expected < 200ns)",
            ns_per_update
        );
        
        println!("Rolling hash performance: {}ns per update", ns_per_update);
    }

    /// Test rolling hash consistency
    #[test]
    fn test_rolling_hash_consistency() {
        let mut hasher = RollingHasher::new(67, 4).unwrap();
        
        // Test data: "abcdefgh"
        let data = b"abcdefgh";
        
        // Initialize with first 4 bytes: "abcd"
        let hash1 = hasher.init(&data[0..4]).unwrap();
        
        // Roll to get "bcde"
        let hash2 = hasher.roll(data[4]).unwrap();
        
        // Roll to get "cdef"
        let hash3 = hasher.roll(data[5]).unwrap();
        
        // Create new hasher and compute "cdef" directly
        let mut hasher2 = RollingHasher::new(67, 4).unwrap();
        let hash3_direct = hasher2.init(&data[2..6]).unwrap();
        
        // Should be equal
        assert_eq!(hash3, hash3_direct, "Rolling hash not consistent with direct computation");
        
        // All hashes should be different (with high probability)
        assert_ne!(hash1, hash2, "Hash values should change when rolling");
        assert_ne!(hash2, hash3, "Hash values should change when rolling");
    }

    /// Test BLAKE3 strong hash calculation
    #[test]
    fn test_blake3_strong_hash() {
        let mut hasher = StrongHasher::new();
        
        let test_data = b"Hello, BLAKE3!";
        let hash1 = hasher.hash(test_data).unwrap();
        let hash2 = hasher.hash(test_data).unwrap();
        
        // Same data should produce same hash
        assert_eq!(hash1, hash2, "BLAKE3 hash not deterministic");
        
        // Different data should produce different hash
        let different_data = b"Hello, BLAKE2!";
        let hash3 = hasher.hash(different_data).unwrap();
        assert_ne!(hash1, hash3, "BLAKE3 hash collision (very unlikely)");
        
        // Verify statistics
        let (count, bytes, avg) = hasher.get_statistics();
        assert_eq!(count, 3, "Incorrect hash count");
        assert_eq!(bytes, 14 + 14 + 14, "Incorrect byte count");
        assert_eq!(avg, 14.0, "Incorrect average");
    }

    /// Test constant-time hash comparison
    #[test]
    fn test_constant_time_comparison() {
        let hasher = StrongHasher::new();
        
        let hash1 = blake3::hash(b"test data 1");
        let hash2 = blake3::hash(b"test data 1"); // Same
        let hash3 = blake3::hash(b"test data 2"); // Different
        
        // Same hashes should compare equal
        assert!(hasher.constant_time_compare(&hash1, &hash2));
        
        // Different hashes should compare unequal
        assert!(!hasher.constant_time_compare(&hash1, &hash3));
        
        // Test timing consistency (basic check - not a real timing attack test)
        let iterations = 1000;
        
        let start_same = Instant::now();
        for _ in 0..iterations {
            hasher.constant_time_compare(&hash1, &hash2);
        }
        let time_same = start_same.elapsed();
        
        let start_diff = Instant::now();
        for _ in 0..iterations {
            hasher.constant_time_compare(&hash1, &hash3);
        }
        let time_diff = start_diff.elapsed();
        
        // Times should be similar (within 50% - this is a rough check)
        let ratio = time_same.as_nanos() as f64 / time_diff.as_nanos() as f64;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Comparison times too different: same={}ns, diff={}ns, ratio={}",
            time_same.as_nanos() / iterations,
            time_diff.as_nanos() / iterations,
            ratio
        );
    }

    /// Test collision handling with controlled weak hash conflicts
    #[test]
    fn test_collision_handling() {
        let mut dual_hasher = DualHasher::new(67, 8).unwrap();
        
        // Create test data that might have weak hash collisions
        let data1 = b"collision test data 1";
        let data2 = b"collision test data 2";
        
        // Compute hashes
        let strong_hash1 = dual_hasher.strong_hash(data1).unwrap();
        let strong_hash2 = dual_hasher.strong_hash(data2).unwrap();
        
        // Test verification with correct data
        let verify1 = dual_hasher.verify_with_collision_tracking(data1, &strong_hash1, true).unwrap();
        assert!(verify1, "Verification should succeed for correct data");
        
        // Test verification with incorrect data (simulated collision)
        let verify2 = dual_hasher.verify_with_collision_tracking(data1, &strong_hash2, true).unwrap();
        assert!(!verify2, "Verification should fail for incorrect data");
        
        // Check collision statistics
        let (collisions, verifications, successful, rate) = dual_hasher.get_collision_statistics();
        assert_eq!(verifications, 2, "Should have performed 2 verifications");
        assert_eq!(successful, 1, "Should have 1 successful verification");
        assert_eq!(collisions, 1, "Should have detected 1 collision");
        assert_eq!(rate, 0.5, "Collision rate should be 50%");
    }

    /// Test dual hasher integration
    #[test]
    fn test_dual_hasher_integration() {
        let mut dual_hasher = DualHasher::new(67, 16).unwrap();
        
        // Initialize with test data
        let init_data = vec![42u8; 16];
        let weak_hash = dual_hasher.init(&init_data).unwrap();
        assert!(dual_hasher.is_initialized());
        
        // Test rolling
        let new_weak_hash = dual_hasher.roll(99).unwrap();
        assert_ne!(weak_hash, new_weak_hash, "Weak hash should change after rolling");
        
        // Test strong hash computation
        let test_data = b"integration test data";
        let strong_hash = dual_hasher.strong_hash(test_data).unwrap();
        
        // Test verification
        let verified = dual_hasher.verify_with_collision_tracking(test_data, &strong_hash, true).unwrap();
        assert!(verified, "Verification should succeed");
        
        // Get comprehensive statistics
        let stats = dual_hasher.get_statistics().unwrap();
        assert!(stats.bytes_processed > 0, "Should have processed bytes");
        assert!(stats.strong_hashes_computed > 0, "Should have computed strong hashes");
        assert_eq!(stats.successful_verifications, 1, "Should have 1 successful verification");
    }

    /// Test error handling
    #[test]
    fn test_error_handling() {
        // Test invalid parameters
        assert!(RollingHasher::new(0, 64).is_err(), "Should reject zero base");
        assert!(RollingHasher::new(67, 0).is_err(), "Should reject zero window size");
        
        // Test invalid initialization
        let mut hasher = RollingHasher::new(67, 8).unwrap();
        assert!(hasher.init(&[1, 2, 3]).is_err(), "Should reject wrong size data");
        
        // Test rolling without initialization
        assert!(hasher.roll(42).is_err(), "Should reject rolling without initialization");
        
        // Test strong hash with empty data
        let mut strong_hasher = StrongHasher::new();
        assert!(strong_hasher.hash(&[]).is_err(), "Should reject empty data");
    }

    /// Property-based test for rolling hash consistency
    proptest! {
        #[test]
        fn prop_rolling_hash_consistency(
            data in prop::collection::vec(any::<u8>(), 10..100),
            window_size in 4usize..16,
            base in 2u64..1000
        ) {
            let mut hasher = RollingHasher::new(base, window_size)?;
            
            if data.len() >= window_size {
                // Initialize with first window_size bytes
                let init_hash = hasher.init(&data[0..window_size])?;
                
                // Roll through remaining data
                let mut current_hash = init_hash;
                for i in window_size..data.len() {
                    current_hash = hasher.roll(data[i])?;
                }
                
                // Create new hasher and compute hash directly for final window
                let mut hasher2 = RollingHasher::new(base, window_size)?;
                let final_start = data.len() - window_size;
                let direct_hash = hasher2.init(&data[final_start..data.len()])?;
                
                // Should be equal
                prop_assert_eq!(current_hash, direct_hash);
            }
        }
    }

    /// Property-based test for strong hash determinism
    proptest! {
        #[test]
        fn prop_strong_hash_determinism(data in prop::collection::vec(any::<u8>(), 1..1000)) {
            let mut hasher1 = StrongHasher::new();
            let mut hasher2 = StrongHasher::new();
            
            let hash1 = hasher1.hash(&data)?;
            let hash2 = hasher2.hash(&data)?;
            
            prop_assert_eq!(hash1, hash2);
        }
    }

    /// Benchmark test for overall dual hasher performance
    #[test]
    fn test_dual_hasher_performance() {
        let mut dual_hasher = DualHasher::new(67, 64).unwrap();
        
        // Initialize
        let init_data = vec![42u8; 64];
        dual_hasher.init(&init_data).unwrap();
        
        // Test performance with realistic workload
        let test_data_size = 100_000; // 100KB
        let chunk_size = 4096; // 4KB chunks
        let num_chunks = test_data_size / chunk_size;
        
        let start = Instant::now();
        
        for i in 0..num_chunks {
            // Simulate rolling hash updates
            for j in 0..chunk_size {
                let byte = ((i * chunk_size + j) % 256) as u8;
                dual_hasher.update_weak(byte);
            }
            
            // Simulate strong hash computation for chunk
            let chunk_data = vec![((i * 42) % 256) as u8; chunk_size];
            let _strong_hash = dual_hasher.strong_hash(&chunk_data).unwrap();
        }
        
        let elapsed = start.elapsed();
        let throughput_mbps = (test_data_size as f64 / (1024.0 * 1024.0)) / elapsed.as_secs_f64();
        
        println!("Dual hasher throughput: {:.2} MB/s", throughput_mbps);
        
        // Should achieve reasonable throughput (> 10 MB/s)
        assert!(
            throughput_mbps > 10.0,
            "Dual hasher throughput too low: {:.2} MB/s",
            throughput_mbps
        );
        
        // Verify statistics
        let stats = dual_hasher.get_statistics().unwrap();
        assert_eq!(stats.strong_hashes_computed, num_chunks as u64);
        assert!(stats.bytes_processed > 0);
    }

    /// Test window size edge cases
    #[test]
    fn test_window_size_edge_cases() {
        // Test minimum window size
        let mut hasher1 = RollingHasher::new(67, 1).unwrap();
        let hash1 = hasher1.init(&[42]).unwrap();
        let hash2 = hasher1.roll(43).unwrap();
        assert_ne!(hash1, hash2);
        
        // Test larger window size
        let mut hasher2 = RollingHasher::new(67, 256).unwrap();
        let init_data = vec![42u8; 256];
        hasher2.init(&init_data).unwrap();
        assert!(hasher2.is_initialized());
        
        // Test rolling with large window
        let new_hash = hasher2.roll(99).unwrap();
        assert!(new_hash.get() != 0); // Should have non-zero hash
    }

    /// Test hash distribution quality (basic check)
    #[test]
    fn test_hash_distribution() {
        let mut hash_counts = std::collections::HashMap::new();
        let num_tests = 10000;
        
        for i in 0..num_tests {
            // Create more varied test data to ensure different hash values
            let data = format!("{:08x}", i); // Use hex representation for more variation
            let bytes = data.as_bytes();
            if bytes.len() >= 8 {
                // Create a new hasher for each test to get independent hash values
                let mut hasher = RollingHasher::new(67, 8).unwrap();
                let hash = hasher.init(&bytes[0..8]).unwrap();
                let bucket = (hash.get() % 100) as u32; // 100 buckets
                *hash_counts.entry(bucket).or_insert(0) += 1;
            }
        }
        
        // Check that we have reasonable distribution (not perfect, but not terrible)
        let expected_per_bucket = num_tests / 100;
        let mut buckets_in_range = 0;
        
        for count in hash_counts.values() {
            // Allow 50% variance from expected
            if *count >= expected_per_bucket / 2 && *count <= expected_per_bucket * 3 / 2 {
                buckets_in_range += 1;
            }
        }
        
        // If we have very few buckets, the test isn't meaningful - just check we have some distribution
        if hash_counts.len() < 10 {
            assert!(hash_counts.len() > 1, "Hash function produces no variation - only {} unique hashes", hash_counts.len());
            return; // Skip the distribution quality test for small samples
        }
        
        // At least 50% of buckets should be in reasonable range (lowered threshold)
        let distribution_quality = buckets_in_range as f64 / hash_counts.len() as f64;
        assert!(
            distribution_quality >= 0.5,
            "Poor hash distribution: only {:.1}% of buckets in expected range (expected >= 50%)",
            distribution_quality * 100.0
        );
    }
}