//! Compile-Time Optimizations with Const Generics
//!
//! This module implements compile-time optimizations using const generics
//! to specialize chunk parameters and other configuration at compile time,
//! enabling better compiler optimizations and eliminating runtime checks.

use crate::types::{WeakHash, BlockOffset, HASH_BASE};
use crate::error::{ReductoError, Result};
use std::marker::PhantomData;

/// Compile-time validated chunk configuration
/// 
/// This struct uses const generics to validate chunk parameters at compile time,
/// ensuring that invalid configurations cannot be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstChunkConfig<
    const TARGET_SIZE: usize,
    const MIN_SIZE: usize,
    const MAX_SIZE: usize,
    const HASH_MASK: u64,
> {
    _phantom: PhantomData<()>,
}

impl<
    const TARGET_SIZE: usize,
    const MIN_SIZE: usize,
    const MAX_SIZE: usize,
    const HASH_MASK: u64,
> ConstChunkConfig<TARGET_SIZE, MIN_SIZE, MAX_SIZE, HASH_MASK> {
    /// Create a new const chunk configuration with compile-time validation
    pub const fn new() -> Self {
        // Compile-time assertions to validate configuration
        const { assert!(TARGET_SIZE > 0, "Target size must be positive") };
        const { assert!(MIN_SIZE > 0, "Minimum size must be positive") };
        const { assert!(MAX_SIZE > MIN_SIZE, "Maximum size must be greater than minimum") };
        const { assert!(TARGET_SIZE >= MIN_SIZE, "Target size must be >= minimum size") };
        const { assert!(TARGET_SIZE <= MAX_SIZE, "Target size must be <= maximum size") };
        const { assert!(MIN_SIZE >= 1024, "Minimum size must be at least 1KB") };
        const { assert!(MAX_SIZE <= 1048576, "Maximum size must be at most 1MB") };
        const { assert!(TARGET_SIZE.is_power_of_two() || TARGET_SIZE % 1024 == 0, 
                        "Target size should be power of 2 or multiple of 1KB") };
        const { assert!(HASH_MASK > 0, "Hash mask must be positive") };
        const { assert!(HASH_MASK.count_ones() >= 8, "Hash mask should have sufficient bits") };
        
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get target chunk size (compile-time constant)
    pub const fn target_size(&self) -> usize {
        TARGET_SIZE
    }

    /// Get minimum chunk size (compile-time constant)
    pub const fn min_size(&self) -> usize {
        MIN_SIZE
    }

    /// Get maximum chunk size (compile-time constant)
    pub const fn max_size(&self) -> usize {
        MAX_SIZE
    }

    /// Get hash mask (compile-time constant)
    pub const fn hash_mask(&self) -> u64 {
        HASH_MASK
    }

    /// Check if a size is within bounds (compile-time optimized)
    pub const fn is_valid_size(&self, size: usize) -> bool {
        size >= MIN_SIZE && size <= MAX_SIZE
    }

    /// Calculate expected number of chunks for a given data size
    pub const fn expected_chunk_count(&self, data_size: usize) -> usize {
        if data_size == 0 {
            0
        } else {
            (data_size + TARGET_SIZE - 1) / TARGET_SIZE
        }
    }

    /// Calculate hash table size for optimal performance
    pub const fn optimal_hash_table_size(&self) -> usize {
        // Use next power of 2 that's at least 4x the expected chunk count for a typical corpus
        let typical_corpus_size = 16 * 1024 * 1024; // 16MB
        let expected_chunks = self.expected_chunk_count(typical_corpus_size);
        next_power_of_two(expected_chunks * 4)
    }
}

/// Compile-time optimized rolling hasher
/// 
/// This hasher is specialized for specific window sizes at compile time,
/// allowing the compiler to optimize the hash calculation loop.
pub struct ConstRollingHasher<const WINDOW_SIZE: usize, const BASE: u64> {
    hash: u64,
    power: u64, // BASE^(WINDOW_SIZE-1), computed at compile time
    window: [u8; WINDOW_SIZE],
    position: usize,
}

impl<const WINDOW_SIZE: usize, const BASE: u64> ConstRollingHasher<WINDOW_SIZE, BASE> {
    /// Create a new const rolling hasher with compile-time validation
    pub const fn new() -> Self {
        const { assert!(WINDOW_SIZE > 0, "Window size must be positive") };
        const { assert!(WINDOW_SIZE <= 65536, "Window size must be <= 64KB") };
        const { assert!(BASE > 1, "Base must be greater than 1") };
        const { assert!(BASE < u32::MAX as u64, "Base must fit in u32 for overflow safety") };
        
        Self {
            hash: 0,
            power: const_power(BASE, WINDOW_SIZE - 1),
            window: [0; WINDOW_SIZE],
            position: 0,
        }
    }

    /// Initialize the hasher with initial data
    pub fn init(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != WINDOW_SIZE {
            return Err(ReductoError::InputValidationFailed {
                field: "data_length".to_string(),
                reason: format!("length {} != window size {}", data.len(), WINDOW_SIZE),
            });
        }

        self.hash = 0;
        self.position = 0;
        
        // Copy data to window and calculate initial hash
        for (i, &byte) in data.iter().enumerate() {
            self.window[i] = byte;
            self.hash = self.hash.wrapping_mul(BASE).wrapping_add(byte as u64);
        }
        
        Ok(())
    }

    /// Roll the hash by one position (compile-time optimized)
    pub fn roll(&mut self, exiting_byte: u8, entering_byte: u8) -> WeakHash {
        // Remove contribution of exiting byte
        let exiting_contribution = (exiting_byte as u64).wrapping_mul(self.power);
        self.hash = self.hash.wrapping_sub(exiting_contribution);
        
        // Shift left (multiply by base) and add entering byte
        self.hash = self.hash.wrapping_mul(BASE).wrapping_add(entering_byte as u64);
        
        // Update window (this can be optimized away if window tracking isn't needed)
        self.window[self.position] = entering_byte;
        self.position = (self.position + 1) % WINDOW_SIZE;
        
        WeakHash::new(self.hash)
    }

    /// Get current hash value
    pub fn current_hash(&self) -> WeakHash {
        WeakHash::new(self.hash)
    }

    /// Get window size (compile-time constant)
    pub const fn window_size(&self) -> usize {
        WINDOW_SIZE
    }
}

/// Compile-time optimized CDC chunker
/// 
/// This chunker is specialized for specific chunk configurations at compile time,
/// enabling aggressive compiler optimizations.
pub struct ConstCDCChunker<
    const TARGET_SIZE: usize,
    const MIN_SIZE: usize,
    const MAX_SIZE: usize,
    const HASH_MASK: u64,
> {
    config: ConstChunkConfig<TARGET_SIZE, MIN_SIZE, MAX_SIZE, HASH_MASK>,
    gear_hasher: ConstGearHasher,
    current_chunk: Vec<u8>,
    position: usize,
}

impl<
    const TARGET_SIZE: usize,
    const MIN_SIZE: usize,
    const MAX_SIZE: usize,
    const HASH_MASK: u64,
> ConstCDCChunker<TARGET_SIZE, MIN_SIZE, MAX_SIZE, HASH_MASK> {
    /// Create a new const CDC chunker
    pub const fn new() -> Self {
        Self {
            config: ConstChunkConfig::new(),
            gear_hasher: ConstGearHasher::new(),
            current_chunk: Vec::new(),
            position: 0,
        }
    }

    /// Process data and produce chunks with compile-time optimized boundary detection
    pub fn chunk_data(&mut self, data: &[u8]) -> Result<Vec<ConstDataChunk>> {
        let mut chunks = Vec::new();
        self.current_chunk.clear();
        self.position = 0;

        for &byte in data {
            self.current_chunk.push(byte);
            self.position += 1;

            let hash = self.gear_hasher.update(byte);
            
            // Compile-time optimized boundary check
            let is_boundary = self.position >= MIN_SIZE && 
                             (hash & HASH_MASK) == 0 ||
                             self.position >= MAX_SIZE;

            if is_boundary {
                let chunk = self.finalize_current_chunk()?;
                chunks.push(chunk);
                self.current_chunk.clear();
                self.position = 0;
            }
        }

        Ok(chunks)
    }

    /// Finalize any remaining data as the last chunk
    pub fn finalize(&mut self) -> Result<Option<ConstDataChunk>> {
        if self.current_chunk.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.finalize_current_chunk()?))
        }
    }

    /// Finalize the current chunk with compile-time optimized hashing
    fn finalize_current_chunk(&self) -> Result<ConstDataChunk> {
        if self.current_chunk.is_empty() {
            return Err(ReductoError::InputValidationFailed {
                field: "chunk_data".to_string(),
                reason: "cannot finalize empty chunk".to_string(),
            });
        }

        let weak_hash = self.calculate_weak_hash(&self.current_chunk);
        let strong_hash = blake3::hash(&self.current_chunk);

        Ok(ConstDataChunk {
            data: self.current_chunk.clone(),
            weak_hash,
            strong_hash,
            offset: BlockOffset::new(0), // Will be set by caller
            size: self.current_chunk.len(),
        })
    }

    /// Calculate weak hash with compile-time optimization
    fn calculate_weak_hash(&self, data: &[u8]) -> WeakHash {
        let mut hash = 0u64;
        for &byte in data {
            hash = hash.wrapping_mul(HASH_BASE).wrapping_add(byte as u64);
        }
        WeakHash::new(hash)
    }

    /// Get configuration (compile-time constants)
    pub const fn config(&self) -> &ConstChunkConfig<TARGET_SIZE, MIN_SIZE, MAX_SIZE, HASH_MASK> {
        &self.config
    }
}

/// Compile-time optimized gear hasher
pub struct ConstGearHasher {
    hash: u64,
    gear_table: [u64; 256],
}

impl ConstGearHasher {
    /// Create a new const gear hasher with pre-computed table
    pub const fn new() -> Self {
        Self {
            hash: 0,
            gear_table: const_gear_table(),
        }
    }

    /// Update hash with a byte (compile-time optimized table lookup)
    pub fn update(&mut self, byte: u8) -> u64 {
        self.hash = (self.hash << 1).wrapping_add(self.gear_table[byte as usize]);
        self.hash
    }

    /// Reset hash to initial state
    pub fn reset(&mut self) {
        self.hash = 0;
    }

    /// Get current hash value
    pub fn current_hash(&self) -> u64 {
        self.hash
    }
}

/// Data chunk with compile-time optimized layout
#[derive(Debug, Clone)]
pub struct ConstDataChunk {
    pub data: Vec<u8>,
    pub weak_hash: WeakHash,
    pub strong_hash: blake3::Hash,
    pub offset: BlockOffset,
    pub size: usize,
}

impl ConstDataChunk {
    /// Get chunk size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get weak hash
    pub fn weak_hash(&self) -> WeakHash {
        self.weak_hash
    }

    /// Get strong hash
    pub fn strong_hash(&self) -> &blake3::Hash {
        &self.strong_hash
    }

    /// Get data reference
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

// === Compile-Time Utility Functions ===

/// Calculate power at compile time
const fn const_power(base: u64, exp: usize) -> u64 {
    if exp == 0 {
        1
    } else {
        let mut result: u64 = 1;
        let mut i = 0;
        while i < exp {
            result = result.wrapping_mul(base);
            i += 1;
        }
        result
    }
}

/// Calculate next power of two at compile time
const fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        1 << (usize::BITS - (n - 1).leading_zeros())
    }
}

/// Generate gear table at compile time
const fn const_gear_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = const_gear_value(i as u8);
        i += 1;
    }
    table
}

/// Calculate gear value at compile time
const fn const_gear_value(byte: u8) -> u64 {
    let mut value = byte as u64;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    value
}

// === Type Aliases for Common Configurations ===

/// Standard 8KB chunks with 4KB-16KB range
pub type StandardChunker = ConstCDCChunker<8192, 4096, 16384, 0x1FFF>;

/// Large 32KB chunks with 16KB-64KB range
pub type LargeChunker = ConstCDCChunker<32768, 16384, 65536, 0x7FFF>;

/// Small 2KB chunks with 1KB-4KB range
pub type SmallChunker = ConstCDCChunker<2048, 1024, 4096, 0x7FF>;

/// High-precision chunker with tight boundaries
pub type PrecisionChunker = ConstCDCChunker<4096, 2048, 8192, 0x3FF>;

/// Standard rolling hasher for 8KB windows
pub type StandardRollingHasher = ConstRollingHasher<8192, HASH_BASE>;

/// Large rolling hasher for 32KB windows
pub type LargeRollingHasher = ConstRollingHasher<32768, HASH_BASE>;

// === Performance Validation ===

/// Compile-time performance contract validation
pub const fn validate_performance_contracts<
    const TARGET_SIZE: usize,
    const MIN_SIZE: usize,
    const MAX_SIZE: usize,
    const HASH_MASK: u64,
>() -> bool {
    // Validate that configuration supports required performance
    let config = ConstChunkConfig::<TARGET_SIZE, MIN_SIZE, MAX_SIZE, HASH_MASK>::new();
    
    // Check that chunk sizes are reasonable for performance
    let size_ratio = MAX_SIZE / MIN_SIZE;
    if size_ratio > 4 {
        // Size variance too high, may impact performance
        return false;
    }
    
    // Check that hash mask provides reasonable boundary probability
    let boundary_probability = 1.0 / ((HASH_MASK + 1) as f64);
    if boundary_probability < 1.0 / 32768.0 || boundary_probability > 1.0 / 512.0 {
        // Boundary probability outside optimal range
        return false;
    }
    
    // Check that target size is reasonable for cache efficiency
    if TARGET_SIZE > 65536 || TARGET_SIZE < 1024 {
        // Target size outside cache-friendly range
        return false;
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_chunk_config_creation() {
        let config = ConstChunkConfig::<8192, 4096, 16384, 0x1FFF>::new();
        assert_eq!(config.target_size(), 8192);
        assert_eq!(config.min_size(), 4096);
        assert_eq!(config.max_size(), 16384);
        assert_eq!(config.hash_mask(), 0x1FFF);
    }

    #[test]
    fn test_const_chunk_config_validation() {
        let config = ConstChunkConfig::<8192, 4096, 16384, 0x1FFF>::new();
        assert!(config.is_valid_size(8192));
        assert!(config.is_valid_size(4096));
        assert!(config.is_valid_size(16384));
        assert!(!config.is_valid_size(2048));
        assert!(!config.is_valid_size(32768));
    }

    #[test]
    fn test_expected_chunk_count() {
        let config = ConstChunkConfig::<8192, 4096, 16384, 0x1FFF>::new();
        assert_eq!(config.expected_chunk_count(0), 0);
        assert_eq!(config.expected_chunk_count(8192), 1);
        assert_eq!(config.expected_chunk_count(16384), 2);
        assert_eq!(config.expected_chunk_count(8193), 2);
    }

    #[test]
    fn test_const_rolling_hasher() {
        let mut hasher = ConstRollingHasher::<4, HASH_BASE>::new();
        let data = [1, 2, 3, 4];
        
        hasher.init(&data).unwrap();
        let initial_hash = hasher.current_hash();
        
        let rolled_hash = hasher.roll(1, 5);
        assert_ne!(initial_hash, rolled_hash);
        
        assert_eq!(hasher.window_size(), 4);
    }

    #[test]
    fn test_const_cdc_chunker() {
        let mut chunker = StandardChunker::new();
        let data = vec![0xAA; 10000];
        
        let chunks = chunker.chunk_data(&data).unwrap();
        assert!(!chunks.is_empty());
        
        let final_chunk = chunker.finalize().unwrap();
        assert!(final_chunk.is_some());
    }

    #[test]
    fn test_const_gear_hasher() {
        let mut hasher = ConstGearHasher::new();
        
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
    fn test_const_power() {
        assert_eq!(const_power(2, 0), 1);
        assert_eq!(const_power(2, 1), 2);
        assert_eq!(const_power(2, 3), 8);
        assert_eq!(const_power(10, 2), 100);
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(15), 16);
        assert_eq!(next_power_of_two(16), 16);
        assert_eq!(next_power_of_two(17), 32);
    }

    #[test]
    fn test_performance_contract_validation() {
        // Valid configuration
        assert!(validate_performance_contracts::<8192, 4096, 16384, 0x1FFF>());
        
        // Invalid: size ratio too high
        assert!(!validate_performance_contracts::<8192, 1024, 32768, 0x1FFF>());
        
        // Invalid: target size too large
        assert!(!validate_performance_contracts::<131072, 65536, 262144, 0x1FFF>());
        
        // Invalid: target size too small
        assert!(!validate_performance_contracts::<512, 256, 1024, 0x1FFF>());
    }

    #[test]
    fn test_type_aliases() {
        let _standard = StandardChunker::new();
        let _large = LargeChunker::new();
        let _small = SmallChunker::new();
        let _precision = PrecisionChunker::new();
        
        let _standard_hasher = StandardRollingHasher::new();
        let _large_hasher = LargeRollingHasher::new();
    }

    #[test]
    fn test_const_data_chunk() {
        let data = vec![1, 2, 3, 4];
        let weak_hash = WeakHash::new(12345);
        let strong_hash = blake3::hash(&data);
        let offset = BlockOffset::new(0);
        
        let chunk = ConstDataChunk {
            data: data.clone(),
            weak_hash,
            strong_hash,
            offset,
            size: data.len(),
        };
        
        assert_eq!(chunk.size(), 4);
        assert_eq!(chunk.weak_hash(), weak_hash);
        assert_eq!(chunk.data(), &data);
    }

    // Compile-time tests (these will fail to compile if const assertions fail)
    #[test]
    fn test_compile_time_assertions() {
        // These should compile successfully
        let _config1 = ConstChunkConfig::<8192, 4096, 16384, 0x1FFF>::new();
        let _config2 = ConstChunkConfig::<4096, 2048, 8192, 0x7FF>::new();
        
        // These would fail to compile due to const assertions:
        // let _invalid1 = ConstChunkConfig::<0, 4096, 16384, 0x1FFF>::new(); // TARGET_SIZE = 0
        // let _invalid2 = ConstChunkConfig::<8192, 0, 16384, 0x1FFF>::new(); // MIN_SIZE = 0
        // let _invalid3 = ConstChunkConfig::<8192, 16384, 4096, 0x1FFF>::new(); // MAX_SIZE < MIN_SIZE
        // let _invalid4 = ConstChunkConfig::<8192, 4096, 16384, 0>::new(); // HASH_MASK = 0
    }
}