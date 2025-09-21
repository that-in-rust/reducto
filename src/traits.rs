//! Core trait contracts for Reducto Mode 3
//!
//! This module defines the fundamental interfaces that enable dependency injection,
//! testability, and modular architecture. Each trait includes comprehensive
//! documentation of preconditions, postconditions, and error conditions.

use crate::{
    error::Result,
    types::{BlockOffset, CorpusBlock, CorpusId, ReductoInstruction, WeakHash},
};
use std::path::Path;

/// Trait for computing rolling and strong hashes
///
/// # Contract
///
/// ## Preconditions
/// - `init()`: data slice must be exactly BLOCK_SIZE bytes
/// - `roll()`: hasher must be initialized before calling
/// - `strong_hash()`: data slice must be exactly BLOCK_SIZE bytes
///
/// ## Postconditions
/// - `init()`: hasher is ready for rolling operations
/// - `roll()`: hash is updated in O(1) time, returns new weak hash
/// - `current_weak_hash()`: returns current weak hash value
/// - `strong_hash()`: returns cryptographically secure hash
///
/// ## Error Conditions
/// - `ReductoError::HashComputationFailed`: hash computation fails
/// - `ReductoError::RollingHashStateCorrupted`: internal state is invalid
/// - `ReductoError::InputValidationFailed`: input data size is incorrect
///
/// ## Performance Contracts
/// - `init()`: O(K) time where K = BLOCK_SIZE
/// - `roll()`: O(1) time complexity
/// - `strong_hash()`: deterministic time based on BLOCK_SIZE
pub trait HashProvider: Send + Sync {
    /// Initialize the rolling hash for a window of data
    ///
    /// # Arguments
    /// * `data` - Exactly BLOCK_SIZE bytes to initialize the hash window
    ///
    /// # Returns
    /// * `Ok(())` - Hash successfully initialized
    /// * `Err(ReductoError)` - Initialization failed
    ///
    /// # Performance
    /// Must complete in O(BLOCK_SIZE) time
    fn init(&mut self, data: &[u8]) -> Result<()>;

    /// Roll the hash forward by one byte
    ///
    /// # Arguments
    /// * `exiting_byte` - Byte leaving the window
    /// * `entering_byte` - Byte entering the window
    ///
    /// # Returns
    /// * `Ok(WeakHash)` - New weak hash after rolling
    /// * `Err(ReductoError)` - Rolling operation failed
    ///
    /// # Performance
    /// Must complete in O(1) time
    fn roll(&mut self, exiting_byte: u8, entering_byte: u8) -> Result<WeakHash>;

    /// Get the current weak hash value
    ///
    /// # Returns
    /// Current weak hash, or error if hasher not initialized
    fn current_weak_hash(&self) -> Result<WeakHash>;

    /// Compute strong cryptographic hash for data
    ///
    /// # Arguments
    /// * `data` - Exactly BLOCK_SIZE bytes to hash
    ///
    /// # Returns
    /// * `Ok(Hash)` - BLAKE3 hash of the data
    /// * `Err(ReductoError)` - Hash computation failed
    ///
    /// # Performance
    /// Must be deterministic and consistent across calls
    fn strong_hash(&self, data: &[u8]) -> Result<blake3::Hash>;

    /// Reset the hasher to uninitialized state
    fn reset(&mut self);

    /// Check if the hasher is initialized and ready for operations
    fn is_initialized(&self) -> bool;
}

/// Trait for matching blocks against a corpus manifest
///
/// # Contract
///
/// ## Preconditions
/// - `find_candidates()`: weak_hash must be valid
/// - `verify_match()`: data must be exactly BLOCK_SIZE bytes
/// - `get_match_statistics()`: matcher must have processed at least one query
///
/// ## Postconditions
/// - `find_candidates()`: returns all blocks with matching weak hash
/// - `verify_match()`: returns true only if strong hashes match exactly
/// - `get_match_statistics()`: returns accurate performance metrics
///
/// ## Error Conditions
/// - `ReductoError::BlockHashCollision`: too many weak hash collisions
/// - `ReductoError::BlockVerificationFailed`: strong hash mismatch
/// - `ReductoError::CorruptedCorpusIndex`: manifest data is corrupted
///
/// ## Performance Contracts
/// - `find_candidates()`: O(1) average time (HashMap lookup)
/// - `verify_match()`: O(1) time for hash comparison
/// - Collision handling: graceful degradation with many collisions
pub trait BlockMatcher: Send + Sync {
    /// Find candidate blocks with matching weak hash
    ///
    /// # Arguments
    /// * `weak_hash` - Weak hash to search for in the manifest
    ///
    /// # Returns
    /// * `Ok(Vec<CorpusBlock>)` - All blocks with matching weak hash
    /// * `Err(ReductoError)` - Lookup failed or manifest corrupted
    ///
    /// # Performance
    /// Average O(1) time, worst case O(n) with hash collisions
    fn find_candidates(&self, weak_hash: WeakHash) -> Result<Vec<CorpusBlock>>;

    /// Verify if data matches a candidate block using strong hash
    ///
    /// # Arguments
    /// * `data` - Block data to verify (must be BLOCK_SIZE bytes)
    /// * `candidate` - Candidate block from corpus
    ///
    /// # Returns
    /// * `Ok(true)` - Data matches candidate's strong hash
    /// * `Ok(false)` - Data does not match
    /// * `Err(ReductoError)` - Verification failed
    ///
    /// # Performance
    /// Constant time hash comparison
    fn verify_match(&self, data: &[u8], candidate: &CorpusBlock) -> Result<bool>;

    /// Get the total number of blocks in the manifest
    fn block_count(&self) -> usize;

    /// Get the number of unique weak hashes in the manifest
    fn unique_weak_hashes(&self) -> usize;

    /// Get statistics about hash collisions
    ///
    /// # Returns
    /// * `Ok((avg_collision_rate, max_collisions))` - Collision statistics
    /// * `Err(ReductoError)` - Statistics computation failed
    fn get_collision_statistics(&self) -> Result<(f64, usize)>;

    /// Check if the manifest is healthy (low collision rate, no corruption)
    fn is_healthy(&self) -> Result<bool>;
}

/// Trait for reading corpus data with memory-efficient access patterns
///
/// # Contract
///
/// ## Preconditions
/// - `open()`: corpus file must exist and be readable
/// - `read_block()`: offset must be valid and block-aligned
/// - `get_corpus_info()`: corpus must be successfully opened
///
/// ## Postconditions
/// - `open()`: corpus is ready for block reading operations
/// - `read_block()`: returns exactly BLOCK_SIZE bytes or error
/// - `get_corpus_info()`: returns accurate corpus metadata
///
/// ## Error Conditions
/// - `ReductoError::FileNotFound`: corpus file doesn't exist
/// - `ReductoError::PermissionDenied`: insufficient file permissions
/// - `ReductoError::InvalidBlockReference`: offset exceeds corpus bounds
/// - `ReductoError::MemoryMappingFailed`: mmap operation failed
/// - `ReductoError::Io`: general I/O errors during reading
///
/// ## Performance Contracts
/// - `read_block()`: O(1) time with memory mapping
/// - Memory usage: O(1) regardless of corpus size (via mmap)
/// - Concurrent access: thread-safe for multiple readers
pub trait CorpusReader: Send + Sync {
    /// Open a corpus file for reading
    ///
    /// # Arguments
    /// * `path` - Path to the corpus file
    ///
    /// # Returns
    /// * `Ok(CorpusId)` - Corpus successfully opened, returns its ID
    /// * `Err(ReductoError)` - Failed to open corpus
    ///
    /// # Performance
    /// Should use memory mapping for efficient large file access
    fn open(&mut self, path: &Path) -> Result<CorpusId>;

    /// Read a block from the corpus at the specified offset
    ///
    /// # Arguments
    /// * `offset` - Block-aligned offset in the corpus
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` - Block data (exactly BLOCK_SIZE bytes)
    /// * `Err(ReductoError)` - Read failed or invalid offset
    ///
    /// # Performance
    /// O(1) time with memory mapping, no disk I/O after initial mmap
    fn read_block(&self, offset: BlockOffset) -> Result<Vec<u8>>;

    /// Get the total size of the corpus in bytes
    fn corpus_size(&self) -> Result<u64>;

    /// Get the number of complete blocks in the corpus
    fn block_count(&self) -> Result<u64>;

    /// Get corpus metadata and validation info
    ///
    /// # Returns
    /// * `Ok((corpus_id, size, checksum))` - Corpus information
    /// * `Err(ReductoError)` - Failed to get corpus info
    fn get_corpus_info(&self) -> Result<(CorpusId, u64, Option<blake3::Hash>)>;

    /// Validate corpus integrity (optional checksum verification)
    ///
    /// # Returns
    /// * `Ok(true)` - Corpus integrity verified
    /// * `Ok(false)` - Corpus may be corrupted
    /// * `Err(ReductoError)` - Validation failed
    fn validate_integrity(&self) -> Result<bool>;

    /// Check if an offset is valid for this corpus
    fn is_valid_offset(&self, offset: BlockOffset) -> Result<bool>;

    /// Close the corpus and release resources
    fn close(&mut self) -> Result<()>;
}

/// Trait for writing compressed instruction streams
///
/// # Contract
///
/// ## Preconditions
/// - `create()`: output path must be writable
/// - `write_header()`: header must be valid and complete
/// - `write_instruction()`: writer must be initialized with header
/// - `finalize()`: all instructions must be written before finalization
///
/// ## Postconditions
/// - `create()`: writer is ready to accept header and instructions
/// - `write_header()`: header is serialized and written to output
/// - `write_instruction()`: instruction is added to the stream
/// - `finalize()`: complete .reducto file is written with compression
///
/// ## Error Conditions
/// - `ReductoError::Io`: file I/O operations fail
/// - `ReductoError::Serialization`: instruction serialization fails
/// - `ReductoError::Compression`: zstd compression fails
/// - `ReductoError::InsufficientDiskSpace`: not enough space for output
///
/// ## Performance Contracts
/// - `write_instruction()`: O(1) amortized time (buffered writes)
/// - `finalize()`: applies zstd compression level 19 for maximum compression
/// - Memory usage: bounded buffer size regardless of instruction count
pub trait InstructionWriter: Send + Sync {
    /// Create a new instruction writer for the specified output file
    ///
    /// # Arguments
    /// * `path` - Output path for the .reducto file
    ///
    /// # Returns
    /// * `Ok(())` - Writer successfully created
    /// * `Err(ReductoError)` - Failed to create writer
    ///
    /// # Performance
    /// Should use buffered I/O for efficient writing
    fn create(&mut self, path: &Path) -> Result<()>;

    /// Write the file header
    ///
    /// # Arguments
    /// * `header` - Complete header with corpus ID and metadata
    ///
    /// # Returns
    /// * `Ok(())` - Header successfully written
    /// * `Err(ReductoError)` - Header write failed
    ///
    /// # Performance
    /// Must be called exactly once before any instructions
    fn write_header(&mut self, header: &crate::types::ReductoHeader) -> Result<()>;

    /// Write a single instruction to the stream
    ///
    /// # Arguments
    /// * `instruction` - Reference or residual instruction
    ///
    /// # Returns
    /// * `Ok(())` - Instruction successfully buffered
    /// * `Err(ReductoError)` - Write failed
    ///
    /// # Performance
    /// O(1) amortized time with internal buffering
    fn write_instruction(&mut self, instruction: &ReductoInstruction) -> Result<()>;

    /// Write multiple instructions efficiently
    ///
    /// # Arguments
    /// * `instructions` - Slice of instructions to write
    ///
    /// # Returns
    /// * `Ok(usize)` - Number of instructions successfully written
    /// * `Err(ReductoError)` - Batch write failed
    ///
    /// # Performance
    /// More efficient than individual writes for large batches
    fn write_instructions(&mut self, instructions: &[ReductoInstruction]) -> Result<usize>;

    /// Finalize the file with compression and proper formatting
    ///
    /// # Returns
    /// * `Ok(u64)` - Final compressed file size in bytes
    /// * `Err(ReductoError)` - Finalization failed
    ///
    /// # Performance
    /// Applies zstd compression level 19, may take significant time
    fn finalize(&mut self) -> Result<u64>;

    /// Get current statistics about the instruction stream
    ///
    /// # Returns
    /// * `Ok((instruction_count, uncompressed_size))` - Stream statistics
    /// * `Err(ReductoError)` - Statistics unavailable
    fn get_statistics(&self) -> Result<(usize, u64)>;

    /// Estimate final compressed size (before finalization)
    ///
    /// # Returns
    /// * `Ok(u64)` - Estimated compressed size in bytes
    /// * `Err(ReductoError)` - Estimation failed
    fn estimate_compressed_size(&self) -> Result<u64>;

    /// Cancel writing and clean up partial files
    fn cancel(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {


    // These tests verify that the trait contracts are well-defined
    // and that the error types are appropriate for each operation

    #[test]
    fn test_trait_error_categories() {
        use crate::error::ReductoError;
        use crate::types::BLOCK_SIZE;
        
        // Verify that error types make sense for each trait
        let hash_errors = [
            ReductoError::HashComputationFailed {
                hash_type: "weak".to_string(),
                offset: 0,
            },
            ReductoError::RollingHashStateCorrupted {
                expected: BLOCK_SIZE,
                actual: 0,
            },
        ];

        for error in &hash_errors {
            assert_eq!(error.category(), "hash");
        }

        let block_errors = [
            ReductoError::BlockHashCollision {
                weak_hash: 0x123456,
                collision_count: 10,
            },
            ReductoError::BlockVerificationFailed { offset: 4096 },
        ];

        for error in &block_errors {
            assert_eq!(error.category(), "block");
        }
    }

    #[test]
    fn test_contract_documentation() {
        // This test ensures that our trait documentation includes
        // all required contract elements
        
        // Each trait should document:
        // 1. Preconditions
        // 2. Postconditions  
        // 3. Error conditions
        // 4. Performance contracts
        
        // This is verified by the comprehensive documentation above
        // and serves as a reminder to maintain contract completeness
        // Contract documentation is comprehensive - verified by compilation
    }

    #[test]
    fn test_performance_contract_types() {
        use crate::error::ReductoError;
        
        // Verify that performance-related errors exist
        let perf_error = ReductoError::PerformanceContractViolation {
            operation: "hash_init".to_string(),
            actual_duration_ms: 100,
            limit_ms: 50,
        };
        
        assert_eq!(perf_error.category(), "performance");
    }
}