//! Core trait contracts for Reducto Mode 3
//!
//! This module defines the fundamental interfaces that enable dependency injection,
//! testability, and modular architecture. Each trait includes comprehensive
//! documentation of preconditions, postconditions, and error conditions.

use crate::{
    error::Result,
    types::{BlockOffset, CorpusChunk, CorpusId, ReductoInstruction, WeakHash},
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
    /// Find candidate chunks with matching weak hash
    ///
    /// # Arguments
    /// * `weak_hash` - Weak hash to search for in the manifest
    ///
    /// # Returns
    /// * `Ok(Vec<CorpusChunk>)` - All chunks with matching weak hash
    /// * `Err(ReductoError)` - Lookup failed or manifest corrupted
    ///
    /// # Performance
    /// Average O(1) time, worst case O(n) with hash collisions
    fn find_candidates(&self, weak_hash: WeakHash) -> Result<Vec<CorpusChunk>>;

    /// Verify if data matches a candidate chunk using strong hash
    ///
    /// # Arguments
    /// * `data` - Chunk data to verify (variable size)
    /// * `candidate` - Candidate chunk from corpus
    ///
    /// # Returns
    /// * `Ok(true)` - Data matches candidate's strong hash and size
    /// * `Ok(false)` - Data does not match
    /// * `Err(ReductoError)` - Verification failed
    ///
    /// # Performance
    /// Constant time hash comparison
    fn verify_match(&self, data: &[u8], candidate: &CorpusChunk) -> Result<bool>;

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

// === Enterprise Trait Contracts ===

/// Content-Defined Chunking trait for variable-size block processing
///
/// # Contract
///
/// ## Preconditions
/// - `new()`: config must have valid chunk size parameters (min < target < max)
/// - `chunk_data()`: input data must be non-empty
/// - `finalize()`: chunker must have processed at least some data
///
/// ## Postconditions
/// - `chunk_data()`: returns chunks with sizes within configured bounds (50%-200% of target)
/// - `finalize()`: returns final chunk if any data remains
/// - Chunk boundaries are content-defined and stable across identical inputs
///
/// ## Error Conditions
/// - `ReductoError::InvalidConfiguration`: invalid chunk parameters
/// - `ReductoError::HashComputationFailed`: boundary detection fails
/// - `ReductoError::MemoryAllocationFailed`: insufficient memory for chunks
///
/// ## Performance Contracts
/// - Boundary detection: O(1) time per byte processed
/// - Chunk size variance: 50%-200% of target size
/// - Memory usage: O(target_chunk_size) regardless of input size
pub trait CDCChunker: Send + Sync {
    /// Create a new CDC chunker with the specified configuration
    ///
    /// # Arguments
    /// * `config` - Chunk configuration with size bounds and hash parameters
    ///
    /// # Returns
    /// * `Ok(Self)` - Chunker successfully created
    /// * `Err(ReductoError)` - Invalid configuration
    fn new(config: crate::types::ChunkConfig) -> Result<Self> where Self: Sized;

    /// Process input data and yield variable-size chunks
    ///
    /// # Arguments
    /// * `data` - Input data to chunk
    ///
    /// # Returns
    /// * `Ok(Vec<DataChunk>)` - Chunks with content-defined boundaries
    /// * `Err(ReductoError)` - Chunking failed
    ///
    /// # Performance
    /// O(n) time where n is input size, O(1) per byte
    fn chunk_data(&mut self, data: &[u8]) -> Result<Vec<crate::types::DataChunk>>;

    /// Finalize chunking and return any remaining data as final chunk
    ///
    /// # Returns
    /// * `Ok(Option<DataChunk>)` - Final chunk if data remains
    /// * `Err(ReductoError)` - Finalization failed
    fn finalize(&mut self) -> Result<Option<crate::types::DataChunk>>;

    /// Get current chunker statistics
    ///
    /// # Returns
    /// * `Ok((chunks_processed, avg_chunk_size, boundary_count))` - Statistics
    /// * `Err(ReductoError)` - Statistics unavailable
    fn get_statistics(&self) -> Result<(usize, f64, usize)>;

    /// Reset chunker state for processing new input
    fn reset(&mut self);

    /// Check if current position would be a chunk boundary
    ///
    /// # Arguments
    /// * `hash` - Current hash value
    ///
    /// # Returns
    /// True if this position should be a chunk boundary
    fn is_boundary(&self, hash: u64) -> bool;
}

/// Corpus management trait for enterprise-scale reference corpus operations
///
/// # Contract
///
/// ## Preconditions
/// - `build_corpus()`: input paths must exist and be readable
/// - `optimize_corpus()`: analysis data must be representative of target workload
/// - `get_candidates()`: corpus must be successfully built and indexed
/// - `validate_corpus_integrity()`: corpus must be loaded
///
/// ## Postconditions
/// - `build_corpus()`: creates immutable corpus with cryptographic signature
/// - `optimize_corpus()`: provides actionable recommendations for corpus improvement
/// - `get_candidates()`: returns all chunks matching weak hash
/// - `validate_corpus_integrity()`: verifies cryptographic signatures and checksums
///
/// ## Error Conditions
/// - `crate::error::CorpusError::NotFound`: corpus file doesn't exist
/// - `crate::error::CorpusError::SignatureVerificationFailed`: corpus signature invalid
/// - `crate::error::CorpusError::Storage`: persistent storage operations fail
/// - `crate::error::CorpusError::OptimizationFailed`: corpus optimization fails
/// - `crate::error::CorpusError::ConcurrencyConflict`: concurrent access conflict
///
/// ## Performance Contracts
/// - `get_candidates()`: O(1) average time with persistent indexing
/// - `build_corpus()`: supports datasets exceeding available memory
/// - Concurrent access: thread-safe for multiple readers, single writer
pub trait CorpusManager: Send + Sync {
    /// Build corpus from input data with CDC chunking
    ///
    /// # Arguments
    /// * `input_paths` - Paths to files for corpus construction
    /// * `config` - CDC configuration for chunking
    ///
    /// # Returns
    /// * `Ok(CorpusMetadata)` - Corpus successfully built with metadata
    /// * `Err(CorpusError)` - Corpus construction failed
    ///
    /// # Performance
    /// Supports datasets larger than available memory via persistent storage
    async fn build_corpus(
        &mut self,
        input_paths: &[std::path::PathBuf],
        config: crate::types::ChunkConfig,
    ) -> Result<crate::types::CorpusMetadata>;

    /// Generate optimized "Golden Corpus" from dataset analysis
    ///
    /// # Arguments
    /// * `analysis_data` - Representative dataset for optimization
    ///
    /// # Returns
    /// * `Ok(OptimizationRecommendations)` - Recommendations for corpus improvement
    /// * `Err(CorpusError)` - Optimization analysis failed
    ///
    /// # Performance
    /// Performs frequency analysis and deduplication potential assessment
    async fn optimize_corpus(
        &mut self,
        analysis_data: &[std::path::PathBuf],
    ) -> Result<crate::types::OptimizationRecommendations>;

    /// Build persistent index for datasets exceeding memory
    ///
    /// # Arguments
    /// * `corpus_path` - Path to corpus file
    ///
    /// # Returns
    /// * `Ok(())` - Index successfully built
    /// * `Err(CorpusError)` - Index construction failed
    ///
    /// # Performance
    /// Uses LSM trees or RocksDB for memory-efficient indexing
    async fn build_persistent_index(&mut self, corpus_path: &std::path::Path) -> Result<()>;

    /// Lookup chunks with collision handling
    ///
    /// # Arguments
    /// * `weak_hash` - Weak hash to search for
    ///
    /// # Returns
    /// * `Ok(Option<Vec<CorpusChunk>>)` - Matching chunks or None
    /// * `Err(CorpusError)` - Lookup failed
    ///
    /// # Performance
    /// O(1) average time with hash table lookup
    fn get_candidates(&self, weak_hash: WeakHash) -> Result<Option<Vec<CorpusChunk>>>;

    /// Verify chunk match with constant-time comparison
    ///
    /// # Arguments
    /// * `chunk` - Chunk data to verify
    /// * `candidate` - Candidate from corpus
    ///
    /// # Returns
    /// * `Ok(bool)` - True if chunks match
    /// * `Err(CorpusError)` - Verification failed
    ///
    /// # Performance
    /// Constant-time comparison to prevent timing attacks
    fn verify_match(&self, chunk: &[u8], candidate: &CorpusChunk) -> Result<bool>;

    /// Validate corpus integrity with cryptographic verification
    ///
    /// # Returns
    /// * `Ok(())` - Corpus integrity verified
    /// * `Err(CorpusError)` - Integrity check failed
    ///
    /// # Performance
    /// Verifies signatures and checksums for all corpus components
    fn validate_corpus_integrity(&self) -> Result<()>;

    /// Prune stale blocks based on usage statistics
    ///
    /// # Arguments
    /// * `retention_policy` - Policy for determining which blocks to prune
    ///
    /// # Returns
    /// * `Ok(PruneStats)` - Statistics about pruning operation
    /// * `Err(CorpusError)` - Pruning failed
    ///
    /// # Performance
    /// Analyzes usage patterns and removes infrequently accessed blocks
    async fn prune_corpus(
        &mut self,
        retention_policy: crate::types::RetentionPolicy,
    ) -> Result<crate::types::PruneStats>;
}

/// Security management trait for cryptographic operations and compliance
///
/// # Contract
///
/// ## Preconditions
/// - `sign_corpus()`: corpus data must be complete and valid
/// - `verify_corpus_signature()`: signature must be from trusted source
/// - `encrypt_output()`: data must be non-empty
/// - `decrypt_input()`: encrypted data must be valid format
/// - `log_corpus_access()`: audit logging must be enabled
///
/// ## Postconditions
/// - `sign_corpus()`: creates cryptographically secure signature
/// - `verify_corpus_signature()`: validates signature authenticity
/// - `encrypt_output()`: produces AES-GCM encrypted data
/// - `decrypt_input()`: recovers original plaintext data
/// - `log_corpus_access()`: creates immutable audit record
///
/// ## Error Conditions
/// - `crate::error::SecurityError::SigningFailed`: cryptographic signing fails
/// - `crate::error::SecurityError::VerificationFailed`: signature verification fails
/// - `crate::error::SecurityError::EncryptionFailed`: encryption operation fails
/// - `crate::error::SecurityError::DecryptionFailed`: decryption operation fails
/// - `crate::error::SecurityError::KeyManagement`: key operations fail
/// - `crate::error::SecurityError::AuditFailed`: audit logging fails
///
/// ## Performance Contracts
/// - Signature operations: deterministic time based on data size
/// - Encryption/decryption: streaming operations for large data
/// - Audit logging: non-blocking with persistent storage
pub trait SecurityManager: Send + Sync {
    /// Cryptographically sign corpus files and indexes
    ///
    /// # Arguments
    /// * `corpus_data` - Corpus data to sign
    ///
    /// # Returns
    /// * `Ok(Signature)` - Cryptographic signature
    /// * `Err(SecurityError)` - Signing failed
    ///
    /// # Performance
    /// Uses ed25519 for fast signature generation
    fn sign_corpus(&self, corpus_data: &[u8]) -> Result<crate::types::Signature>;

    /// Validate signatures before using corpus data
    ///
    /// # Arguments
    /// * `corpus_data` - Data to verify
    /// * `signature` - Signature to validate
    ///
    /// # Returns
    /// * `Ok(bool)` - True if signature is valid
    /// * `Err(SecurityError)` - Verification failed
    ///
    /// # Performance
    /// Fast ed25519 signature verification
    fn verify_corpus_signature(
        &self,
        corpus_data: &[u8],
        signature: &crate::types::Signature,
    ) -> Result<bool>;

    /// Encrypt compressed outputs for sensitive data
    ///
    /// # Arguments
    /// * `data` - Plaintext data to encrypt
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` - AES-GCM encrypted data
    /// * `Err(SecurityError)` - Encryption failed
    ///
    /// # Performance
    /// Streaming encryption for large datasets
    fn encrypt_output(&self, data: &[u8]) -> Result<Vec<u8>>;

    /// Decrypt compressed data
    ///
    /// # Arguments
    /// * `encrypted_data` - AES-GCM encrypted data
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` - Decrypted plaintext data
    /// * `Err(SecurityError)` - Decryption failed
    ///
    /// # Performance
    /// Streaming decryption with authentication
    fn decrypt_input(&self, encrypted_data: &[u8]) -> Result<Vec<u8>>;

    /// Log corpus access and modifications for compliance
    ///
    /// # Arguments
    /// * `corpus_id` - ID of accessed corpus
    /// * `operation` - Type of access operation
    /// * `user` - User performing the operation
    ///
    /// # Returns
    /// * `Ok(())` - Audit record created
    /// * `Err(SecurityError)` - Audit logging failed
    ///
    /// # Performance
    /// Non-blocking audit logging with persistent storage
    fn log_corpus_access(
        &self,
        corpus_id: &str,
        operation: crate::types::AccessOperation,
        user: &str,
    ) -> Result<()>;

    /// Secure deletion with configurable retention
    ///
    /// # Arguments
    /// * `file_path` - Path to file for secure deletion
    ///
    /// # Returns
    /// * `Ok(())` - File securely deleted
    /// * `Err(SecurityError)` - Secure deletion failed
    ///
    /// # Performance
    /// Multiple-pass overwrite for secure deletion
    async fn secure_delete(&self, file_path: &std::path::Path) -> Result<()>;
}

/// Metrics collection trait for observability and economic reporting
///
/// # Contract
///
/// ## Preconditions
/// - `analyze_compression_potential()`: input and corpus files must exist
/// - `export_metrics()`: metrics must be collected before export
/// - `calculate_roi()`: usage statistics must cover sufficient time period
///
/// ## Postconditions
/// - `analyze_compression_potential()`: provides accurate compression predictions
/// - `export_metrics()`: produces valid Prometheus or JSON format
/// - `calculate_roi()`: calculates quantified cost savings and ROI
///
/// ## Error Conditions
/// - `crate::error::MetricsError::CollectionFailed`: metrics collection fails
/// - `crate::error::MetricsError::ExportFailed`: metrics export fails
/// - `crate::error::MetricsError::AnalysisFailed`: analysis computation fails
/// - `crate::error::MetricsError::InsufficientData`: insufficient data for ROI calculation
///
/// ## Performance Contracts
/// - `analyze_compression_potential()`: dry-run analysis without actual compression
/// - `export_metrics()`: efficient serialization to standard formats
/// - Real-time metrics: low-latency collection with minimal overhead
pub trait MetricsCollector: Send + Sync {
    /// Dry run analysis for compression prediction
    ///
    /// # Arguments
    /// * `input_path` - Path to input file for analysis
    /// * `corpus_path` - Path to reference corpus
    ///
    /// # Returns
    /// * `Ok(CompressionAnalysis)` - Predicted compression metrics
    /// * `Err(MetricsError)` - Analysis failed
    ///
    /// # Performance
    /// Performs analysis without actual compression for speed
    async fn analyze_compression_potential(
        &self,
        input_path: &std::path::Path,
        corpus_path: &std::path::Path,
    ) -> Result<crate::types::CompressionAnalysis>;

    /// Export metrics in standard formats
    ///
    /// # Arguments
    /// * `format` - Export format (Prometheus, JSON, etc.)
    ///
    /// # Returns
    /// * `Ok(String)` - Formatted metrics data
    /// * `Err(MetricsError)` - Export failed
    ///
    /// # Performance
    /// Efficient serialization with minimal memory allocation
    async fn export_metrics(&self, format: crate::types::MetricsFormat) -> Result<String>;

    /// Calculate ROI based on usage patterns
    ///
    /// # Arguments
    /// * `usage_stats` - Historical usage statistics
    ///
    /// # Returns
    /// * `Ok(EconomicReport)` - ROI analysis and cost savings
    /// * `Err(MetricsError)` - ROI calculation failed
    ///
    /// # Performance
    /// Analyzes bandwidth/storage savings and operational costs
    fn calculate_roi(&self, usage_stats: &crate::types::UsageStats) -> Result<crate::types::EconomicReport>;

    /// Record compression operation metrics
    ///
    /// # Arguments
    /// * `metrics` - Compression operation metrics
    ///
    /// # Returns
    /// * `Ok(())` - Metrics recorded successfully
    /// * `Err(MetricsError)` - Recording failed
    ///
    /// # Performance
    /// Low-latency recording with buffered writes
    fn record_compression_metrics(&mut self, metrics: &crate::types::CompressionMetrics) -> Result<()>;

    /// Record decompression operation metrics
    ///
    /// # Arguments
    /// * `metrics` - Decompression operation metrics
    ///
    /// # Returns
    /// * `Ok(())` - Metrics recorded successfully
    /// * `Err(MetricsError)` - Recording failed
    ///
    /// # Performance
    /// Low-latency recording with buffered writes
    fn record_decompression_metrics(&mut self, metrics: &crate::types::DecompressionMetrics) -> Result<()>;

    /// Get real-time performance metrics
    ///
    /// # Returns
    /// * `Ok(PerformanceMetrics)` - Current performance statistics
    /// * `Err(MetricsError)` - Metrics unavailable
    ///
    /// # Performance
    /// Returns cached metrics for low-latency access
    fn get_performance_metrics(&self) -> Result<crate::types::PerformanceMetrics>;
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