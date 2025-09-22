//! Core data types and constants for Reducto Mode 3
//!
//! This module defines the fundamental data structures used throughout the system,
//! following the newtype pattern for type safety and compile-time validation.

use blake3::Hash;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Block size in bytes (4 KiB for optimal performance)
pub const BLOCK_SIZE: usize = 4096;

/// Prime base for rolling hash computation
pub const HASH_BASE: u64 = 67;

/// Magic bytes for .reducto file format identification (enhanced format)
pub const MAGIC_BYTES: [u8; 8] = *b"R3_AB202";

/// Newtype wrapper for block offsets to prevent confusion with other numeric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockOffset(pub u64);

impl BlockOffset {
    /// Create a new block offset
    pub fn new(offset: u64) -> Self {
        Self(offset)
    }

    /// Get the raw offset value
    pub fn get(self) -> u64 {
        self.0
    }

    /// Check if this offset is aligned to block boundaries
    pub fn is_block_aligned(self) -> bool {
        self.0 % BLOCK_SIZE as u64 == 0
    }

    /// Get the block index (offset / BLOCK_SIZE)
    pub fn block_index(self) -> u64 {
        self.0 / BLOCK_SIZE as u64
    }
}

impl fmt::Display for BlockOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockOffset({})", self.0)
    }
}

impl From<u64> for BlockOffset {
    fn from(offset: u64) -> Self {
        Self(offset)
    }
}

impl From<BlockOffset> for u64 {
    fn from(offset: BlockOffset) -> Self {
        offset.0
    }
}

/// Newtype wrapper for weak hash values (rolling polynomial hash)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WeakHash(pub u64);

impl WeakHash {
    /// Create a new weak hash
    pub fn new(hash: u64) -> Self {
        Self(hash)
    }

    /// Get the raw hash value
    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for WeakHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WeakHash(0x{:016x})", self.0)
    }
}

impl From<u64> for WeakHash {
    fn from(hash: u64) -> Self {
        Self(hash)
    }
}

impl From<WeakHash> for u64 {
    fn from(hash: WeakHash) -> Self {
        hash.0
    }
}

/// Newtype wrapper for corpus identifiers to prevent ID confusion
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CorpusId(pub String);

impl CorpusId {
    /// Create a new corpus ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the raw ID string
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Check if the corpus ID is valid (non-empty, reasonable length)
    pub fn is_valid(&self) -> bool {
        !self.0.is_empty() && self.0.len() <= 256 && self.0.chars().all(|c| c.is_ascii_graphic())
    }
}

impl fmt::Display for CorpusId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CorpusId({})", self.0)
    }
}

impl From<String> for CorpusId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for CorpusId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

impl From<CorpusId> for String {
    fn from(id: CorpusId) -> Self {
        id.0
    }
}

/// Represents a variable-size chunk in the Reference Corpus with metadata
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusChunk {
    /// Offset of the chunk in the corpus file
    pub offset: u64,
    /// Size of the chunk in bytes (variable size for CDC)
    pub size: u32,
    /// Strong cryptographic hash (BLAKE3) for verification
    pub strong_hash: Hash,
}

impl CorpusChunk {
    /// Create a new corpus chunk with variable size
    pub fn new(offset: u64, size: u32, strong_hash: Hash) -> Self {
        Self {
            offset,
            size,
            strong_hash,
        }
    }

    /// Verify if the given data matches this chunk's strong hash and size
    pub fn verify_data(&self, data: &[u8]) -> bool {
        if data.len() != self.size as usize {
            return false;
        }
        blake3::hash(data) == self.strong_hash
    }

    /// Get the end offset of this chunk
    pub fn end_offset(&self) -> u64 {
        self.offset + self.size as u64
    }

    /// Check if this chunk overlaps with another chunk
    pub fn overlaps_with(&self, other: &CorpusChunk) -> bool {
        let self_end = self.end_offset();
        let other_end = other.end_offset();
        
        !(self_end <= other.offset || other_end <= self.offset)
    }

    /// Check if the chunk size is within acceptable CDC bounds
    pub fn is_valid_size(&self, config: &ChunkConfig) -> bool {
        config.is_valid_chunk_size(self.size as usize)
    }
}

/// Legacy type alias for backward compatibility
pub type CorpusBlock = CorpusChunk;

/// Instructions for the decompressor to reconstruct the original file
/// Enhanced for variable-size chunks with explicit size information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductoInstruction {
    /// Reference a variable-size chunk from the corpus
    Reference { 
        /// Offset in the corpus file
        offset: u64, 
        /// Size of the referenced chunk in bytes
        size: u32 
    },
    /// Literal data not found in the corpus
    Residual(Vec<u8>),
}

impl ReductoInstruction {
    /// Get the size in bytes that this instruction represents in the output
    pub fn output_size(&self) -> usize {
        match self {
            Self::Reference { size, .. } => *size as usize,
            Self::Residual(data) => data.len(),
        }
    }

    /// Check if this instruction is a reference to the corpus
    pub fn is_reference(&self) -> bool {
        matches!(self, Self::Reference { .. })
    }

    /// Check if this instruction contains residual data
    pub fn is_residual(&self) -> bool {
        matches!(self, Self::Residual(_))
    }

    /// Get the referenced chunk offset and size if this is a reference instruction
    pub fn reference_info(&self) -> Option<(u64, u32)> {
        match self {
            Self::Reference { offset, size } => Some((*offset, *size)),
            Self::Residual(_) => None,
        }
    }

    /// Get the referenced chunk offset if this is a reference instruction
    pub fn reference_offset(&self) -> Option<u64> {
        match self {
            Self::Reference { offset, .. } => Some(*offset),
            Self::Residual(_) => None,
        }
    }

    /// Get the referenced chunk size if this is a reference instruction
    pub fn reference_size(&self) -> Option<u32> {
        match self {
            Self::Reference { size, .. } => Some(*size),
            Self::Residual(_) => None,
        }
    }

    /// Get the residual data if this is a residual instruction
    pub fn residual_data(&self) -> Option<&[u8]> {
        match self {
            Self::Reference { .. } => None,
            Self::Residual(data) => Some(data),
        }
    }

    /// Create a reference instruction from a corpus chunk
    pub fn from_corpus_chunk(chunk: &CorpusChunk) -> Self {
        Self::Reference {
            offset: chunk.offset,
            size: chunk.size,
        }
    }

    /// Validate that a reference instruction has valid parameters
    pub fn validate_reference(&self, corpus_size: u64) -> crate::Result<()> {
        use crate::error::ReductoError;

        if let Self::Reference { offset, size } = self {
            let end_offset = *offset + *size as u64;
            if end_offset > corpus_size {
                return Err(ReductoError::InvalidBlockReference {
                    offset: *offset,
                    max_offset: corpus_size,
                });
            }

            if *size == 0 {
                return Err(ReductoError::InvalidConfiguration {
                    parameter: "reference_size".to_string(),
                    value: size.to_string(),
                    reason: "reference size cannot be zero".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Enhanced header structure for .reducto files with enterprise features
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReductoHeader {
    /// Magic bytes for format identification (b"R3_AB2025")
    pub magic: [u8; 8],
    /// Format version for evolution and compatibility
    pub version: u32,
    /// Immutable corpus GUID for precise identification
    pub corpus_id: uuid::Uuid,
    /// Cryptographic signature for corpus integrity
    pub corpus_signature: Vec<u8>,
    /// CDC configuration parameters used for chunking
    pub chunk_config: ChunkConfig,
    /// End-to-end integrity hash for verification
    pub integrity_hash: Hash,
    /// Zstandard compression level used (1-22)
    pub compression_level: u8,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Optional metadata for extensions
    pub metadata: std::collections::HashMap<String, String>,
}

impl ReductoHeader {
    /// Create a new enhanced header with enterprise features
    pub fn new(
        corpus_id: uuid::Uuid,
        corpus_signature: Vec<u8>,
        chunk_config: ChunkConfig,
        integrity_hash: Hash,
        compression_level: u8,
    ) -> Self {
        Self {
            magic: *b"R3_AB202",  // Updated magic for enhanced format
            version: 2,  // Version 2 for enhanced format
            corpus_id,
            corpus_signature,
            chunk_config,
            integrity_hash,
            compression_level,
            created_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Create a basic header for testing/compatibility
    pub fn basic(corpus_id: uuid::Uuid, chunk_config: ChunkConfig) -> Self {
        Self::new(
            corpus_id,
            Vec::new(),  // Empty signature for basic mode
            chunk_config,
            blake3::hash(b""),  // Empty integrity hash for basic mode
            19,  // Default high compression
        )
    }

    /// Validate that this header is compatible with the current implementation
    pub fn validate(&self) -> crate::Result<()> {
        use crate::error::ReductoError;

        // Check magic bytes for enhanced format
        if self.magic != *b"R3_AB202" {
            return Err(ReductoError::InvalidMagicBytes {
                expected: b"R3_AB202".to_vec(),
                found: self.magic.to_vec(),
            });
        }

        // Check version compatibility (support v1 and v2)
        if self.version < 1 || self.version > 2 {
            return Err(ReductoError::UnsupportedVersion {
                found: self.version.to_string(),
                supported: "1-2".to_string(),
            });
        }

        // Validate chunk configuration
        self.chunk_config.validate()?;

        // Validate compression level
        if self.compression_level < 1 || self.compression_level > 22 {
            return Err(ReductoError::ParameterOutOfRange {
                parameter: "compression_level".to_string(),
                value: self.compression_level as i64,
                min: 1,
                max: 22,
            });
        }

        // Validate corpus ID (UUID should always be valid, but check for nil)
        if self.corpus_id.is_nil() {
            return Err(ReductoError::InputValidationFailed {
                field: "corpus_id".to_string(),
                reason: "Corpus ID cannot be nil UUID".to_string(),
            });
        }

        Ok(())
    }

    /// Add metadata to the header
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Check if this header has a valid signature
    pub fn has_signature(&self) -> bool {
        !self.corpus_signature.is_empty()
    }

    /// Get the expected chunk size variance bounds from the configuration
    pub fn chunk_size_bounds(&self) -> (usize, usize) {
        (self.chunk_config.min_size, self.chunk_config.max_size)
    }

    /// Verify integrity hash against provided data
    pub fn verify_integrity(&self, data: &[u8]) -> bool {
        blake3::hash(data) == self.integrity_hash
    }

    /// Update integrity hash with new data
    pub fn update_integrity_hash(&mut self, data: &[u8]) {
        self.integrity_hash = blake3::hash(data);
    }
}

impl Default for ReductoHeader {
    fn default() -> Self {
        Self::basic(
            uuid::Uuid::new_v4(),
            ChunkConfig::default(),
        )
    }
}

// === Enterprise Types ===

/// Configuration for Content-Defined Chunking with compile-time validation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Target average chunk size (4KB-64KB range enforced)
    pub target_size: usize,
    /// Minimum chunk size (50% of target, enforced at compile-time where possible)
    pub min_size: usize,
    /// Maximum chunk size (200% of target, enforced at compile-time where possible)
    pub max_size: usize,
    /// Gear hash mask for boundary detection (content-defined boundaries)
    pub hash_mask: u64,
    /// Prime base for rolling hash computation
    pub hash_base: u64,
}

/// Compile-time constants for chunk size validation
pub const MIN_CHUNK_SIZE: usize = 4 * 1024;  // 4KB minimum
pub const MAX_CHUNK_SIZE: usize = 64 * 1024; // 64KB maximum
pub const MIN_TARGET_RATIO: f64 = 0.5;       // 50% of target for min_size
pub const MAX_TARGET_RATIO: f64 = 2.0;       // 200% of target for max_size

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_size: 8192,        // 8KB default
            min_size: 4096,           // 4KB minimum
            max_size: 16384,          // 16KB maximum
            hash_mask: 0x07FF,        // 11-bit mask for ~2KB average (more boundaries)
            hash_base: HASH_BASE,     // Use global constant
        }
    }
}

impl ChunkConfig {
    /// Create a new ChunkConfig with validation
    pub fn new(target_size: usize) -> crate::Result<Self> {
        Self::with_ratios(target_size, MIN_TARGET_RATIO, MAX_TARGET_RATIO)
    }

    /// Create ChunkConfig with custom min/max ratios
    pub fn with_ratios(target_size: usize, min_ratio: f64, max_ratio: f64) -> crate::Result<Self> {
        use crate::error::ReductoError;

        // Validate target size is within global bounds
        if target_size < MIN_CHUNK_SIZE || target_size > MAX_CHUNK_SIZE {
            return Err(ReductoError::ParameterOutOfRange {
                parameter: "target_size".to_string(),
                value: target_size as i64,
                min: MIN_CHUNK_SIZE as i64,
                max: MAX_CHUNK_SIZE as i64,
            });
        }

        let min_size = (target_size as f64 * min_ratio) as usize;
        let max_size = (target_size as f64 * max_ratio) as usize;

        // Calculate appropriate hash mask for target size
        // Use a mask that gives approximately the target chunk size
        let mask_bits = (target_size as f64).log2() as u32;
        let hash_mask = (1u64 << mask_bits) - 1;

        let config = Self {
            target_size,
            min_size,
            max_size,
            hash_mask,
            hash_base: HASH_BASE,
        };

        config.validate()?;
        Ok(config)
    }

    /// Validate chunk configuration parameters with comprehensive checks
    pub fn validate(&self) -> crate::Result<()> {
        use crate::error::ReductoError;

        // Check size relationships
        if self.min_size >= self.target_size {
            return Err(ReductoError::InvalidConfiguration {
                parameter: "min_size".to_string(),
                value: self.min_size.to_string(),
                reason: "min_size must be less than target_size".to_string(),
            });
        }

        if self.target_size >= self.max_size {
            return Err(ReductoError::InvalidConfiguration {
                parameter: "max_size".to_string(),
                value: self.max_size.to_string(),
                reason: "max_size must be greater than target_size".to_string(),
            });
        }

        // Check global bounds
        if self.target_size < MIN_CHUNK_SIZE || self.target_size > MAX_CHUNK_SIZE {
            return Err(ReductoError::ParameterOutOfRange {
                parameter: "target_size".to_string(),
                value: self.target_size as i64,
                min: MIN_CHUNK_SIZE as i64,
                max: MAX_CHUNK_SIZE as i64,
            });
        }

        // Check variance bounds (50%-200% of target)
        let min_expected = (self.target_size as f64 * MIN_TARGET_RATIO) as usize;
        let max_expected = (self.target_size as f64 * MAX_TARGET_RATIO) as usize;

        if self.min_size < min_expected || self.min_size > self.target_size {
            return Err(ReductoError::ConstraintViolation {
                constraint: "chunk_size_variance".to_string(),
                details: format!(
                    "min_size {} outside acceptable range [{}, {}]",
                    self.min_size, min_expected, self.target_size
                ),
            });
        }

        if self.max_size > max_expected || self.max_size < self.target_size {
            return Err(ReductoError::ConstraintViolation {
                constraint: "chunk_size_variance".to_string(),
                details: format!(
                    "max_size {} outside acceptable range [{}, {}]",
                    self.max_size, self.target_size, max_expected
                ),
            });
        }

        // Validate hash parameters
        if self.hash_base == 0 {
            return Err(ReductoError::InvalidConfiguration {
                parameter: "hash_base".to_string(),
                value: self.hash_base.to_string(),
                reason: "hash_base must be non-zero".to_string(),
            });
        }

        if self.hash_mask == 0 {
            return Err(ReductoError::InvalidConfiguration {
                parameter: "hash_mask".to_string(),
                value: self.hash_mask.to_string(),
                reason: "hash_mask must be non-zero".to_string(),
            });
        }

        Ok(())
    }

    /// Check if a chunk size is within acceptable variance bounds
    pub fn is_valid_chunk_size(&self, size: usize) -> bool {
        size >= self.min_size && size <= self.max_size
    }

    /// Get the expected average boundary frequency (1 / expected_chunk_size)
    pub fn boundary_frequency(&self) -> f64 {
        1.0 / self.target_size as f64
    }
}

/// Variable-size data chunk from CDC processing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataChunk {
    /// Chunk data
    pub data: Vec<u8>,
    /// Weak hash of content
    pub weak_hash: WeakHash,
    /// Strong hash for verification
    pub strong_hash: Hash,
    /// Position in original data
    pub offset: u64,
}

impl DataChunk {
    /// Create a new data chunk
    pub fn new(data: Vec<u8>, offset: u64) -> Self {
        let weak_hash = WeakHash::new(0); // Will be computed by chunker
        let strong_hash = blake3::hash(&data);
        Self {
            data,
            weak_hash,
            strong_hash,
            offset,
        }
    }

    /// Get chunk size
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Verify chunk integrity
    pub fn verify(&self) -> bool {
        blake3::hash(&self.data) == self.strong_hash
    }
}

/// Corpus metadata and information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusMetadata {
    /// Immutable corpus GUID
    pub corpus_id: uuid::Uuid,
    /// Cryptographic signature
    pub signature: Vec<u8>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Number of chunks in corpus
    pub chunk_count: u64,
    /// Total corpus size in bytes
    pub total_size: u64,
    /// CDC configuration used
    pub chunk_config: ChunkConfig,
    /// Optimization statistics
    pub optimization_stats: OptimizationStats,
}

/// Corpus optimization statistics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Deduplication ratio achieved
    pub deduplication_ratio: u32, // Fixed-point: value / 10000 = actual ratio
    /// Average chunk size
    pub avg_chunk_size: u32,
    /// Chunk size variance
    pub chunk_size_variance: u32,
    /// Hash collision rate
    pub collision_rate: u32, // Fixed-point: value / 10000 = actual rate
}

/// Corpus optimization recommendations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationRecommendations {
    /// Recommended chunk size
    pub recommended_chunk_size: usize,
    /// Frequency analysis results
    pub frequency_analysis: std::collections::HashMap<u64, u32>,
    /// Deduplication potential (0.0 to 1.0)
    pub deduplication_potential: f64,
    /// Suggested chunks to prune
    pub suggested_pruning: Vec<u64>,
}

/// Retention policy for corpus pruning
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Retention period in days
    pub retention_days: u32,
    /// Enable secure deletion
    pub secure_deletion: bool,
    /// Audit log retention in days
    pub audit_retention_days: u32,
}

/// Statistics from corpus pruning operation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PruneStats {
    /// Number of chunks removed
    pub chunks_removed: u64,
    /// Bytes freed
    pub bytes_freed: u64,
    /// Time taken for operation
    pub duration_ms: u64,
    /// Chunks analyzed
    pub chunks_analyzed: u64,
}

/// Cryptographic signature wrapper
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature {
    /// Signature algorithm used
    pub algorithm: String,
    /// Signature bytes
    pub signature: Vec<u8>,
    /// Public key identifier
    pub key_id: Option<String>,
}

/// Access operation types for audit logging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessOperation {
    Read,
    Write,
    Delete,
    Modify,
    Create,
    Verify,
}

impl std::fmt::Display for AccessOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::Delete => write!(f, "delete"),
            Self::Modify => write!(f, "modify"),
            Self::Create => write!(f, "create"),
            Self::Verify => write!(f, "verify"),
        }
    }
}

/// Compression analysis results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionAnalysis {
    /// Input size in bytes
    pub input_size: u64,
    /// Predicted output size
    pub predicted_output_size: u64,
    /// Predicted compression ratio
    pub predicted_ratio: f64,
    /// Corpus hit rate prediction
    pub predicted_hit_rate: f64,
    /// Analysis duration
    pub analysis_duration_ms: u64,
}

/// Metrics export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricsFormat {
    Prometheus,
    Json,
    Csv,
}

impl std::fmt::Display for MetricsFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Prometheus => write!(f, "prometheus"),
            Self::Json => write!(f, "json"),
            Self::Csv => write!(f, "csv"),
        }
    }
}

/// Usage statistics for ROI calculation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageStats {
    /// Time period start
    pub period_start: chrono::DateTime<chrono::Utc>,
    /// Time period end
    pub period_end: chrono::DateTime<chrono::Utc>,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Total bytes saved
    pub bytes_saved: u64,
    /// Number of operations
    pub operation_count: u64,
    /// Average processing time per operation
    pub avg_processing_time_ms: f64,
}

/// Economic analysis report
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EconomicReport {
    /// Total cost savings in USD
    pub cost_savings_usd: f64,
    /// Bandwidth savings in bytes
    pub bandwidth_saved: u64,
    /// Storage savings in bytes
    pub storage_saved: u64,
    /// ROI percentage
    pub roi_percentage: f64,
    /// Payback period in months
    pub payback_period_months: f64,
}

/// Compression operation metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionMetrics {
    /// Input size
    pub input_size: u64,
    /// Output size
    pub output_size: u64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Corpus hit rate
    pub corpus_hit_rate: f64,
    /// Processing time
    pub processing_time_ms: u64,
    /// Memory usage peak
    pub memory_usage_bytes: u64,
}

/// Decompression operation metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecompressionMetrics {
    /// Input size
    pub input_size: u64,
    /// Output size
    pub output_size: u64,
    /// Decompression ratio
    pub decompression_ratio: f64,
    /// Processing time
    pub processing_time_ms: u64,
    /// Memory usage peak
    pub memory_usage_bytes: u64,
    /// Corpus access count
    pub corpus_access_count: u64,
}

/// Real-time performance metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Throughput in MB/s
    pub throughput_mbps: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// I/O wait time in milliseconds
    pub io_wait_time_ms: u64,
    /// Current bottleneck type
    pub bottleneck_type: BottleneckType,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    CpuBound,
    IoBound,
    MemoryBound,
    NetworkBound,
    None,
}

impl std::fmt::Display for BottleneckType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuBound => write!(f, "cpu_bound"),
            Self::IoBound => write!(f, "io_bound"),
            Self::MemoryBound => write!(f, "memory_bound"),
            Self::NetworkBound => write!(f, "network_bound"),
            Self::None => write!(f, "none"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Unit tests for basic functionality
    #[test]
    fn test_block_offset() {
        let offset = BlockOffset::new(8192);
        assert_eq!(offset.get(), 8192);
        assert!(offset.is_block_aligned());
        assert_eq!(offset.block_index(), 2);

        let unaligned = BlockOffset::new(100);
        assert!(!unaligned.is_block_aligned());
    }

    #[test]
    fn test_weak_hash() {
        let hash = WeakHash::new(0x1234567890abcdef);
        assert_eq!(hash.get(), 0x1234567890abcdef);
        
        let display = format!("{}", hash);
        assert!(display.contains("0x1234567890abcdef"));
    }

    #[test]
    fn test_corpus_id() {
        let id = CorpusId::new("test-corpus-123");
        assert!(id.is_valid());
        assert_eq!(id.as_str(), "test-corpus-123");

        let empty_id = CorpusId::new("");
        assert!(!empty_id.is_valid());

        let long_id = CorpusId::new("a".repeat(300));
        assert!(!long_id.is_valid());
    }

    #[test]
    fn test_corpus_block() {
        let test_data = vec![0u8; BLOCK_SIZE];
        let hash = blake3::hash(&test_data);
        let chunk = CorpusChunk::new(0, 4096, hash);
        
        assert!(chunk.verify_data(&test_data));
        
        let wrong_data = vec![1u8; BLOCK_SIZE];
        assert!(!chunk.verify_data(&wrong_data));
        
        let wrong_size = vec![0u8; BLOCK_SIZE - 1];
        assert!(!chunk.verify_data(&wrong_size));
    }

    #[test]
    fn test_reducto_instruction() {
        let reference = ReductoInstruction::Reference { offset: 4096, size: BLOCK_SIZE as u32 };
        assert!(reference.is_reference());
        assert!(!reference.is_residual());
        assert_eq!(reference.output_size(), BLOCK_SIZE);
        assert_eq!(reference.reference_offset(), Some(4096));

        let residual = ReductoInstruction::Residual(vec![1, 2, 3, 4]);
        assert!(!residual.is_reference());
        assert!(residual.is_residual());
        assert_eq!(residual.output_size(), 4);
        assert_eq!(residual.residual_data(), Some([1, 2, 3, 4].as_slice()));
    }

    #[test]
    fn test_reducto_header() {
        let header = ReductoHeader::default();
        assert!(header.validate().is_ok());

        let invalid_magic = ReductoHeader {
            magic: *b"INVALID!",
            ..header.clone()
        };
        assert!(invalid_magic.validate().is_err());

        let invalid_compression = ReductoHeader {
            compression_level: 0,  // Invalid compression level
            ..header.clone()
        };
        assert!(invalid_compression.validate().is_err());

        let with_metadata = header.with_metadata("compression_level", "19");
        assert_eq!(with_metadata.get_metadata("compression_level"), Some("19"));
    }

    #[test]
    fn test_serialization() {
        let instruction = ReductoInstruction::Reference { offset: 8192, size: 4096 };
        let serialized = bincode::serialize(&instruction).unwrap();
        let deserialized: ReductoInstruction = bincode::deserialize(&serialized).unwrap();
        assert_eq!(instruction, deserialized);

        let header = ReductoHeader::default();
        let serialized = bincode::serialize(&header).unwrap();
        let deserialized: ReductoHeader = bincode::deserialize(&serialized).unwrap();
        assert_eq!(header, deserialized);
    }

    #[test]
    fn test_chunk_config_validation() {
        // Valid configuration
        let config = ChunkConfig::new(8192).unwrap();
        assert!(config.validate().is_ok());
        assert_eq!(config.target_size, 8192);
        assert_eq!(config.min_size, 4096);  // 50% of target
        assert_eq!(config.max_size, 16384); // 200% of target

        // Invalid target size (too small)
        assert!(ChunkConfig::new(1024).is_err());

        // Invalid target size (too large)
        assert!(ChunkConfig::new(128 * 1024).is_err());
    }

    #[test]
    fn test_corpus_chunk_variable_size() {
        let hash = blake3::hash(b"test data");
        let chunk = CorpusChunk::new(1024, 512, hash);
        
        assert_eq!(chunk.offset, 1024);
        assert_eq!(chunk.size, 512);
        assert_eq!(chunk.end_offset(), 1536);

        // Test overlap detection
        let chunk2 = CorpusChunk::new(1500, 100, hash);
        assert!(chunk.overlaps_with(&chunk2));

        let chunk3 = CorpusChunk::new(2000, 100, hash);
        assert!(!chunk.overlaps_with(&chunk3));
    }

    #[test]
    fn test_enhanced_reducto_instruction() {
        let reference = ReductoInstruction::Reference { offset: 4096, size: 2048 };
        assert!(reference.is_reference());
        assert!(!reference.is_residual());
        assert_eq!(reference.output_size(), 2048);
        assert_eq!(reference.reference_info(), Some((4096, 2048)));

        let residual = ReductoInstruction::Residual(vec![1, 2, 3, 4]);
        assert!(!residual.is_reference());
        assert!(residual.is_residual());
        assert_eq!(residual.output_size(), 4);
        assert_eq!(residual.residual_data(), Some([1, 2, 3, 4].as_slice()));
    }

    #[test]
    fn test_enhanced_reducto_header() {
        let corpus_id = uuid::Uuid::new_v4();
        let config = ChunkConfig::new(8192).unwrap();
        let signature = b"test_signature".to_vec();
        let integrity_hash = blake3::hash(b"test_data");
        
        let header = ReductoHeader::new(
            corpus_id,
            signature.clone(),
            config.clone(),
            integrity_hash,
            19,
        );

        assert!(header.validate().is_ok());
        assert_eq!(header.corpus_id, corpus_id);
        assert_eq!(header.corpus_signature, signature);
        assert_eq!(header.chunk_config, config);
        assert_eq!(header.compression_level, 19);
        assert!(header.has_signature());

        // Test integrity verification
        assert!(header.verify_integrity(b"test_data"));
        assert!(!header.verify_integrity(b"wrong_data"));
    }

    // Property-based tests for data model invariants
    mod property_tests {
        use super::*;

        proptest! {
            /// Property: BlockOffset roundtrip conversion preserves value
            #[test]
            fn block_offset_roundtrip(offset in any::<u64>()) {
                let block_offset = BlockOffset::new(offset);
                prop_assert_eq!(block_offset.get(), offset);
                
                let from_u64: BlockOffset = offset.into();
                let back_to_u64: u64 = from_u64.into();
                prop_assert_eq!(back_to_u64, offset);
            }

            /// Property: Block alignment is consistent with modulo arithmetic
            #[test]
            fn block_offset_alignment_invariant(offset in any::<u64>()) {
                let block_offset = BlockOffset::new(offset);
                let is_aligned = block_offset.is_block_aligned();
                let manual_check = offset % BLOCK_SIZE as u64 == 0;
                prop_assert_eq!(is_aligned, manual_check);
            }

            /// Property: Block index calculation is consistent
            #[test]
            fn block_index_invariant(offset in any::<u64>()) {
                let block_offset = BlockOffset::new(offset);
                let index = block_offset.block_index();
                let manual_index = offset / BLOCK_SIZE as u64;
                prop_assert_eq!(index, manual_index);
            }

            /// Property: WeakHash roundtrip conversion preserves value
            #[test]
            fn weak_hash_roundtrip(hash in any::<u64>()) {
                let weak_hash = WeakHash::new(hash);
                prop_assert_eq!(weak_hash.get(), hash);
                
                let from_u64: WeakHash = hash.into();
                let back_to_u64: u64 = from_u64.into();
                prop_assert_eq!(back_to_u64, hash);
            }

            /// Property: CorpusId validation is consistent
            #[test]
            fn corpus_id_validation_invariant(id in ".*") {
                let corpus_id = CorpusId::new(id.clone());
                let is_valid = corpus_id.is_valid();
                
                // Manual validation logic
                let expected_valid = !id.is_empty() 
                    && id.len() <= 256 
                    && id.chars().all(|c| c.is_ascii_graphic());
                
                prop_assert_eq!(is_valid, expected_valid);
            }

            /// Property: CorpusId string conversion preserves content
            #[test]
            fn corpus_id_string_roundtrip(id in "[a-zA-Z0-9_-]{1,100}") {
                let corpus_id = CorpusId::new(id.clone());
                prop_assert_eq!(corpus_id.as_str(), &id);
                
                let back_to_string: String = corpus_id.into();
                prop_assert_eq!(back_to_string, id);
            }

            /// Property: CorpusBlock data verification is deterministic
            #[test]
            fn corpus_block_verification_invariant(
                offset in any::<u64>(),
                data in prop::collection::vec(any::<u8>(), BLOCK_SIZE..=BLOCK_SIZE)
            ) {
                let hash = blake3::hash(&data);
                let chunk = CorpusChunk::new(offset, BLOCK_SIZE as u32, hash);
                
                // Should always verify the same data
                prop_assert!(chunk.verify_data(&data));
                
                // Should always reject different data (if we modify it)
                if !data.is_empty() {
                    let mut modified_data = data.clone();
                    modified_data[0] = modified_data[0].wrapping_add(1);
                    prop_assert!(!chunk.verify_data(&modified_data));
                }
            }

            /// Property: ReductoInstruction output size calculation is correct
            #[test]
            fn instruction_output_size_invariant(
                offset in any::<u64>(),
                size in 1u32..65536u32,
                residual_data in prop::collection::vec(any::<u8>(), 0..10000)
            ) {
                let reference = ReductoInstruction::Reference { offset, size };
                prop_assert_eq!(reference.output_size(), size as usize);
                
                let residual = ReductoInstruction::Residual(residual_data.clone());
                prop_assert_eq!(residual.output_size(), residual_data.len());
            }

            /// Property: ChunkConfig maintains 50%-200% variance bounds
            #[test]
            fn chunk_config_variance_bounds(
                target_size in (MIN_CHUNK_SIZE..=MAX_CHUNK_SIZE)
            ) {
                let config = ChunkConfig::new(target_size).unwrap();
                
                // Verify size relationships
                prop_assert!(config.min_size < config.target_size);
                prop_assert!(config.target_size < config.max_size);
                
                // Verify variance bounds (50%-200% of target)
                let min_expected = (target_size as f64 * MIN_TARGET_RATIO) as usize;
                let max_expected = (target_size as f64 * MAX_TARGET_RATIO) as usize;
                
                prop_assert_eq!(config.min_size, min_expected);
                prop_assert_eq!(config.max_size, max_expected);
                
                // Verify validation passes
                prop_assert!(config.validate().is_ok());
            }

            /// Property: Chunk size validation is consistent with bounds
            #[test]
            fn chunk_size_validation_invariant(
                target_size in (MIN_CHUNK_SIZE..=MAX_CHUNK_SIZE),
                test_size in 1usize..131072usize
            ) {
                let config = ChunkConfig::new(target_size).unwrap();
                let is_valid = config.is_valid_chunk_size(test_size);
                
                let expected_valid = test_size >= config.min_size && test_size <= config.max_size;
                prop_assert_eq!(is_valid, expected_valid);
            }

            /// Property: CorpusChunk overlap detection is symmetric and transitive
            #[test]
            fn corpus_chunk_overlap_invariant(
                offset1 in 0u64..1000000u64,
                size1 in 1u32..65536u32,
                offset2 in 0u64..1000000u64,
                size2 in 1u32..65536u32
            ) {
                let hash = blake3::hash(b"test");
                let chunk1 = CorpusChunk::new(offset1, size1, hash);
                let chunk2 = CorpusChunk::new(offset2, size2, hash);
                
                // Overlap detection should be symmetric
                prop_assert_eq!(chunk1.overlaps_with(&chunk2), chunk2.overlaps_with(&chunk1));
                
                // Manual overlap calculation for verification
                let end1 = offset1 + size1 as u64;
                let end2 = offset2 + size2 as u64;
                let expected_overlap = !(end1 <= offset2 || end2 <= offset1);
                
                prop_assert_eq!(chunk1.overlaps_with(&chunk2), expected_overlap);
            }

            /// Property: ReductoInstruction reference validation is consistent
            #[test]
            fn instruction_reference_validation_invariant(
                offset in 0u64..1000000u64,
                size in 1u32..65536u32,
                corpus_size in 1u64..2000000u64
            ) {
                let instruction = ReductoInstruction::Reference { offset, size };
                let validation_result = instruction.validate_reference(corpus_size);
                
                let end_offset = offset + size as u64;
                let should_be_valid = end_offset <= corpus_size;
                
                prop_assert_eq!(validation_result.is_ok(), should_be_valid);
            }

            /// Property: Enhanced ReductoHeader validation is comprehensive
            #[test]
            fn enhanced_header_validation_invariant(
                target_size in (MIN_CHUNK_SIZE..=MAX_CHUNK_SIZE),
                compression_level in 1u8..=22u8
            ) {
                let corpus_id = uuid::Uuid::new_v4();
                let config = ChunkConfig::new(target_size).unwrap();
                let signature = b"test_signature".to_vec();
                let integrity_hash = blake3::hash(b"test_data");
                
                let header = ReductoHeader::new(
                    corpus_id,
                    signature,
                    config,
                    integrity_hash,
                    compression_level,
                );
                
                // Valid header should always pass validation
                prop_assert!(header.validate().is_ok());
                
                // Verify properties
                prop_assert_eq!(header.compression_level, compression_level);
                prop_assert!(!header.corpus_id.is_nil());
                prop_assert!(header.has_signature());
            }

            /// Property: Chunk size variance stays within 50%-200% bounds
            #[test]
            fn chunk_variance_bounds_property(
                target_size in (MIN_CHUNK_SIZE..=MAX_CHUNK_SIZE)
            ) {
                let config = ChunkConfig::new(target_size).unwrap();
                
                // Calculate expected bounds
                let min_bound = (target_size as f64 * MIN_TARGET_RATIO) as usize;
                let max_bound = (target_size as f64 * MAX_TARGET_RATIO) as usize;
                
                // Verify actual bounds match expected
                prop_assert_eq!(config.min_size, min_bound);
                prop_assert_eq!(config.max_size, max_bound);
                
                // Verify variance ratios
                let min_ratio = config.min_size as f64 / target_size as f64;
                let max_ratio = config.max_size as f64 / target_size as f64;
                
                prop_assert!((min_ratio - MIN_TARGET_RATIO).abs() < 0.01);
                prop_assert!((max_ratio - MAX_TARGET_RATIO).abs() < 0.01);
            }

            /// Property: ReductoInstruction type checking is exhaustive
            #[test]
            fn instruction_type_checking_exhaustive(
                offset in any::<u64>(),
                size in 1u32..65536u32,
                residual_data in prop::collection::vec(any::<u8>(), 0..1000)
            ) {
                let reference = ReductoInstruction::Reference { offset, size };
                prop_assert!(reference.is_reference());
                prop_assert!(!reference.is_residual());
                prop_assert_eq!(reference.reference_info(), Some((offset, size)));
                prop_assert_eq!(reference.reference_offset(), Some(offset));
                prop_assert_eq!(reference.reference_size(), Some(size));
                prop_assert_eq!(reference.residual_data(), None);
                
                let residual = ReductoInstruction::Residual(residual_data.clone());
                prop_assert!(!residual.is_reference());
                prop_assert!(residual.is_residual());
                prop_assert_eq!(residual.reference_info(), None);
                prop_assert_eq!(residual.reference_offset(), None);
                prop_assert_eq!(residual.reference_size(), None);
                prop_assert_eq!(residual.residual_data(), Some(residual_data.as_slice()));
            }

            /// Property: ReductoHeader validation is consistent
            #[test]
            fn header_validation_invariant(
                target_size in (MIN_CHUNK_SIZE..=MAX_CHUNK_SIZE),
                compression_level in 1u8..=22u8
            ) {
                let corpus_id = uuid::Uuid::new_v4();
                let config = ChunkConfig::new(target_size).unwrap();
                let header = ReductoHeader::basic(corpus_id, config);
                
                // Valid header should always pass validation
                prop_assert!(header.validate().is_ok());
                
                // Header with wrong magic should always fail
                let invalid_magic = ReductoHeader {
                    magic: *b"INVALID!",
                    ..header.clone()
                };
                prop_assert!(invalid_magic.validate().is_err());
                
                // Header with invalid compression level should fail
                let invalid_compression = ReductoHeader {
                    compression_level: 0,  // Invalid level
                    ..header.clone()
                };
                prop_assert!(invalid_compression.validate().is_err());
            }

            /// Property: Serialization roundtrip preserves data
            #[test]
            fn serialization_roundtrip_invariant(
                offset in any::<u64>(),
                size in 1u32..65536u32,
                residual_data in prop::collection::vec(any::<u8>(), 0..1000),
                target_size in (MIN_CHUNK_SIZE..=MAX_CHUNK_SIZE)
            ) {
                // Test ReductoInstruction serialization
                let reference = ReductoInstruction::Reference { offset, size };
                let serialized = bincode::serialize(&reference).unwrap();
                let deserialized: ReductoInstruction = bincode::deserialize(&serialized).unwrap();
                prop_assert_eq!(reference, deserialized);
                
                let residual = ReductoInstruction::Residual(residual_data);
                let serialized = bincode::serialize(&residual).unwrap();
                let deserialized: ReductoInstruction = bincode::deserialize(&serialized).unwrap();
                prop_assert_eq!(residual, deserialized);
                
                // Test ReductoHeader serialization
                let corpus_id = uuid::Uuid::new_v4();
                let config = ChunkConfig::new(target_size).unwrap();
                let header = ReductoHeader::basic(corpus_id, config);
                let serialized = bincode::serialize(&header).unwrap();
                let deserialized: ReductoHeader = bincode::deserialize(&serialized).unwrap();
                prop_assert_eq!(header, deserialized);
            }

            /// Property: Metadata operations preserve header validity
            #[test]
            fn header_metadata_invariant(
                target_size in (MIN_CHUNK_SIZE..=MAX_CHUNK_SIZE),
                key in "[a-zA-Z0-9_-]{1,50}",
                value in "[a-zA-Z0-9_-]{1,100}"
            ) {
                let corpus_id = uuid::Uuid::new_v4();
                let config = ChunkConfig::new(target_size).unwrap();
                let header = ReductoHeader::basic(corpus_id, config);
                let with_metadata = header.with_metadata(key.clone(), value.clone());
                
                // Header should still be valid after adding metadata
                prop_assert!(with_metadata.validate().is_ok());
                
                // Metadata should be retrievable
                prop_assert_eq!(with_metadata.get_metadata(&key), Some(value.as_str()));
            }

            /// Property: Constants are within expected ranges
            #[test]
            fn constants_invariant(_dummy in any::<u8>()) {
                // BLOCK_SIZE should be a power of 2 and reasonable size
                prop_assert!(BLOCK_SIZE > 0);
                prop_assert!(BLOCK_SIZE.is_power_of_two());
                prop_assert!(BLOCK_SIZE >= 1024); // At least 1KB
                prop_assert!(BLOCK_SIZE <= 1024 * 1024); // At most 1MB
                
                // HASH_BASE should be a reasonable prime
                prop_assert!(HASH_BASE > 1);
                prop_assert!(HASH_BASE < 1000); // Reasonable range for rolling hash
                
                // MAGIC_BYTES should be exactly 8 bytes
                prop_assert_eq!(MAGIC_BYTES.len(), 8);
            }
        }
    }
}