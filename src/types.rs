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

/// Magic bytes for .reducto file format identification
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

/// Represents a block in the Reference Corpus with its metadata
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusBlock {
    /// Offset of the block in the corpus file
    pub offset: BlockOffset,
    /// Strong cryptographic hash (BLAKE3) for verification
    pub strong_hash: Hash,
}

impl CorpusBlock {
    /// Create a new corpus block
    pub fn new(offset: BlockOffset, strong_hash: Hash) -> Self {
        Self {
            offset,
            strong_hash,
        }
    }

    /// Verify if the given data matches this block's strong hash
    pub fn verify_data(&self, data: &[u8]) -> bool {
        if data.len() != BLOCK_SIZE {
            return false;
        }
        blake3::hash(data) == self.strong_hash
    }
}

/// Instructions for the decompressor to reconstruct the original file
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductoInstruction {
    /// Reference a block from the corpus by offset
    /// The block size is implicitly BLOCK_SIZE
    Reference(BlockOffset),
    /// Literal data not found in the corpus
    Residual(Vec<u8>),
}

impl ReductoInstruction {
    /// Get the size in bytes that this instruction represents in the output
    pub fn output_size(&self) -> usize {
        match self {
            Self::Reference(_) => BLOCK_SIZE,
            Self::Residual(data) => data.len(),
        }
    }

    /// Check if this instruction is a reference to the corpus
    pub fn is_reference(&self) -> bool {
        matches!(self, Self::Reference(_))
    }

    /// Check if this instruction contains residual data
    pub fn is_residual(&self) -> bool {
        matches!(self, Self::Residual(_))
    }

    /// Get the referenced block offset if this is a reference instruction
    pub fn reference_offset(&self) -> Option<BlockOffset> {
        match self {
            Self::Reference(offset) => Some(*offset),
            Self::Residual(_) => None,
        }
    }

    /// Get the residual data if this is a residual instruction
    pub fn residual_data(&self) -> Option<&[u8]> {
        match self {
            Self::Reference(_) => None,
            Self::Residual(data) => Some(data),
        }
    }
}

/// Header structure for .reducto files
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReductoHeader {
    /// Magic bytes for format identification
    pub magic: [u8; 8],
    /// Unique identifier of the corpus used for compression
    pub corpus_id: CorpusId,
    /// Block size used during compression
    pub block_size: u32,
    /// Format version for future compatibility
    pub version: u32,
    /// Optional metadata for extensions
    pub metadata: std::collections::HashMap<String, String>,
}

impl ReductoHeader {
    /// Create a new header with the current format version
    pub fn new(corpus_id: CorpusId) -> Self {
        Self {
            magic: MAGIC_BYTES,
            corpus_id,
            block_size: BLOCK_SIZE as u32,
            version: 1,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Validate that this header is compatible with the current implementation
    pub fn validate(&self) -> crate::Result<()> {
        use crate::error::ReductoError;

        // Check magic bytes
        if self.magic != MAGIC_BYTES {
            return Err(ReductoError::InvalidMagicBytes {
                expected: MAGIC_BYTES.to_vec(),
                found: self.magic.to_vec(),
            });
        }

        // Check block size
        if self.block_size != BLOCK_SIZE as u32 {
            return Err(ReductoError::BlockSizeMismatch {
                expected: BLOCK_SIZE as u32,
                found: self.block_size,
            });
        }

        // Check version compatibility
        if self.version > 1 {
            return Err(ReductoError::UnsupportedVersion {
                found: self.version.to_string(),
                supported: "1".to_string(),
            });
        }

        // Validate corpus ID
        if !self.corpus_id.is_valid() {
            return Err(ReductoError::InputValidationFailed {
                field: "corpus_id".to_string(),
                reason: "Invalid corpus ID format".to_string(),
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
}

impl Default for ReductoHeader {
    fn default() -> Self {
        Self::new(CorpusId::new("default"))
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
        let block = CorpusBlock::new(BlockOffset::new(0), hash);
        
        assert!(block.verify_data(&test_data));
        
        let wrong_data = vec![1u8; BLOCK_SIZE];
        assert!(!block.verify_data(&wrong_data));
        
        let wrong_size = vec![0u8; BLOCK_SIZE - 1];
        assert!(!block.verify_data(&wrong_size));
    }

    #[test]
    fn test_reducto_instruction() {
        let reference = ReductoInstruction::Reference(BlockOffset::new(4096));
        assert!(reference.is_reference());
        assert!(!reference.is_residual());
        assert_eq!(reference.output_size(), BLOCK_SIZE);
        assert_eq!(reference.reference_offset(), Some(BlockOffset::new(4096)));

        let residual = ReductoInstruction::Residual(vec![1, 2, 3, 4]);
        assert!(!residual.is_reference());
        assert!(residual.is_residual());
        assert_eq!(residual.output_size(), 4);
        assert_eq!(residual.residual_data(), Some([1, 2, 3, 4].as_slice()));
    }

    #[test]
    fn test_reducto_header() {
        let header = ReductoHeader::new(CorpusId::new("test-corpus"));
        assert!(header.validate().is_ok());

        let invalid_magic = ReductoHeader {
            magic: *b"INVALID!",
            ..header.clone()
        };
        assert!(invalid_magic.validate().is_err());

        let invalid_block_size = ReductoHeader {
            block_size: 1024,
            ..header.clone()
        };
        assert!(invalid_block_size.validate().is_err());

        let with_metadata = header.with_metadata("compression_level", "19");
        assert_eq!(with_metadata.get_metadata("compression_level"), Some("19"));
    }

    #[test]
    fn test_serialization() {
        let instruction = ReductoInstruction::Reference(BlockOffset::new(8192));
        let serialized = bincode::serialize(&instruction).unwrap();
        let deserialized: ReductoInstruction = bincode::deserialize(&serialized).unwrap();
        assert_eq!(instruction, deserialized);

        let header = ReductoHeader::new(CorpusId::new("test"));
        let serialized = bincode::serialize(&header).unwrap();
        let deserialized: ReductoHeader = bincode::deserialize(&serialized).unwrap();
        assert_eq!(header, deserialized);
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
                let block = CorpusBlock::new(BlockOffset::new(offset), hash);
                
                // Should always verify the same data
                prop_assert!(block.verify_data(&data));
                
                // Should always reject different data (if we modify it)
                if !data.is_empty() {
                    let mut modified_data = data.clone();
                    modified_data[0] = modified_data[0].wrapping_add(1);
                    prop_assert!(!block.verify_data(&modified_data));
                }
            }

            /// Property: ReductoInstruction output size calculation is correct
            #[test]
            fn instruction_output_size_invariant(
                offset in any::<u64>(),
                residual_data in prop::collection::vec(any::<u8>(), 0..10000)
            ) {
                let reference = ReductoInstruction::Reference(BlockOffset::new(offset));
                prop_assert_eq!(reference.output_size(), BLOCK_SIZE);
                
                let residual = ReductoInstruction::Residual(residual_data.clone());
                prop_assert_eq!(residual.output_size(), residual_data.len());
            }

            /// Property: ReductoInstruction type checking is exhaustive
            #[test]
            fn instruction_type_checking_exhaustive(
                offset in any::<u64>(),
                residual_data in prop::collection::vec(any::<u8>(), 0..1000)
            ) {
                let reference = ReductoInstruction::Reference(BlockOffset::new(offset));
                prop_assert!(reference.is_reference());
                prop_assert!(!reference.is_residual());
                prop_assert_eq!(reference.reference_offset(), Some(BlockOffset::new(offset)));
                prop_assert_eq!(reference.residual_data(), None);
                
                let residual = ReductoInstruction::Residual(residual_data.clone());
                prop_assert!(!residual.is_reference());
                prop_assert!(residual.is_residual());
                prop_assert_eq!(residual.reference_offset(), None);
                prop_assert_eq!(residual.residual_data(), Some(residual_data.as_slice()));
            }

            /// Property: ReductoHeader validation is consistent
            #[test]
            fn header_validation_invariant(corpus_id in "[a-zA-Z0-9_-]{1,100}") {
                let header = ReductoHeader::new(CorpusId::new(corpus_id));
                
                // Valid header should always pass validation
                prop_assert!(header.validate().is_ok());
                
                // Header with wrong magic should always fail
                let invalid_magic = ReductoHeader {
                    magic: *b"INVALID!",
                    ..header.clone()
                };
                prop_assert!(invalid_magic.validate().is_err());
                
                // Header with wrong block size should always fail
                let invalid_block_size = ReductoHeader {
                    block_size: 1024,
                    ..header.clone()
                };
                prop_assert!(invalid_block_size.validate().is_err());
            }

            /// Property: Serialization roundtrip preserves data
            #[test]
            fn serialization_roundtrip_invariant(
                offset in any::<u64>(),
                residual_data in prop::collection::vec(any::<u8>(), 0..1000),
                corpus_id in "[a-zA-Z0-9_-]{1,100}"
            ) {
                // Test ReductoInstruction serialization
                let reference = ReductoInstruction::Reference(BlockOffset::new(offset));
                let serialized = bincode::serialize(&reference).unwrap();
                let deserialized: ReductoInstruction = bincode::deserialize(&serialized).unwrap();
                prop_assert_eq!(reference, deserialized);
                
                let residual = ReductoInstruction::Residual(residual_data);
                let serialized = bincode::serialize(&residual).unwrap();
                let deserialized: ReductoInstruction = bincode::deserialize(&serialized).unwrap();
                prop_assert_eq!(residual, deserialized);
                
                // Test ReductoHeader serialization
                let header = ReductoHeader::new(CorpusId::new(corpus_id));
                let serialized = bincode::serialize(&header).unwrap();
                let deserialized: ReductoHeader = bincode::deserialize(&serialized).unwrap();
                prop_assert_eq!(header, deserialized);
            }

            /// Property: Metadata operations preserve header validity
            #[test]
            fn header_metadata_invariant(
                corpus_id in "[a-zA-Z0-9_-]{1,100}",
                key in "[a-zA-Z0-9_-]{1,50}",
                value in "[a-zA-Z0-9_-]{1,100}"
            ) {
                let header = ReductoHeader::new(CorpusId::new(corpus_id));
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