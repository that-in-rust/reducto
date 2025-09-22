//! # Reducto Mode 3 - Differential Synchronization
//!
//! High-performance compression system that achieves extreme compression ratios by identifying
//! data blocks in target files that already exist in a shared Reference Corpus (RC) and
//! replacing them with efficient pointer references.
//!
//! ## Architecture
//!
//! The system follows a layered architecture:
//! - **L1 Core**: Block processing, rolling hash implementation, RAII resource management
//! - **L2 Standard**: Collections (HashMap), iterators, smart pointers (Arc), error handling  
//! - **L3 External**: Serialization (bincode), compression (zstd), memory mapping (memmap2), cryptographic hashing (BLAKE3)

pub mod error;
pub mod traits;
pub mod types;
pub mod stubs;
pub mod cdc_chunker;
pub mod rolling_hash;
pub mod corpus_manager;
pub mod compressor;
pub mod serializer;
pub mod security_manager;

#[cfg(test)]
pub mod contract_tests;

// Re-export commonly used types
pub use error::{ReductoError, Result};
pub use traits::{HashProvider, BlockMatcher, CorpusReader, InstructionWriter, CDCChunker, CorpusManager, SecurityManager, MetricsCollector};
pub use types::{
    BlockOffset, WeakHash, CorpusId, CorpusBlock, ReductoInstruction, ReductoHeader,
    ChunkConfig, DataChunk, CorpusMetadata, Signature, AccessOperation, MetricsFormat,
    BLOCK_SIZE, HASH_BASE,
};
pub use cdc_chunker::{FastCDCChunker, GearHasher};
pub use rolling_hash::{RollingHasher, StrongHasher, DualHasher, DualHashStatistics};
pub use corpus_manager::{EnterpriseCorpusManager, InMemoryStorage, StorageStats, StorageBackend};
pub use compressor::{Compressor, CompressionStats};
pub use serializer::{AdvancedSerializer, SerializerConfig, CompressionProfile, SerializationStats};
pub use security_manager::{EnterpriseSecurityManager, KeyManagementConfig, AuditLogger, AuditEvent, OperationResult, AuditSeverity};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        error::{ReductoError, Result},
        traits::{HashProvider, BlockMatcher, CorpusReader, InstructionWriter, CDCChunker, CorpusManager, SecurityManager, MetricsCollector},
        types::{
            BlockOffset, WeakHash, CorpusId, CorpusBlock, ReductoInstruction, ReductoHeader,
            ChunkConfig, DataChunk, CorpusMetadata, Signature, AccessOperation, MetricsFormat,
            BLOCK_SIZE, HASH_BASE,
        },
        cdc_chunker::{FastCDCChunker, GearHasher},
        rolling_hash::{RollingHasher, StrongHasher, DualHasher, DualHashStatistics},
        corpus_manager::{EnterpriseCorpusManager, InMemoryStorage, StorageStats, StorageBackend},
        compressor::{Compressor, CompressionStats},
        serializer::{AdvancedSerializer, SerializerConfig, CompressionProfile, SerializationStats},
        security_manager::{EnterpriseSecurityManager, KeyManagementConfig, AuditLogger, AuditEvent, OperationResult, AuditSeverity},
    };
}