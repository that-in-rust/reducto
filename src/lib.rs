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
pub mod cli_error;
pub mod simd_optimizations;
pub mod const_optimizations;

#[cfg(feature = "metrics")]
pub mod metrics_collector;

#[cfg(feature = "sdk")]
pub mod ecosystem_decompressor;

#[cfg(feature = "sdk")]
pub mod sdk;

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
pub use simd_optimizations::{SimdHashCalculator, SimdGearHasher};
pub use const_optimizations::{
    ConstChunkConfig, ConstRollingHasher, ConstCDCChunker, ConstGearHasher, ConstDataChunk,
    StandardChunker, LargeChunker, SmallChunker, PrecisionChunker,
    StandardRollingHasher, LargeRollingHasher,
};

#[cfg(feature = "metrics")]
pub use metrics_collector::{EnterpriseMetricsCollector, MetricsConfig, DryRunAnalysis, MetricsSnapshot, EconomicSummary};

#[cfg(feature = "sdk")]
pub use ecosystem_decompressor::{EcosystemDecompressor, CorpusRepository, CorpusSource, DecompressionResult, DecompressionMetrics, EcosystemConfig, StandardCompressor, ZstdFallbackCompressor, CacheStats};

#[cfg(feature = "sdk")]
pub use sdk::{
    ReductoSDK, SDKConfig, TimeoutConfig, StreamConfig, SecurityConfig, PipelineConfig,
    CompressionResult, CompressionMetrics, StreamStats, TarFilter, SshWrapper, CloudCliPlugin, CloudProvider,
    // C FFI exports
    CSDKConfig, CCompressionResult, CDecompressionResult, CResult, CErrorCode,
};

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
        simd_optimizations::{SimdHashCalculator, SimdGearHasher},
        const_optimizations::{
            ConstChunkConfig, ConstRollingHasher, ConstCDCChunker, ConstGearHasher, ConstDataChunk,
            StandardChunker, LargeChunker, SmallChunker, PrecisionChunker,
            StandardRollingHasher, LargeRollingHasher,
        },
    };
    
    #[cfg(feature = "metrics")]
    pub use crate::metrics_collector::{EnterpriseMetricsCollector, MetricsConfig, DryRunAnalysis, MetricsSnapshot, EconomicSummary};
    
    #[cfg(feature = "sdk")]
    pub use crate::ecosystem_decompressor::{EcosystemDecompressor, CorpusRepository, CorpusSource, DecompressionResult, DecompressionMetrics, EcosystemConfig, StandardCompressor, ZstdFallbackCompressor, CacheStats};
    
    #[cfg(feature = "sdk")]
    pub use crate::sdk::{
        ReductoSDK, SDKConfig, TimeoutConfig, StreamConfig, SecurityConfig, PipelineConfig,
        CompressionResult, CompressionMetrics, StreamStats, TarFilter, SshWrapper, CloudCliPlugin, CloudProvider,
    };
}