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

#[cfg(test)]
pub mod contract_tests;

// Re-export commonly used types
pub use error::{ReductoError, Result};
pub use traits::{HashProvider, BlockMatcher, CorpusReader, InstructionWriter};
pub use types::{
    BlockOffset, WeakHash, CorpusId, CorpusBlock, ReductoInstruction, ReductoHeader,
    BLOCK_SIZE, HASH_BASE,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        error::{ReductoError, Result},
        traits::{HashProvider, BlockMatcher, CorpusReader, InstructionWriter},
        types::{
            BlockOffset, WeakHash, CorpusId, CorpusBlock, ReductoInstruction, ReductoHeader,
            BLOCK_SIZE, HASH_BASE,
        },
    };
}