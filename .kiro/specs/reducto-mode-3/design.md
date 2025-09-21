# Design Document - Reducto Mode 3 (Differential Synchronization)

## Overview

Reducto Mode 3 implements a high-performance block-based differential compression system that achieves extreme compression ratios by identifying data blocks in target files that already exist in a shared Reference Corpus (RC) and replacing them with efficient pointer references. The system uses a two-tier hashing approach with rolling hash algorithms for optimal performance.

### Core Innovation

The system employs a dual-hash strategy:
- **Weak Hash**: Fast polynomial rolling hash (Rabin-Karp style) for initial candidate identification
- **Strong Hash**: BLAKE3 cryptographic hash for verification and collision resolution

This approach enables efficient scanning of input data while maintaining high accuracy in block matching.

## Architecture

### High-Level System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input File    │    │ Reference Corpus│    │  Output File    │
│                 │    │     (RC)        │    │   (.reducto)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Compressor    │◄──►│ Corpus Manifest │    │  Decompressor   │
│                 │    │    (Index)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Layer Architecture

Following the L1→L2→L3 pattern:

- **L1 Core**: Block processing, rolling hash implementation, RAII resource management
- **L2 Standard**: Collections (HashMap), iterators, smart pointers (Arc), error handling
- **L3 External**: Serialization (bincode), compression (zstd), memory mapping (memmap2), cryptographic hashing (BLAKE3)

## Components and Interfaces

### 1. Core Data Structures

```rust
// Configuration constants
const BLOCK_SIZE: usize = 4096; // 4 KiB blocks for optimal performance
const HASH_BASE: u64 = 67; // Prime base for rolling hash

// Corpus block representation in the index
#[derive(Debug, Clone)]
struct CorpusBlock {
    offset: u64,
    strong_hash: Hash, // BLAKE3 Hash for verification
}

// Corpus manifest - the core index structure
type CorpusManifest = HashMap<u64, Vec<CorpusBlock>>;

// Output instruction format
#[derive(Serialize, Deserialize, Debug)]
pub enum ReductoInstruction {
    Reference(u64),      // Reference to corpus block by offset
    Residual(Vec<u8>),   // Literal data not found in corpus
}

// File format header
#[derive(Serialize, Deserialize, Debug)]
pub struct ReductoHeader {
    magic: [u8; 8],      // b"R3_AB2025"
    corpus_id: String,   // Unique corpus identifier
    block_size: u32,     // Block size used for compression
}
```

### 2. Rolling Hash Engine

**Design Decision**: Polynomial rolling hash provides O(1) hash updates as the scanning window advances, enabling efficient block matching without rescanning overlapping data.

```rust
struct RollingHasher {
    hash: u64,
    base: u64,
    power: u64, // Pre-calculated base^(BLOCK_SIZE-1)
}

impl RollingHasher {
    // Initialize hash for a window (O(K) time where K = BLOCK_SIZE)
    fn init(&mut self, data: &[u8]);
    
    // Roll hash one byte forward (O(1) time)
    fn roll(&mut self, exiting_byte: u8, entering_byte: u8);
    
    fn current_hash(&self) -> u64;
}
```

**Rationale**: The rolling hash eliminates the need to recalculate hashes for overlapping windows, providing significant performance improvements over naive approaches.

### 3. Corpus Indexer

**Design Decision**: Pre-compute a complete manifest of all blocks in the reference corpus using both weak and strong hashes to enable fast lookups with collision handling.

```rust
pub struct CorpusIndexer {
    manifest: CorpusManifest,
    corpus_id: String,
}

impl CorpusIndexer {
    pub fn build_manifest(corpus_path: &Path) -> Result<Self, IndexError>;
    pub fn get_candidates(&self, weak_hash: u64) -> Option<&Vec<CorpusBlock>>;
    pub fn verify_match(&self, block: &[u8], candidate: &CorpusBlock) -> bool;
}
```

**Rationale**: Separating indexing from compression allows the manifest to be built once and reused for multiple compression operations, improving overall system efficiency.

### 4. Compression Engine

**Design Decision**: Use a sliding window approach with rolling hash to scan the input file, checking for matches at every byte position while maintaining optimal performance.

```rust
pub struct Compressor {
    hasher: RollingHasher,
    manifest: Arc<CorpusManifest>,
}

impl Compressor {
    pub fn compress(&mut self, input_data: &[u8]) -> Vec<ReductoInstruction>;
    
    // Core matching logic
    fn find_block_match(&self, window: &[u8], weak_hash: u64) -> Option<u64>;
    fn advance_window(&mut self, input: &[u8], cursor: &mut usize) -> bool;
}
```

**Algorithm Flow**:
1. Initialize rolling hash for first BLOCK_SIZE bytes
2. For each position:
   - Check manifest for weak hash matches
   - Verify candidates with strong hash (BLAKE3)
   - If match found: emit Reference instruction, advance by BLOCK_SIZE
   - If no match: add byte to residual buffer, advance by 1 byte, roll hash
3. Flush remaining residual data

### 5. Decompression Engine

**Design Decision**: Use memory mapping for corpus access to handle large reference corpora efficiently without loading entirely into memory.

```rust
pub struct Decompressor {
    corpus_mmap: Mmap,
    corpus_id: String,
}

impl Decompressor {
    pub fn new(corpus_path: &Path, expected_corpus_id: &str) -> Result<Self, DecompressionError>;
    pub fn decompress(&self, compressed_file: &Path) -> Result<Vec<u8>, DecompressionError>;
    
    // Core reconstruction logic
    fn process_instruction(&self, instruction: &ReductoInstruction, output: &mut Vec<u8>) -> Result<(), DecompressionError>;
}
```

**Rationale**: Memory mapping allows the OS to manage corpus data efficiently, providing fast random access while supporting corpora larger than available RAM.

## Data Models

### File Format Specification (.reducto)

```
┌─────────────────┐
│ Header Length   │ 4 bytes (u32, little-endian)
├─────────────────┤
│ Header Data     │ Variable length (bincode serialized)
├─────────────────┤
│ Compressed      │ Variable length (zstd compressed
│ Instructions    │ bincode serialized instructions)
└─────────────────┘
```

### Instruction Stream Format

The instruction stream consists of a sequence of `ReductoInstruction` enums:
- **Reference(offset)**: Copy BLOCK_SIZE bytes from corpus at given offset
- **Residual(data)**: Copy literal data directly to output

### Corpus Manifest Structure

```rust
// Key: Weak hash (u64)
// Value: Vector of blocks with matching weak hash
HashMap<u64, Vec<CorpusBlock>>

// Handles weak hash collisions by storing multiple candidates
// Strong hash verification resolves actual matches
```

## Error Handling

### Structured Error Hierarchy

Following the thiserror pattern for library-style structured errors:

```rust
#[derive(Error, Debug)]
pub enum ReductoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    
    #[error("Compression error: {0}")]
    Compression(#[from] zstd::Error),
    
    #[error("Invalid file format: {reason}")]
    InvalidFormat { reason: String },
    
    #[error("Corpus ID mismatch: expected {expected}, found {found}")]
    CorpusIdMismatch { expected: String, found: String },
    
    #[error("Block size mismatch: expected {expected}, found {found}")]
    BlockSizeMismatch { expected: u32, found: u32 },
    
    #[error("Invalid reference offset: {offset} exceeds corpus bounds {max}")]
    InvalidReference { offset: u64, max: u64 },
    
    #[error("Memory mapping failed: {0}")]
    MemoryMapping(String),
}
```

### Error Recovery Strategies

- **Validation Errors**: Fail fast with descriptive messages
- **IO Errors**: Propagate with context about the operation being performed
- **Format Errors**: Provide specific details about what was expected vs. found
- **Memory Errors**: Handle gracefully without crashing

## Testing Strategy

### Unit Testing Approach

Following TDD principles with comprehensive test coverage:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    mod rolling_hash_tests {
        #[test]
        fn test_hash_initialization() { /* ... */ }
        
        #[test]
        fn test_hash_rolling() { /* ... */ }
        
        #[test]
        fn test_hash_consistency() { /* ... */ }
    }
    
    mod compression_tests {
        #[test]
        fn test_exact_block_matches() { /* ... */ }
        
        #[test]
        fn test_partial_matches() { /* ... */ }
        
        #[test]
        fn test_no_matches() { /* ... */ }
        
        #[test]
        fn test_hash_collisions() { /* ... */ }
    }
    
    mod decompression_tests {
        #[test]
        fn test_roundtrip_compression() { /* ... */ }
        
        #[test]
        fn test_corpus_id_validation() { /* ... */ }
        
        #[test]
        fn test_invalid_references() { /* ... */ }
    }
}
```

### Property-Based Testing

Using proptest for invariant validation:

```rust
proptest! {
    #[test]
    fn compression_roundtrip_property(
        input_data in prop::collection::vec(any::<u8>(), 0..10000)
    ) {
        // Property: compress(data) -> decompress -> original data
        let compressed = compress(&input_data, &test_corpus);
        let decompressed = decompress(&compressed, &test_corpus);
        prop_assert_eq!(input_data, decompressed);
    }
    
    #[test]
    fn rolling_hash_equivalence(
        data in prop::collection::vec(any::<u8>(), BLOCK_SIZE..BLOCK_SIZE*2)
    ) {
        // Property: Rolling hash equals direct hash calculation
        let direct_hash = calculate_direct_hash(&data[0..BLOCK_SIZE]);
        let mut roller = RollingHasher::new(HASH_BASE);
        roller.init(&data[0..BLOCK_SIZE]);
        prop_assert_eq!(direct_hash, roller.current_hash());
    }
}
```

### Integration Testing

End-to-end testing with real file scenarios:

```rust
#[test]
fn test_large_file_compression() {
    // Test with files larger than available memory
    let large_file = create_test_file(100 * 1024 * 1024); // 100MB
    let corpus = create_test_corpus(50 * 1024 * 1024);    // 50MB
    
    let compressed = compress_file(&large_file, &corpus).unwrap();
    let decompressed = decompress_file(&compressed, &corpus).unwrap();
    
    assert_files_equal(&large_file, &decompressed);
}

#[test]
fn test_performance_contracts() {
    let input_size = 2_100_000; // 2.1MB as specified in requirements
    let test_data = create_test_data(input_size);
    
    let start = Instant::now();
    let _compressed = compress(&test_data, &test_corpus);
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_secs(5), 
            "Compression took {:?}, expected <5s", elapsed);
}
```

### Performance Testing

Benchmark critical paths to validate performance claims:

```rust
#[bench]
fn bench_rolling_hash_performance(b: &mut Bencher) {
    let data = vec![0u8; BLOCK_SIZE * 1000];
    let mut hasher = RollingHasher::new(HASH_BASE);
    
    b.iter(|| {
        hasher.init(&data[0..BLOCK_SIZE]);
        for i in BLOCK_SIZE..data.len() {
            hasher.roll(data[i - BLOCK_SIZE], data[i]);
        }
    });
}

#[bench]
fn bench_block_matching(b: &mut Bencher) {
    let manifest = create_test_manifest(10000); // 10k blocks
    let test_blocks = create_test_blocks(1000);
    
    b.iter(|| {
        for block in &test_blocks {
            black_box(find_block_match(&manifest, block));
        }
    });
}
```

## Performance Considerations

### Memory Efficiency

- **Corpus Manifest**: Uses HashMap for O(1) average lookup time
- **Memory Mapping**: Enables handling of large corpora without full memory load
- **Streaming Processing**: Processes input files without loading entirely into memory
- **RAII Resource Management**: Automatic cleanup of file handles and memory maps

### CPU Optimization

- **Rolling Hash**: O(1) hash updates vs O(K) recalculation
- **Block-Aligned Processing**: 4KB blocks optimize for CPU cache lines
- **Minimal Allocations**: Reuse buffers where possible
- **SIMD-Friendly Operations**: Hash calculations can leverage vectorization

### I/O Optimization

- **Sequential Corpus Reading**: Build manifest with single sequential pass
- **Buffered Output**: Batch instruction writes for better I/O performance
- **Memory Mapping**: Leverage OS page cache for corpus access
- **Compression Pipeline**: Overlap compression with I/O operations

## Security Considerations

### Hash Collision Resistance

- **Two-Tier Hashing**: Weak hash for speed, strong hash (BLAKE3) for security
- **Collision Handling**: Multiple candidates per weak hash prevent false matches
- **Cryptographic Strength**: BLAKE3 provides strong collision resistance

### Input Validation

- **File Format Validation**: Strict header and magic number checking
- **Bounds Checking**: Validate all offsets against corpus size
- **Corpus ID Verification**: Prevent decompression with wrong corpus
- **Size Limits**: Prevent memory exhaustion attacks

### Resource Protection

- **Memory Limits**: Bounded allocations prevent DoS attacks
- **Timeout Handling**: Prevent infinite processing loops
- **Error Boundaries**: Fail safely without exposing internal state

## Deployment and Operations

### Build Configuration

```toml
[dependencies]
blake3 = "1.5"        # Strong cryptographic hashing
bincode = "1.3"       # Efficient binary serialization
serde = { version = "1.0", features = ["derive"] }
zstd = "0.13"         # High-performance compression
memmap2 = "0.9"       # Memory-mapped file access
hashbrown = "0.14"    # High-performance HashMap
anyhow = "1.0"        # Application error handling
thiserror = "1.0"     # Library error definitions

[dev-dependencies]
proptest = "1.0"      # Property-based testing
criterion = "0.5"     # Benchmarking framework
tempfile = "3.0"      # Temporary files for testing
```

### CLI Interface Design

```rust
// Command-line interface structure
pub enum ReductoCommand {
    Index {
        corpus_path: PathBuf,
        output_manifest: PathBuf,
    },
    Compress {
        input_file: PathBuf,
        corpus_path: PathBuf,
        output_file: PathBuf,
    },
    Decompress {
        compressed_file: PathBuf,
        corpus_path: PathBuf,
        output_file: PathBuf,
    },
}
```

### Monitoring and Metrics

- **Compression Ratios**: Track effectiveness across different file types
- **Processing Speed**: Monitor throughput (MB/s) for performance regression detection
- **Memory Usage**: Track peak memory consumption during operations
- **Error Rates**: Monitor and alert on compression/decompression failures

This design provides a robust, high-performance implementation of differential synchronization that meets all specified requirements while maintaining excellent performance characteristics and operational reliability.