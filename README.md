# Reducto Mode 3 - Differential Synchronization

High-performance compression system that achieves extreme compression ratios by identifying data blocks in target files that already exist in a shared Reference Corpus (RC) and replacing them with efficient pointer references.

## Architecture Overview

The system follows a layered architecture pattern:

- **L1 Core**: Block processing, rolling hash implementation, RAII resource management
- **L2 Standard**: Collections (HashMap), iterators, smart pointers (Arc), error handling  
- **L3 External**: Serialization (bincode), compression (zstd), memory mapping (memmap2), cryptographic hashing (BLAKE3)

## Current Implementation Status

### ✅ Task 1: Core Architecture and Contracts (COMPLETED)

- **Cargo.toml**: Configured with layered dependencies (L1→L2→L3)
- **Error Hierarchy**: Exhaustive `ReductoError` enum with thiserror for all failure modes
- **Core Traits**: Defined with comprehensive contracts
  - `HashProvider`: Rolling and strong hash computation
  - `BlockMatcher`: Block matching against corpus manifest
  - `CorpusReader`: Memory-efficient corpus data access
  - `InstructionWriter`: Compressed instruction stream writing
- **Type System**: Newtype wrappers for type safety
  - `BlockOffset`, `WeakHash`, `CorpusId`
  - `CorpusBlock`, `ReductoInstruction`, `ReductoHeader`
- **STUB Implementations**: All traits return `unimplemented!()` for TDD
- **Contract Tests**: Comprehensive tests for preconditions, postconditions, and error conditions

### 🔄 Next Tasks

- Task 2: L1 core data models with type safety
- Task 3: Rolling hash engine with performance contracts
- Task 4: Corpus indexing with dependency injection
- And more...

## Key Features

### Type Safety Through Design
- Newtype pattern prevents ID confusion (`UserId(Uuid)`, `BlockOffset(u64)`)
- Exhaustive error hierarchies with structured error handling
- Compile-time validation with const generics

### Performance Contracts
- Rolling hash: O(1) hash updates vs O(K) recalculation
- Block matching: O(1) average lookup time with HashMap
- Memory mapping: O(1) corpus access regardless of size

### Contract-Driven Development
All traits include comprehensive documentation of:
- **Preconditions**: What must be true before calling
- **Postconditions**: What will be true after successful completion  
- **Error conditions**: What errors can occur and why
- **Performance contracts**: Timing and complexity guarantees

## Building and Testing

```bash
# Check compilation
cargo check

# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Check specific module
cargo test error::tests
```

## Dependencies

### L1 Core (No external dependencies)
- Rust standard library features only

### L2 Standard Library
- Collections, iterators, smart pointers, thread safety

### L3 External Dependencies
- `blake3`: Strong cryptographic hashing with serde support
- `bincode`: Efficient binary serialization
- `serde`: Serialization framework with derive macros
- `zstd`: High-performance compression
- `memmap2`: Memory-mapped file access
- `hashbrown`: High-performance HashMap
- `anyhow`: Application error handling
- `thiserror`: Library error definitions
- `uuid`: Unique identifiers with v4 and serde features

### Development Dependencies
- `proptest`: Property-based testing
- `criterion`: Benchmarking framework
- `tempfile`: Temporary files for testing
- `tokio-test`: Async testing utilities

## Design Principles

1. **Executable Specifications**: All requirements are testable contracts
2. **Layered Architecture**: Clear separation of concerns (L1→L2→L3)
3. **Dependency Injection**: All components depend on traits, not concrete types
4. **RAII Resource Management**: Automatic cleanup with Drop implementations
5. **Performance Validation**: All performance claims backed by automated tests
6. **Structured Error Handling**: thiserror for libraries, anyhow for applications
7. **Type Safety**: Make invalid states unrepresentable
8. **Thread Safety**: All traits require Send + Sync for concurrency

## Error Categories

The system defines comprehensive error categories:
- `io`: File system and I/O operations
- `serialization`: Data serialization/deserialization
- `compression`: Compression and decompression
- `format`: File format validation
- `corpus`: Corpus management and indexing
- `block`: Block processing and verification
- `memory`: Memory management and mapping
- `hash`: Hash computation and validation
- `configuration`: Parameter and configuration validation
- `concurrency`: Thread synchronization and safety
- `performance`: Performance contract violations
- `resource`: Resource exhaustion and limits
- `validation`: Input validation and constraints
- `system`: System and environment issues
- `internal`: Internal logic errors and bugs

## License

MIT License - see LICENSE file for details.