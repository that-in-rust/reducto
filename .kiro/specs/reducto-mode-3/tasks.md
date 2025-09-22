# Implementation Plan - Reducto Mode 3

## Phase 1: L1 Core Architecture and Contracts (STUB → RED → GREEN)

- [x] 1. Define core trait contracts and error hierarchy
  - Create Cargo.toml with L1 (core), L2 (std), L3 (external) dependency layers
  - Define trait contracts for HashProvider, BlockMatcher, CorpusReader, InstructionWriter
  - Implement exhaustive ReductoError hierarchy with thiserror for all failure modes
  - Write STUB implementations for all traits returning unimplemented!()
  - Create contract tests with preconditions, postconditions, and error conditions
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 2. Implement L1 core data models with type safety
  - Define newtype wrappers: BlockOffset(u64), WeakHash(u64), CorpusId(String)
  - Create CorpusBlock struct with compile-time size validation
  - Implement ReductoInstruction enum with exhaustive pattern matching
  - Add ReductoHeader with magic bytes validation and version compatibility
  - Write property-based tests for all data model invariants using proptest
  - _Requirements: 1.5, 5.1, 5.2, 5.3, 5.4, 5.5_

## Phase 2: L2 Standard Library Implementation (RED → GREEN)

- [ ] 3. Build rolling hash engine with performance contracts
  - Create src/rolling_hash.rs module with RollingHasher struct
  - Implement RollingHasher with explicit performance contracts in documentation
  - Write failing tests for O(K) init() and O(1) roll() performance requirements
  - Implement polynomial rolling hash with wrapping arithmetic for safety
  - Add comprehensive unit tests validating hash consistency across window slides
  - Create property-based tests ensuring rolling hash equals direct calculation
  - Add benchmark tests validating <1μs per roll operation performance contract
  - Replace StubHashProvider with production RollingHasher implementation
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 4. Create corpus indexing with dependency injection
  - Create src/corpus_indexer.rs module with CorpusIndexer struct
  - Implement production version using HashMap<WeakHash, Vec<CorpusBlock>>
  - Create mock implementation for testing with controlled collision scenarios
  - Write failing tests for O(1) lookup performance and collision handling
  - Implement BLAKE3 strong hash verification with constant-time comparison
  - Add RAII resource management for file handles with Drop implementation
  - Replace StubBlockMatcher with production CorpusIndexer implementation
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

## Phase 3: L3 External Integration (GREEN → REFACTOR)

- [ ] 5. Implement compression engine with trait-based architecture
  - Create src/compressor.rs module with Compressor struct
  - Define Compressor trait with compress() method contract
  - Write comprehensive failing tests for all compression scenarios
  - Implement sliding window algorithm with rolling hash integration
  - Add block matching logic using two-tier hash verification
  - Handle residual data accumulation with bounded buffer management
  - Create stress tests for large file processing and memory constraints
  - Integrate with HashProvider and BlockMatcher traits
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 6. Build serialization layer with format validation
  - Create src/serializer.rs module with FileSerializer struct
  - Write failing tests for header validation and corruption detection
  - Implement bincode serialization with error boundary handling
  - Add Zstandard compression with configurable levels and error recovery
  - Create format version compatibility tests and migration strategies
  - Validate file integrity with checksums and magic number verification
  - Replace StubInstructionWriter with production FileSerializer implementation
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

## Phase 4: Memory Management and Performance (REFACTOR)

- [ ] 7. Implement decompression with memory mapping
  - Create src/decompressor.rs module with Decompressor struct
  - Write failing tests for bounds checking and invalid reference handling
  - Implement memory mapping with RAII cleanup and error boundaries
  - Add instruction processing with zero-copy optimization where possible
  - Create comprehensive validation for corpus ID and format compatibility
  - Add performance tests ensuring <100ms decompression for 10MB files
  - Replace StubCorpusReader with production MemoryMappedCorpusReader implementation
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 8. Add comprehensive property-based testing
  - Implement compression roundtrip property: compress(data) → decompress → data
  - Create rolling hash equivalence property across all input patterns
  - Add corpus manifest consistency properties with collision scenarios
  - Test instruction stream integrity with malformed input fuzzing
  - Validate performance properties with generated workloads
  - Add concurrency safety tests with multiple readers/writers
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.2, 3.3, 3.4, 3.5_

## Phase 5: Integration and Performance Validation

- [ ] 8.5. Create main library API and integration layer
  - Create src/api.rs module with high-level compress/decompress functions
  - Implement ReductoCompressor struct that orchestrates all components
  - Implement ReductoDecompressor struct for file decompression
  - Add comprehensive integration tests with real file scenarios
  - Create example usage documentation and code samples
  - Validate end-to-end compression/decompression roundtrip
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 9. Build performance-validated CLI interface
  - Create src/bin/reducto.rs CLI application
  - Create CLI with clap using builder pattern for type safety
  - Implement command handlers with comprehensive error context using anyhow
  - Add progress reporting and cancellation support for long operations
  - Write integration tests with real file scenarios and edge cases
  - Create performance regression tests with automated benchmarking
  - Add memory profiling and leak detection in test suite
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 10. Implement production-ready error handling
  - Create context-aware error messages with actionable guidance
  - Add structured logging with tracing crate for observability
  - Implement graceful degradation for partial failures
  - Create error recovery strategies for corrupted files and network issues
  - Add comprehensive error boundary testing with fault injection
  - Validate error message clarity with user experience testing
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

## Phase 6: Final Optimization and Validation

- [ ] 11. Performance optimization with measurement
  - Profile critical paths using criterion benchmarks and flamegraphs
  - Optimize memory layout with #[repr] annotations for cache efficiency
  - Implement SIMD optimizations for hash calculations where beneficial
  - Add compile-time optimizations with const generics and const fn
  - Create performance regression detection in CI pipeline
  - Validate all performance claims with automated measurement
  - _Requirements: 1.1, 2.5, 3.1, 3.2, 3.3_

- [ ] 12. Production readiness and documentation
  - Add clap dependency for CLI argument parsing
  - Add comprehensive API documentation with executable examples
  - Create deployment guides with performance tuning recommendations
  - Add configuration validation and environment-specific settings
  - Create troubleshooting guides with common error scenarios
  - Perform final security audit and dependency vulnerability scanning
  - Create README.md with usage examples and performance benchmarks
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5_