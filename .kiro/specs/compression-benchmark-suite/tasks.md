# Implementation Plan

- [ ] 1. Set up project structure and core interfaces
  - Create binary entry point at `src/bin/benchmark.rs`
  - Define core data structures in `src/models.rs`
  - Set up error hierarchy in `src/errors.rs`
  - Create trait definitions for compression testers
  - _Requirements: 6.1, 6.2_

- [ ] 2. Implement data loading and generation components
  - [ ] 2.1 Create DataLoader for directory processing
    - Implement directory traversal with 100MB size limit
    - Add file type detection and filtering
    - Create DataSet construction from file collection
    - Write unit tests for size limits and file filtering
    - _Requirements: 1.1, 1.2_

  - [ ] 2.2 Implement DataGenerator for test data creation
    - Generate realistic source code patterns (functions, classes, imports)
    - Generate JSON log entries with common fields and patterns
    - Generate documentation with repeated sections and boilerplate
    - Create 20MB mixed dataset when no path provided
    - Write tests to verify generated data characteristics
    - _Requirements: 1.2_

- [ ] 3. Build data analysis engine
  - [ ] 3.1 Implement redundancy detection algorithms
    - Create rolling hash implementation for pattern detection
    - Implement duplicate block identification with configurable block sizes
    - Calculate redundancy percentage based on duplicate content
    - Write unit tests with known redundancy patterns
    - _Requirements: 1.3, 1.6_

  - [ ] 3.2 Create early exit logic for low redundancy data
    - Implement 10% redundancy threshold checking
    - Create early termination with "not recommended" message
    - Add logging for redundancy analysis results
    - Write tests for threshold boundary conditions
    - _Requirements: 1.4, 1.5_

- [ ] 4. Implement corpus building system
  - [ ] 4.1 Create CorpusBuilder with smart selection
    - Implement most-redundant-data selection algorithm
    - Build corpus from 30% of input data by redundancy score
    - Add 10-second time limit with graceful degradation
    - Write tests for corpus quality and build time limits
    - _Requirements: 7.1, 7.2_

  - [ ] 4.2 Add corpus effectiveness validation
    - Implement compression advantage measurement (>10% threshold)
    - Create corpus quality scoring based on block frequency
    - Add early exit when corpus provides insufficient benefit
    - Write tests for effectiveness calculation accuracy
    - _Requirements: 7.3, 7.4_

- [ ] 5. Build compression testing framework
  - [ ] 5.1 Implement GzipTester with level 6 compression
    - Create gzip compression using flate2 crate with level 6
    - Implement decompression with error handling
    - Measure compression and decompression timing
    - Calculate compression ratios and store results
    - Write tests for compression consistency and performance
    - _Requirements: 2.1, 2.2, 2.4_

  - [ ] 5.2 Implement ReductoTester with corpus integration
    - Integrate existing Reducto Mode 3 implementation
    - Add corpus building time to total compression time
    - Implement compression with reference corpus
    - Add decompression with corpus dependency
    - Write tests for corpus-based compression accuracy
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 5.3 Create compression comparison logic
    - Implement speed ratio calculation (10x threshold)
    - Compare compression ratios between formats
    - Determine winner based on ratio improvement and speed
    - Create structured comparison results
    - Write tests for winner determination edge cases
    - _Requirements: 2.5, 2.6, 2.7_

- [ ] 6. Build reliability verification system
  - [ ] 6.1 Implement BLAKE3-based integrity checking
    - Add BLAKE3 hashing for original and decompressed data
    - Create hash comparison for corruption detection
    - Implement immediate failure on hash mismatch
    - Write tests with intentionally corrupted data
    - _Requirements: 3.1, 3.4_

  - [ ] 6.2 Add decompression speed validation
    - Measure decompression timing for both formats
    - Implement 5x speed limit checking for decompression
    - Create reliability failure reporting
    - Write tests for speed threshold enforcement
    - _Requirements: 3.2, 3.3_

- [ ] 7. Create recommendation engine
  - [ ] 7.1 Implement decision logic with conservative bias
    - Create winner-to-recommendation mapping
    - Implement conservative bias for close results
    - Generate clear YES/NO recommendations
    - Add rationale explanation for each decision
    - Write tests for decision boundary conditions
    - _Requirements: 4.1, 4.3, 4.4_

  - [ ] 7.2 Add cost savings calculation
    - Implement storage cost estimation based on compression improvement
    - Create monthly savings calculation with configurable rates
    - Add cost savings to recommendation output
    - Write tests for cost calculation accuracy
    - _Requirements: 4.2_

  - [ ] 7.3 Generate next steps guidance
    - Add "cargo add reducto" instruction for positive recommendations
    - Include specific reasons for negative recommendations
    - Create actionable guidance based on test results
    - Write tests for guidance message generation
    - _Requirements: 4.2, 4.5_

- [ ] 8. Build output formatting system
  - [ ] 8.1 Create console output formatter
    - Implement progress reporting during benchmark execution
    - Format final results in readable text format
    - Display data analysis, compression results, and recommendation
    - Add timing information and performance metrics
    - Write tests for output format consistency
    - _Requirements: 5.1, 5.2_

  - [ ] 8.2 Implement file output generation
    - Create benchmark_results.txt with detailed results
    - Include all metrics, timing, and recommendation rationale
    - Format for easy parsing and archival
    - Write tests for file output completeness
    - _Requirements: 5.3_

- [ ] 9. Implement CLI orchestration and timing
  - [ ] 9.1 Create main CLI entry point with argument parsing
    - Use clap for command-line argument parsing
    - Support optional data directory path
    - Add help text and usage examples
    - Implement graceful error handling and user feedback
    - Write tests for argument parsing edge cases
    - _Requirements: 6.1, 6.2_

  - [ ] 9.2 Add 60-second timeout enforcement
    - Implement tokio-based timeout for entire benchmark
    - Add graceful degradation when approaching time limit
    - Create smaller data sampling when time-constrained
    - Ensure clean exit on timeout with partial results
    - Write tests for timeout behavior and data sampling
    - _Requirements: 6.1, 6.3, 6.4_

  - [ ] 9.3 Integrate all components into complete workflow
    - Wire together data loading, analysis, compression, and output
    - Add error propagation and recovery strategies
    - Implement progress reporting throughout pipeline
    - Create end-to-end integration with all components
    - Write comprehensive integration tests for full workflow
    - _Requirements: 6.5_

- [ ] 10. Add comprehensive testing and validation
  - [ ] 10.1 Create performance contract tests
    - Write tests to validate 60-second time limit compliance
    - Test memory usage stays within reasonable bounds
    - Validate compression ratio calculation accuracy
    - Test reliability verification catches all corruption types
    - _Requirements: 6.1, 3.1, 2.2_

  - [ ] 10.2 Add property-based testing for core algorithms
    - Test compression/decompression round-trip properties
    - Validate redundancy detection with generated patterns
    - Test corpus building effectiveness across data types
    - Verify recommendation consistency with same inputs
    - _Requirements: 1.3, 7.1, 4.1_

  - [ ] 10.3 Create integration tests with real-world data
    - Test with actual source code repositories
    - Test with log files and documentation
    - Validate end-to-end workflow with various data types
    - Test error handling with invalid inputs and edge cases
    - _Requirements: 1.1, 1.2, 6.5_