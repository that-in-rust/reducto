# Implementation Plan

- [x] 1. Create basic CLI structure
  - Set up `src/bin/benchmark.rs` with clap for argument parsing
  - Add `src/lib.rs` with main benchmark function
  - Update `Cargo.toml` with dependencies: clap, flate2, anyhow
  - _Requirements: 6.1, 6.2_

- [-] 2. Implement data loading
  - Create function to load files from directory (max 100MB)
  - Create function to generate 20MB test data if no path given
  - Add simple file reading with size limits
  - _Requirements: 1.1, 1.2_

- [x] 3. Add gzip compression testing
  - Implement gzip compression with flate2 (level 6)
  - Measure compression time and ratio
  - Test decompression and verify data integrity
  - _Requirements: 2.1, 2.4, 3.1_

- [x] 4. Add Reducto Mode 3 compression testing
  - Integrate existing Reducto Mode 3 implementation
  - Build reference corpus from 30% of data
  - Measure compression time (including corpus build time)
  - Test decompression with corpus
  - _Requirements: 2.1, 2.3, 7.1_

- [x] 5. Implement comparison and recommendation logic
  - Compare compression ratios and speeds
  - Apply 10x speed limit rule
  - Generate "RECOMMENDED" or "NOT RECOMMENDED" decision
  - Add simple cost savings calculation
  - _Requirements: 2.5, 2.6, 4.1, 4.2_

- [ ] 6. Add output formatting
  - Print results to console in simple text format
  - Save results to benchmark_results.txt
  - Include timing, ratios, and recommendation
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 7. Add 60-second timeout and error handling
  - Implement overall timeout with graceful exit
  - Add basic error handling for file operations
  - Test with various data sizes and types
  - _Requirements: 6.1, 6.3, 6.5_