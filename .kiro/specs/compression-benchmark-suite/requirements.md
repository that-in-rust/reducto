# Requirements Document

## Introduction

This feature creates a **dead-simple benchmark** that answers ONE critical question: **"Will Reducto Mode 3 save me storage/bandwidth costs on MY data?"**

The system tests the user's actual data (or generates realistic samples), compares Reducto Mode 3 against gzip (the universal baseline), and gives a clear YES/NO recommendation with quantified savings.

**Success Metric**: User runs `cargo run --bin benchmark [my-data-directory]` and gets a definitive answer in under 60 seconds.

**Anti-Goals**: 
- Complex multi-format comparisons (gzip is the only baseline that matters)
- Elaborate visualizations (simple text output is sufficient)
- Organizational deployment simulation (premature - prove value first)

## Requirements

### Requirement 1: User Data Analysis

**User Story:** As a developer with actual data files, I want to test Reducto Mode 3 on MY data, so that I know if it will save me money in my specific use case.

#### Acceptance Criteria

1. WHEN I run `cargo run --bin benchmark /path/to/my/data` THEN the system SHALL analyze all files in that directory up to 100MB total
2. WHEN I don't provide a data path THEN the system SHALL generate 20MB of realistic test data representing common scenarios: source code, JSON logs, and documentation
3. WHEN analyzing my data THEN the system SHALL identify redundancy patterns and predict if differential compression will be effective
4. WHEN my data has low redundancy THEN the system SHALL immediately report "Reducto Mode 3 not recommended for this data type" and exit
5. WHEN my data shows promise THEN the system SHALL proceed with compression testing
6. WHEN data analysis completes THEN the system SHALL output: "Data analysis: X% redundancy detected, proceeding with compression test"

### Requirement 2: Head-to-Head Compression Test

**User Story:** As a developer evaluating compression options, I need to see if Reducto Mode 3 beats gzip on my data, so that I can make a simple go/no-go decision.

#### Acceptance Criteria

1. WHEN running compression tests THEN the system SHALL test only 2 formats: Reducto Mode 3 vs gzip (the universal baseline)
2. WHEN measuring effectiveness THEN the system SHALL calculate compression ratio and compression speed for both formats
3. WHEN testing Reducto Mode 3 THEN the system SHALL build a reference corpus from 30% of the data and include corpus build time in total time
4. WHEN testing gzip THEN the system SHALL use level 6 (production default)
5. WHEN tests complete THEN the system SHALL determine the winner based on: better compression ratio AND reasonable speed (not >10x slower)
6. WHEN Reducto wins THEN the system SHALL report: "Reducto Mode 3: X% better compression, Y% speed difference"
7. WHEN gzip wins THEN the system SHALL report: "Stick with gzip: Reducto provides insufficient benefit"

### Requirement 3: Reliability Verification

**User Story:** As a developer considering Reducto Mode 3 for production, I need to know it won't corrupt my data, so that I can trust it with important files.

#### Acceptance Criteria

1. WHEN compression completes THEN the system SHALL decompress both formats and verify data integrity using BLAKE3 hashing
2. WHEN any decompression fails THEN the system SHALL immediately report "RELIABILITY FAILURE" and recommend against using that format
3. WHEN decompression succeeds THEN the system SHALL measure decompression speed to ensure it's reasonable (not >5x slower than gzip)
4. WHEN reliability testing completes THEN the system SHALL report: "Reliability: PASS" or "Reliability: FAIL - data corruption detected"

### Requirement 4: Clear Recommendation

**User Story:** As a busy developer, I want a simple YES/NO answer on whether to use Reducto Mode 3, so that I can make a decision and move on.

#### Acceptance Criteria

1. WHEN all tests complete THEN the system SHALL output ONE clear recommendation: "RECOMMENDED: Use Reducto Mode 3" or "NOT RECOMMENDED: Stick with gzip"
2. WHEN recommending Reducto THEN the system SHALL quantify the benefit: "X% smaller files, saves $Y/month in storage costs"
3. WHEN not recommending Reducto THEN the system SHALL explain why: "Insufficient compression improvement" or "Too slow for practical use"
4. WHEN the decision is close THEN the system SHALL default to "NOT RECOMMENDED" (conservative bias)
5. WHEN recommendation is made THEN the system SHALL include next steps: "Add Reducto to your project with: cargo add reducto"

### Requirement 5: Simple Text Output

**User Story:** As a developer running a benchmark, I want clear text output I can understand immediately, so that I don't need to interpret charts or diagrams.

#### Acceptance Criteria

1. WHEN benchmark completes THEN the system SHALL output results in plain text format that's immediately readable
2. WHEN displaying results THEN the system SHALL show: data type analyzed, compression ratios, speed comparison, and final recommendation
3. WHEN saving results THEN the system SHALL write a simple `benchmark_results.txt` file with the same information
4. WHEN results are unclear THEN the system SHALL err on the side of being too simple rather than too complex

### Requirement 6: 60-Second Benchmark

**User Story:** As a developer evaluating Reducto, I want to run `cargo run --bin benchmark` and get a definitive answer in under 60 seconds, so that I can quickly decide without waiting around.

#### Acceptance Criteria

1. WHEN I run `cargo run --bin benchmark` THEN the system SHALL complete all testing in under 60 seconds
2. WHEN I run `cargo run --bin benchmark /my/data` THEN the system SHALL test my specific data instead of generating test data
3. WHEN testing takes longer than 60 seconds THEN the system SHALL use a smaller data sample to stay within the time limit
4. WHEN benchmark completes THEN the system SHALL display the recommendation immediately and save details to `benchmark_results.txt`
5. WHEN any step fails THEN the system SHALL report the failure clearly and exit gracefully

### Requirement 7: Smart Corpus Building

**User Story:** As a user testing Reducto Mode 3, I want the system to automatically build the best possible reference corpus from my data, so that the test represents optimal performance.

#### Acceptance Criteria

1. WHEN building reference corpus THEN the system SHALL use 30% of the most redundant data as corpus material
2. WHEN corpus building takes longer than 10 seconds THEN the system SHALL use a smaller corpus to stay within time limits
3. WHEN corpus provides less than 10% compression advantage THEN the system SHALL report "Your data doesn't benefit from differential compression"
4. WHEN corpus building succeeds THEN the system SHALL report "Corpus built: X blocks, Y% redundancy detected"