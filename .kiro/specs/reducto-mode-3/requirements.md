# Requirements Document

## Introduction

Reducto Mode 3 (Differential Synchronization) is a high-performance compression system that achieves extreme compression ratios by identifying data blocks in target files that already exist in a shared Reference Corpus (RC) and replacing them with efficient pointer references. The system uses block-based differential encoding with rolling hashes (similar to rsync) to provide an optimal balance of compression speed, memory efficiency, and implementation complexity.

The core innovation is the use of a two-tier hashing system: fast "weak" hashes for initial candidate identification and strong cryptographic hashes for verification, combined with a rolling hash algorithm that enables efficient scanning of input data.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to compress files using differential synchronization against a reference corpus, so that I can achieve extreme compression ratios for files that share common data blocks.

#### Acceptance Criteria

1. WHEN a user provides an input file and reference corpus THEN the system SHALL compress the file using block-based differential encoding
2. WHEN the system processes the input file THEN it SHALL use 4KB fixed-size blocks for optimal performance
3. WHEN a block in the input matches a block in the reference corpus THEN the system SHALL replace it with a pointer reference
4. WHEN no match is found for data THEN the system SHALL store it as residual literal data
5. WHEN compression is complete THEN the system SHALL output a .reducto format file containing serialized instructions

### Requirement 2

**User Story:** As a system administrator, I want to build and maintain a reference corpus index, so that compression operations can efficiently locate matching blocks.

#### Acceptance Criteria

1. WHEN a reference corpus is provided THEN the system SHALL build a manifest index of all blocks
2. WHEN indexing blocks THEN the system SHALL calculate both weak hash (rolling polynomial) and strong hash (BLAKE3) for each block
3. WHEN storing the index THEN the system SHALL use a HashMap with weak hash as key and list of matching blocks as value
4. WHEN multiple blocks have the same weak hash THEN the system SHALL handle collisions by storing all candidates
5. WHEN the index is complete THEN the system SHALL provide efficient O(1) lookup for block matching

### Requirement 3

**User Story:** As a performance-conscious user, I want the compression to use rolling hash algorithms, so that block matching can be performed efficiently without rescanning overlapping data.

#### Acceptance Criteria

1. WHEN scanning input data THEN the system SHALL use a polynomial rolling hash (Rabin-Karp style)
2. WHEN the scanning window advances by one byte THEN the system SHALL update the hash in O(1) time
3. WHEN initializing a new window THEN the system SHALL calculate the hash in O(K) time where K is block size
4. WHEN rolling the hash THEN the system SHALL subtract exiting byte contribution, multiply by base, and add entering byte
5. WHEN hash collisions occur THEN the system SHALL verify matches using strong cryptographic hashes

### Requirement 4

**User Story:** As a user, I want to decompress .reducto files back to their original form, so that I can retrieve the original data using the reference corpus.

#### Acceptance Criteria

1. WHEN a .reducto file is provided for decompression THEN the system SHALL validate the file format and header
2. WHEN processing reference instructions THEN the system SHALL copy the corresponding blocks from the memory-mapped reference corpus
3. WHEN processing residual instructions THEN the system SHALL copy the literal data directly to output
4. WHEN the corpus ID doesn't match THEN the system SHALL reject the decompression with an error
5. WHEN decompression is complete THEN the system SHALL produce output identical to the original input file

### Requirement 5

**User Story:** As a developer, I want the system to use efficient serialization and secondary compression, so that the instruction stream itself is optimized for storage and transmission.

#### Acceptance Criteria

1. WHEN serializing instructions THEN the system SHALL use bincode for efficient binary serialization
2. WHEN finalizing the compressed file THEN the system SHALL apply Zstandard compression to the instruction stream
3. WHEN creating the output file THEN the system SHALL include a header with magic bytes, corpus ID, and block size
4. WHEN writing the file THEN the system SHALL use the format: header length + header + compressed payload
5. WHEN compression level is set THEN the system SHALL use Zstandard level 19 for maximum compression

### Requirement 6

**User Story:** As a system integrator, I want the decompressor to use memory mapping for corpus access, so that large reference corpora can be handled efficiently without loading entirely into memory.

#### Acceptance Criteria

1. WHEN opening a reference corpus for decompression THEN the system SHALL use memory mapping (mmap)
2. WHEN accessing corpus blocks THEN the system SHALL read directly from the mapped memory region
3. WHEN validating block references THEN the system SHALL ensure offsets don't exceed corpus bounds
4. WHEN the corpus file is modified during decompression THEN the system SHALL handle this as an error condition
5. WHEN memory mapping fails THEN the system SHALL provide clear error messages

### Requirement 7

**User Story:** As a quality assurance engineer, I want comprehensive error handling and validation, so that the system fails gracefully and provides clear diagnostics.

#### Acceptance Criteria

1. WHEN invalid file formats are encountered THEN the system SHALL return descriptive error messages
2. WHEN corpus ID mismatches occur THEN the system SHALL prevent decompression and explain the issue
3. WHEN block size mismatches are detected THEN the system SHALL reject the operation with clear feedback
4. WHEN file I/O operations fail THEN the system SHALL propagate errors with context information
5. WHEN memory allocation fails THEN the system SHALL handle the condition gracefully without crashing