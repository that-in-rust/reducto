This is a practical guide to coding **Reducto Mode 3 (Differential Synchronization)** in Rust. This mode achieves extreme compression by identifying data in the target file that already exists in a shared Reference Corpus (RC) and replacing that data with pointers.

The most practical approach for this is **Block-Based Differential Encoding**, utilizing rolling hashes (similar to `rsync`), which provides an excellent balance of speed, memory efficiency, and implementation complexity.

### The Architecture: Blocks and Rolling Hashes

We divide the Reference Corpus into fixed-size blocks (e.g., 4KB) and index them using a combination of a fast "weak" hash and a strong "cryptographic" hash. During compression, we use a rolling hash to efficiently scan the input file for matches in the index.

### Prerequisites and Dependencies

```toml
[dependencies]
blake3 = "1.5"        # Strong hash (e.g., BLAKE3) for verification
bincode = "1.3"       # Serialization of instructions
serde = { version = "1.0", features = ["derive"] }
zstd = "0.13"         # Secondary compression of the instruction stream
memmap2 = "0.9"       # Efficient access to the Corpus during decompression
hashbrown = "0.14"    # Efficient HashMap for the index
anyhow = "1.0"        # Error handling
```

### 1\. Data Structures and Configuration

```rust
use serde::{Serialize, Deserialize};
use hashbrown::HashMap;
use blake3::Hash;

// Configuration
const BLOCK_SIZE: usize = 4096; // 4 KiB
const HASH_BASE: u64 = 67; // Prime base for the rolling hash

// Represents a block in the Reference Corpus (used in the index)
#[derive(Debug, Clone)]
struct CorpusBlock {
    offset: u64,
    strong_hash: Hash, // BLAKE3 Hash
}

// The Corpus Manifest (Index)
// Key: Weak Hash (u64), Value: List of blocks matching that weak hash
type CorpusManifest = HashMap<u64, Vec<CorpusBlock>>;

// The output instructions for the decompressor
#[derive(Serialize, Deserialize, Debug)]
pub enum ReductoInstruction {
    // Reference a block from the Corpus (by offset). Length is implicitly BLOCK_SIZE.
    Reference(u64),
    // Literal data not found in the Corpus
    Residual(Vec<u8>),
}
```

### 2\. The Rolling Hash Implementation

We implement a polynomial rolling hash (Rabin-Karp style) which allows us to update the hash in O(1) time as the window slides forward by one byte.

```rust
struct RollingHasher {
    hash: u64,
    base: u64,
    // Pre-calculated power: base ^ (BLOCK_SIZE - 1)
    power: u64,
}

impl RollingHasher {
    fn new(base: u64) -> Self {
        let mut power = 1u64;
        // Calculate base ^ (BLOCK_SIZE - 1) using wrapping arithmetic
        for _ in 0..(BLOCK_SIZE - 1) {
            power = power.wrapping_mul(base);
        }
        Self { hash: 0, base, power }
    }

    // Initialize the hash for a window (O(K) time)
    fn init(&mut self, data: &[u8]) {
        assert_eq!(data.len(), BLOCK_SIZE);
        self.hash = 0;
        for &byte in data {
            self.hash = self.hash.wrapping_mul(self.base).wrapping_add(byte as u64);
        }
    }

    // Roll the hash one byte forward (O(1) time)
    fn roll(&mut self, exiting_byte: u8, entering_byte: u8) {
        // 1. Subtract the contribution of the exiting byte: (exiting_byte * power)
        let exiting_contribution = (exiting_byte as u64).wrapping_mul(self.power);
        self.hash = self.hash.wrapping_sub(exiting_contribution);
        
        // 2. Shift left (multiply by base)
        self.hash = self.hash.wrapping_mul(self.base);
        
        // 3. Add the entering byte
        self.hash = self.hash.wrapping_add(entering_byte as u64);
    }

    fn current_hash(&self) -> u64 {
        self.hash
    }
}
```

### 3\. The Indexer (Pre-computation)

This phase processes the Reference Corpus once to build the `CorpusManifest`.

```rust
use std::fs;
use std::path::Path;
use anyhow::Result;

fn build_manifest(corpus_path: &Path) -> Result<CorpusManifest> {
    println!("Indexing Reference Corpus...");
    // Note: For very large corpora, consider using memmap or stream processing.
    let corpus_data = fs::read(corpus_path)?;
    let mut manifest = CorpusManifest::new();
    let mut offset: u64 = 0;
    
    let mut hasher = RollingHasher::new(HASH_BASE);

    // Iterate over the corpus in non-overlapping blocks
    for block in corpus_data.chunks(BLOCK_SIZE) {
        // Only index full-size blocks
        if block.len() == BLOCK_SIZE {
            hasher.init(block);
            let weak_hash = hasher.current_hash();
            let strong_hash = blake3::hash(block);

            let entry = CorpusBlock {
                offset,
                strong_hash,
            };

            // Store the block in the manifest, handling potential weak hash collisions
            manifest.entry(weak_hash).or_default().push(entry);
        }
        offset += block.len() as u64;
    }
    println!("Index built.");
    Ok(manifest)
}
```

### 4\. The Compressor (Matching Logic)

The compressor scans the input file, using the rolling hash to efficiently identify potential matches in the manifest.

```rust
pub fn compress(input_data: &[u8], manifest: &CorpusManifest) -> Vec<ReductoInstruction> {
    let mut instructions = Vec::new();
    let mut cursor = 0; // Start of the current window
    let mut residual_buffer = Vec::new();
    
    if input_data.len() < BLOCK_SIZE {
        return vec![ReductoInstruction::Residual(input_data.to_vec())];
    }

    let mut hasher = RollingHasher::new(HASH_BASE);
    hasher.init(&input_data[0..BLOCK_SIZE]);

    // Loop while a full block remains
    while cursor <= input_data.len() - BLOCK_SIZE {
        let weak_hash = hasher.current_hash();
        let mut match_found = false;

        // 1. Check the manifest (Weak Hash Lookup)
        if let Some(candidates) = manifest.get(&weak_hash) {
            // 2. Verify potential matches (Strong Hash Check)
            let current_block = &input_data[cursor..cursor + BLOCK_SIZE];
            let strong_hash = blake3::hash(current_block);
            
            if let Some(matched_block) = candidates.iter().find(|c| c.strong_hash == strong_hash) {
                // 3. Strong match confirmed.
                
                // Flush pending residuals
                if !residual_buffer.is_empty() {
                    instructions.push(ReductoInstruction::Residual(residual_buffer.clone()));
                    residual_buffer.clear();
                }
                
                // Emit Reference
                instructions.push(ReductoInstruction::Reference(matched_block.offset));
                
                // Advance cursor past the block
                cursor += BLOCK_SIZE;
                match_found = true;
                
                // Re-initialize hash for the new position (if possible)
                if cursor <= input_data.len() - BLOCK_SIZE {
                    hasher.init(&input_data[cursor..cursor + BLOCK_SIZE]);
                }
            }
        }

        // 4. Handle no match
        if !match_found {
            // Add the first byte of the window to residuals.
            residual_buffer.push(input_data[cursor]);
            
            // Advance cursor by one byte
            cursor += 1;
            
            // Roll the hash efficiently (if possible)
            if cursor <= input_data.len() - BLOCK_SIZE {
                let exiting_byte = input_data[cursor - 1];
                let entering_byte = input_data[cursor + BLOCK_SIZE - 1];
                hasher.roll(exiting_byte, entering_byte);
            }
        }
    }

    // Handle remaining data (tail)
    if cursor < input_data.len() {
        residual_buffer.extend_from_slice(&input_data[cursor..]);
    }

    // Flush final residuals
    if !residual_buffer.is_empty() {
        instructions.push(ReductoInstruction::Residual(residual_buffer));
    }

    instructions
}
```

### 5\. Finalization (.ab2025 Format)

We serialize the instruction stream and compress it using Zstandard. This secondary compression is highly effective.

```rust
use std::io::Write;

#[derive(Serialize, Deserialize, Debug)]
pub struct ReductoHeader {
    magic: [u8; 8], // b"R3_AB2025"
    corpus_id: String, // Unique ID/Hash of the Corpus used
    block_size: u32,
}

fn finalize_file(instructions: Vec<ReductoInstruction>, corpus_id: String, output_path: &Path) -> Result<()> {
    // 1. Serialize the instruction stream
    let serialized_instructions = bincode::serialize(&instructions)?;

    // 2. Compress the stream using Zstandard (High level)
    let compressed_data = zstd::encode_all(&serialized_instructions[..], 19)?;

    // 3. Create the header
    let header = ReductoHeader {
        magic: *b"R3_AB2025",
        corpus_id,
        block_size: BLOCK_SIZE as u32,
    };

    // 4. Write the file (Header Length + Header + Payload)
    let mut file = fs::File::create(output_path)?;
    let header_bytes = bincode::serialize(&header)?;
    
    file.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    file.write_all(&compressed_data)?;

    Ok(())
}
```

### 6\. The Decompressor

Decompression is fast. It involves reading the instructions and reconstructing the file by copying data from the memory-mapped Reference Corpus.

```rust
use std::io::Read;
use memmap2::Mmap;
use anyhow::anyhow;

fn decompress(file_path: &Path, corpus_path: &Path, expected_corpus_id: &str) -> Result<Vec<u8>> {
    let mut file = fs::File::open(file_path)?;
    
    // 1. Read and Parse Header
    let mut len_bytes = [0u8; 4];
    file.read_exact(&mut len_bytes)?;
    let header_len = u32::from_le_bytes(len_bytes) as usize;

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header: ReductoHeader = bincode::deserialize(&header_bytes)?;

    // 2. Verification
    if &header.magic != b"R3_AB2025" || header.block_size as usize != BLOCK_SIZE || header.corpus_id != expected_corpus_id {
        return Err(anyhow!("Format, block size, or corpus ID mismatch."));
    }

    // 3. Read, Decompress, and Deserialize instructions
    let mut compressed_data = Vec::new();
    file.read_to_end(&mut compressed_data)?;
    let serialized_instructions = zstd::decode_all(&compressed_data[..])?;
    let instructions: Vec<ReductoInstruction> = bincode::deserialize(&serialized_instructions)?;

    // 4. Load the Reference Corpus using Memory Mapping
    let corpus_file = fs::File::open(corpus_path)?;
    // Safety: We assume the corpus file is not modified during decompression.
    let corpus_mmap = unsafe { Mmap::map(&corpus_file)? };

    // 5. Reconstruct the file
    let mut output = Vec::new();
    for instruction in instructions {
        match instruction {
            ReductoInstruction::Reference(offset) => {
                let start = offset as usize;
                let end = start + BLOCK_SIZE;
                
                if end > corpus_mmap.len() {
                     return Err(anyhow!("Invalid reference offset: exceeds corpus bounds."));
                }
                output.extend_from_slice(&corpus_mmap[start..end]);
            }
            ReductoInstruction::Residual(data) => {
                output.extend_from_slice(&data);
            }
        }
    }

    Ok(output)
}
```