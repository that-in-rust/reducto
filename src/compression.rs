
/// Mock implementation of Reducto Mode 3 compression.
/// Until the real CDC engine is wired, this uses Zstd as a stand-in and records
/// corpus build time (simulated as 30 % of total data preparation).
#[derive(Debug, Clone)]
pub struct ReductoMockTester {
    /// Zstd compression level
    level: i32,
    /// Fraction of data used for corpus (0.0‒1.0)
    corpus_fraction: f64,
}

impl Default for ReductoMockTester {
    fn default() -> Self {
        Self { level: 19, corpus_fraction: 0.3 }
    }
}

impl ReductoMockTester {
    pub fn new(level: i32, corpus_fraction: f64) -> Self {
        Self { level, corpus_fraction }
    }
}

impl CompressionTester for ReductoMockTester {
    fn test(&self, data: &[u8]) -> Result<CompressResult> {
        use std::io::{Read, Write};
        use zstd::stream::{encode_all, decode_all};

        // Simulate corpus build cost: slice first N bytes
        let corpus_size = (data.len() as f64 * self.corpus_fraction) as usize;
        let (corpus_data, payload) = data.split_at(corpus_size.min(data.len()));

        // Corpus build time simulation – proportional to corpus size (ns)
        let corpus_start = Instant::now();
        if !corpus_data.is_empty() {
            // simple hash to simulate work
            let _digest = blake3::hash(corpus_data);
        }
        let corpus_ms = corpus_start.elapsed().as_millis();

        // Compress payload with Zstd
        let start = Instant::now();
        let mut compressed = encode_all(payload, self.level)?;
        let elapsed_ms = start.elapsed().as_millis() + corpus_ms;

        // prepend corpus slice to compressed data to keep round-trip simple
        let mut full_blob = corpus_data.to_vec();
        full_blob.append(&mut compressed);

        // Decompress & reconstruct
        let mut decompressed_payload = decode_all(&full_blob[corpus_size..])?;
        let mut reconstructed = corpus_data.to_vec();
        reconstructed.append(&mut decompressed_payload);
        anyhow::ensure!(reconstructed == data, "Reducto mock round-trip failed");

        Ok(CompressResult {
            ratio: (data.len() as f64) / (full_blob.len() as f64),
            compressed_size: full_blob.len(),
            time_ms: elapsed_ms,
        })
    }
}

// Generic compression interfaces and gzip tester implementation
//
// Follows L1→L2→L3 layered approach:
// - L1: `CompressResult` (pure data, no std deps)
// - L2: `CompressionTester` trait (std only)
// - L3: `GzipTester` implementation (external crate `flate2`)

use anyhow::Result;
use std::time::Instant;

/// Result of a compression test
#[derive(Debug, Clone, PartialEq)]
pub struct CompressResult {
    /// Compression ratio original_size / compressed_size (higher is better)
    pub ratio: f64,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Time taken in milliseconds
    pub time_ms: u128,
}

/// Trait abstraction for compression testers (dependency-injection friendly)
pub trait CompressionTester {
    /// Compress `data` and verify decompression round-trip.
    fn test(&self, data: &[u8]) -> Result<CompressResult>;
}

/// Gzip implementation using `flate2` crate (level 6 default)
#[derive(Debug, Clone)]
pub struct GzipTester {
    level: flate2::Compression,
}

impl Default for GzipTester {
    fn default() -> Self {
        Self {
            level: flate2::Compression::new(6),
        }
    }
}

impl GzipTester {
    pub fn new(level: u32) -> Self {
        Self {
            level: flate2::Compression::new(level),
        }
    }
}

impl CompressionTester for GzipTester {
    fn test(&self, data: &[u8]) -> Result<CompressResult> {
        use flate2::{write::GzEncoder, read::GzDecoder};
        use std::io::{Read, Write};

        // --- Compress ---
        let start = Instant::now();
        let mut encoder = GzEncoder::new(Vec::new(), self.level);
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        let elapsed = start.elapsed().as_millis();

        // --- Decompress & verify ---
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut roundtrip = Vec::with_capacity(data.len());
        decoder.read_to_end(&mut roundtrip)?;
        anyhow::ensure!(roundtrip == data, "gzip round-trip verification failed");

        let result = CompressResult {
            ratio: (data.len() as f64) / (compressed.len() as f64),
            compressed_size: compressed.len(),
            time_ms: elapsed,
        };
        Ok(result)
    }
}