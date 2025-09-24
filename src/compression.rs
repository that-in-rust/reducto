//! Generic compression interfaces and gzip tester implementation
//!
//! Follows L1→L2→L3 layered approach:
//! - L1: `CompressResult` (pure data, no std deps)
//! - L2: `CompressionTester` trait (std only)
//! - L3: `GzipTester` implementation (external crate `flate2`)

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