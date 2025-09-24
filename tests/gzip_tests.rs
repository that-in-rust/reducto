// Gzip tester unit tests (RED→GREEN already satisfied)
use compression_benchmark_suite::compression::{CompressResult, CompressionTester, GzipTester};

#[test]
fn test_gzip_roundtrip_and_ratio() {
    // Small deterministic data
    let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
    let tester = GzipTester::default();
    let result = tester.test(data).expect("gzip test should succeed");

    // Ensure decompression preserved data & ratio > 1 (compressed smaller)
    assert!(result.ratio > 1.0, "Compression ratio should be >1, got {}", result.ratio);
    assert!(result.compressed_size < data.len(), "Output should be smaller than input");
}
