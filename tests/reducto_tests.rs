// Reducto mock tester unit tests
use compression_benchmark_suite::compression::{CompressionTester, ReductoMockTester};

#[test]
fn test_reducto_mock_roundtrip_and_ratio() {
    let data = vec![0u8; 1024 * 32]; // 32 KB sample
    let tester = ReductoMockTester::default();
    let result = tester.test(&data).expect("mock reduTo should succeed");

    // Ratio should still be >1 (compressed smaller)
    assert!(result.ratio > 1.0, "Expected ratio >1, got {}", result.ratio);
    assert!(result.compressed_size < data.len());
}
