// Formatter tests
use compression_benchmark_suite::compression::CompressResult;
use compression_benchmark_suite::decision::{Recommendation, Decision};
use compression_benchmark_suite::formatter::{format_console, save_report};
use tempfile::tempdir;

fn res(ratio: f64, time: u128) -> CompressResult {
    CompressResult { ratio, compressed_size: 0, time_ms: time }
}

#[test]
fn test_format_and_save_report() {
    let red = res(2.5, 400);
    let gz = res(1.8, 300);
    let rec = Recommendation { decision: Decision::Recommended, reason: "Test reason".into() };

    let text = format_console(&red, &gz, &rec);
    assert!(text.contains("Reducto Mode 3"));
    assert!(text.contains("Recommendation"));

    let dir = tempdir().unwrap();
    let path = dir.path().join("benchmark_results.txt");
    save_report(&path, &text).unwrap();
    let contents = std::fs::read_to_string(path).unwrap();
    assert_eq!(contents, text);
}
