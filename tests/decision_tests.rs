// Decision engine tests
use compression_benchmark_suite::decision::{compare_results, Decision};
use compression_benchmark_suite::compression::CompressResult;

fn res(ratio: f64, time: u128) -> CompressResult {
    CompressResult { ratio, compressed_size: 0, time_ms: time }
}

#[test]
fn test_recommended_when_better_and_fast_enough() {
    let red = res(2.5, 500); // 2.5x, 0.5s
    let gz  = res(1.8, 300); // 1.8x, 0.3s
    let rec = compare_results(&red, &gz);
    assert_eq!(rec.decision, Decision::Recommended);
}

#[test]
fn test_not_recommended_when_ratio_insufficient() {
    let red = res(1.82, 500);
    let gz  = res(1.8, 300);
    let rec = compare_results(&red, &gz);
    assert_eq!(rec.decision, Decision::NotRecommended);
}

#[test]
fn test_not_recommended_when_too_slow() {
    let red = res(3.0, 5000); // 5s
    let gz  = res(2.0, 200);  // 0.2s => 25x slower
    let rec = compare_results(&red, &gz);
    assert_eq!(rec.decision, Decision::NotRecommended);
}
