// Timeout integration test (uses 1s limit with sleep to simulate long work)
use compression_benchmark_suite::{compression::GzipTester, errors::BenchmarkError, run_benchmark};
use std::time::Duration;

#[tokio::test]
async fn test_global_timeout_triggers() {
    let data = None; // generate test data (fast) but we'll set timeout very small
    let output = std::path::PathBuf::from("/tmp/bench_dummy.txt");
    let res = tokio::time::timeout(Duration::from_secs(1), async {
        run_benchmark(data, 1, output, false).await
    }).await;
    match res {
        Err(_) => (), // outer timeout triggered
        Ok(Err(BenchmarkError::Timeout(_))) => (),
        _ => panic!("Expected timeout"),
    }
}
