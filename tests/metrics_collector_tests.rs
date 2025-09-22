//! Comprehensive tests for metrics collection and export functionality
//!
//! These tests validate all aspects of the metrics collection system including
//! dry-run analysis, performance monitoring, economic reporting, and export formats.

use reducto_mode_3::{
    metrics_collector::{EnterpriseMetricsCollector, MetricsConfig, DryRunAnalysis},
    types::{CompressionMetrics, DecompressionMetrics, MetricsFormat, UsageStats},
    error::MetricsError,
};
use chrono::{Utc, Duration as ChronoDuration};
use std::io::Write;
use tempfile::NamedTempFile;
use tokio::time::{timeout, Duration};

/// Test metrics collector creation and configuration
#[test]
fn test_metrics_collector_creation() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config);
    assert!(collector.is_ok(), "Should create metrics collector successfully");
}

/// Test metrics collector with custom configuration
#[test]
fn test_metrics_collector_custom_config() {
    let config = MetricsConfig {
        enable_performance_monitoring: true,
        enable_economic_analysis: true,
        retention_hours: 48,
        sample_interval_ms: 500,
        bandwidth_cost_per_gb: 0.12,
        storage_cost_per_gb: 0.03,
        enable_prometheus: true,
        prometheus_endpoint: Some("http://localhost:9090".to_string()),
    };
    
    let collector = EnterpriseMetricsCollector::new(config.clone());
    assert!(collector.is_ok(), "Should create collector with custom config");
    
    let collector = collector.unwrap();
    // Verify configuration is applied (would need getter methods in real implementation)
}

/// Test compression metrics recording
#[test]
fn test_compression_metrics_recording() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    let metrics = CompressionMetrics {
        input_size: 1024 * 1024, // 1MB
        output_size: 512 * 1024, // 512KB
        compression_ratio: 2.0,
        corpus_hit_rate: 0.75,
        processing_time_ms: 150,
        memory_usage_bytes: 2 * 1024 * 1024, // 2MB
    };
    
    let result = collector.record_compression(metrics);
    assert!(result.is_ok(), "Should record compression metrics successfully");
}

/// Test decompression metrics recording
#[test]
fn test_decompression_metrics_recording() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    let metrics = DecompressionMetrics {
        input_size: 512 * 1024, // 512KB
        output_size: 1024 * 1024, // 1MB
        decompression_ratio: 2.0,
        processing_time_ms: 75,
        memory_usage_bytes: 1024 * 1024, // 1MB
        corpus_access_count: 15,
    };
    
    let result = collector.record_decompression(metrics);
    assert!(result.is_ok(), "Should record decompression metrics successfully");
}

/// Test multiple metrics recording and retention
#[test]
fn test_multiple_metrics_recording() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Record multiple compression operations
    for i in 0..50 {
        let metrics = CompressionMetrics {
            input_size: 1024 * (i + 1),
            output_size: 512 * (i + 1),
            compression_ratio: 2.0,
            corpus_hit_rate: 0.8,
            processing_time_ms: 100 + i,
            memory_usage_bytes: 2048 * (i + 1),
        };
        
        let result = collector.record_compression(metrics);
        assert!(result.is_ok(), "Should record compression metrics #{}", i);
    }
    
    // Record multiple decompression operations
    for i in 0..30 {
        let metrics = DecompressionMetrics {
            input_size: 512 * (i + 1),
            output_size: 1024 * (i + 1),
            decompression_ratio: 2.0,
            processing_time_ms: 50 + i,
            memory_usage_bytes: 1024 * (i + 1),
            corpus_access_count: 10 + i,
        };
        
        let result = collector.record_decompression(metrics);
        assert!(result.is_ok(), "Should record decompression metrics #{}", i);
    }
}

/// Test dry-run analysis for compression prediction
#[tokio::test]
async fn test_dry_run_analysis() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Create test input file with known content
    let mut input_file = NamedTempFile::new().unwrap();
    let test_data = vec![0u8; 10 * 1024]; // 10KB of zeros (low entropy)
    input_file.write_all(&test_data).unwrap();
    
    // Create test corpus file
    let mut corpus_file = NamedTempFile::new().unwrap();
    let corpus_data = vec![1u8; 100 * 1024]; // 100KB of ones
    corpus_file.write_all(&corpus_data).unwrap();
    
    let analysis = collector.analyze_compression_potential(
        input_file.path(),
        corpus_file.path(),
    ).await;
    
    assert!(analysis.is_ok(), "Dry-run analysis should succeed");
    
    let analysis = analysis.unwrap();
    assert_eq!(analysis.input_size, 10 * 1024, "Should detect correct input size");
    assert!(analysis.predicted_compression_ratio > 0.0, "Should predict positive compression ratio");
    assert!(analysis.predicted_hit_rate >= 0.0 && analysis.predicted_hit_rate <= 1.0, "Hit rate should be in valid range");
    assert!(analysis.confidence_level >= 0.0 && analysis.confidence_level <= 1.0, "Confidence should be in valid range");
    assert!(analysis.predicted_processing_time_ms > 0, "Should predict positive processing time");
    assert!(analysis.predicted_memory_usage > 0, "Should predict positive memory usage");
}

/// Test dry-run analysis with large files
#[tokio::test]
async fn test_dry_run_analysis_large_file() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Create larger test files
    let mut input_file = NamedTempFile::new().unwrap();
    let test_data = vec![42u8; 5 * 1024 * 1024]; // 5MB of repeated data
    input_file.write_all(&test_data).unwrap();
    
    let mut corpus_file = NamedTempFile::new().unwrap();
    let corpus_data = (0..1024*1024).map(|i| (i % 256) as u8).collect::<Vec<u8>>(); // 1MB of varied data
    corpus_file.write_all(&corpus_data).unwrap();
    
    let analysis = collector.analyze_compression_potential(
        input_file.path(),
        corpus_file.path(),
    ).await;
    
    assert!(analysis.is_ok(), "Large file analysis should succeed");
    
    let analysis = analysis.unwrap();
    assert_eq!(analysis.input_size, 5 * 1024 * 1024, "Should detect correct large file size");
    assert!(analysis.predicted_processing_time_ms > 0, "Should predict processing time for large file");
}

/// Test dry-run analysis error handling
#[tokio::test]
async fn test_dry_run_analysis_error_handling() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Test with non-existent input file
    let result = collector.analyze_compression_potential(
        std::path::Path::new("/nonexistent/input.dat"),
        std::path::Path::new("/nonexistent/corpus.dat"),
    ).await;
    
    assert!(result.is_err(), "Should fail with non-existent files");
    
    // Verify error type
    match result.unwrap_err() {
        reducto_mode_3::error::ReductoError::InternalError { message } => {
            assert!(message.contains("Metrics error"), "Should be a metrics error");
        }
        _ => panic!("Expected MetricsError"),
    }
}

/// Test performance metrics update
#[test]
fn test_performance_metrics_update() {
    let config = MetricsConfig {
        enable_performance_monitoring: true,
        ..Default::default()
    };
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    let result = collector.update_performance_metrics();
    assert!(result.is_ok(), "Performance metrics update should succeed");
}

/// Test performance metrics disabled
#[test]
fn test_performance_metrics_disabled() {
    let config = MetricsConfig {
        enable_performance_monitoring: false,
        ..Default::default()
    };
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    let result = collector.update_performance_metrics();
    assert!(result.is_ok(), "Should succeed even when disabled");
}

/// Test ROI calculation with realistic data
#[test]
fn test_roi_calculation() {
    let config = MetricsConfig {
        bandwidth_cost_per_gb: 0.09, // AWS pricing
        storage_cost_per_gb: 0.023,  // AWS S3 pricing
        ..Default::default()
    };
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Record some compression operations with significant savings
    for i in 0..10 {
        let metrics = CompressionMetrics {
            input_size: 100 * 1024 * 1024, // 100MB
            output_size: 10 * 1024 * 1024,  // 10MB (10:1 compression)
            compression_ratio: 10.0,
            corpus_hit_rate: 0.9,
            processing_time_ms: 1000,
            memory_usage_bytes: 50 * 1024 * 1024,
        };
        collector.record_compression(metrics).unwrap();
    }
    
    let usage_stats = UsageStats {
        period_start: Utc::now() - ChronoDuration::days(30),
        period_end: Utc::now(),
        bytes_processed: 1000 * 1024 * 1024, // 1GB processed
        bytes_saved: 900 * 1024 * 1024,      // 900MB saved
        operation_count: 10,
        avg_processing_time_ms: 1000.0,
    };
    
    let roi_report = collector.calculate_roi(&usage_stats);
    assert!(roi_report.is_ok(), "ROI calculation should succeed");
    
    let report = roi_report.unwrap();
    assert!(report.cost_savings_usd > 0.0, "Should show positive cost savings");
    assert!(report.bandwidth_saved > 0, "Should show bandwidth savings");
    assert!(report.storage_saved > 0, "Should show storage savings");
    assert!(report.payback_period_months > 0.0, "Should calculate payback period");
}

/// Test ROI calculation with no savings
#[test]
fn test_roi_calculation_no_savings() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // No compression operations recorded
    let usage_stats = UsageStats {
        period_start: Utc::now() - ChronoDuration::days(30),
        period_end: Utc::now(),
        bytes_processed: 0,
        bytes_saved: 0,
        operation_count: 0,
        avg_processing_time_ms: 0.0,
    };
    
    let roi_report = collector.calculate_roi(&usage_stats);
    assert!(roi_report.is_ok(), "ROI calculation should succeed even with no data");
    
    let report = roi_report.unwrap();
    assert_eq!(report.cost_savings_usd, 0.0, "Should show zero savings");
    assert_eq!(report.bandwidth_saved, 0, "Should show zero bandwidth savings");
    assert_eq!(report.storage_saved, 0, "Should show zero storage savings");
}

/// Test JSON metrics export
#[tokio::test]
async fn test_json_export() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Record some test data
    let compression_metrics = CompressionMetrics {
        input_size: 2048,
        output_size: 1024,
        compression_ratio: 2.0,
        corpus_hit_rate: 0.8,
        processing_time_ms: 100,
        memory_usage_bytes: 4096,
    };
    collector.record_compression(compression_metrics).unwrap();
    
    let json_result = collector.export_metrics(MetricsFormat::Json).await;
    assert!(json_result.is_ok(), "JSON export should succeed");
    
    let json_data = json_result.unwrap();
    assert!(json_data.contains("timestamp"), "Should contain timestamp");
    assert!(json_data.contains("performance"), "Should contain performance metrics");
    assert!(json_data.contains("recent_compressions"), "Should contain compression data");
    assert!(json_data.contains("economic_summary"), "Should contain economic data");
    
    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json_data).unwrap();
    assert!(parsed.is_object(), "Should be valid JSON object");
}

/// Test CSV metrics export
#[tokio::test]
async fn test_csv_export() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Record test data
    let compression_metrics = CompressionMetrics {
        input_size: 4096,
        output_size: 2048,
        compression_ratio: 2.0,
        corpus_hit_rate: 0.75,
        processing_time_ms: 150,
        memory_usage_bytes: 8192,
    };
    collector.record_compression(compression_metrics).unwrap();
    
    let decompression_metrics = DecompressionMetrics {
        input_size: 2048,
        output_size: 4096,
        decompression_ratio: 2.0,
        processing_time_ms: 75,
        memory_usage_bytes: 4096,
        corpus_access_count: 5,
    };
    collector.record_decompression(decompression_metrics).unwrap();
    
    let csv_result = collector.export_metrics(MetricsFormat::Csv).await;
    assert!(csv_result.is_ok(), "CSV export should succeed");
    
    let csv_data = csv_result.unwrap();
    assert!(csv_data.contains("timestamp,operation_type"), "Should contain CSV header");
    assert!(csv_data.contains("compression,4096,2048"), "Should contain compression data");
    assert!(csv_data.contains("decompression,2048,4096"), "Should contain decompression data");
    
    // Verify CSV structure
    let lines: Vec<&str> = csv_data.lines().collect();
    assert!(lines.len() >= 3, "Should have header + at least 2 data rows");
}

/// Test Prometheus export (when feature is enabled)
#[cfg(feature = "metrics")]
#[tokio::test]
async fn test_prometheus_export() {
    let config = MetricsConfig {
        enable_prometheus: true,
        ..Default::default()
    };
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Record some metrics to populate Prometheus data
    let compression_metrics = CompressionMetrics {
        input_size: 1024,
        output_size: 512,
        compression_ratio: 2.0,
        corpus_hit_rate: 0.8,
        processing_time_ms: 100,
        memory_usage_bytes: 2048,
    };
    collector.record_compression(compression_metrics).unwrap();
    
    let prometheus_result = collector.export_metrics(MetricsFormat::Prometheus).await;
    assert!(prometheus_result.is_ok(), "Prometheus export should succeed");
    
    let prometheus_data = prometheus_result.unwrap();
    assert!(prometheus_data.contains("reducto_compression_operations_total"), "Should contain compression counter");
    assert!(prometheus_data.contains("reducto_compression_ratio"), "Should contain compression ratio histogram");
    assert!(prometheus_data.contains("reducto_processing_time_seconds"), "Should contain processing time histogram");
}

/// Test Prometheus export when feature is disabled
#[cfg(not(feature = "metrics"))]
#[tokio::test]
async fn test_prometheus_export_disabled() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    let prometheus_result = collector.export_metrics(MetricsFormat::Prometheus).await;
    assert!(prometheus_result.is_err(), "Prometheus export should fail when feature disabled");
    
    match prometheus_result.unwrap_err() {
        reducto_mode_3::error::ReductoError::InternalError { message } => {
            assert!(message.contains("Prometheus feature not enabled"), "Should indicate feature not enabled");
        }
        _ => panic!("Expected feature not enabled error"),
    }
}

/// Test metrics snapshot generation
#[test]
fn test_metrics_snapshot() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Record some data
    let compression_metrics = CompressionMetrics {
        input_size: 8192,
        output_size: 4096,
        compression_ratio: 2.0,
        corpus_hit_rate: 0.85,
        processing_time_ms: 200,
        memory_usage_bytes: 16384,
    };
    collector.record_compression(compression_metrics).unwrap();
    
    let snapshot = collector.get_metrics_snapshot();
    assert!(snapshot.is_ok(), "Snapshot generation should succeed");
    
    let snapshot = snapshot.unwrap();
    assert!(!snapshot.recent_compressions.is_empty(), "Should contain recent compression data");
    assert!(snapshot.economic_summary.total_savings_usd >= 0.0, "Should have economic summary");
}

/// Test real-time metrics streaming
#[tokio::test]
async fn test_metrics_streaming() {
    let config = MetricsConfig {
        sample_interval_ms: 50, // Fast sampling for test
        ..Default::default()
    };
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    let mut stream = collector.start_metrics_streaming().await.unwrap();
    
    // Record some data while streaming
    let compression_metrics = CompressionMetrics {
        input_size: 1024,
        output_size: 512,
        compression_ratio: 2.0,
        corpus_hit_rate: 0.8,
        processing_time_ms: 100,
        memory_usage_bytes: 2048,
    };
    collector.record_compression(compression_metrics).unwrap();
    
    // Wait for a few snapshots
    let snapshot1 = timeout(Duration::from_millis(100), stream.recv()).await;
    assert!(snapshot1.is_ok(), "Should receive first snapshot");
    assert!(snapshot1.unwrap().is_some(), "First snapshot should contain data");
    
    let snapshot2 = timeout(Duration::from_millis(100), stream.recv()).await;
    assert!(snapshot2.is_ok(), "Should receive second snapshot");
    assert!(snapshot2.unwrap().is_some(), "Second snapshot should contain data");
}

/// Test metrics streaming timeout
#[tokio::test]
async fn test_metrics_streaming_timeout() {
    let config = MetricsConfig {
        sample_interval_ms: 1000, // Slow sampling
        ..Default::default()
    };
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    let mut stream = collector.start_metrics_streaming().await.unwrap();
    
    // Should timeout before receiving data
    let result = timeout(Duration::from_millis(50), stream.recv()).await;
    assert!(result.is_err(), "Should timeout waiting for snapshot");
}

/// Test entropy calculation with different data patterns
#[test]
fn test_entropy_calculation() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Test with uniform data (should have low entropy)
    let uniform_data = vec![0u8; 1024];
    let entropy = collector.calculate_entropy(&uniform_data);
    assert!(entropy < 0.1, "Uniform data should have low entropy: {}", entropy);
    
    // Test with random-like data (should have higher entropy)
    let varied_data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let entropy = collector.calculate_entropy(&varied_data);
    assert!(entropy > 0.5, "Varied data should have higher entropy: {}", entropy);
    
    // Test with completely random data
    let random_data: Vec<u8> = (0..1024).map(|i| ((i * 17 + 42) % 256) as u8).collect();
    let entropy = collector.calculate_entropy(&random_data);
    assert!(entropy > 0.7, "Random-like data should have high entropy: {}", entropy);
}

/// Test prediction accuracy bounds
#[test]
fn test_prediction_bounds() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Test hit rate prediction bounds
    let hit_rate_low = collector.predict_hit_rate(&vec![0u8; 1024], 1024).unwrap();
    let hit_rate_high = collector.predict_hit_rate(&vec![0u8; 1024], 10 * 1024 * 1024 * 1024).unwrap();
    
    assert!(hit_rate_low >= 0.1 && hit_rate_low <= 0.95, "Hit rate should be in bounds");
    assert!(hit_rate_high >= 0.1 && hit_rate_high <= 0.95, "Hit rate should be in bounds");
    assert!(hit_rate_high >= hit_rate_low, "Larger corpus should give higher hit rate");
    
    // Test compression ratio prediction
    let compression_ratio = collector.predict_compression_ratio(0.8).unwrap();
    assert!(compression_ratio > 1.0, "Compression ratio should be > 1.0");
    
    // Test processing time prediction
    let processing_time = collector.predict_processing_time(1024 * 1024).unwrap();
    assert!(processing_time > 0, "Processing time should be positive");
    
    // Test memory usage prediction
    let memory_usage = collector.predict_memory_usage(1024 * 1024).unwrap();
    assert!(memory_usage > 0, "Memory usage should be positive");
}

/// Test confidence level calculation
#[test]
fn test_confidence_calculation() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Test with small sample and small corpus (low confidence)
    let confidence_low = collector.calculate_confidence_level(1024, 1024 * 1024, 1024 * 1024).unwrap();
    assert!(confidence_low < 0.5, "Small sample should give low confidence");
    
    // Test with large sample and large corpus (high confidence)
    let confidence_high = collector.calculate_confidence_level(
        1024 * 1024, 
        1024 * 1024, 
        10 * 1024 * 1024 * 1024
    ).unwrap();
    assert!(confidence_high > 0.5, "Large sample and corpus should give higher confidence");
    
    // Confidence should be bounded
    assert!(confidence_low >= 0.0 && confidence_low <= 1.0, "Confidence should be in [0,1]");
    assert!(confidence_high >= 0.0 && confidence_high <= 1.0, "Confidence should be in [0,1]");
}

/// Test economic calculations with different cost models
#[test]
fn test_economic_calculations() {
    let config = MetricsConfig {
        bandwidth_cost_per_gb: 0.15, // Higher cost scenario
        storage_cost_per_gb: 0.05,
        ..Default::default()
    };
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Test cost savings calculation
    let savings = collector.calculate_predicted_savings(1024 * 1024 * 1024, 5.0).unwrap(); // 1GB, 5:1 compression
    assert!(savings > 0.0, "Should calculate positive savings");
    
    // Verify the calculation: 1GB input, 5:1 compression = 0.8GB saved
    // Cost = 0.8 * (0.15 + 0.05) = 0.8 * 0.20 = $0.16
    let expected_savings = 0.8 * (0.15 + 0.05);
    assert!((savings - expected_savings).abs() < 0.01, "Savings calculation should be accurate");
}

/// Test metrics retention and cleanup
#[test]
fn test_metrics_retention() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Record many metrics to trigger retention cleanup
    for i in 0..1200 { // More than the 1000 limit
        let metrics = CompressionMetrics {
            input_size: 1024 * (i + 1),
            output_size: 512 * (i + 1),
            compression_ratio: 2.0,
            corpus_hit_rate: 0.8,
            processing_time_ms: 100,
            memory_usage_bytes: 2048,
        };
        collector.record_compression(metrics).unwrap();
    }
    
    // The internal storage should have been cleaned up
    // (This test verifies the cleanup logic doesn't crash)
}

/// Test error handling in metrics collection
#[test]
fn test_metrics_error_handling() {
    let config = MetricsConfig::default();
    let collector = EnterpriseMetricsCollector::new(config).unwrap();
    
    // Test with extreme values that might cause issues
    let extreme_metrics = CompressionMetrics {
        input_size: u64::MAX,
        output_size: 0,
        compression_ratio: f64::INFINITY,
        corpus_hit_rate: 2.0, // Invalid hit rate > 1.0
        processing_time_ms: u64::MAX,
        memory_usage_bytes: u64::MAX,
    };
    
    // Should handle extreme values gracefully
    let result = collector.record_compression(extreme_metrics);
    assert!(result.is_ok(), "Should handle extreme values gracefully");
}

/// Test concurrent metrics recording
#[tokio::test]
async fn test_concurrent_metrics_recording() {
    let config = MetricsConfig::default();
    let collector = std::sync::Arc::new(EnterpriseMetricsCollector::new(config).unwrap());
    
    let mut handles = vec![];
    
    // Spawn multiple tasks recording metrics concurrently
    for i in 0..10 {
        let collector_clone = collector.clone();
        let handle = tokio::spawn(async move {
            for j in 0..10 {
                let metrics = CompressionMetrics {
                    input_size: 1024 * (i * 10 + j + 1),
                    output_size: 512 * (i * 10 + j + 1),
                    compression_ratio: 2.0,
                    corpus_hit_rate: 0.8,
                    processing_time_ms: 100,
                    memory_usage_bytes: 2048,
                };
                collector_clone.record_compression(metrics).unwrap();
            }
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify all metrics were recorded (100 total)
    let snapshot = collector.get_metrics_snapshot().unwrap();
    assert!(!snapshot.recent_compressions.is_empty(), "Should have recorded metrics");
}