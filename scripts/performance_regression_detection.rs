#!/usr/bin/env rust-script
//! Performance Regression Detection for CI Pipeline
//!
//! This script runs performance benchmarks and detects regressions by comparing
//! results against baseline measurements. It's designed to be integrated into
//! CI pipelines to catch performance regressions early.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Performance benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub throughput_mbps: f64,
    pub latency_ns: f64,
    pub memory_usage_mb: f64,
    pub cpu_utilization: f64,
    pub timestamp: String,
    pub git_commit: String,
    pub build_config: String,
}

/// Performance regression threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionThresholds {
    pub throughput_degradation_percent: f64,
    pub latency_increase_percent: f64,
    pub memory_increase_percent: f64,
    pub cpu_increase_percent: f64,
}

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            throughput_degradation_percent: 10.0, // 10% throughput loss is significant
            latency_increase_percent: 15.0,       // 15% latency increase is concerning
            memory_increase_percent: 20.0,        // 20% memory increase is notable
            cpu_increase_percent: 25.0,           // 25% CPU increase may be acceptable
        }
    }
}

/// Performance regression detection result
#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    pub benchmark_name: String,
    pub has_regression: bool,
    pub throughput_change_percent: f64,
    pub latency_change_percent: f64,
    pub memory_change_percent: f64,
    pub cpu_change_percent: f64,
    pub severity: RegressionSeverity,
    pub details: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegressionSeverity {
    None,
    Minor,
    Moderate,
    Severe,
    Critical,
}

/// Performance baseline database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub benchmarks: HashMap<String, BenchmarkResult>,
    pub last_updated: String,
    pub baseline_commit: String,
}

/// Main performance regression detector
pub struct RegressionDetector {
    baseline_path: PathBuf,
    thresholds: RegressionThresholds,
    workspace_root: PathBuf,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        let workspace_root = workspace_root.as_ref().to_path_buf();
        let baseline_path = workspace_root.join("performance_baseline.json");
        
        Self {
            baseline_path,
            thresholds: RegressionThresholds::default(),
            workspace_root,
        }
    }

    /// Run performance benchmarks and detect regressions
    pub fn run_regression_detection(&self) -> Result<Vec<RegressionAnalysis>, Box<dyn std::error::Error>> {
        println!("🚀 Starting performance regression detection...");
        
        // Load or create baseline
        let baseline = self.load_or_create_baseline()?;
        
        // Run current benchmarks
        let current_results = self.run_benchmarks()?;
        
        // Analyze for regressions
        let mut analyses = Vec::new();
        for (name, current) in &current_results {
            if let Some(baseline_result) = baseline.benchmarks.get(name) {
                let analysis = self.analyze_regression(baseline_result, current);
                analyses.push(analysis);
            } else {
                println!("⚠️  No baseline found for benchmark: {}", name);
            }
        }
        
        // Update baseline if no regressions or if explicitly requested
        if !analyses.iter().any(|a| a.has_regression) || std::env::var("UPDATE_BASELINE").is_ok() {
            self.update_baseline(current_results)?;
        }
        
        // Report results
        self.report_results(&analyses);
        
        Ok(analyses)
    }

    /// Load existing baseline or create a new one
    fn load_or_create_baseline(&self) -> Result<PerformanceBaseline, Box<dyn std::error::Error>> {
        if self.baseline_path.exists() {
            let content = fs::read_to_string(&self.baseline_path)?;
            let baseline: PerformanceBaseline = serde_json::from_str(&content)?;
            println!("📊 Loaded baseline from {}", self.baseline_path.display());
            Ok(baseline)
        } else {
            println!("📊 No baseline found, creating new baseline");
            let baseline = PerformanceBaseline {
                benchmarks: HashMap::new(),
                last_updated: chrono::Utc::now().to_rfc3339(),
                baseline_commit: self.get_git_commit()?,
            };
            Ok(baseline)
        }
    }

    /// Run performance benchmarks
    fn run_benchmarks(&self) -> Result<HashMap<String, BenchmarkResult>, Box<dyn std::error::Error>> {
        println!("🔬 Running performance benchmarks...");
        
        let mut results = HashMap::new();
        
        // Run Criterion benchmarks
        let criterion_results = self.run_criterion_benchmarks()?;
        results.extend(criterion_results);
        
        // Run custom performance tests
        let custom_results = self.run_custom_benchmarks()?;
        results.extend(custom_results);
        
        println!("✅ Completed {} benchmarks", results.len());
        Ok(results)
    }

    /// Run Criterion benchmarks and parse results
    fn run_criterion_benchmarks(&self) -> Result<HashMap<String, BenchmarkResult>, Box<dyn std::error::Error>> {
        let mut results = HashMap::new();
        
        // Run criterion benchmarks with JSON output
        let output = Command::new("cargo")
            .args(&["bench", "--bench", "criterion_benchmarks", "--", "--output-format", "json"])
            .current_dir(&self.workspace_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Criterion benchmarks failed: {}", stderr).into());
        }
        
        // Parse criterion output (simplified - in practice you'd parse the actual JSON format)
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if let Some(result) = self.parse_criterion_line(line)? {
                results.insert(result.name.clone(), result);
            }
        }
        
        Ok(results)
    }

    /// Parse a single line of Criterion output
    fn parse_criterion_line(&self, line: &str) -> Result<Option<BenchmarkResult>, Box<dyn std::error::Error>> {
        // This is a simplified parser - in practice you'd parse the actual Criterion JSON format
        if line.contains("time:") && line.contains("throughput:") {
            // Extract benchmark name and metrics
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 6 {
                let name = parts[0].to_string();
                let throughput_str = parts.iter().find(|s| s.contains("MB/s")).unwrap_or(&"0MB/s");
                let throughput = throughput_str.replace("MB/s", "").parse::<f64>().unwrap_or(0.0);
                
                let result = BenchmarkResult {
                    name,
                    throughput_mbps: throughput,
                    latency_ns: 0.0, // Would be extracted from actual output
                    memory_usage_mb: 0.0, // Would be measured separately
                    cpu_utilization: 0.0, // Would be measured separately
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    git_commit: self.get_git_commit()?,
                    build_config: "release".to_string(),
                };
                
                return Ok(Some(result));
            }
        }
        Ok(None)
    }

    /// Run custom performance benchmarks
    fn run_custom_benchmarks(&self) -> Result<HashMap<String, BenchmarkResult>, Box<dyn std::error::Error>> {
        let mut results = HashMap::new();
        
        // Rolling hash performance test
        let rolling_hash_result = self.benchmark_rolling_hash()?;
        results.insert("rolling_hash_performance".to_string(), rolling_hash_result);
        
        // CDC chunking performance test
        let cdc_result = self.benchmark_cdc_chunking()?;
        results.insert("cdc_chunking_performance".to_string(), cdc_result);
        
        // Compression pipeline performance test
        let compression_result = self.benchmark_compression_pipeline()?;
        results.insert("compression_pipeline_performance".to_string(), compression_result);
        
        // Memory usage test
        let memory_result = self.benchmark_memory_usage()?;
        results.insert("memory_usage_efficiency".to_string(), memory_result);
        
        Ok(results)
    }

    /// Benchmark rolling hash performance
    fn benchmark_rolling_hash(&self) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let data_size = 1024 * 1024; // 1MB
        let data = vec![0xAA; data_size];
        
        // Simulate rolling hash operations
        let window_size = 4096;
        let mut hash = 0u64;
        let base = 67u64;
        
        for i in 0..data.len() {
            hash = hash.wrapping_mul(base).wrapping_add(data[i] as u64);
            if i >= window_size {
                // Simulate rolling
                let exiting_byte = data[i - window_size] as u64;
                hash = hash.wrapping_sub(exiting_byte.wrapping_mul(base.pow(window_size as u32)));
            }
        }
        
        let elapsed = start.elapsed();
        let throughput = (data_size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
        
        Ok(BenchmarkResult {
            name: "rolling_hash_performance".to_string(),
            throughput_mbps: throughput,
            latency_ns: elapsed.as_nanos() as f64 / data.len() as f64,
            memory_usage_mb: (data_size as f64) / (1024.0 * 1024.0),
            cpu_utilization: 95.0, // Estimated
            timestamp: chrono::Utc::now().to_rfc3339(),
            git_commit: self.get_git_commit()?,
            build_config: "release".to_string(),
        })
    }

    /// Benchmark CDC chunking performance
    fn benchmark_cdc_chunking(&self) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let data_size = 4 * 1024 * 1024; // 4MB
        let data = vec![0xBB; data_size];
        
        // Simulate CDC chunking
        let mut chunks = 0;
        let mut position = 0;
        let hash_mask = 0x1FFF;
        let mut gear_hash = 0u64;
        
        for &byte in &data {
            gear_hash = (gear_hash << 1).wrapping_add(byte as u64);
            position += 1;
            
            if position >= 4096 && (gear_hash & hash_mask) == 0 {
                chunks += 1;
                position = 0;
            }
        }
        
        let elapsed = start.elapsed();
        let throughput = (data_size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
        
        Ok(BenchmarkResult {
            name: "cdc_chunking_performance".to_string(),
            throughput_mbps: throughput,
            latency_ns: elapsed.as_nanos() as f64 / chunks as f64,
            memory_usage_mb: (data_size as f64) / (1024.0 * 1024.0),
            cpu_utilization: 90.0, // Estimated
            timestamp: chrono::Utc::now().to_rfc3339(),
            git_commit: self.get_git_commit()?,
            build_config: "release".to_string(),
        })
    }

    /// Benchmark compression pipeline performance
    fn benchmark_compression_pipeline(&self) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let data_size = 2 * 1024 * 1024; // 2MB
        let data = vec![0xCC; data_size];
        
        // Simulate compression pipeline
        let compressed = zstd::encode_all(&data[..], 6)?;
        let _decompressed = zstd::decode_all(&compressed[..])?;
        
        let elapsed = start.elapsed();
        let throughput = (data_size as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
        let compression_ratio = data_size as f64 / compressed.len() as f64;
        
        Ok(BenchmarkResult {
            name: "compression_pipeline_performance".to_string(),
            throughput_mbps: throughput,
            latency_ns: elapsed.as_nanos() as f64,
            memory_usage_mb: ((data_size + compressed.len()) as f64) / (1024.0 * 1024.0),
            cpu_utilization: 85.0, // Estimated
            timestamp: chrono::Utc::now().to_rfc3339(),
            git_commit: self.get_git_commit()?,
            build_config: "release".to_string(),
        })
    }

    /// Benchmark memory usage efficiency
    fn benchmark_memory_usage(&self) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let start = Instant::now();
        
        // Simulate memory-intensive operations
        let mut allocations = Vec::new();
        for i in 0..1000 {
            let size = 1024 + (i % 1024);
            allocations.push(vec![i as u8; size]);
        }
        
        // Simulate processing
        let mut total = 0u64;
        for allocation in &allocations {
            for &byte in allocation {
                total = total.wrapping_add(byte as u64);
            }
        }
        
        let elapsed = start.elapsed();
        let total_memory = allocations.iter().map(|v| v.len()).sum::<usize>();
        
        Ok(BenchmarkResult {
            name: "memory_usage_efficiency".to_string(),
            throughput_mbps: (total_memory as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64(),
            latency_ns: elapsed.as_nanos() as f64 / allocations.len() as f64,
            memory_usage_mb: (total_memory as f64) / (1024.0 * 1024.0),
            cpu_utilization: 70.0, // Estimated
            timestamp: chrono::Utc::now().to_rfc3339(),
            git_commit: self.get_git_commit()?,
            build_config: "release".to_string(),
        })
    }

    /// Analyze a benchmark for regressions
    fn analyze_regression(&self, baseline: &BenchmarkResult, current: &BenchmarkResult) -> RegressionAnalysis {
        let throughput_change = ((current.throughput_mbps - baseline.throughput_mbps) / baseline.throughput_mbps) * 100.0;
        let latency_change = ((current.latency_ns - baseline.latency_ns) / baseline.latency_ns) * 100.0;
        let memory_change = ((current.memory_usage_mb - baseline.memory_usage_mb) / baseline.memory_usage_mb) * 100.0;
        let cpu_change = ((current.cpu_utilization - baseline.cpu_utilization) / baseline.cpu_utilization) * 100.0;
        
        let has_throughput_regression = throughput_change < -self.thresholds.throughput_degradation_percent;
        let has_latency_regression = latency_change > self.thresholds.latency_increase_percent;
        let has_memory_regression = memory_change > self.thresholds.memory_increase_percent;
        let has_cpu_regression = cpu_change > self.thresholds.cpu_increase_percent;
        
        let has_regression = has_throughput_regression || has_latency_regression || 
                           has_memory_regression || has_cpu_regression;
        
        let severity = if !has_regression {
            RegressionSeverity::None
        } else if throughput_change < -25.0 || latency_change > 50.0 {
            RegressionSeverity::Critical
        } else if throughput_change < -20.0 || latency_change > 30.0 {
            RegressionSeverity::Severe
        } else if throughput_change < -15.0 || latency_change > 20.0 {
            RegressionSeverity::Moderate
        } else {
            RegressionSeverity::Minor
        };
        
        let mut details = Vec::new();
        if has_throughput_regression {
            details.push(format!("Throughput decreased by {:.1}%", -throughput_change));
        }
        if has_latency_regression {
            details.push(format!("Latency increased by {:.1}%", latency_change));
        }
        if has_memory_regression {
            details.push(format!("Memory usage increased by {:.1}%", memory_change));
        }
        if has_cpu_regression {
            details.push(format!("CPU usage increased by {:.1}%", cpu_change));
        }
        
        RegressionAnalysis {
            benchmark_name: current.name.clone(),
            has_regression,
            throughput_change_percent: throughput_change,
            latency_change_percent: latency_change,
            memory_change_percent: memory_change,
            cpu_change_percent: cpu_change,
            severity,
            details: details.join("; "),
        }
    }

    /// Update the performance baseline
    fn update_baseline(&self, results: HashMap<String, BenchmarkResult>) -> Result<(), Box<dyn std::error::Error>> {
        let baseline = PerformanceBaseline {
            benchmarks: results,
            last_updated: chrono::Utc::now().to_rfc3339(),
            baseline_commit: self.get_git_commit()?,
        };
        
        let content = serde_json::to_string_pretty(&baseline)?;
        fs::write(&self.baseline_path, content)?;
        
        println!("📊 Updated performance baseline at {}", self.baseline_path.display());
        Ok(())
    }

    /// Report regression analysis results
    fn report_results(&self, analyses: &[RegressionAnalysis]) {
        println!("\n📈 Performance Regression Analysis Results");
        println!("==========================================");
        
        let mut has_any_regression = false;
        let mut critical_count = 0;
        let mut severe_count = 0;
        let mut moderate_count = 0;
        let mut minor_count = 0;
        
        for analysis in analyses {
            match analysis.severity {
                RegressionSeverity::None => {
                    println!("✅ {}: No regression detected", analysis.benchmark_name);
                    if analysis.throughput_change_percent > 0.0 {
                        println!("   📈 Throughput improved by {:.1}%", analysis.throughput_change_percent);
                    }
                }
                RegressionSeverity::Minor => {
                    println!("⚠️  {}: Minor regression", analysis.benchmark_name);
                    println!("   📉 {}", analysis.details);
                    minor_count += 1;
                    has_any_regression = true;
                }
                RegressionSeverity::Moderate => {
                    println!("🔶 {}: Moderate regression", analysis.benchmark_name);
                    println!("   📉 {}", analysis.details);
                    moderate_count += 1;
                    has_any_regression = true;
                }
                RegressionSeverity::Severe => {
                    println!("🔴 {}: Severe regression", analysis.benchmark_name);
                    println!("   📉 {}", analysis.details);
                    severe_count += 1;
                    has_any_regression = true;
                }
                RegressionSeverity::Critical => {
                    println!("🚨 {}: CRITICAL regression", analysis.benchmark_name);
                    println!("   📉 {}", analysis.details);
                    critical_count += 1;
                    has_any_regression = true;
                }
            }
        }
        
        println!("\n📊 Summary:");
        println!("  Total benchmarks: {}", analyses.len());
        println!("  Critical regressions: {}", critical_count);
        println!("  Severe regressions: {}", severe_count);
        println!("  Moderate regressions: {}", moderate_count);
        println!("  Minor regressions: {}", minor_count);
        
        if has_any_regression {
            println!("\n🚨 Performance regressions detected!");
            if critical_count > 0 || severe_count > 0 {
                println!("   Consider blocking this change until performance is restored.");
            }
        } else {
            println!("\n✅ No performance regressions detected!");
        }
    }

    /// Get current git commit hash
    fn get_git_commit(&self) -> Result<String, Box<dyn std::error::Error>> {
        let output = Command::new("git")
            .args(&["rev-parse", "HEAD"])
            .current_dir(&self.workspace_root)
            .output()?;
        
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Ok("unknown".to_string())
        }
    }
}

/// Main entry point for the regression detection script
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = std::env::current_dir()?;
    let detector = RegressionDetector::new(workspace_root);
    
    let analyses = detector.run_regression_detection()?;
    
    // Exit with error code if critical or severe regressions are found
    let has_critical_regression = analyses.iter().any(|a| {
        matches!(a.severity, RegressionSeverity::Critical | RegressionSeverity::Severe)
    });
    
    if has_critical_regression {
        std::process::exit(1);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_regression_analysis() {
        let detector = RegressionDetector::new(".");
        
        let baseline = BenchmarkResult {
            name: "test_benchmark".to_string(),
            throughput_mbps: 100.0,
            latency_ns: 1000.0,
            memory_usage_mb: 10.0,
            cpu_utilization: 50.0,
            timestamp: "2023-01-01T00:00:00Z".to_string(),
            git_commit: "abc123".to_string(),
            build_config: "release".to_string(),
        };
        
        // Test no regression
        let current_good = BenchmarkResult {
            throughput_mbps: 105.0, // 5% improvement
            latency_ns: 950.0,      // 5% improvement
            ..baseline.clone()
        };
        
        let analysis = detector.analyze_regression(&baseline, &current_good);
        assert!(!analysis.has_regression);
        assert_eq!(analysis.severity, RegressionSeverity::None);
        
        // Test minor regression
        let current_minor = BenchmarkResult {
            throughput_mbps: 95.0,  // 5% degradation (below 10% threshold)
            latency_ns: 1100.0,     // 10% increase (below 15% threshold)
            ..baseline.clone()
        };
        
        let analysis = detector.analyze_regression(&baseline, &current_minor);
        assert!(!analysis.has_regression); // Should not trigger with these thresholds
        
        // Test severe regression
        let current_severe = BenchmarkResult {
            throughput_mbps: 80.0,  // 20% degradation
            latency_ns: 1300.0,     // 30% increase
            ..baseline.clone()
        };
        
        let analysis = detector.analyze_regression(&baseline, &current_severe);
        assert!(analysis.has_regression);
        assert_eq!(analysis.severity, RegressionSeverity::Severe);
    }

    #[test]
    fn test_baseline_creation() {
        let temp_dir = TempDir::new().unwrap();
        let detector = RegressionDetector::new(temp_dir.path());
        
        // Should create new baseline when none exists
        let baseline = detector.load_or_create_baseline().unwrap();
        assert!(baseline.benchmarks.is_empty());
        assert!(!baseline.baseline_commit.is_empty());
    }

    #[test]
    fn test_threshold_configuration() {
        let thresholds = RegressionThresholds::default();
        assert_eq!(thresholds.throughput_degradation_percent, 10.0);
        assert_eq!(thresholds.latency_increase_percent, 15.0);
        assert_eq!(thresholds.memory_increase_percent, 20.0);
        assert_eq!(thresholds.cpu_increase_percent, 25.0);
    }
}