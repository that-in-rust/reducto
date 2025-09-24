// BenchmarkError extended with Timeout
use thiserror::Error;

use std::time::Duration;

/// Comprehensive error hierarchy for the benchmark system
#[derive(Error, Debug)]
pub enum BenchmarkError {
    #[error("Data collection failed from {data_source}: {details}")]
    DataCollection { data_source: String, details: String },
    
    #[error("Data loading failed: {0}")]
    DataLoading(String),
    
    #[error("API request failed: {url} - {status}: {message}")]
    ApiRequest { url: String, status: u16, message: String },
    
    #[error("Rate limit exceeded for {service}: retry after {retry_after_seconds}s")]
    RateLimit { service: String, retry_after_seconds: u64 },
    
    #[error("Corpus building failed: {reason}")]
    CorpusBuilding { reason: String },
    
    #[error("Corpus quality insufficient: {score:.2} < {threshold:.2}")]
    CorpusQuality { score: f64, threshold: f64 },
    
    #[error("Compression format {format} failed: {error}")]
    CompressionFailed { format: String, error: String },
    
    #[error("Decompression failed for {format}: data integrity check failed")]
    DecompressionFailed { format: String },
    
    #[error("Performance analysis failed: {details}")]
    AnalysisFailed { details: String },
    
    #[error("Report generation failed at {stage}: {error}")]
    ReportGeneration { stage: String, error: String },
    
    #[error("System resource exhausted: {resource} - {details}")]
    ResourceExhausted { resource: String, details: String },
    
    #[error("Configuration error: {parameter} - {issue}")]
    Configuration { parameter: String, issue: String },
    
    #[error("I/O error: {operation}")]
    Io { operation: String, #[source] source: std::io::Error },
    
    #[error("Serialization error: {context}")]
    Serialization { context: String, #[source] source: serde_json::Error },
    
    #[error("Timeout exceeded: {operation} took longer than {timeout_seconds}s")]
    Timeout { operation: String, timeout_seconds: u64 },
    
    #[error("Invalid data format: {expected} expected, got {actual}")]
    InvalidFormat { expected: String, actual: String },
    
    #[error("Missing dependency: {tool} is required but not found")]
    MissingDependency { tool: String },
}

// Note: BenchmarkResult struct is defined in lib.rs for the main benchmark result

/// Error recovery strategies and context
impl BenchmarkError {
    /// Determine if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            BenchmarkError::RateLimit { .. } => true,
            BenchmarkError::ApiRequest { status, .. } if *status >= 500 => true,
            BenchmarkError::Timeout { .. } => true,
            BenchmarkError::ResourceExhausted { .. } => false,
            BenchmarkError::MissingDependency { .. } => false,
            BenchmarkError::Configuration { .. } => false,
            _ => false,
        }
    }
    
    /// Get suggested retry delay for recoverable errors
    pub fn retry_delay(&self) -> Option<Duration> {
        match self {
            BenchmarkError::RateLimit { retry_after_seconds, .. } => {
                Some(Duration::from_secs(*retry_after_seconds))
            }
            BenchmarkError::ApiRequest { status, .. } if *status >= 500 => {
                Some(Duration::from_secs(5))
            }
            BenchmarkError::Timeout { .. } => Some(Duration::from_secs(2)),
            _ => None,
        }
    }
    
    /// Get user-friendly error message with suggested actions
    pub fn user_message(&self) -> String {
        match self {
            BenchmarkError::DataCollection { data_source, .. } => {
                format!("Failed to collect data from {}. Check network connection and API tokens.", data_source)
            }
            BenchmarkError::RateLimit { service, retry_after_seconds } => {
                format!("Rate limited by {}. Will retry in {}s. Consider using API tokens for higher limits.", service, retry_after_seconds)
            }
            BenchmarkError::MissingDependency { tool } => {
                format!("Missing required tool: {}. Install it or run with --skip-{}", tool, tool.to_lowercase())
            }
            BenchmarkError::CorpusQuality { score, threshold } => {
                format!("Corpus quality too low ({:.2} < {:.2}). Try different data sources or increase corpus size.", score, threshold)
            }
            _ => self.to_string(),
        }
    }
}

/// Convert from common error types
impl From<std::io::Error> for BenchmarkError {
    fn from(err: std::io::Error) -> Self {
        BenchmarkError::Io {
            operation: "file operation".to_string(),
            source: err,
        }
    }
}

impl From<serde_json::Error> for BenchmarkError {
    fn from(err: serde_json::Error) -> Self {
        BenchmarkError::Serialization {
            context: "JSON processing".to_string(),
            source: err,
        }
    }
}

impl From<reqwest::Error> for BenchmarkError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            BenchmarkError::Timeout {
                operation: "HTTP request".to_string(),
                timeout_seconds: 30,
            }
        } else if let Some(status) = err.status() {
            BenchmarkError::ApiRequest {
                url: err.url().map(|u| u.to_string()).unwrap_or_default(),
                status: status.as_u16(),
                message: err.to_string(),
            }
        } else {
            BenchmarkError::DataCollection {
                data_source: "HTTP client".to_string(),
                details: err.to_string(),
            }
        }
    }
}