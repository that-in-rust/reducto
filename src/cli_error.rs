//! CLI Error Handling
//!
//! Comprehensive error handling for the Reducto CLI with actionable guidance
//! and structured error reporting for enterprise environments.

use std::path::PathBuf;
use thiserror::Error;
use console::style;

/// CLI-specific errors with actionable remediation guidance
#[derive(Error, Debug)]
pub enum CliError {
    #[error("Configuration error: {message}")]
    Configuration {
        message: String,
        remediation: String,
        config_path: Option<PathBuf>,
    },
    
    #[error("Corpus not found: {corpus_id}")]
    CorpusNotFound {
        corpus_id: String,
        searched_paths: Vec<PathBuf>,
        remediation: String,
    },
    
    #[error("Invalid input file: {path}")]
    InvalidInputFile {
        path: PathBuf,
        reason: String,
        remediation: String,
    },
    
    #[error("Compression failed: {reason}")]
    CompressionFailed {
        reason: String,
        input_path: PathBuf,
        corpus_id: String,
        remediation: String,
    },
    
    #[error("Decompression failed: {reason}")]
    DecompressionFailed {
        reason: String,
        input_path: PathBuf,
        remediation: String,
    },
    
    #[error("Corpus operation failed: {operation} - {reason}")]
    CorpusOperationFailed {
        operation: String,
        reason: String,
        corpus_path: Option<PathBuf>,
        remediation: String,
    },
    
    #[error("Stream processing error: {reason}")]
    StreamProcessingError {
        reason: String,
        operation: String,
        remediation: String,
    },
    
    #[error("Analysis failed: {reason}")]
    AnalysisFailed {
        reason: String,
        input_paths: Vec<PathBuf>,
        remediation: String,
    },
    
    #[error("Network error: {reason}")]
    NetworkError {
        reason: String,
        url: Option<String>,
        remediation: String,
    },
    
    #[error("Permission denied: {path}")]
    PermissionDenied {
        path: PathBuf,
        operation: String,
        remediation: String,
    },
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted {
        resource: String,
        current_usage: String,
        limit: String,
        remediation: String,
    },
    
    #[error("Validation error: {field} - {reason}")]
    ValidationError {
        field: String,
        reason: String,
        provided_value: String,
        valid_range: String,
        remediation: String,
    },
    
    #[error("Operation cancelled by user")]
    OperationCancelled {
        operation: String,
        partial_results: Option<String>,
    },
    
    #[error("Feature not available: {feature}")]
    FeatureNotAvailable {
        feature: String,
        required_build_flags: Vec<String>,
        remediation: String,
    },
    
    #[error("Internal error: {message}")]
    InternalError {
        message: String,
        context: String,
        remediation: String,
    },
}

impl CliError {
    /// Create a configuration error with remediation guidance
    pub fn configuration_error(message: &str, config_path: Option<PathBuf>) -> Self {
        let remediation = match config_path.as_ref() {
            Some(path) => format!(
                "Check configuration file at '{}' or run 'reducto config validate --fix' to auto-fix issues",
                path.display()
            ),
            None => "Run 'reducto config init --interactive' to create a configuration file".to_string(),
        };
        
        Self::Configuration {
            message: message.to_string(),
            remediation,
            config_path,
        }
    }
    
    /// Create a corpus not found error with search guidance
    pub fn corpus_not_found(corpus_id: &str, searched_paths: Vec<PathBuf>) -> Self {
        let remediation = if searched_paths.is_empty() {
            format!("Run 'reducto corpus fetch {}' to download from repository or 'reducto corpus build' to create locally", corpus_id)
        } else {
            format!(
                "Corpus '{}' not found in searched paths. Run 'reducto corpus fetch {}' or check paths: {}",
                corpus_id,
                corpus_id,
                searched_paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
            )
        };
        
        Self::CorpusNotFound {
            corpus_id: corpus_id.to_string(),
            searched_paths,
            remediation,
        }
    }
    
    /// Create an invalid input file error
    pub fn invalid_input_file(path: PathBuf, reason: &str) -> Self {
        let remediation = match reason {
            r if r.contains("not found") => "Check that the file path is correct and the file exists".to_string(),
            r if r.contains("permission") => "Check file permissions or run with appropriate privileges".to_string(),
            r if r.contains("empty") => "Provide a non-empty input file or use a different source".to_string(),
            r if r.contains("format") => "Ensure the input file is in the expected format".to_string(),
            _ => "Check the input file and try again".to_string(),
        };
        
        Self::InvalidInputFile {
            path,
            reason: reason.to_string(),
            remediation,
        }
    }
    
    /// Create a compression failed error with diagnostic guidance
    pub fn compression_failed(reason: &str, input_path: PathBuf, corpus_id: &str) -> Self {
        let remediation = match reason {
            r if r.contains("corpus") => format!("Verify corpus '{}' exists and is valid with 'reducto corpus verify {}'", corpus_id, corpus_id),
            r if r.contains("memory") => "Reduce chunk size or increase available memory".to_string(),
            r if r.contains("disk") => "Free up disk space or specify a different output location".to_string(),
            r if r.contains("format") => "Check input file format and ensure it's supported".to_string(),
            _ => "Run with --verbose for detailed error information".to_string(),
        };
        
        Self::CompressionFailed {
            reason: reason.to_string(),
            input_path,
            corpus_id: corpus_id.to_string(),
            remediation,
        }
    }
    
    /// Create a validation error for CLI arguments
    pub fn validation_error(field: &str, reason: &str, provided_value: &str, valid_range: &str) -> Self {
        let remediation = format!(
            "Provide a valid value for '{}' in range {}. Use --help for more information",
            field, valid_range
        );
        
        Self::ValidationError {
            field: field.to_string(),
            reason: reason.to_string(),
            provided_value: provided_value.to_string(),
            valid_range: valid_range.to_string(),
            remediation,
        }
    }
    
    /// Create a feature not available error
    pub fn feature_not_available(feature: &str, required_flags: Vec<String>) -> Self {
        let remediation = if required_flags.is_empty() {
            "This feature is not available in the current build".to_string()
        } else {
            format!(
                "Rebuild with features: {}. Example: cargo build --features \"{}\"",
                required_flags.join(", "),
                required_flags.join(" ")
            )
        };
        
        Self::FeatureNotAvailable {
            feature: feature.to_string(),
            required_build_flags: required_flags,
            remediation,
        }
    }
    
    /// Create a network error with retry guidance
    pub fn network_error(reason: &str, url: Option<String>) -> Self {
        let remediation = match reason {
            r if r.contains("timeout") => "Check network connectivity and try again with increased timeout".to_string(),
            r if r.contains("dns") => "Check DNS settings and network configuration".to_string(),
            r if r.contains("certificate") => "Check SSL/TLS certificate configuration".to_string(),
            r if r.contains("authentication") => "Check authentication credentials and permissions".to_string(),
            _ => "Check network connectivity and try again".to_string(),
        };
        
        Self::NetworkError {
            reason: reason.to_string(),
            url,
            remediation,
        }
    }
    
    /// Create a resource exhausted error with scaling guidance
    pub fn resource_exhausted(resource: &str, current: &str, limit: &str) -> Self {
        let remediation = match resource {
            "memory" => "Reduce chunk size, close other applications, or increase available memory".to_string(),
            "disk" => "Free up disk space or use a different storage location".to_string(),
            "network" => "Reduce concurrent operations or increase network bandwidth".to_string(),
            "cpu" => "Reduce compression level or limit concurrent operations".to_string(),
            _ => format!("Reduce {} usage or increase available {}", resource, resource),
        };
        
        Self::ResourceExhausted {
            resource: resource.to_string(),
            current_usage: current.to_string(),
            limit: limit.to_string(),
            remediation,
        }
    }
    
    /// Print a formatted error message with remediation guidance
    pub fn print_error(&self) {
        eprintln!("{} {}", style("Error:").red().bold(), self);
        
        match self {
            Self::Configuration { remediation, config_path, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
                if let Some(path) = config_path {
                    eprintln!("{} {}", style("Config:").blue(), path.display());
                }
            }
            
            Self::CorpusNotFound { remediation, searched_paths, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
                if !searched_paths.is_empty() {
                    eprintln!("{} {}", style("Searched:").blue(), 
                        searched_paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", "));
                }
            }
            
            Self::InvalidInputFile { remediation, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
            }
            
            Self::CompressionFailed { remediation, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
            }
            
            Self::DecompressionFailed { remediation, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
            }
            
            Self::CorpusOperationFailed { remediation, corpus_path, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
                if let Some(path) = corpus_path {
                    eprintln!("{} {}", style("Corpus:").blue(), path.display());
                }
            }
            
            Self::StreamProcessingError { remediation, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
            }
            
            Self::AnalysisFailed { remediation, input_paths, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
                if !input_paths.is_empty() {
                    eprintln!("{} {}", style("Inputs:").blue(), 
                        input_paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", "));
                }
            }
            
            Self::NetworkError { remediation, url, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
                if let Some(url) = url {
                    eprintln!("{} {}", style("URL:").blue(), url);
                }
            }
            
            Self::PermissionDenied { remediation, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
            }
            
            Self::ResourceExhausted { remediation, current_usage, limit, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
                eprintln!("{} {} / {}", style("Usage:").blue(), current_usage, limit);
            }
            
            Self::ValidationError { remediation, valid_range, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
                eprintln!("{} {}", style("Valid:").blue(), valid_range);
            }
            
            Self::OperationCancelled { partial_results, .. } => {
                if let Some(results) = partial_results {
                    eprintln!("{} {}", style("Partial:").blue(), results);
                }
            }
            
            Self::FeatureNotAvailable { remediation, required_build_flags, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
                if !required_build_flags.is_empty() {
                    eprintln!("{} {}", style("Features:").blue(), required_build_flags.join(", "));
                }
            }
            
            Self::InternalError { remediation, context, .. } => {
                eprintln!("{} {}", style("Fix:").yellow().bold(), remediation);
                eprintln!("{} {}", style("Context:").blue(), context);
                eprintln!("{} Please report this issue with the above context", style("Note:").cyan());
            }
        }
    }
    
    /// Get the exit code for this error
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Configuration { .. } => 1,
            Self::CorpusNotFound { .. } => 2,
            Self::InvalidInputFile { .. } => 3,
            Self::CompressionFailed { .. } => 4,
            Self::DecompressionFailed { .. } => 5,
            Self::CorpusOperationFailed { .. } => 6,
            Self::StreamProcessingError { .. } => 7,
            Self::AnalysisFailed { .. } => 8,
            Self::NetworkError { .. } => 9,
            Self::PermissionDenied { .. } => 10,
            Self::ResourceExhausted { .. } => 11,
            Self::ValidationError { .. } => 12,
            Self::OperationCancelled { .. } => 130, // Standard SIGINT exit code
            Self::FeatureNotAvailable { .. } => 13,
            Self::InternalError { .. } => 14,
        }
    }
}

/// Helper trait for converting standard errors to CLI errors with context
pub trait CliErrorExt<T> {
    fn with_cli_context(self, context: &str) -> Result<T, CliError>;
    fn with_input_context(self, path: PathBuf) -> Result<T, CliError>;
    fn with_corpus_context(self, corpus_id: &str) -> Result<T, CliError>;
}

impl<T, E> CliErrorExt<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_cli_context(self, context: &str) -> Result<T, CliError> {
        self.map_err(|e| CliError::InternalError {
            message: e.to_string(),
            context: context.to_string(),
            remediation: "Run with --verbose for more details or report this issue".to_string(),
        })
    }
    
    fn with_input_context(self, path: PathBuf) -> Result<T, CliError> {
        self.map_err(|e| {
            let reason = e.to_string();
            CliError::invalid_input_file(path, &reason)
        })
    }
    
    fn with_corpus_context(self, corpus_id: &str) -> Result<T, CliError> {
        self.map_err(|e| CliError::corpus_not_found(corpus_id, vec![]))
    }
}

/// Validation helpers for CLI arguments
pub mod validation {
    use super::CliError;
    use std::path::Path;
    
    /// Validate compression level (1-22)
    pub fn validate_compression_level(level: u8) -> Result<(), CliError> {
        if level < 1 || level > 22 {
            Err(CliError::validation_error(
                "compression-level",
                "value out of range",
                &level.to_string(),
                "1-22"
            ))
        } else {
            Ok(())
        }
    }
    
    /// Validate chunk size (4-64 KB)
    pub fn validate_chunk_size(size: u32) -> Result<(), CliError> {
        if size < 4 || size > 64 {
            Err(CliError::validation_error(
                "chunk-size",
                "value out of range",
                &size.to_string(),
                "4-64 KB"
            ))
        } else {
            Ok(())
        }
    }
    
    /// Validate input file exists and is readable
    pub fn validate_input_file(path: &Path) -> Result<(), CliError> {
        if !path.exists() {
            return Err(CliError::invalid_input_file(
                path.to_path_buf(),
                "file not found"
            ));
        }
        
        if !path.is_file() {
            return Err(CliError::invalid_input_file(
                path.to_path_buf(),
                "not a regular file"
            ));
        }
        
        // Check if file is readable
        match std::fs::File::open(path) {
            Ok(_) => Ok(()),
            Err(e) => Err(CliError::invalid_input_file(
                path.to_path_buf(),
                &format!("cannot read file: {}", e)
            )),
        }
    }
    
    /// Validate output directory is writable
    pub fn validate_output_path(path: &Path) -> Result<(), CliError> {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                return Err(CliError::invalid_input_file(
                    path.to_path_buf(),
                    "output directory does not exist"
                ));
            }
            
            // Try to create a temporary file to test writability
            match tempfile::NamedTempFile::new_in(parent) {
                Ok(_) => Ok(()),
                Err(e) => Err(CliError::PermissionDenied {
                    path: parent.to_path_buf(),
                    operation: "write".to_string(),
                    remediation: format!("Check directory permissions: {}", e),
                }),
            }
        } else {
            Ok(()) // Current directory
        }
    }
    
    /// Validate retention period (1-365 days)
    pub fn validate_retention_days(days: u32) -> Result<(), CliError> {
        if days < 1 || days > 365 {
            Err(CliError::validation_error(
                "retention-days",
                "value out of range",
                &days.to_string(),
                "1-365 days"
            ))
        } else {
            Ok(())
        }
    }
    
    /// Validate buffer size (1-1024 KB)
    pub fn validate_buffer_size(size: u32) -> Result<(), CliError> {
        if size < 1 || size > 1024 {
            Err(CliError::validation_error(
                "buffer-size",
                "value out of range",
                &size.to_string(),
                "1-1024 KB"
            ))
        } else {
            Ok(())
        }
    }
    
    /// Validate URL format
    pub fn validate_url(url: &str) -> Result<(), CliError> {
        if !url.starts_with("http://") && !url.starts_with("https://") {
            Err(CliError::validation_error(
                "url",
                "invalid URL format",
                url,
                "http:// or https:// URLs"
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_configuration_error() {
        let error = CliError::configuration_error("Invalid setting", Some(PathBuf::from("/test/config.toml")));
        assert!(error.to_string().contains("Configuration error"));
        assert_eq!(error.exit_code(), 1);
    }
    
    #[test]
    fn test_corpus_not_found_error() {
        let error = CliError::corpus_not_found("test-corpus", vec![PathBuf::from("/test/path")]);
        assert!(error.to_string().contains("Corpus not found"));
        assert_eq!(error.exit_code(), 2);
    }
    
    #[test]
    fn test_validation_error() {
        let error = CliError::validation_error("level", "out of range", "25", "1-22");
        assert!(error.to_string().contains("Validation error"));
        assert_eq!(error.exit_code(), 12);
    }
    
    #[test]
    fn test_feature_not_available_error() {
        let error = CliError::feature_not_available("metrics", vec!["metrics".to_string()]);
        assert!(error.to_string().contains("Feature not available"));
        assert_eq!(error.exit_code(), 13);
    }
    
    #[test]
    fn test_validation_compression_level() {
        use validation::*;
        
        assert!(validate_compression_level(1).is_ok());
        assert!(validate_compression_level(22).is_ok());
        assert!(validate_compression_level(0).is_err());
        assert!(validate_compression_level(23).is_err());
    }
    
    #[test]
    fn test_validation_chunk_size() {
        use validation::*;
        
        assert!(validate_chunk_size(4).is_ok());
        assert!(validate_chunk_size(64).is_ok());
        assert!(validate_chunk_size(3).is_err());
        assert!(validate_chunk_size(65).is_err());
    }
    
    #[test]
    fn test_validation_url() {
        use validation::*;
        
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("http://example.com").is_ok());
        assert!(validate_url("ftp://example.com").is_err());
        assert!(validate_url("example.com").is_err());
    }
}