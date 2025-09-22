//! Exhaustive error hierarchy for Reducto Mode 3
//!
//! This module defines all possible failure modes in the system using structured
//! error handling with thiserror for library-style errors.

use std::io;
use thiserror::Error;
use chrono::{DateTime, Utc};

/// Comprehensive error type covering all failure modes in Reducto Mode 3
///
/// This exhaustive hierarchy ensures all error conditions are handled explicitly
/// and provides actionable error messages for debugging and user feedback.
#[derive(Error, Debug)]
pub enum ReductoError {
    // === I/O and File System Errors ===
    #[error("I/O operation failed: {operation} - {source}")]
    Io {
        operation: String,
        #[source]
        source: io::Error,
    },

    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Permission denied accessing: {path}")]
    PermissionDenied { path: String },

    #[error("Disk space insufficient for operation: {required_bytes} bytes needed")]
    InsufficientDiskSpace { required_bytes: u64 },

    // === Serialization and Deserialization Errors ===
    #[error("Serialization failed: {context} - {source}")]
    Serialization {
        context: String,
        #[source]
        source: bincode::Error,
    },

    #[error("Deserialization failed: {context} - {source}")]
    Deserialization {
        context: String,
        #[source]
        source: bincode::Error,
    },

    // === Compression and Decompression Errors ===
    #[error("Compression failed: {algorithm} - {source}")]
    Compression {
        algorithm: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Decompression failed: {algorithm} - {source}")]
    Decompression {
        algorithm: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    // === File Format and Validation Errors ===
    #[error("Invalid file format: {reason}")]
    InvalidFormat { reason: String },

    #[error("Unsupported file version: found {found}, supported versions: {supported}")]
    UnsupportedVersion { found: String, supported: String },

    #[error("Magic bytes mismatch: expected {expected:?}, found {found:?}")]
    InvalidMagicBytes { expected: Vec<u8>, found: Vec<u8> },

    #[error("File header corrupted: {details}")]
    CorruptedHeader { details: String },

    #[error("File payload corrupted: checksum mismatch")]
    CorruptedPayload,

    // === Corpus Management Errors ===
    #[error("Corpus ID mismatch: expected {expected}, found {found}")]
    CorpusIdMismatch { expected: String, found: String },

    #[error("Corpus not found: {corpus_id}")]
    CorpusNotFound { corpus_id: String },

    #[error("Corpus index corrupted: {details}")]
    CorruptedCorpusIndex { details: String },

    #[error("Corpus manifest build failed: {reason}")]
    ManifestBuildFailed { reason: String },

    // === Block Processing Errors ===
    #[error("Block size mismatch: expected {expected}, found {found}")]
    BlockSizeMismatch { expected: u32, found: u32 },

    #[error("Invalid block reference: offset {offset} exceeds corpus bounds {max_offset}")]
    InvalidBlockReference { offset: u64, max_offset: u64 },

    #[error("Block hash collision: weak hash {weak_hash} has {collision_count} candidates")]
    BlockHashCollision {
        weak_hash: u64,
        collision_count: usize,
    },

    #[error("Block verification failed: strong hash mismatch for offset {offset}")]
    BlockVerificationFailed { offset: u64 },

    // === Memory Management Errors ===
    #[error("Memory mapping failed: {path} - {reason}")]
    MemoryMappingFailed { path: String, reason: String },

    #[error("Memory allocation failed: requested {requested_bytes} bytes")]
    MemoryAllocationFailed { requested_bytes: usize },

    #[error("Memory limit exceeded: {current_usage} bytes (limit: {limit_bytes} bytes)")]
    MemoryLimitExceeded {
        current_usage: usize,
        limit_bytes: usize,
    },

    // === Hash Computation Errors ===
    #[error("Hash computation failed: {hash_type} for block at offset {offset}")]
    HashComputationFailed { hash_type: String, offset: u64 },

    #[error("Rolling hash state corrupted: expected window size {expected}, got {actual}")]
    RollingHashStateCorrupted { expected: usize, actual: usize },

    // === Configuration and Parameter Errors ===
    #[error("Invalid configuration: {parameter} = {value} (reason: {reason})")]
    InvalidConfiguration {
        parameter: String,
        value: String,
        reason: String,
    },

    #[error("Parameter out of range: {parameter} = {value} (valid range: {min}..={max})")]
    ParameterOutOfRange {
        parameter: String,
        value: i64,
        min: i64,
        max: i64,
    },

    // === Concurrency and Threading Errors ===
    #[error("Thread synchronization failed: {operation}")]
    ThreadSynchronizationFailed { operation: String },

    #[error("Deadlock detected in operation: {operation}")]
    DeadlockDetected { operation: String },

    #[error("Resource contention: {resource} is locked by another operation")]
    ResourceContention { resource: String },

    // === Performance and Timeout Errors ===
    #[error("Operation timeout: {operation} exceeded {timeout_seconds}s limit")]
    OperationTimeout {
        operation: String,
        timeout_seconds: u64,
    },

    #[error("Performance contract violation: {operation} took {actual_duration_ms}ms (limit: {limit_ms}ms)")]
    PerformanceContractViolation {
        operation: String,
        actual_duration_ms: u64,
        limit_ms: u64,
    },

    // === Resource Exhaustion Errors ===
    #[error("Resource exhausted: {resource} (current: {current}, limit: {limit})")]
    ResourceExhausted {
        resource: String,
        current: u64,
        limit: u64,
    },

    #[error("File descriptor limit reached: {current_fds} (system limit: {max_fds})")]
    FileDescriptorLimitReached { current_fds: u32, max_fds: u32 },

    // === Validation and Constraint Errors ===
    #[error("Input validation failed: {field} - {reason}")]
    InputValidationFailed { field: String, reason: String },

    #[error("Constraint violation: {constraint} - {details}")]
    ConstraintViolation { constraint: String, details: String },

    #[error("Invariant violation: {invariant} - {context}")]
    InvariantViolation { invariant: String, context: String },

    // === System and Environment Errors ===
    #[error("System resource unavailable: {resource}")]
    SystemResourceUnavailable { resource: String },

    #[error("Environment error: {variable} is not set or invalid")]
    EnvironmentError { variable: String },

    #[error("Platform not supported: {platform} (supported: {supported_platforms})")]
    PlatformNotSupported {
        platform: String,
        supported_platforms: String,
    },

    // === Internal Logic Errors ===
    #[error("Internal error: {message} (this is a bug, please report it)")]
    InternalError { message: String },

    #[error("Unimplemented feature: {feature}")]
    UnimplementedFeature { feature: String },

    #[error("Assertion failed: {assertion} in {location}")]
    AssertionFailed { assertion: String, location: String },
}

/// Convenience type alias for Results in this crate
pub type Result<T> = std::result::Result<T, ReductoError>;

impl ReductoError {
    /// Create an I/O error with operation context
    pub fn io_error(operation: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            operation: operation.into(),
            source,
        }
    }

    /// Create a serialization error with context
    pub fn serialization_error(context: impl Into<String>, source: bincode::Error) -> Self {
        Self::Serialization {
            context: context.into(),
            source,
        }
    }

    /// Create a deserialization error with context
    pub fn deserialization_error(context: impl Into<String>, source: bincode::Error) -> Self {
        Self::Deserialization {
            context: context.into(),
            source,
        }
    }

    /// Create an internal error (should be used sparingly)
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Recoverable errors - can retry or handle gracefully
            Self::Io { .. }
            | Self::OperationTimeout { .. }
            | Self::ResourceContention { .. }
            | Self::MemoryAllocationFailed { .. }
            | Self::ThreadSynchronizationFailed { .. } => true,

            // Non-recoverable errors - indicate fundamental problems
            Self::InvalidFormat { .. }
            | Self::CorpusIdMismatch { .. }
            | Self::BlockSizeMismatch { .. }
            | Self::CorruptedHeader { .. }
            | Self::CorruptedPayload
            | Self::InternalError { .. }
            | Self::AssertionFailed { .. } => false,

            // Context-dependent - depends on specific situation
            _ => false,
        }
    }

    /// Get error category for logging and metrics
    pub fn category(&self) -> &'static str {
        match self {
            Self::Io { .. }
            | Self::FileNotFound { .. }
            | Self::PermissionDenied { .. }
            | Self::InsufficientDiskSpace { .. } => "io",

            Self::Serialization { .. } | Self::Deserialization { .. } => "serialization",

            Self::Compression { .. } | Self::Decompression { .. } => "compression",

            Self::InvalidFormat { .. }
            | Self::UnsupportedVersion { .. }
            | Self::InvalidMagicBytes { .. }
            | Self::CorruptedHeader { .. }
            | Self::CorruptedPayload => "format",

            Self::CorpusIdMismatch { .. }
            | Self::CorpusNotFound { .. }
            | Self::CorruptedCorpusIndex { .. }
            | Self::ManifestBuildFailed { .. } => "corpus",

            Self::BlockSizeMismatch { .. }
            | Self::InvalidBlockReference { .. }
            | Self::BlockHashCollision { .. }
            | Self::BlockVerificationFailed { .. } => "block",

            Self::MemoryMappingFailed { .. }
            | Self::MemoryAllocationFailed { .. }
            | Self::MemoryLimitExceeded { .. } => "memory",

            Self::HashComputationFailed { .. } | Self::RollingHashStateCorrupted { .. } => "hash",

            Self::InvalidConfiguration { .. } | Self::ParameterOutOfRange { .. } => "configuration",

            Self::ThreadSynchronizationFailed { .. }
            | Self::DeadlockDetected { .. }
            | Self::ResourceContention { .. } => "concurrency",

            Self::OperationTimeout { .. } | Self::PerformanceContractViolation { .. } => {
                "performance"
            }

            Self::ResourceExhausted { .. } | Self::FileDescriptorLimitReached { .. } => "resource",

            Self::InputValidationFailed { .. }
            | Self::ConstraintViolation { .. }
            | Self::InvariantViolation { .. } => "validation",

            Self::SystemResourceUnavailable { .. }
            | Self::EnvironmentError { .. }
            | Self::PlatformNotSupported { .. } => "system",

            Self::InternalError { .. }
            | Self::UnimplementedFeature { .. }
            | Self::AssertionFailed { .. } => "internal",
        }
    }
}

// === Enterprise Module Error Hierarchies ===

/// Corpus management specific errors
#[derive(Error, Debug)]
pub enum CorpusError {
    #[error("Corpus not found: {corpus_id}")]
    NotFound { corpus_id: String },
    
    #[error("Corpus signature verification failed for {corpus_id}")]
    SignatureVerificationFailed { corpus_id: String },
    
    #[error("Corpus repository unreachable: {url} - {cause}")]
    RepositoryUnreachable { url: String, cause: String },
    
    #[error("Persistent storage error: {operation} - {cause}")]
    Storage { operation: String, cause: String },
    
    #[error("Optimization failed: {reason}")]
    OptimizationFailed { reason: String },
    
    #[error("Corpus version incompatible: expected {expected}, found {found}")]
    IncompatibleVersion { expected: String, found: String },
    
    #[error("Corpus index corruption detected: {details}")]
    IndexCorruption { details: String },
    
    #[error("Concurrent access conflict for corpus {corpus_id}")]
    ConcurrencyConflict { corpus_id: String },
    
    #[error("Corpus capacity exceeded: {current_size} bytes (limit: {max_size} bytes)")]
    CapacityExceeded { current_size: u64, max_size: u64 },
    
    #[error("Chunk frequency analysis failed: {reason}")]
    FrequencyAnalysisFailed { reason: String },
}

/// Security framework specific errors
#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Cryptographic signing failed: {algorithm} - {cause}")]
    SigningFailed { algorithm: String, cause: String },
    
    #[error("Signature verification failed for {resource_id}")]
    VerificationFailed { resource_id: String },
    
    #[error("Encryption failed: {algorithm} - {cause}")]
    EncryptionFailed { algorithm: String, cause: String },
    
    #[error("Decryption failed: {algorithm} - {cause}")]
    DecryptionFailed { algorithm: String, cause: String },
    
    #[error("Key management error: {operation} - {cause}")]
    KeyManagement { operation: String, cause: String },
    
    #[error("Audit logging failed: {event_type} - {cause}")]
    AuditFailed { event_type: String, cause: String },
    
    #[error("Retention policy violation: {policy} - {details}")]
    RetentionViolation { policy: String, details: String },
    
    #[error("Authentication failed for user {user_id}")]
    AuthenticationFailed { user_id: String },
    
    #[error("Authorization denied: user {user_id} lacks permission {permission}")]
    AuthorizationDenied { user_id: String, permission: String },
    
    #[error("Secure deletion failed for {resource}: {cause}")]
    SecureDeletionFailed { resource: String, cause: String },
}

/// Metrics and observability specific errors
#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("Metrics collection failed for component {component}: {cause}")]
    CollectionFailed { component: String, cause: String },
    
    #[error("Metrics export failed: format {format} - {cause}")]
    ExportFailed { format: String, cause: String },
    
    #[error("Analysis error: {operation} - {cause}")]
    AnalysisFailed { operation: String, cause: String },
    
    #[error("ROI calculation error: insufficient data for period {start} to {end}")]
    InsufficientData { start: DateTime<Utc>, end: DateTime<Utc> },
    
    #[error("Performance monitoring failed: {metric} - {cause}")]
    MonitoringFailed { metric: String, cause: String },
    
    #[error("Prometheus export error: {endpoint} - {cause}")]
    PrometheusExportFailed { endpoint: String, cause: String },
    
    #[error("Metrics storage full: {current_size} bytes (limit: {max_size} bytes)")]
    StorageFull { current_size: u64, max_size: u64 },
    
    #[error("Time series data corruption detected: {metric} at {timestamp}")]
    TimeSeriesCorruption { metric: String, timestamp: DateTime<Utc> },
}

/// Storage abstraction specific errors
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Database error: {operation} - {cause}")]
    Database { operation: String, cause: String },
    
    #[error("Index corruption detected in {index_name}")]
    IndexCorruption { index_name: String },
    
    #[error("Storage capacity exceeded: {used} bytes used of {limit} bytes")]
    CapacityExceeded { used: u64, limit: u64 },
    
    #[error("Concurrent access conflict for resource {resource_id}")]
    ConcurrencyConflict { resource_id: String },
    
    #[error("Transaction failed: {transaction_id} - {cause}")]
    TransactionFailed { transaction_id: String, cause: String },
    
    #[error("Backup operation failed: {backup_id} - {cause}")]
    BackupFailed { backup_id: String, cause: String },
    
    #[error("Recovery operation failed: {recovery_id} - {cause}")]
    RecoveryFailed { recovery_id: String, cause: String },
    
    #[error("Replication lag exceeded threshold: {current_lag_ms}ms (max: {max_lag_ms}ms)")]
    ReplicationLagExceeded { current_lag_ms: u64, max_lag_ms: u64 },
}

/// SDK and integration specific errors
#[derive(Error, Debug)]
pub enum SDKError {
    #[error("Configuration error: {parameter} - {message}\nRemediation: {remediation}")]
    Configuration { parameter: String, message: String, remediation: String },
    
    #[error("Corpus not found: {corpus_id}\nRemediation: Check corpus repositories or run 'reducto corpus fetch {corpus_id}'")]
    CorpusNotFound { corpus_id: String },
    
    #[error("API version mismatch: client={client_version}, server={server_version}\nRemediation: Update SDK to version {required_version}")]
    VersionMismatch { client_version: String, server_version: String, required_version: String },
    
    #[error("Stream processing error: {operation} - {cause}")]
    StreamProcessing { operation: String, cause: String },
    
    #[error("Pipeline integration failed: {tool} - {cause}")]
    PipelineIntegration { tool: String, cause: String },
    
    #[error("FFI error: {function} - {cause}")]
    FFI { function: String, cause: String },
    
    #[error("Timeout in operation {operation}: {elapsed_ms}ms (limit: {timeout_ms}ms)")]
    Timeout { operation: String, elapsed_ms: u64, timeout_ms: u64 },
    
    #[error("Rate limit exceeded: {current_rate} requests/sec (limit: {max_rate} requests/sec)")]
    RateLimitExceeded { current_rate: f64, max_rate: f64 },
}

// Implement From conversions for the main ReductoError
impl From<CorpusError> for ReductoError {
    fn from(err: CorpusError) -> Self {
        match err {
            CorpusError::NotFound { corpus_id } => Self::CorpusNotFound { corpus_id },
            CorpusError::IndexCorruption { details } => Self::CorruptedCorpusIndex { details },
            CorpusError::OptimizationFailed { reason } => Self::ManifestBuildFailed { reason },
            _ => Self::InternalError { message: format!("Corpus error: {}", err) },
        }
    }
}

impl From<SecurityError> for ReductoError {
    fn from(err: SecurityError) -> Self {
        Self::InternalError { message: format!("Security error: {}", err) }
    }
}

impl From<MetricsError> for ReductoError {
    fn from(err: MetricsError) -> Self {
        Self::InternalError { message: format!("Metrics error: {}", err) }
    }
}

impl From<StorageError> for ReductoError {
    fn from(err: StorageError) -> Self {
        match err {
            StorageError::CapacityExceeded { used, limit } => Self::ResourceExhausted {
                resource: "storage".to_string(),
                current: used,
                limit,
            },
            StorageError::ConcurrencyConflict { resource_id } => Self::ResourceContention {
                resource: resource_id,
            },
            _ => Self::InternalError { message: format!("Storage error: {}", err) },
        }
    }
}

impl From<SDKError> for ReductoError {
    fn from(err: SDKError) -> Self {
        match err {
            SDKError::CorpusNotFound { corpus_id } => Self::CorpusNotFound { corpus_id },
            SDKError::Timeout { operation, elapsed_ms, timeout_ms: _ } => Self::OperationTimeout {
                operation,
                timeout_seconds: elapsed_ms / 1000,
            },
            _ => Self::InternalError { message: format!("SDK error: {}", err) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_categories() {
        let io_error = ReductoError::io_error("test operation", io::Error::from(io::ErrorKind::NotFound));
        assert_eq!(io_error.category(), "io");

        let format_error = ReductoError::InvalidFormat {
            reason: "test".to_string(),
        };
        assert_eq!(format_error.category(), "format");
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable = ReductoError::OperationTimeout {
            operation: "test".to_string(),
            timeout_seconds: 30,
        };
        assert!(recoverable.is_recoverable());

        let non_recoverable = ReductoError::CorruptedPayload;
        assert!(!non_recoverable.is_recoverable());
    }

    #[test]
    fn test_error_display() {
        let error = ReductoError::CorpusIdMismatch {
            expected: "corpus-123".to_string(),
            found: "corpus-456".to_string(),
        };
        let display = format!("{}", error);
        assert!(display.contains("corpus-123"));
        assert!(display.contains("corpus-456"));
    }

    #[test]
    fn test_corpus_error_conversion() {
        let corpus_error = CorpusError::NotFound {
            corpus_id: "test-corpus".to_string(),
        };
        let reducto_error: ReductoError = corpus_error.into();
        
        match reducto_error {
            ReductoError::CorpusNotFound { corpus_id } => {
                assert_eq!(corpus_id, "test-corpus");
            }
            _ => panic!("Expected CorpusNotFound variant"),
        }
    }

    #[test]
    fn test_security_error_display() {
        let error = SecurityError::SigningFailed {
            algorithm: "ed25519".to_string(),
            cause: "key not found".to_string(),
        };
        let display = format!("{}", error);
        assert!(display.contains("ed25519"));
        assert!(display.contains("key not found"));
    }

    #[test]
    fn test_metrics_error_with_timestamp() {
        let now = Utc::now();
        let error = MetricsError::InsufficientData {
            start: now,
            end: now,
        };
        let display = format!("{}", error);
        assert!(display.contains("insufficient data"));
    }

    #[test]
    fn test_sdk_error_remediation() {
        let error = SDKError::CorpusNotFound {
            corpus_id: "missing-corpus".to_string(),
        };
        let display = format!("{}", error);
        assert!(display.contains("Remediation:"));
        assert!(display.contains("reducto corpus fetch"));
    }
}