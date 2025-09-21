//! Exhaustive error hierarchy for Reducto Mode 3
//!
//! This module defines all possible failure modes in the system using structured
//! error handling with thiserror for library-style errors.

use std::io;
use thiserror::Error;

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
}