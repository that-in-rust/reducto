//! Comprehensive tests for the Enterprise SDK
//!
//! These tests validate SDK functionality including stream processing,
//! C FFI bindings, pipeline integrations, and error handling.

use reducto_mode_3::{
    sdk::*,
    types::*,
    error::*,
    ecosystem_decompressor::*,
};
use std::{
    ffi::{CStr, CString},
    path::PathBuf,
    time::Duration,
};
use tempfile::TempDir;
use std::io::Cursor;

/// Test SDK creation and configuration validation
#[tokio::test]
async fn test_sdk_creation_and_validation() {
    // Test successful creation with default config
    let config = SDKConfig::default();
    let sdk = ReductoSDK::new(config).await;
    assert!(sdk.is_ok(), "SDK creation should succeed with default config");

    // Test configuration validation - invalid compression level
    let mut invalid_config = SDKConfig::default();
    invalid_config.compression_level = 25; // Invalid level (max is 22)
    
    let result = ReductoSDK::new(invalid_config).await;
    assert!(result.is_err(), "SDK creation should fail with invalid compression level");
    
    match result.unwrap_err() {
        ReductoError::ParameterOutOfRange { parameter, .. } => {
            assert_eq!(parameter, "compression_level");
        }
        _ => panic!("Expected ParameterOutOfRange error"),
    }

    // Test configuration validation - invalid buffer size
    let mut invalid_config = SDKConfig::default();
    invalid_config.stream_config.buffer_size = 0; // Invalid size
    
    let result = ReductoSDK::new(invalid_config).await;
    assert!(result.is_err(), "SDK creation should fail with zero buffer size");
}

/// Test API version compatibility checking
#[tokio::test]
async fn test_api_version_compatibility() {
    // Test compatible version
    assert!(ReductoSDK::is_api_version_compatible(SDK_API_VERSION));
    
    // Test incompatible versions
    assert!(!ReductoSDK::is_api_version_compatible("2.0.0"));
    assert!(!ReductoSDK::is_api_version_compatible("0.9.0"));
    assert!(!ReductoSDK::is_api_version_compatible("invalid"));

    // Test SDK creation with incompatible API version
    let mut config = SDKConfig::default();
    config.api_version = "2.0.0".to_string();
    
    let result = ReductoSDK::new(config).await;
    assert!(result.is_err(), "SDK creation should fail with incompatible API version");
}

/// Test stream-based compression functionality
#[tokio::test]
async fn test_stream_compression() {
    let config = SDKConfig::default();
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    // Create test data
    let test_data = b"Hello, world! This is test data for stream compression. ".repeat(100);
    let input_cursor = Cursor::new(&test_data);
    let mut output_buffer = Vec::new();
    let output_cursor = Cursor::new(&mut output_buffer);
    
    // Create temporary corpus file
    let temp_dir = TempDir::new().unwrap();
    let corpus_path = temp_dir.path().join("test_corpus.bin");
    std::fs::write(&corpus_path, b"test corpus data for compression").unwrap();
    
    // Perform stream compression
    let result = sdk.compress_stream(input_cursor, output_cursor, &corpus_path).await;
    assert!(result.is_ok(), "Stream compression should succeed");
    
    let compression_result = result.unwrap();
    assert!(compression_result.success, "Compression should be successful");
    assert_eq!(compression_result.input_size, test_data.len() as u64);
    assert!(compression_result.output_size > 0, "Output should have non-zero size");
    assert!(compression_result.processing_time > Duration::ZERO, "Processing time should be recorded");
    
    // Verify output buffer contains data
    assert!(!output_buffer.is_empty(), "Output buffer should contain compressed data");
}

/// Test stream-based decompression functionality
#[tokio::test]
async fn test_stream_decompression() {
    let config = SDKConfig::default();
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    // Create test compressed data (minimal valid format)
    let header = ReductoHeader::basic(
        uuid::Uuid::new_v4(),
        ChunkConfig::default(),
    );
    
    let instructions = vec![
        ReductoInstruction::Residual(b"Hello, ".to_vec()),
        ReductoInstruction::Residual(b"world!".to_vec()),
    ];
    
    // Serialize header and instructions
    let serialized_header = bincode::serialize(&header).unwrap();
    let serialized_instructions = bincode::serialize(&instructions).unwrap();
    let compressed_instructions = zstd::encode_all(&serialized_instructions[..], 19).unwrap();
    
    // Create compressed data format
    let mut compressed_data = Vec::new();
    compressed_data.extend_from_slice(&(serialized_header.len() as u32).to_le_bytes());
    compressed_data.extend_from_slice(&serialized_header);
    compressed_data.extend_from_slice(&compressed_instructions);
    
    let input_cursor = Cursor::new(&compressed_data);
    let mut output_buffer = Vec::new();
    let output_cursor = Cursor::new(&mut output_buffer);
    
    // Perform stream decompression
    let result = sdk.decompress_stream(input_cursor, output_cursor).await;
    assert!(result.is_ok(), "Stream decompression should succeed");
    
    let decompression_result = result.unwrap();
    assert!(decompression_result.success, "Decompression should be successful");
    
    // Verify output
    assert_eq!(output_buffer, b"Hello, world!", "Decompressed data should match expected");
}

/// Test timeout handling in stream operations
#[tokio::test]
async fn test_stream_timeout_handling() {
    let mut config = SDKConfig::default();
    config.timeout_config.compression_timeout = Duration::from_millis(1); // Very short timeout
    
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    // Create large test data that will likely exceed the timeout
    let large_data = vec![0u8; 10 * 1024 * 1024]; // 10MB
    let input_cursor = Cursor::new(&large_data);
    let mut output_buffer = Vec::new();
    let output_cursor = Cursor::new(&mut output_buffer);
    
    let temp_dir = TempDir::new().unwrap();
    let corpus_path = temp_dir.path().join("test_corpus.bin");
    std::fs::write(&corpus_path, b"test corpus").unwrap();
    
    // This should timeout
    let result = sdk.compress_stream(input_cursor, output_cursor, &corpus_path).await;
    assert!(result.is_err(), "Compression should timeout");
    
    // Verify it's a timeout error
    match result.unwrap_err() {
        ReductoError::OperationTimeout { operation, .. } => {
            assert_eq!(operation, "stream_compression");
        }
        _ => panic!("Expected timeout error"),
    }
}

/// Test SDK configuration updates
#[tokio::test]
async fn test_sdk_config_updates() {
    let config = SDKConfig::default();
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    // Get initial config
    let initial_config = sdk.get_config().await;
    assert_eq!(initial_config.compression_level, 19);
    
    // Update config
    let mut new_config = initial_config.clone();
    new_config.compression_level = 15;
    new_config.enable_metrics = false;
    
    let result = sdk.update_config(new_config.clone()).await;
    assert!(result.is_ok(), "Config update should succeed");
    
    // Verify config was updated
    let updated_config = sdk.get_config().await;
    assert_eq!(updated_config.compression_level, 15);
    assert!(!updated_config.enable_metrics);
    
    // Test invalid config update
    let mut invalid_config = new_config;
    invalid_config.compression_level = 25; // Invalid
    
    let result = sdk.update_config(invalid_config).await;
    assert!(result.is_err(), "Invalid config update should fail");
}

/// Test pipeline tool management
#[tokio::test]
async fn test_pipeline_tool_management() {
    let config = SDKConfig::default();
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    // Get available tools
    let tools = sdk.get_pipeline_tools().await;
    assert!(tools.contains(&"tar".to_string()), "Should have tar tool");
    assert!(tools.contains(&"ssh".to_string()), "Should have ssh tool");
    assert!(tools.contains(&"aws".to_string()), "Should have aws tool");
    assert!(tools.contains(&"gcp".to_string()), "Should have gcp tool");
    assert!(tools.contains(&"azure".to_string()), "Should have azure tool");
    
    // Test adding custom tool
    struct CustomTool;
    impl PipelineTool for CustomTool {
        fn name(&self) -> &str { "custom" }
        fn execute(&self, _args: &[String], _sdk: &ReductoSDK) -> Result<String> {
            Ok("custom tool executed".to_string())
        }
        fn help(&self) -> &str { "Custom test tool" }
    }
    
    sdk.add_pipeline_tool("custom".to_string(), Box::new(CustomTool)).await;
    
    let updated_tools = sdk.get_pipeline_tools().await;
    assert!(updated_tools.contains(&"custom".to_string()), "Should have custom tool");
}

/// Test pipeline integration helpers
#[tokio::test]
async fn test_pipeline_integration_helpers() {
    let config = SDKConfig::default();
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    // Test tar filter creation
    let tar_filter = sdk.create_tar_filter();
    // Note: We can't test actual tar processing without implementing it
    
    // Test SSH wrapper creation
    let ssh_wrapper = sdk.create_ssh_wrapper();
    // Note: We can't test actual SSH operations without implementing it
    
    // Test cloud CLI plugin creation
    let aws_plugin = sdk.create_cloud_cli_plugin(CloudProvider::Aws);
    let gcp_plugin = sdk.create_cloud_cli_plugin(CloudProvider::Gcp);
    let azure_plugin = sdk.create_cloud_cli_plugin(CloudProvider::Azure);
    
    // These are just creation tests - actual functionality would need implementation
}

/// Test C FFI configuration conversion
#[test]
fn test_c_ffi_config_conversion() {
    let c_config = CSDKConfig {
        target_chunk_size: 8192,
        compression_level: 19,
        enable_metrics: 1,
        timeout_seconds: 300,
        buffer_size: 65536,
        verify_signatures: 1,
    };
    
    let rust_config = c_config_to_rust_config(&c_config);
    assert!(rust_config.is_ok(), "C config conversion should succeed");
    
    let config = rust_config.unwrap();
    assert_eq!(config.chunk_config.target_size, 8192);
    assert_eq!(config.compression_level, 19);
    assert!(config.enable_metrics);
    assert_eq!(config.stream_config.buffer_size, 65536);
    assert!(config.security_config.verify_signatures);
    assert_eq!(config.timeout_config.compression_timeout, Duration::from_secs(300));
}

/// Test C FFI error code conversion
#[test]
fn test_c_ffi_error_conversion() {
    let test_cases = vec![
        (ReductoError::InputValidationFailed { 
            field: "test".to_string(), 
            reason: "test".to_string() 
        }, CErrorCode::InvalidInput),
        (ReductoError::CompressionFailed { 
            algorithm: "test".to_string(), 
            cause: "test".to_string() 
        }, CErrorCode::CompressionFailed),
        (ReductoError::CorpusNotFound { 
            corpus_id: "test".to_string() 
        }, CErrorCode::CorpusNotFound),
        (ReductoError::OperationTimeout { 
            operation: "test".to_string(), 
            timeout_seconds: 30 
        }, CErrorCode::TimeoutError),
        (ReductoError::InvalidConfiguration { 
            parameter: "test".to_string(), 
            value: "test".to_string(), 
            reason: "test".to_string() 
        }, CErrorCode::ConfigurationError),
    ];
    
    for (error, expected_code) in test_cases {
        let c_error_code = CErrorCode::from(&error);
        assert_eq!(c_error_code as i32, expected_code as i32, 
                   "Error conversion failed for: {:?}", error);
    }
}

/// Test C string creation and cleanup
#[test]
fn test_c_string_management() {
    let test_string = "Test error message";
    let c_string_ptr = create_c_string(test_string);
    assert!(!c_string_ptr.is_null(), "C string pointer should not be null");
    
    unsafe {
        let c_str = CStr::from_ptr(c_string_ptr);
        let rust_str = c_str.to_str().unwrap();
        assert_eq!(rust_str, test_string, "C string should match original");
        
        // Test cleanup
        reducto_free_string(c_string_ptr as *mut std::ffi::c_char);
    }
    
    // Test string with null bytes (should be handled gracefully)
    let string_with_null = "Test\0with\0nulls";
    let c_string_ptr = create_c_string(string_with_null);
    assert!(!c_string_ptr.is_null(), "Should handle strings with null bytes");
    
    unsafe {
        reducto_free_string(c_string_ptr as *mut std::ffi::c_char);
    }
}

/// Test structured error responses with remediation
#[tokio::test]
async fn test_structured_error_responses() {
    // Test corpus not found error with remediation
    let error = SDKError::CorpusNotFound {
        corpus_id: "missing-corpus-123".to_string(),
    };
    
    let error_message = format!("{}", error);
    assert!(error_message.contains("Remediation:"), "Error should include remediation");
    assert!(error_message.contains("reducto corpus fetch"), "Should suggest fetch command");
    assert!(error_message.contains("missing-corpus-123"), "Should include corpus ID");
    
    // Test version mismatch error with remediation
    let error = SDKError::VersionMismatch {
        client_version: "1.0.0".to_string(),
        server_version: "1.1.0".to_string(),
        required_version: "1.1.0".to_string(),
    };
    
    let error_message = format!("{}", error);
    assert!(error_message.contains("Remediation:"), "Error should include remediation");
    assert!(error_message.contains("Update SDK"), "Should suggest SDK update");
    
    // Test configuration error with remediation
    let error = SDKError::Configuration {
        parameter: "compression_level".to_string(),
        message: "Invalid value 25".to_string(),
        remediation: "Use value between 1 and 22".to_string(),
    };
    
    let error_message = format!("{}", error);
    assert!(error_message.contains("Remediation:"), "Error should include remediation");
    assert!(error_message.contains("Use value between 1 and 22"), "Should include specific remediation");
}

/// Test memory management and resource cleanup
#[tokio::test]
async fn test_memory_management() {
    let mut config = SDKConfig::default();
    config.stream_config.max_memory_usage = 1024 * 1024; // 1MB limit
    
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    // Create data that exceeds memory limit
    let large_data = vec![0u8; 2 * 1024 * 1024]; // 2MB
    let input_cursor = Cursor::new(&large_data);
    let mut output_buffer = Vec::new();
    let output_cursor = Cursor::new(&mut output_buffer);
    
    let temp_dir = TempDir::new().unwrap();
    let corpus_path = temp_dir.path().join("test_corpus.bin");
    std::fs::write(&corpus_path, b"test corpus").unwrap();
    
    // This should fail due to memory limit
    let result = sdk.compress_stream(input_cursor, output_cursor, &corpus_path).await;
    assert!(result.is_err(), "Should fail due to memory limit");
    
    match result.unwrap_err() {
        ReductoError::MemoryLimitExceeded { current_usage, limit_bytes } => {
            assert!(current_usage > limit_bytes, "Should exceed memory limit");
        }
        _ => panic!("Expected memory limit exceeded error"),
    }
}

/// Test concurrent SDK operations
#[tokio::test]
async fn test_concurrent_operations() {
    let config = SDKConfig::default();
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    let temp_dir = TempDir::new().unwrap();
    let corpus_path = temp_dir.path().join("test_corpus.bin");
    std::fs::write(&corpus_path, b"test corpus data").unwrap();
    
    // Create multiple concurrent compression tasks
    let mut tasks = Vec::new();
    
    for i in 0..5 {
        let sdk_clone = sdk.clone();
        let corpus_path_clone = corpus_path.clone();
        
        let task = tokio::spawn(async move {
            let test_data = format!("Test data for task {}", i).repeat(100);
            let input_cursor = Cursor::new(test_data.as_bytes());
            let mut output_buffer = Vec::new();
            let output_cursor = Cursor::new(&mut output_buffer);
            
            sdk_clone.compress_stream(input_cursor, output_cursor, &corpus_path_clone).await
        });
        
        tasks.push(task);
    }
    
    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;
    
    // Verify all tasks succeeded
    for (i, result) in results.into_iter().enumerate() {
        let task_result = result.unwrap();
        assert!(task_result.is_ok(), "Task {} should succeed", i);
        
        let compression_result = task_result.unwrap();
        assert!(compression_result.success, "Compression {} should be successful", i);
    }
}

/// Test SDK with different chunk configurations
#[tokio::test]
async fn test_different_chunk_configurations() {
    let chunk_sizes = vec![4096, 8192, 16384, 32768];
    
    for chunk_size in chunk_sizes {
        let mut config = SDKConfig::default();
        config.chunk_config = ChunkConfig::new(chunk_size).unwrap();
        
        let sdk = ReductoSDK::new(config).await.unwrap();
        
        let test_data = b"Test data for chunk configuration testing. ".repeat(200);
        let input_cursor = Cursor::new(&test_data);
        let mut output_buffer = Vec::new();
        let output_cursor = Cursor::new(&mut output_buffer);
        
        let temp_dir = TempDir::new().unwrap();
        let corpus_path = temp_dir.path().join("test_corpus.bin");
        std::fs::write(&corpus_path, b"test corpus data").unwrap();
        
        let result = sdk.compress_stream(input_cursor, output_cursor, &corpus_path).await;
        assert!(result.is_ok(), "Compression should succeed with chunk size {}", chunk_size);
        
        let compression_result = result.unwrap();
        assert!(compression_result.success, "Compression should be successful");
        assert_eq!(compression_result.input_size, test_data.len() as u64);
    }
}

/// Test SDK metrics collection
#[tokio::test]
async fn test_metrics_collection() {
    let mut config = SDKConfig::default();
    config.enable_metrics = true;
    
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    let test_data = b"Test data for metrics collection. ".repeat(100);
    let input_cursor = Cursor::new(&test_data);
    let mut output_buffer = Vec::new();
    let output_cursor = Cursor::new(&mut output_buffer);
    
    let temp_dir = TempDir::new().unwrap();
    let corpus_path = temp_dir.path().join("test_corpus.bin");
    std::fs::write(&corpus_path, b"test corpus data").unwrap();
    
    let result = sdk.compress_stream(input_cursor, output_cursor, &corpus_path).await;
    assert!(result.is_ok(), "Compression should succeed");
    
    let compression_result = result.unwrap();
    
    // Verify metrics are collected
    assert!(compression_result.metrics.chunks_processed > 0, "Should process chunks");
    assert!(compression_result.processing_time > Duration::ZERO, "Should record processing time");
    assert!(compression_result.compression_ratio > 0.0, "Should calculate compression ratio");
    
    // Test with metrics disabled
    let mut config = SDKConfig::default();
    config.enable_metrics = false;
    
    let sdk = ReductoSDK::new(config).await.unwrap();
    // Should still work but with minimal metrics
}

/// Integration test for real enterprise scenarios
#[tokio::test]
async fn test_enterprise_integration_scenario() {
    // Simulate enterprise configuration
    let mut config = SDKConfig::default();
    config.compression_level = 22; // Maximum compression
    config.enable_metrics = true;
    config.security_config.verify_signatures = true;
    config.timeout_config.compression_timeout = Duration::from_secs(300);
    config.stream_config.buffer_size = 1024 * 1024; // 1MB buffer
    
    // Add corpus repositories (would be real URLs in production)
    config.corpus_repositories = vec![
        CorpusRepository::new("https://corpus-repo-1.example.com")
            .with_priority(1)
            .with_timeout(Duration::from_secs(30)),
        CorpusRepository::new("https://corpus-repo-2.example.com")
            .with_priority(2)
            .with_timeout(Duration::from_secs(60)),
    ];
    
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    // Simulate VM image data (highly redundant)
    let vm_image_data = create_simulated_vm_image_data();
    let input_cursor = Cursor::new(&vm_image_data);
    let mut output_buffer = Vec::new();
    let output_cursor = Cursor::new(&mut output_buffer);
    
    let temp_dir = TempDir::new().unwrap();
    let corpus_path = temp_dir.path().join("enterprise_corpus.bin");
    std::fs::write(&corpus_path, &create_simulated_corpus_data()).unwrap();
    
    let result = sdk.compress_stream(input_cursor, output_cursor, &corpus_path).await;
    assert!(result.is_ok(), "Enterprise compression should succeed");
    
    let compression_result = result.unwrap();
    assert!(compression_result.success, "Enterprise compression should be successful");
    
    // Verify enterprise-level metrics
    assert!(compression_result.input_size > 0, "Should process input data");
    assert!(compression_result.output_size > 0, "Should produce output");
    assert!(compression_result.processing_time > Duration::ZERO, "Should record processing time");
    
    // In a real enterprise scenario, we'd expect high compression ratios
    // due to redundancy in VM images, CI/CD artifacts, etc.
}

/// Helper function to create simulated VM image data
fn create_simulated_vm_image_data() -> Vec<u8> {
    let mut data = Vec::new();
    
    // Simulate boot sector (highly redundant)
    let boot_sector = vec![0x55, 0xAA].repeat(256);
    data.extend_from_slice(&boot_sector);
    
    // Simulate file system metadata (repetitive patterns)
    let fs_metadata = b"EXT4_SUPERBLOCK".repeat(100);
    data.extend_from_slice(&fs_metadata);
    
    // Simulate application binaries (some redundancy)
    let binary_data = include_bytes!("../Cargo.toml").repeat(50);
    data.extend_from_slice(&binary_data);
    
    // Simulate log files (highly compressible)
    let log_data = b"[INFO] Application started successfully\n".repeat(1000);
    data.extend_from_slice(&log_data);
    
    data
}

/// Helper function to create simulated corpus data
fn create_simulated_corpus_data() -> Vec<u8> {
    let mut corpus = Vec::new();
    
    // Add common patterns that would appear in VM images
    corpus.extend_from_slice(&vec![0x00; 4096]); // Empty blocks
    corpus.extend_from_slice(&vec![0xFF; 4096]); // Full blocks
    corpus.extend_from_slice(b"EXT4_SUPERBLOCK".repeat(256).as_slice());
    corpus.extend_from_slice(include_bytes!("../Cargo.toml"));
    
    corpus
}

/// Test error handling and recovery scenarios
#[tokio::test]
async fn test_error_handling_and_recovery() {
    let config = SDKConfig::default();
    let sdk = ReductoSDK::new(config).await.unwrap();
    
    // Test with non-existent corpus file
    let test_data = b"Test data";
    let input_cursor = Cursor::new(&test_data);
    let mut output_buffer = Vec::new();
    let output_cursor = Cursor::new(&mut output_buffer);
    
    let non_existent_corpus = PathBuf::from("/non/existent/corpus.bin");
    
    let result = sdk.compress_stream(input_cursor, output_cursor, &non_existent_corpus).await;
    assert!(result.is_err(), "Should fail with non-existent corpus");
    
    // Verify error type
    match result.unwrap_err() {
        ReductoError::Io { operation, .. } => {
            assert!(operation.contains("read") || operation.contains("open"), 
                   "Should be an I/O error for reading corpus");
        }
        _ => panic!("Expected I/O error"),
    }
}