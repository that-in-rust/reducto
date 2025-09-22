//! Enterprise SDK for Reducto Mode 3
//!
//! This module provides a comprehensive SDK for integrating Reducto Mode 3 into enterprise
//! applications and pipelines. It includes stream-based compression/decompression, C FFI
//! bindings for multi-language support, and pipeline integration helpers.

use crate::{
    error::{ReductoError, Result, SDKError},
    types::{ChunkConfig, ReductoHeader, ReductoInstruction, CorpusId},
    traits::{CDCChunker, CorpusManager, SecurityManager, MetricsCollector},
    compressor::Compressor,
    ecosystem_decompressor::{EcosystemDecompressor, CorpusRepository, DecompressionResult},
    security_manager::EnterpriseSecurityManager,
};

#[cfg(feature = "metrics")]
use crate::metrics_collector::EnterpriseMetricsCollector;

use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char, c_int, c_void},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncRead, AsyncWrite, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter},
    sync::{Mutex, RwLock},
    time::timeout,
};

/// Current SDK API version for compatibility tracking
pub const SDK_API_VERSION: &str = "1.0.0";

/// Maximum stream buffer size (16MB)
const MAX_STREAM_BUFFER_SIZE: usize = 16 * 1024 * 1024;

/// Default operation timeout (30 seconds)
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// SDK configuration with enterprise features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDKConfig {
    /// CDC chunking configuration
    pub chunk_config: ChunkConfig,
    /// Corpus repositories for automatic fetching
    pub corpus_repositories: Vec<CorpusRepository>,
    /// Zstandard compression level (1-22)
    pub compression_level: u8,
    /// Enable comprehensive metrics collection
    pub enable_metrics: bool,
    /// API version for compatibility checking
    pub api_version: String,
    /// Operation timeout settings
    pub timeout_config: TimeoutConfig,
    /// Stream processing configuration
    pub stream_config: StreamConfig,
    /// Security configuration
    pub security_config: SecurityConfig,
    /// Pipeline integration settings
    pub pipeline_config: PipelineConfig,
}

/// Timeout configuration for various operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Compression operation timeout
    pub compression_timeout: Duration,
    /// Decompression operation timeout
    pub decompression_timeout: Duration,
    /// Corpus fetch timeout
    pub corpus_fetch_timeout: Duration,
    /// Stream operation timeout
    pub stream_timeout: Duration,
}

/// Stream processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Buffer size for stream operations
    pub buffer_size: usize,
    /// Enable progressive compression
    pub progressive_compression: bool,
    /// Maximum memory usage for streaming
    pub max_memory_usage: usize,
    /// Enable backpressure handling
    pub enable_backpressure: bool,
}

/// Security configuration for SDK operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable corpus signature verification
    pub verify_signatures: bool,
    /// Enable output encryption
    pub enable_encryption: bool,
    /// Audit logging configuration
    pub audit_config: AuditConfig,
    /// Key management settings
    pub key_config: KeyConfig,
}

/// Audit logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Audit log file path
    pub log_path: Option<PathBuf>,
    /// Log level for audit events
    pub log_level: AuditLevel,
    /// Retention period for audit logs
    pub retention_days: u32,
}

/// Audit logging levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditLevel {
    Error,
    Warn,
    Info,
    Debug,
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyConfig {
    /// Key storage path
    pub key_store_path: Option<PathBuf>,
    /// Key rotation interval in days
    pub rotation_interval_days: u32,
    /// Enable key derivation
    pub enable_key_derivation: bool,
}

/// Pipeline integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Enable tar integration
    pub enable_tar: bool,
    /// Enable SSH integration
    pub enable_ssh: bool,
    /// Enable cloud CLI integration
    pub enable_cloud_cli: bool,
    /// Custom pipeline tools
    pub custom_tools: HashMap<String, ToolConfig>,
}

/// Configuration for custom pipeline tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Tool executable path
    pub executable: PathBuf,
    /// Command line arguments template
    pub args_template: Vec<String>,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
    /// Working directory
    pub working_dir: Option<PathBuf>,
}

impl Default for SDKConfig {
    fn default() -> Self {
        Self {
            chunk_config: ChunkConfig::default(),
            corpus_repositories: Vec::new(),
            compression_level: 19,
            enable_metrics: true,
            api_version: SDK_API_VERSION.to_string(),
            timeout_config: TimeoutConfig::default(),
            stream_config: StreamConfig::default(),
            security_config: SecurityConfig::default(),
            pipeline_config: PipelineConfig::default(),
        }
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            compression_timeout: Duration::from_secs(300),  // 5 minutes
            decompression_timeout: Duration::from_secs(180), // 3 minutes
            corpus_fetch_timeout: Duration::from_secs(60),   // 1 minute
            stream_timeout: Duration::from_secs(30),         // 30 seconds
        }
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024,           // 64KB buffer
            progressive_compression: true,
            max_memory_usage: 256 * 1024 * 1024, // 256MB
            enable_backpressure: true,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            verify_signatures: true,
            enable_encryption: false,
            audit_config: AuditConfig::default(),
            key_config: KeyConfig::default(),
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_path: None,
            log_level: AuditLevel::Info,
            retention_days: 90,
        }
    }
}

impl Default for KeyConfig {
    fn default() -> Self {
        Self {
            key_store_path: None,
            rotation_interval_days: 90,
            enable_key_derivation: true,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_tar: true,
            enable_ssh: true,
            enable_cloud_cli: true,
            custom_tools: HashMap::new(),
        }
    }
}

/// Compression operation result with detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionResult {
    /// Success status
    pub success: bool,
    /// Input size in bytes
    pub input_size: u64,
    /// Output size in bytes
    pub output_size: u64,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Corpus hit rate
    pub corpus_hit_rate: f64,
    /// Processing time
    pub processing_time: Duration,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Detailed metrics
    pub metrics: CompressionMetrics,
}

/// Detailed compression metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetrics {
    /// Number of chunks processed
    pub chunks_processed: u64,
    /// Number of corpus hits
    pub corpus_hits: u64,
    /// Residual data size
    pub residual_size: u64,
    /// Average chunk size
    pub avg_chunk_size: f64,
    /// Hash computation time
    pub hash_time: Duration,
    /// Corpus lookup time
    pub lookup_time: Duration,
    /// Serialization time
    pub serialization_time: Duration,
}

/// Stream processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStats {
    /// Bytes processed
    pub bytes_processed: u64,
    /// Processing rate in bytes/second
    pub processing_rate: f64,
    /// Buffer utilization percentage
    pub buffer_utilization: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Number of backpressure events
    pub backpressure_events: u64,
}

/// Main SDK interface for enterprise integration
pub struct ReductoSDK {
    /// SDK configuration
    config: Arc<RwLock<SDKConfig>>,
    /// Compressor instance
    compressor: Arc<Mutex<Compressor>>,
    /// Ecosystem decompressor
    decompressor: Arc<Mutex<EcosystemDecompressor>>,
    /// Metrics collector
    #[cfg(feature = "metrics")]
    metrics: Option<Arc<Mutex<EnterpriseMetricsCollector>>>,
    #[cfg(not(feature = "metrics"))]
    metrics: Option<()>,
    /// Security manager
    security: Option<Arc<Mutex<EnterpriseSecurityManager>>>,
    /// Pipeline integrations
    pipeline_tools: Arc<RwLock<HashMap<String, Box<dyn PipelineTool + Send + Sync>>>>,
}

impl ReductoSDK {
    /// Create a new SDK instance with configuration
    pub async fn new(config: SDKConfig) -> Result<Self> {
        // Validate configuration
        Self::validate_config(&config)?;

        // Initialize compressor with a dummy corpus manager for now
        // In a real implementation, this would be properly initialized
        let storage = crate::corpus_manager::StorageBackend::InMemory(
            crate::corpus_manager::InMemoryStorage::new()
        );
        let corpus_manager = Arc::new(std::sync::RwLock::new(
            crate::corpus_manager::EnterpriseCorpusManager::new(storage, config.chunk_config.clone())?
        ));
        let compressor = Compressor::new(config.chunk_config.clone(), corpus_manager)?;

        // Initialize ecosystem decompressor
        let decompressor = EcosystemDecompressor::new_async(
            config.corpus_repositories.clone(),
            None, // Will be set when needed
        ).await?;

        // Initialize metrics collector if enabled
        let metrics = if config.enable_metrics {
            #[cfg(feature = "metrics")]
            {
                let metrics_config = crate::metrics_collector::MetricsConfig::default();
                Some(Arc::new(Mutex::new(
                    EnterpriseMetricsCollector::new(metrics_config)?
                )))
            }
            #[cfg(not(feature = "metrics"))]
            {
                None
            }
        } else {
            None
        };

        // Initialize security manager if needed
        let security = if config.security_config.verify_signatures || config.security_config.enable_encryption {
            Some(Arc::new(Mutex::new(
                EnterpriseSecurityManager::new()?
            )))
        } else {
            None
        };

        // Initialize pipeline tools
        let mut pipeline_tools: HashMap<String, Box<dyn PipelineTool + Send + Sync>> = HashMap::new();
        
        if config.pipeline_config.enable_tar {
            pipeline_tools.insert("tar".to_string(), Box::new(TarIntegration::new()));
        }
        
        if config.pipeline_config.enable_ssh {
            pipeline_tools.insert("ssh".to_string(), Box::new(SshIntegration::new()));
        }
        
        if config.pipeline_config.enable_cloud_cli {
            pipeline_tools.insert("aws".to_string(), Box::new(AwsCliIntegration::new()));
            pipeline_tools.insert("gcp".to_string(), Box::new(GcpCliIntegration::new()));
            pipeline_tools.insert("azure".to_string(), Box::new(AzureCliIntegration::new()));
        }

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            compressor: Arc::new(Mutex::new(compressor)),
            decompressor: Arc::new(Mutex::new(decompressor)),
            metrics,
            security,
            pipeline_tools: Arc::new(RwLock::new(pipeline_tools)),
        })
    }

    /// Validate SDK configuration
    fn validate_config(config: &SDKConfig) -> Result<()> {
        // Validate API version compatibility
        if !Self::is_api_version_compatible(&config.api_version) {
            return Err(ReductoError::from(SDKError::VersionMismatch {
                client_version: config.api_version.clone(),
                server_version: SDK_API_VERSION.to_string(),
                required_version: SDK_API_VERSION.to_string(),
            }));
        }

        // Validate chunk configuration
        config.chunk_config.validate()?;

        // Validate compression level
        if config.compression_level < 1 || config.compression_level > 22 {
            return Err(ReductoError::ParameterOutOfRange {
                parameter: "compression_level".to_string(),
                value: config.compression_level as i64,
                min: 1,
                max: 22,
            });
        }

        // Validate stream configuration
        if config.stream_config.buffer_size == 0 {
            return Err(ReductoError::InvalidConfiguration {
                parameter: "stream_buffer_size".to_string(),
                value: config.stream_config.buffer_size.to_string(),
                reason: "Buffer size must be greater than zero".to_string(),
            });
        }

        if config.stream_config.buffer_size > MAX_STREAM_BUFFER_SIZE {
            return Err(ReductoError::ParameterOutOfRange {
                parameter: "stream_buffer_size".to_string(),
                value: config.stream_config.buffer_size as i64,
                min: 1,
                max: MAX_STREAM_BUFFER_SIZE as i64,
            });
        }

        Ok(())
    }

    /// Check if API version is compatible
    fn is_api_version_compatible(version: &str) -> bool {
        // For now, only support exact version match
        // In the future, implement semantic versioning compatibility
        version == SDK_API_VERSION
    }

    /// Stream-based compression for stdin/stdout operations
    pub async fn compress_stream<R, W>(
        &self,
        mut input: R,
        mut output: W,
        corpus_path: &Path,
    ) -> Result<CompressionResult>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let start_time = Instant::now();
        let config = self.config.read().await;
        
        // Set up timeout
        let compression_future = self.compress_stream_internal(input, output, corpus_path);
        let result = timeout(config.timeout_config.compression_timeout, compression_future).await;

        match result {
            Ok(compression_result) => compression_result,
            Err(_) => Err(ReductoError::from(SDKError::Timeout {
                operation: "stream_compression".to_string(),
                elapsed_ms: start_time.elapsed().as_millis() as u64,
                timeout_ms: config.timeout_config.compression_timeout.as_millis() as u64,
            })),
        }
    }

    /// Internal stream compression implementation
    async fn compress_stream_internal<R, W>(
        &self,
        mut input: R,
        mut output: W,
        corpus_path: &Path,
    ) -> Result<CompressionResult>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let start_time = Instant::now();
        let config = self.config.read().await.clone();

        // Create buffered readers/writers for efficiency
        let mut buffered_input = BufReader::with_capacity(
            config.stream_config.buffer_size,
            input
        );
        let mut buffered_output = BufWriter::with_capacity(
            config.stream_config.buffer_size,
            output
        );

        let mut total_input_size = 0u64;
        let mut total_output_size = 0u64;
        let mut chunks_processed = 0u64;
        let mut corpus_hits = 0u64;
        let mut residual_size = 0u64;

        // Read input data in chunks
        let mut buffer = vec![0u8; config.stream_config.buffer_size];
        let mut input_data = Vec::new();

        loop {
            let bytes_read = buffered_input.read(&mut buffer).await
                .map_err(|e| ReductoError::io_error("stream_read", e))?;
            
            if bytes_read == 0 {
                break; // End of input
            }

            input_data.extend_from_slice(&buffer[..bytes_read]);
            total_input_size += bytes_read as u64;

            // Check memory usage
            if input_data.len() > config.stream_config.max_memory_usage {
                return Err(ReductoError::MemoryLimitExceeded {
                    current_usage: input_data.len(),
                    limit_bytes: config.stream_config.max_memory_usage,
                });
            }
        }

        // Perform compression
        let mut compressor = self.compressor.lock().await;
        let instructions = compressor.compress(&input_data)?;
        drop(compressor); // Release lock

        // Calculate metrics
        for instruction in &instructions {
            chunks_processed += 1;
            match instruction {
                ReductoInstruction::Reference { .. } => {
                    corpus_hits += 1;
                }
                ReductoInstruction::Residual(data) => {
                    residual_size += data.len() as u64;
                }
            }
        }

        // Create header
        let corpus_id = uuid::Uuid::new_v4(); // TODO: Get from corpus
        let header = ReductoHeader::new(
            corpus_id,
            Vec::new(), // TODO: Add signature if security enabled
            config.chunk_config.clone(),
            blake3::hash(&input_data),
            config.compression_level,
        );

        // Serialize and compress output
        let serialized_header = bincode::serialize(&header)
            .map_err(|e| ReductoError::serialization_error("header", e))?;
        let serialized_instructions = bincode::serialize(&instructions)
            .map_err(|e| ReductoError::serialization_error("instructions", e))?;
        
        let compressed_instructions = zstd::encode_all(
            &serialized_instructions[..],
            config.compression_level as i32
        ).map_err(|e| ReductoError::CompressionFailed {
            algorithm: "zstd".to_string(),
            cause: e.to_string(),
        })?;

        // Write output
        buffered_output.write_all(&(serialized_header.len() as u32).to_le_bytes()).await
            .map_err(|e| ReductoError::io_error("write_header_length", e))?;
        buffered_output.write_all(&serialized_header).await
            .map_err(|e| ReductoError::io_error("write_header", e))?;
        buffered_output.write_all(&compressed_instructions).await
            .map_err(|e| ReductoError::io_error("write_instructions", e))?;
        
        buffered_output.flush().await
            .map_err(|e| ReductoError::io_error("flush_output", e))?;

        total_output_size = 4 + serialized_header.len() as u64 + compressed_instructions.len() as u64;

        let processing_time = start_time.elapsed();
        let compression_ratio = if total_input_size > 0 {
            total_output_size as f64 / total_input_size as f64
        } else {
            1.0
        };
        let corpus_hit_rate = if chunks_processed > 0 {
            corpus_hits as f64 / chunks_processed as f64
        } else {
            0.0
        };

        // Record metrics if enabled
        #[cfg(feature = "metrics")]
        if let Some(metrics) = &self.metrics {
            let mut metrics_collector = metrics.lock().await;
            // TODO: Record compression metrics
        }

        Ok(CompressionResult {
            success: true,
            input_size: total_input_size,
            output_size: total_output_size,
            compression_ratio,
            corpus_hit_rate,
            processing_time,
            error_message: None,
            metrics: CompressionMetrics {
                chunks_processed,
                corpus_hits,
                residual_size,
                avg_chunk_size: if chunks_processed > 0 {
                    total_input_size as f64 / chunks_processed as f64
                } else {
                    0.0
                },
                hash_time: Duration::from_millis(0), // TODO: Measure actual time
                lookup_time: Duration::from_millis(0), // TODO: Measure actual time
                serialization_time: Duration::from_millis(0), // TODO: Measure actual time
            },
        })
    }

    /// Stream-based decompression
    pub async fn decompress_stream<R, W>(
        &self,
        mut input: R,
        mut output: W,
    ) -> Result<DecompressionResult>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let start_time = Instant::now();
        let config = self.config.read().await;
        
        // Set up timeout
        let decompression_future = self.decompress_stream_internal(input, output);
        let result = timeout(config.timeout_config.decompression_timeout, decompression_future).await;

        match result {
            Ok(decompression_result) => decompression_result,
            Err(_) => Err(ReductoError::from(SDKError::Timeout {
                operation: "stream_decompression".to_string(),
                elapsed_ms: start_time.elapsed().as_millis() as u64,
                timeout_ms: config.timeout_config.decompression_timeout.as_millis() as u64,
            })),
        }
    }

    /// Internal stream decompression implementation
    async fn decompress_stream_internal<R, W>(
        &self,
        mut input: R,
        mut output: W,
    ) -> Result<DecompressionResult>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let start_time = Instant::now();
        let config = self.config.read().await.clone();

        // Create buffered readers/writers
        let mut buffered_input = BufReader::with_capacity(
            config.stream_config.buffer_size,
            input
        );
        let mut buffered_output = BufWriter::with_capacity(
            config.stream_config.buffer_size,
            output
        );

        // Read header length
        let mut header_len_bytes = [0u8; 4];
        buffered_input.read_exact(&mut header_len_bytes).await
            .map_err(|e| ReductoError::io_error("read_header_length", e))?;
        let header_len = u32::from_le_bytes(header_len_bytes) as usize;

        // Read header
        let mut header_bytes = vec![0u8; header_len];
        buffered_input.read_exact(&mut header_bytes).await
            .map_err(|e| ReductoError::io_error("read_header", e))?;
        
        let header: ReductoHeader = bincode::deserialize(&header_bytes)
            .map_err(|e| ReductoError::deserialization_error("header", e))?;

        // Validate header
        header.validate()?;

        // Read compressed instructions
        let mut compressed_instructions = Vec::new();
        buffered_input.read_to_end(&mut compressed_instructions).await
            .map_err(|e| ReductoError::io_error("read_instructions", e))?;

        // Decompress instructions
        let serialized_instructions = zstd::decode_all(&compressed_instructions[..])
            .map_err(|e| ReductoError::DecompressionFailed {
                algorithm: "zstd".to_string(),
                cause: e.to_string(),
            })?;

        let instructions: Vec<ReductoInstruction> = bincode::deserialize(&serialized_instructions)
            .map_err(|e| ReductoError::deserialization_error("instructions", e))?;

        // Perform decompression using ecosystem decompressor
        let mut decompressor = self.decompressor.lock().await;
        let result = decompressor.decompress_instructions(&instructions).await?;
        drop(decompressor);

        // Write output
        buffered_output.write_all(&result.data).await
            .map_err(|e| ReductoError::io_error("write_output", e))?;
        buffered_output.flush().await
            .map_err(|e| ReductoError::io_error("flush_output", e))?;

        // Verify integrity if enabled
        if config.security_config.verify_signatures && header.has_signature() {
            if !header.verify_integrity(&result.data) {
                return Err(ReductoError::IntegrityCheckFailed {
                    expected: format!("{:?}", header.integrity_hash),
                    calculated: format!("{:?}", blake3::hash(&result.data)),
                });
            }
        }

        Ok(result)
    }

    /// Get current SDK configuration
    pub async fn get_config(&self) -> SDKConfig {
        self.config.read().await.clone()
    }

    /// Update SDK configuration
    pub async fn update_config(&self, new_config: SDKConfig) -> Result<()> {
        Self::validate_config(&new_config)?;
        *self.config.write().await = new_config;
        Ok(())
    }

    /// Get available pipeline tools
    pub async fn get_pipeline_tools(&self) -> Vec<String> {
        self.pipeline_tools.read().await.keys().cloned().collect()
    }

    /// Add custom pipeline tool
    pub async fn add_pipeline_tool(&self, name: String, tool: Box<dyn PipelineTool + Send + Sync>) {
        self.pipeline_tools.write().await.insert(name, tool);
    }

    /// Create tar integration filter
    pub fn create_tar_filter(&self) -> TarFilter {
        TarFilter::new(self.clone())
    }

    /// Create SSH wrapper
    pub fn create_ssh_wrapper(&self) -> SshWrapper {
        SshWrapper::new(self.clone())
    }

    /// Create cloud CLI plugin
    pub fn create_cloud_cli_plugin(&self, provider: CloudProvider) -> CloudCliPlugin {
        CloudCliPlugin::new(self.clone(), provider)
    }
}

// Clone implementation for SDK (needed for pipeline tools)
impl Clone for ReductoSDK {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            compressor: Arc::clone(&self.compressor),
            decompressor: Arc::clone(&self.decompressor),
            #[cfg(feature = "metrics")]
            metrics: self.metrics.as_ref().map(Arc::clone),
            #[cfg(not(feature = "metrics"))]
            metrics: None,
            security: self.security.as_ref().map(Arc::clone),
            pipeline_tools: Arc::clone(&self.pipeline_tools),
        }
    }
}

/// Trait for pipeline tool integrations
pub trait PipelineTool {
    /// Get tool name
    fn name(&self) -> &str;
    
    /// Execute tool with Reducto integration
    fn execute(&self, args: &[String], sdk: &ReductoSDK) -> Result<String>;
    
    /// Get tool-specific help text
    fn help(&self) -> &str;
}

/// Tar integration for pipeline processing
pub struct TarIntegration;

impl TarIntegration {
    pub fn new() -> Self {
        Self
    }
}

impl PipelineTool for TarIntegration {
    fn name(&self) -> &str {
        "tar"
    }
    
    fn execute(&self, args: &[String], _sdk: &ReductoSDK) -> Result<String> {
        // TODO: Implement tar integration
        Ok(format!("Tar integration with args: {:?}", args))
    }
    
    fn help(&self) -> &str {
        "Tar archive integration with Reducto compression"
    }
}

/// SSH integration for remote compression
pub struct SshIntegration;

impl SshIntegration {
    pub fn new() -> Self {
        Self
    }
}

impl PipelineTool for SshIntegration {
    fn name(&self) -> &str {
        "ssh"
    }
    
    fn execute(&self, args: &[String], _sdk: &ReductoSDK) -> Result<String> {
        // TODO: Implement SSH integration
        Ok(format!("SSH integration with args: {:?}", args))
    }
    
    fn help(&self) -> &str {
        "SSH remote compression and transfer"
    }
}

/// AWS CLI integration
pub struct AwsCliIntegration;

impl AwsCliIntegration {
    pub fn new() -> Self {
        Self
    }
}

impl PipelineTool for AwsCliIntegration {
    fn name(&self) -> &str {
        "aws"
    }
    
    fn execute(&self, args: &[String], _sdk: &ReductoSDK) -> Result<String> {
        // TODO: Implement AWS CLI integration
        Ok(format!("AWS CLI integration with args: {:?}", args))
    }
    
    fn help(&self) -> &str {
        "AWS CLI integration for S3 and other services"
    }
}

/// GCP CLI integration
pub struct GcpCliIntegration;

impl GcpCliIntegration {
    pub fn new() -> Self {
        Self
    }
}

impl PipelineTool for GcpCliIntegration {
    fn name(&self) -> &str {
        "gcp"
    }
    
    fn execute(&self, args: &[String], _sdk: &ReductoSDK) -> Result<String> {
        // TODO: Implement GCP CLI integration
        Ok(format!("GCP CLI integration with args: {:?}", args))
    }
    
    fn help(&self) -> &str {
        "Google Cloud CLI integration"
    }
}

/// Azure CLI integration
pub struct AzureCliIntegration;

impl AzureCliIntegration {
    pub fn new() -> Self {
        Self
    }
}

impl PipelineTool for AzureCliIntegration {
    fn name(&self) -> &str {
        "azure"
    }
    
    fn execute(&self, args: &[String], _sdk: &ReductoSDK) -> Result<String> {
        // TODO: Implement Azure CLI integration
        Ok(format!("Azure CLI integration with args: {:?}", args))
    }
    
    fn help(&self) -> &str {
        "Azure CLI integration"
    }
}

/// Pipeline integration helpers
pub struct TarFilter {
    sdk: ReductoSDK,
}

impl TarFilter {
    pub fn new(sdk: ReductoSDK) -> Self {
        Self { sdk }
    }
    
    /// Process tar archive with Reducto compression
    pub async fn process_archive(&self, input_path: &Path, output_path: &Path) -> Result<CompressionResult> {
        // TODO: Implement tar archive processing
        todo!("Implement tar archive processing")
    }
}

/// SSH wrapper for remote operations
pub struct SshWrapper {
    sdk: ReductoSDK,
}

impl SshWrapper {
    pub fn new(sdk: ReductoSDK) -> Self {
        Self { sdk }
    }
    
    /// Execute remote compression via SSH
    pub async fn remote_compress(&self, host: &str, remote_path: &str, local_path: &Path) -> Result<CompressionResult> {
        // TODO: Implement SSH remote compression
        todo!("Implement SSH remote compression")
    }
}

/// Cloud provider types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudProvider {
    Aws,
    Gcp,
    Azure,
}

/// Cloud CLI plugin
pub struct CloudCliPlugin {
    sdk: ReductoSDK,
    provider: CloudProvider,
}

impl CloudCliPlugin {
    pub fn new(sdk: ReductoSDK, provider: CloudProvider) -> Self {
        Self { sdk, provider }
    }
    
    /// Upload compressed data to cloud storage
    pub async fn upload_compressed(&self, local_path: &Path, remote_path: &str) -> Result<String> {
        // TODO: Implement cloud upload with compression
        todo!("Implement cloud upload with compression")
    }
    
    /// Download and decompress from cloud storage
    pub async fn download_decompress(&self, remote_path: &str, local_path: &Path) -> Result<DecompressionResult> {
        // TODO: Implement cloud download with decompression
        todo!("Implement cloud download with decompression")
    }
}
// === C FFI Bindings for Multi-Language Support ===

/// C-compatible SDK configuration
#[repr(C)]
pub struct CSDKConfig {
    /// Target chunk size
    pub target_chunk_size: u32,
    /// Compression level (1-22)
    pub compression_level: u8,
    /// Enable metrics collection
    pub enable_metrics: u8, // 0 = false, 1 = true
    /// Operation timeout in seconds
    pub timeout_seconds: u32,
    /// Buffer size for streaming
    pub buffer_size: u32,
    /// Enable signature verification
    pub verify_signatures: u8, // 0 = false, 1 = true
}

/// C-compatible compression result
#[repr(C)]
pub struct CCompressionResult {
    /// Success status (0 = failure, 1 = success)
    pub success: u8,
    /// Input size in bytes
    pub input_size: u64,
    /// Output size in bytes
    pub output_size: u64,
    /// Compression ratio (fixed-point: value / 10000)
    pub compression_ratio: u32,
    /// Corpus hit rate (fixed-point: value / 10000)
    pub corpus_hit_rate: u32,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Error code (0 = no error)
    pub error_code: i32,
    /// Error message (null-terminated string, or null if no error)
    pub error_message: *const c_char,
}

/// C-compatible decompression result
#[repr(C)]
pub struct CDecompressionResult {
    /// Success status (0 = failure, 1 = success)
    pub success: u8,
    /// Output size in bytes
    pub output_size: u64,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Corpus source (0 = local, 1 = remote, 2 = fallback)
    pub corpus_source: u8,
    /// Error code (0 = no error)
    pub error_code: i32,
    /// Error message (null-terminated string, or null if no error)
    pub error_message: *const c_char,
}

/// C-compatible result type
#[repr(C)]
pub struct CResult {
    /// Success status (0 = failure, 1 = success)
    pub success: u8,
    /// Error code (0 = no error)
    pub error_code: i32,
    /// Error message (null-terminated string, or null if no error)
    pub error_message: *const c_char,
    /// Result data pointer (type depends on operation)
    pub data: *mut c_void,
}

/// Error codes for C FFI
#[repr(C)]
pub enum CErrorCode {
    Success = 0,
    InvalidInput = 1,
    CompressionFailed = 2,
    DecompressionFailed = 3,
    CorpusNotFound = 4,
    TimeoutError = 5,
    ConfigurationError = 6,
    MemoryError = 7,
    IoError = 8,
    SecurityError = 9,
    UnknownError = 99,
}

impl From<&ReductoError> for CErrorCode {
    fn from(error: &ReductoError) -> Self {
        match error {
            ReductoError::InputValidationFailed { .. } => CErrorCode::InvalidInput,
            ReductoError::CompressionFailed { .. } | ReductoError::Compression { .. } => CErrorCode::CompressionFailed,
            ReductoError::DecompressionFailed { .. } | ReductoError::Decompression { .. } => CErrorCode::DecompressionFailed,
            ReductoError::CorpusNotFound { .. } => CErrorCode::CorpusNotFound,
            ReductoError::OperationTimeout { .. } => CErrorCode::TimeoutError,
            ReductoError::InvalidConfiguration { .. } | ReductoError::ParameterOutOfRange { .. } => CErrorCode::ConfigurationError,
            ReductoError::MemoryAllocationFailed { .. } | ReductoError::MemoryLimitExceeded { .. } => CErrorCode::MemoryError,
            ReductoError::Io { .. } | ReductoError::FileNotFound { .. } => CErrorCode::IoError,
            _ => CErrorCode::UnknownError,
        }
    }
}

/// Global SDK instance for C FFI (thread-safe)
static mut GLOBAL_SDK: Option<Arc<Mutex<ReductoSDK>>> = None;
static SDK_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize the SDK with configuration
/// 
/// # Safety
/// This function is safe to call multiple times, but only the first call will initialize the SDK.
/// The config parameter must point to a valid CSDKConfig structure.
#[no_mangle]
pub unsafe extern "C" fn reducto_sdk_init(config: *const CSDKConfig) -> CResult {
    if config.is_null() {
        return CResult {
            success: 0,
            error_code: CErrorCode::InvalidInput as i32,
            error_message: create_c_string("Config pointer is null"),
            data: std::ptr::null_mut(),
        };
    }

    let c_config = &*config;
    
    // Convert C config to Rust config
    let rust_config = match c_config_to_rust_config(c_config) {
        Ok(config) => config,
        Err(e) => {
            return CResult {
                success: 0,
                error_code: CErrorCode::from(&e) as i32,
                error_message: create_c_string(&e.to_string()),
                data: std::ptr::null_mut(),
            };
        }
    };

    let mut init_result = CResult {
        success: 1,
        error_code: CErrorCode::Success as i32,
        error_message: std::ptr::null(),
        data: std::ptr::null_mut(),
    };

    SDK_INIT.call_once(|| {
        // Initialize async runtime for SDK
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                init_result = CResult {
                    success: 0,
                    error_code: CErrorCode::ConfigurationError as i32,
                    error_message: create_c_string(&format!("Failed to create async runtime: {}", e)),
                    data: std::ptr::null_mut(),
                };
                return;
            }
        };

        // Create SDK instance
        let sdk_result = rt.block_on(async {
            ReductoSDK::new(rust_config).await
        });

        match sdk_result {
            Ok(sdk) => {
                GLOBAL_SDK = Some(Arc::new(Mutex::new(sdk)));
            }
            Err(e) => {
                init_result = CResult {
                    success: 0,
                    error_code: CErrorCode::from(&e) as i32,
                    error_message: create_c_string(&e.to_string()),
                    data: std::ptr::null_mut(),
                };
            }
        }
    });

    init_result
}

/// Compress data from input file to output file
/// 
/// # Safety
/// The input_path, corpus_path, and output_path parameters must be valid null-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn reducto_compress_file(
    input_path: *const c_char,
    corpus_path: *const c_char,
    output_path: *const c_char,
) -> CCompressionResult {
    if input_path.is_null() || corpus_path.is_null() || output_path.is_null() {
        return CCompressionResult {
            success: 0,
            input_size: 0,
            output_size: 0,
            compression_ratio: 0,
            corpus_hit_rate: 0,
            processing_time_ms: 0,
            error_code: CErrorCode::InvalidInput as i32,
            error_message: create_c_string("One or more path parameters are null"),
        };
    }

    let input_path_str = match CStr::from_ptr(input_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            return CCompressionResult {
                success: 0,
                input_size: 0,
                output_size: 0,
                compression_ratio: 0,
                corpus_hit_rate: 0,
                processing_time_ms: 0,
                error_code: CErrorCode::InvalidInput as i32,
                error_message: create_c_string("Invalid input path encoding"),
            };
        }
    };

    let corpus_path_str = match CStr::from_ptr(corpus_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            return CCompressionResult {
                success: 0,
                input_size: 0,
                output_size: 0,
                compression_ratio: 0,
                corpus_hit_rate: 0,
                processing_time_ms: 0,
                error_code: CErrorCode::InvalidInput as i32,
                error_message: create_c_string("Invalid corpus path encoding"),
            };
        }
    };

    let output_path_str = match CStr::from_ptr(output_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            return CCompressionResult {
                success: 0,
                input_size: 0,
                output_size: 0,
                compression_ratio: 0,
                corpus_hit_rate: 0,
                processing_time_ms: 0,
                error_code: CErrorCode::InvalidInput as i32,
                error_message: create_c_string("Invalid output path encoding"),
            };
        }
    };

    // Get global SDK instance
    let sdk = match &GLOBAL_SDK {
        Some(sdk) => sdk,
        None => {
            return CCompressionResult {
                success: 0,
                input_size: 0,
                output_size: 0,
                compression_ratio: 0,
                corpus_hit_rate: 0,
                processing_time_ms: 0,
                error_code: CErrorCode::ConfigurationError as i32,
                error_message: create_c_string("SDK not initialized. Call reducto_sdk_init first."),
            };
        }
    };

    // Create async runtime for this operation
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            return CCompressionResult {
                success: 0,
                input_size: 0,
                output_size: 0,
                compression_ratio: 0,
                corpus_hit_rate: 0,
                processing_time_ms: 0,
                error_code: CErrorCode::ConfigurationError as i32,
                error_message: create_c_string(&format!("Failed to create runtime: {}", e)),
            };
        }
    };

    // Perform compression
    let result = rt.block_on(async {
        let sdk_guard = sdk.lock().await;
        
        // Open files
        let input_file = match tokio::fs::File::open(input_path_str).await {
            Ok(f) => f,
            Err(e) => return Err(ReductoError::io_error("open_input_file", e)),
        };

        let output_file = match tokio::fs::File::create(output_path_str).await {
            Ok(f) => f,
            Err(e) => return Err(ReductoError::io_error("create_output_file", e)),
        };

        // Perform stream compression
        sdk_guard.compress_stream(input_file, output_file, Path::new(corpus_path_str)).await
    });

    match result {
        Ok(compression_result) => CCompressionResult {
            success: if compression_result.success { 1 } else { 0 },
            input_size: compression_result.input_size,
            output_size: compression_result.output_size,
            compression_ratio: (compression_result.compression_ratio * 10000.0) as u32,
            corpus_hit_rate: (compression_result.corpus_hit_rate * 10000.0) as u32,
            processing_time_ms: compression_result.processing_time.as_millis() as u64,
            error_code: CErrorCode::Success as i32,
            error_message: std::ptr::null(),
        },
        Err(e) => CCompressionResult {
            success: 0,
            input_size: 0,
            output_size: 0,
            compression_ratio: 0,
            corpus_hit_rate: 0,
            processing_time_ms: 0,
            error_code: CErrorCode::from(&e) as i32,
            error_message: create_c_string(&e.to_string()),
        },
    }
}

/// Decompress data from input file to output file
/// 
/// # Safety
/// The input_path and output_path parameters must be valid null-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn reducto_decompress_file(
    input_path: *const c_char,
    output_path: *const c_char,
) -> CDecompressionResult {
    if input_path.is_null() || output_path.is_null() {
        return CDecompressionResult {
            success: 0,
            output_size: 0,
            processing_time_ms: 0,
            corpus_source: 0,
            error_code: CErrorCode::InvalidInput as i32,
            error_message: create_c_string("One or more path parameters are null"),
        };
    }

    let input_path_str = match CStr::from_ptr(input_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            return CDecompressionResult {
                success: 0,
                output_size: 0,
                processing_time_ms: 0,
                corpus_source: 0,
                error_code: CErrorCode::InvalidInput as i32,
                error_message: create_c_string("Invalid input path encoding"),
            };
        }
    };

    let output_path_str = match CStr::from_ptr(output_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            return CDecompressionResult {
                success: 0,
                output_size: 0,
                processing_time_ms: 0,
                corpus_source: 0,
                error_code: CErrorCode::InvalidInput as i32,
                error_message: create_c_string("Invalid output path encoding"),
            };
        }
    };

    // Get global SDK instance
    let sdk = match &GLOBAL_SDK {
        Some(sdk) => sdk,
        None => {
            return CDecompressionResult {
                success: 0,
                output_size: 0,
                processing_time_ms: 0,
                corpus_source: 0,
                error_code: CErrorCode::ConfigurationError as i32,
                error_message: create_c_string("SDK not initialized. Call reducto_sdk_init first."),
            };
        }
    };

    // Create async runtime for this operation
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            return CDecompressionResult {
                success: 0,
                output_size: 0,
                processing_time_ms: 0,
                corpus_source: 0,
                error_code: CErrorCode::ConfigurationError as i32,
                error_message: create_c_string(&format!("Failed to create runtime: {}", e)),
            };
        }
    };

    // Perform decompression
    let result = rt.block_on(async {
        let sdk_guard = sdk.lock().await;
        
        // Open files
        let input_file = match tokio::fs::File::open(input_path_str).await {
            Ok(f) => f,
            Err(e) => return Err(ReductoError::io_error("open_input_file", e)),
        };

        let output_file = match tokio::fs::File::create(output_path_str).await {
            Ok(f) => f,
            Err(e) => return Err(ReductoError::io_error("create_output_file", e)),
        };

        // Perform stream decompression
        sdk_guard.decompress_stream(input_file, output_file).await
    });

    match result {
        Ok(decompression_result) => {
            let corpus_source_code = match decompression_result.corpus_source {
                crate::ecosystem_decompressor::CorpusSource::LocalCache => 0,
                crate::ecosystem_decompressor::CorpusSource::Repository(_) => 1,
                crate::ecosystem_decompressor::CorpusSource::NotFound => 2,
            };

            CDecompressionResult {
                success: if decompression_result.success { 1 } else { 0 },
                output_size: decompression_result.data.len() as u64,
                processing_time_ms: decompression_result.metrics.processing_time.as_millis() as u64,
                corpus_source: corpus_source_code,
                error_code: CErrorCode::Success as i32,
                error_message: std::ptr::null(),
            }
        },
        Err(e) => CDecompressionResult {
            success: 0,
            output_size: 0,
            processing_time_ms: 0,
            corpus_source: 0,
            error_code: CErrorCode::from(&e) as i32,
            error_message: create_c_string(&e.to_string()),
        },
    }
}

/// Get SDK version information
/// 
/// # Safety
/// The returned string pointer is valid until the next call to this function.
#[no_mangle]
pub extern "C" fn reducto_get_version() -> *const c_char {
    static mut VERSION_CSTRING: Option<CString> = None;
    
    unsafe {
        if VERSION_CSTRING.is_none() {
            VERSION_CSTRING = Some(CString::new(SDK_API_VERSION).unwrap());
        }
        VERSION_CSTRING.as_ref().unwrap().as_ptr()
    }
}

/// Free memory allocated by the SDK
/// 
/// # Safety
/// The ptr parameter must be a pointer returned by a previous SDK function call.
/// After calling this function, the pointer becomes invalid and must not be used.
#[no_mangle]
pub unsafe extern "C" fn reducto_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}

/// Cleanup SDK resources
/// 
/// # Safety
/// This function should be called before program termination to properly cleanup resources.
/// After calling this function, all other SDK functions will fail until reducto_sdk_init is called again.
#[no_mangle]
pub unsafe extern "C" fn reducto_sdk_cleanup() -> CResult {
    GLOBAL_SDK = None;
    
    CResult {
        success: 1,
        error_code: CErrorCode::Success as i32,
        error_message: std::ptr::null(),
        data: std::ptr::null_mut(),
    }
}

// === Helper Functions for C FFI ===

/// Convert C configuration to Rust configuration
fn c_config_to_rust_config(c_config: &CSDKConfig) -> Result<SDKConfig> {
    let chunk_config = ChunkConfig::new(c_config.target_chunk_size as usize)?;
    
    let mut rust_config = SDKConfig {
        chunk_config,
        compression_level: c_config.compression_level,
        enable_metrics: c_config.enable_metrics != 0,
        ..Default::default()
    };

    // Set timeout configuration
    rust_config.timeout_config.compression_timeout = Duration::from_secs(c_config.timeout_seconds as u64);
    rust_config.timeout_config.decompression_timeout = Duration::from_secs(c_config.timeout_seconds as u64);
    
    // Set stream configuration
    rust_config.stream_config.buffer_size = c_config.buffer_size as usize;
    
    // Set security configuration
    rust_config.security_config.verify_signatures = c_config.verify_signatures != 0;

    Ok(rust_config)
}

/// Create a C string from a Rust string (for error messages)
/// 
/// # Safety
/// The returned pointer must be freed with reducto_free_string
fn create_c_string(s: &str) -> *const c_char {
    match CString::new(s) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => {
            // Fallback for strings with null bytes
            match CString::new("Error message contains null bytes") {
                Ok(c_string) => c_string.into_raw(),
                Err(_) => std::ptr::null(),
            }
        }
    }
}

// === Language Binding Examples ===

/// Python binding example (using ctypes)
/// 
/// ```python
/// import ctypes
/// from ctypes import Structure, c_uint32, c_uint8, c_uint64, c_int32, c_char_p, POINTER
/// 
/// # Load the library
/// lib = ctypes.CDLL('./target/release/libreducto_mode_3.so')
/// 
/// # Define structures
/// class CSDKConfig(Structure):
///     _fields_ = [
///         ("target_chunk_size", c_uint32),
///         ("compression_level", c_uint8),
///         ("enable_metrics", c_uint8),
///         ("timeout_seconds", c_uint32),
///         ("buffer_size", c_uint32),
///         ("verify_signatures", c_uint8),
///     ]
/// 
/// class CCompressionResult(Structure):
///     _fields_ = [
///         ("success", c_uint8),
///         ("input_size", c_uint64),
///         ("output_size", c_uint64),
///         ("compression_ratio", c_uint32),
///         ("corpus_hit_rate", c_uint32),
///         ("processing_time_ms", c_uint64),
///         ("error_code", c_int32),
///         ("error_message", c_char_p),
///     ]
/// 
/// # Define function signatures
/// lib.reducto_sdk_init.argtypes = [POINTER(CSDKConfig)]
/// lib.reducto_sdk_init.restype = CResult
/// 
/// lib.reducto_compress_file.argtypes = [c_char_p, c_char_p, c_char_p]
/// lib.reducto_compress_file.restype = CCompressionResult
/// 
/// # Usage example
/// config = CSDKConfig(
///     target_chunk_size=8192,
///     compression_level=19,
///     enable_metrics=1,
///     timeout_seconds=300,
///     buffer_size=65536,
///     verify_signatures=1
/// )
/// 
/// result = lib.reducto_sdk_init(ctypes.byref(config))
/// if result.success:
///     compression_result = lib.reducto_compress_file(
///         b"input.txt",
///         b"corpus.bin", 
///         b"output.reducto"
///     )
///     print(f"Compression ratio: {compression_result.compression_ratio / 10000.0}")
/// ```

/// Go binding example (using cgo)
/// 
/// ```go
/// package main
/// 
/// /*
/// #cgo LDFLAGS: -L./target/release -lreducto_mode_3
/// #include <stdint.h>
/// 
/// typedef struct {
///     uint32_t target_chunk_size;
///     uint8_t compression_level;
///     uint8_t enable_metrics;
///     uint32_t timeout_seconds;
///     uint32_t buffer_size;
///     uint8_t verify_signatures;
/// } CSDKConfig;
/// 
/// typedef struct {
///     uint8_t success;
///     uint64_t input_size;
///     uint64_t output_size;
///     uint32_t compression_ratio;
///     uint32_t corpus_hit_rate;
///     uint64_t processing_time_ms;
///     int32_t error_code;
///     const char* error_message;
/// } CCompressionResult;
/// 
/// extern CResult reducto_sdk_init(const CSDKConfig* config);
/// extern CCompressionResult reducto_compress_file(const char* input_path, const char* corpus_path, const char* output_path);
/// extern void reducto_free_string(char* ptr);
/// extern CResult reducto_sdk_cleanup();
/// */
/// import "C"
/// import (
///     "fmt"
///     "unsafe"
/// )
/// 
/// func main() {
///     config := C.CSDKConfig{
///         target_chunk_size: 8192,
///         compression_level: 19,
///         enable_metrics: 1,
///         timeout_seconds: 300,
///         buffer_size: 65536,
///         verify_signatures: 1,
///     }
/// 
///     result := C.reducto_sdk_init(&config)
///     if result.success == 1 {
///         inputPath := C.CString("input.txt")
///         corpusPath := C.CString("corpus.bin")
///         outputPath := C.CString("output.reducto")
///         
///         defer C.free(unsafe.Pointer(inputPath))
///         defer C.free(unsafe.Pointer(corpusPath))
///         defer C.free(unsafe.Pointer(outputPath))
/// 
///         compressionResult := C.reducto_compress_file(inputPath, corpusPath, outputPath)
///         if compressionResult.success == 1 {
///             ratio := float64(compressionResult.compression_ratio) / 10000.0
///             fmt.Printf("Compression ratio: %.4f\n", ratio)
///         }
///     }
/// 
///     C.reducto_sdk_cleanup()
/// }
/// ```

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::io::Cursor;

    #[tokio::test]
    async fn test_sdk_creation() {
        let config = SDKConfig::default();
        let sdk = ReductoSDK::new(config).await;
        assert!(sdk.is_ok());
    }

    #[tokio::test]
    async fn test_sdk_config_validation() {
        let mut config = SDKConfig::default();
        config.compression_level = 25; // Invalid level
        
        let result = ReductoSDK::new(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_stream_compression() {
        let config = SDKConfig::default();
        let sdk = ReductoSDK::new(config).await.unwrap();
        
        let input_data = b"Hello, world! This is test data for compression.";
        let input_cursor = Cursor::new(input_data);
        let mut output_buffer = Vec::new();
        let output_cursor = Cursor::new(&mut output_buffer);
        
        let temp_dir = TempDir::new().unwrap();
        let corpus_path = temp_dir.path().join("test_corpus.bin");
        
        // Create a minimal corpus file for testing
        std::fs::write(&corpus_path, b"test corpus data").unwrap();
        
        let result = sdk.compress_stream(input_cursor, output_cursor, &corpus_path).await;
        assert!(result.is_ok());
        
        let compression_result = result.unwrap();
        assert!(compression_result.success);
        assert_eq!(compression_result.input_size, input_data.len() as u64);
    }

    #[tokio::test]
    async fn test_api_version_compatibility() {
        assert!(ReductoSDK::is_api_version_compatible(SDK_API_VERSION));
        assert!(!ReductoSDK::is_api_version_compatible("2.0.0"));
    }

    #[tokio::test]
    async fn test_pipeline_tools() {
        let config = SDKConfig::default();
        let sdk = ReductoSDK::new(config).await.unwrap();
        
        let tools = sdk.get_pipeline_tools().await;
        assert!(tools.contains(&"tar".to_string()));
        assert!(tools.contains(&"ssh".to_string()));
        assert!(tools.contains(&"aws".to_string()));
    }

    #[test]
    fn test_c_config_conversion() {
        let c_config = CSDKConfig {
            target_chunk_size: 8192,
            compression_level: 19,
            enable_metrics: 1,
            timeout_seconds: 300,
            buffer_size: 65536,
            verify_signatures: 1,
        };
        
        let rust_config = c_config_to_rust_config(&c_config);
        assert!(rust_config.is_ok());
        
        let config = rust_config.unwrap();
        assert_eq!(config.chunk_config.target_size, 8192);
        assert_eq!(config.compression_level, 19);
        assert!(config.enable_metrics);
        assert!(config.security_config.verify_signatures);
    }

    #[test]
    fn test_error_code_conversion() {
        let error = ReductoError::CorpusNotFound {
            corpus_id: "test".to_string(),
        };
        let c_error_code = CErrorCode::from(&error);
        assert_eq!(c_error_code as i32, CErrorCode::CorpusNotFound as i32);
    }

    #[test]
    fn test_c_string_creation() {
        let test_string = "Test error message";
        let c_string_ptr = create_c_string(test_string);
        assert!(!c_string_ptr.is_null());
        
        unsafe {
            let c_str = CStr::from_ptr(c_string_ptr);
            let rust_str = c_str.to_str().unwrap();
            assert_eq!(rust_str, test_string);
            
            // Clean up
            reducto_free_string(c_string_ptr as *mut c_char);
        }
    }
}