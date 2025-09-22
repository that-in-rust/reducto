//! Ecosystem-aware decompression with cold start resolution
//!
//! This module implements automatic corpus fetching, graceful degradation,
//! and comprehensive caching for enterprise deployment scenarios.

use crate::{
    error::{ReductoError, Result},
    types::{ReductoHeader, ReductoInstruction},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

#[cfg(feature = "sdk")]
use {
    lru::LruCache,
    reqwest::Client,
    std::sync::Mutex,
    tokio::time::timeout,
};

/// Configuration for corpus repositories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusRepository {
    /// Repository URL
    pub url: String,
    /// Optional authentication token
    pub auth_token: Option<String>,
    /// Repository priority (lower = higher priority)
    pub priority: u8,
    /// Request timeout
    pub timeout: Duration,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Retry backoff multiplier
    pub retry_backoff_ms: u64,
}

impl CorpusRepository {
    /// Create a new corpus repository configuration
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            auth_token: None,
            priority: 100,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_backoff_ms: 1000,
        }
    }

    /// Set authentication token
    pub fn with_auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Set priority (lower = higher priority)
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set retry configuration
    pub fn with_retries(mut self, max_retries: u32, backoff_ms: u64) -> Self {
        self.max_retries = max_retries;
        self.retry_backoff_ms = backoff_ms;
        self
    }
}

/// Source of corpus data for decompression
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorpusSource {
    /// Found in local cache
    LocalCache,
    /// Downloaded from repository
    Repository(String),
    /// Not found anywhere
    NotFound,
}

/// Decompression operation result with detailed metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecompressionResult {
    /// Whether decompression succeeded
    pub success: bool,
    /// Source of corpus data
    pub corpus_source: CorpusSource,
    /// Whether fallback compression was used
    pub fallback_used: bool,
    /// Decompressed data
    pub data: Vec<u8>,
    /// Decompression metrics
    pub metrics: DecompressionMetrics,
    /// Any warnings encountered
    pub warnings: Vec<String>,
}

/// Detailed metrics for decompression operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecompressionMetrics {
    /// Input size in bytes
    pub input_size: u64,
    /// Output size in bytes
    pub output_size: u64,
    /// Total processing time
    pub processing_time: Duration,
    /// Time spent fetching corpus
    pub corpus_fetch_time: Duration,
    /// Number of corpus accesses
    pub corpus_access_count: u64,
    /// Memory usage peak
    pub memory_usage_bytes: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// LRU cache entry for corpus data
#[derive(Debug, Clone)]
struct CachedCorpus {
    /// Corpus data
    data: Arc<Vec<u8>>,
    /// Cache timestamp
    cached_at: DateTime<Utc>,
    /// Access count
    access_count: u64,
    /// Last access time
    last_accessed: DateTime<Utc>,
}

/// Fallback compression interface for graceful degradation
pub trait StandardCompressor: Send + Sync {
    /// Decompress data using standard compression
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>>;
    
    /// Get compressor name for logging
    fn name(&self) -> &str;
}

/// Simple zstd fallback compressor
#[derive(Debug, Default)]
pub struct ZstdFallbackCompressor;

impl StandardCompressor for ZstdFallbackCompressor {
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::decode_all(data).map_err(|e| ReductoError::DecompressionFailed {
            algorithm: "zstd".to_string(),
            cause: e.to_string(),
        })
    }
    
    fn name(&self) -> &str {
        "zstd"
    }
}

/// Ecosystem-aware decompressor with automatic corpus resolution
pub struct EcosystemDecompressor {
    /// Configured corpus repositories
    repositories: Vec<CorpusRepository>,
    /// Local corpus cache directory
    cache_dir: PathBuf,
    /// Fallback compressor for graceful degradation
    fallback_compressor: Option<Box<dyn StandardCompressor>>,
    /// Configuration
    config: EcosystemConfig,
    
    #[cfg(feature = "sdk")]
    /// HTTP client for corpus fetching
    http_client: Client,
    
    #[cfg(feature = "sdk")]
    /// In-memory LRU cache
    memory_cache: Arc<Mutex<LruCache<String, CachedCorpus>>>,
}

/// Configuration for ecosystem decompressor
#[derive(Debug, Clone)]
pub struct EcosystemConfig {
    /// Maximum cache size in bytes
    pub max_cache_size_bytes: u64,
    /// Maximum number of cached corpora
    pub max_cached_corpora: usize,
    /// Cache entry TTL
    pub cache_ttl: Duration,
    /// Enable fallback compression
    pub enable_fallback: bool,
    /// Verify corpus integrity
    pub verify_integrity: bool,
    /// Parallel corpus fetching
    pub parallel_fetch: bool,
}

impl Default for EcosystemConfig {
    fn default() -> Self {
        Self {
            max_cache_size_bytes: 1024 * 1024 * 1024, // 1GB
            max_cached_corpora: 100,
            cache_ttl: Duration::from_secs(24 * 60 * 60), // 24 hours
            enable_fallback: true,
            verify_integrity: true,
            parallel_fetch: false,
        }
    }
}

impl EcosystemDecompressor {
    /// Create a new ecosystem decompressor
    pub fn new(
        repositories: Vec<CorpusRepository>,
        cache_dir: impl Into<PathBuf>,
    ) -> Result<Self> {
        Self::with_config(repositories, cache_dir, EcosystemConfig::default())
    }

    /// Create a new ecosystem decompressor (async version for SDK)
    pub async fn new_async(
        repositories: Vec<CorpusRepository>,
        cache_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let cache_dir = cache_dir.unwrap_or_else(|| {
            std::env::temp_dir().join("reducto_cache")
        });
        Self::with_config(repositories, cache_dir, EcosystemConfig::default())
    }

    /// Create a new ecosystem decompressor with custom configuration
    pub fn with_config(
        mut repositories: Vec<CorpusRepository>,
        cache_dir: impl Into<PathBuf>,
        config: EcosystemConfig,
    ) -> Result<Self> {
        let cache_dir = cache_dir.into();
        
        // Create cache directory if it doesn't exist
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            ReductoError::io_error("create cache directory", e)
        })?;

        // Sort repositories by priority
        repositories.sort_by_key(|r| r.priority);

        #[cfg(feature = "sdk")]
        let http_client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| ReductoError::InternalError {
                message: format!("Failed to create HTTP client: {}", e),
            })?;

        #[cfg(feature = "sdk")]
        let memory_cache = Arc::new(Mutex::new(
            LruCache::new(std::num::NonZeroUsize::new(config.max_cached_corpora).unwrap())
        ));

        Ok(Self {
            repositories,
            cache_dir,
            fallback_compressor: Some(Box::new(ZstdFallbackCompressor::default())),
            config,
            #[cfg(feature = "sdk")]
            http_client,
            #[cfg(feature = "sdk")]
            memory_cache,
        })
    }

    /// Set fallback compressor
    pub fn with_fallback_compressor(
        mut self,
        compressor: Box<dyn StandardCompressor>,
    ) -> Self {
        self.fallback_compressor = Some(compressor);
        self
    }

    /// Disable fallback compression
    pub fn without_fallback(mut self) -> Self {
        self.fallback_compressor = None;
        self
    }

    /// Decompress with automatic corpus resolution
    pub async fn decompress_with_resolution(
        &mut self,
        compressed_file: &Path,
        output_path: &Path,
    ) -> Result<DecompressionResult> {
        let start_time = Instant::now();
        let mut metrics = DecompressionMetrics {
            input_size: 0,
            output_size: 0,
            processing_time: Duration::ZERO,
            corpus_fetch_time: Duration::ZERO,
            corpus_access_count: 0,
            memory_usage_bytes: 0,
            cache_hit_rate: 0.0,
        };
        let mut warnings = Vec::new();

        // Read and parse the compressed file
        let compressed_data = std::fs::read(compressed_file).map_err(|e| {
            ReductoError::io_error("read compressed file", e)
        })?;
        
        metrics.input_size = compressed_data.len() as u64;

        // Parse header to get corpus ID
        let header = self.parse_header(&compressed_data)?;
        let corpus_id = header.corpus_id.to_string();

        // Try to get corpus data
        let corpus_fetch_start = Instant::now();
        let (corpus_data, corpus_source) = match self.get_corpus(&corpus_id).await {
            Ok((data, source)) => (data, source),
            Err(e) => {
                warnings.push(format!("Corpus fetch failed: {}", e));
                
                if self.config.enable_fallback && self.fallback_compressor.is_some() {
                    return self.fallback_decompress(&compressed_data, output_path, metrics, warnings).await;
                } else {
                    return Err(e);
                }
            }
        };
        
        metrics.corpus_fetch_time = corpus_fetch_start.elapsed();

        // Verify corpus integrity if enabled
        if self.config.verify_integrity && !header.corpus_signature.is_empty() {
            if let Err(e) = self.verify_corpus_integrity(&corpus_data, &header.corpus_signature) {
                warnings.push(format!("Corpus integrity verification failed: {}", e));
                
                if self.config.enable_fallback && self.fallback_compressor.is_some() {
                    return self.fallback_decompress(&compressed_data, output_path, metrics, warnings).await;
                } else {
                    return Err(e);
                }
            }
        }

        // Perform decompression
        let decompressed_data = self.decompress_with_corpus(
            &compressed_data,
            &corpus_data,
            &header,
            &mut metrics,
        )?;

        // Verify end-to-end integrity
        if self.config.verify_integrity {
            let calculated_hash = blake3::hash(&decompressed_data);
            if calculated_hash != header.integrity_hash {
                let error = ReductoError::IntegrityCheckFailed {
                    expected: hex::encode(header.integrity_hash.as_bytes()),
                    calculated: hex::encode(calculated_hash.as_bytes()),
                };
                
                warnings.push(format!("End-to-end integrity check failed: {}", error));
                
                if self.config.enable_fallback && self.fallback_compressor.is_some() {
                    return self.fallback_decompress(&compressed_data, output_path, metrics, warnings).await;
                } else {
                    return Err(error);
                }
            }
        }

        // Write output
        std::fs::write(output_path, &decompressed_data).map_err(|e| {
            ReductoError::io_error("write decompressed file", e)
        })?;

        metrics.output_size = decompressed_data.len() as u64;
        metrics.processing_time = start_time.elapsed();

        Ok(DecompressionResult {
            success: true,
            corpus_source,
            fallback_used: false,
            data: decompressed_data,
            metrics,
            warnings,
        })
    }

    /// Get corpus data from cache or repositories
    async fn get_corpus(&mut self, corpus_id: &str) -> Result<(Arc<Vec<u8>>, CorpusSource)> {
        // Check memory cache first
        #[cfg(feature = "sdk")]
        {
            if let Ok(mut cache) = self.memory_cache.lock() {
                if let Some(cached) = cache.get_mut(corpus_id) {
                    // Check if cache entry is still valid
                    let age = Utc::now().signed_duration_since(cached.cached_at);
                    if age.to_std().unwrap_or(Duration::MAX) < self.config.cache_ttl {
                        cached.access_count += 1;
                        cached.last_accessed = Utc::now();
                        return Ok((cached.data.clone(), CorpusSource::LocalCache));
                    } else {
                        // Remove expired entry
                        cache.pop(corpus_id);
                    }
                }
            }
        }

        // Check local file cache
        let cache_path = self.cache_dir.join(format!("{}.corpus", corpus_id));
        if cache_path.exists() {
            if let Ok(metadata) = std::fs::metadata(&cache_path) {
                if let Ok(modified) = metadata.modified() {
                    let age = modified.elapsed().unwrap_or(Duration::MAX);
                    if age < self.config.cache_ttl {
                        if let Ok(data) = std::fs::read(&cache_path) {
                            let data = Arc::new(data);
                            
                            // Update memory cache
                            #[cfg(feature = "sdk")]
                            {
                                if let Ok(mut cache) = self.memory_cache.lock() {
                                    cache.put(corpus_id.to_string(), CachedCorpus {
                                        data: data.clone(),
                                        cached_at: Utc::now(),
                                        access_count: 1,
                                        last_accessed: Utc::now(),
                                    });
                                }
                            }
                            
                            return Ok((data, CorpusSource::LocalCache));
                        }
                    }
                }
            }
        }

        // Fetch from repositories
        self.fetch_from_repositories(corpus_id).await
    }

    /// Fetch corpus from configured repositories
    #[cfg(feature = "sdk")]
    async fn fetch_from_repositories(&mut self, corpus_id: &str) -> Result<(Arc<Vec<u8>>, CorpusSource)> {
        let mut last_error = None;
        let repositories = self.repositories.clone(); // Clone to avoid borrowing issues

        for repository in &repositories {
            match self.fetch_from_repository(corpus_id, repository).await {
                Ok(data) => {
                    let data = Arc::new(data);
                    
                    // Cache the data
                    self.cache_corpus(corpus_id, &data).await?;
                    
                    return Ok((data, CorpusSource::Repository(repository.url.clone())));
                }
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| ReductoError::CorpusNotFound {
            corpus_id: corpus_id.to_string(),
        }))
    }

    /// Fetch corpus from a specific repository with retry logic
    #[cfg(feature = "sdk")]
    async fn fetch_from_repository(
        &self,
        corpus_id: &str,
        repository: &CorpusRepository,
    ) -> Result<Vec<u8>> {
        let url = format!("{}/corpus/{}", repository.url.trim_end_matches('/'), corpus_id);
        let mut attempts = 0;
        let mut backoff = repository.retry_backoff_ms;

        loop {
            attempts += 1;
            
            let mut request = self.http_client.get(&url);
            
            // Add authentication if configured
            if let Some(ref token) = repository.auth_token {
                request = request.bearer_auth(token);
            }

            let result = timeout(repository.timeout, request.send()).await;
            
            match result {
                Ok(Ok(response)) => {
                    if response.status().is_success() {
                        let data = response.bytes().await.map_err(|e| {
                            ReductoError::InternalError {
                                message: format!("Failed to read response body: {}", e),
                            }
                        })?;
                        return Ok(data.to_vec());
                    } else {
                        let status = response.status();
                        let error = ReductoError::InternalError {
                            message: format!("HTTP {} from {}", status, url),
                        };
                        
                        if attempts >= repository.max_retries || status.is_client_error() {
                            return Err(error);
                        }
                        
                        // Retry on server errors
                    }
                }
                Ok(Err(e)) => {
                    let error = ReductoError::InternalError {
                        message: format!("HTTP request failed: {}", e),
                    };
                    
                    if attempts >= repository.max_retries {
                        return Err(error);
                    }
                }
                Err(_) => {
                    let error = ReductoError::OperationTimeout {
                        operation: format!("fetch corpus from {}", url),
                        timeout_seconds: repository.timeout.as_secs(),
                    };
                    
                    if attempts >= repository.max_retries {
                        return Err(error);
                    }
                }
            }

            // Wait before retry
            tokio::time::sleep(Duration::from_millis(backoff)).await;
            backoff = (backoff * 2).min(30000); // Cap at 30 seconds
        }
    }

    /// Fallback implementation when SDK feature is not enabled
    #[cfg(not(feature = "sdk"))]
    async fn fetch_from_repositories(&mut self, corpus_id: &str) -> Result<(Arc<Vec<u8>>, CorpusSource)> {
        Err(ReductoError::UnimplementedFeature {
            feature: "HTTP corpus fetching requires 'sdk' feature".to_string(),
        })
    }

    /// Cache corpus data locally
    async fn cache_corpus(&mut self, corpus_id: &str, data: &Arc<Vec<u8>>) -> Result<()> {
        // Write to file cache
        let cache_path = self.cache_dir.join(format!("{}.corpus", corpus_id));
        std::fs::write(&cache_path, data.as_ref()).map_err(|e| {
            ReductoError::io_error("write corpus cache", e)
        })?;

        // Update memory cache
        #[cfg(feature = "sdk")]
        {
            if let Ok(mut cache) = self.memory_cache.lock() {
                cache.put(corpus_id.to_string(), CachedCorpus {
                    data: data.clone(),
                    cached_at: Utc::now(),
                    access_count: 1,
                    last_accessed: Utc::now(),
                });
            }
        }

        Ok(())
    }

    /// Parse header from compressed data
    fn parse_header(&self, data: &[u8]) -> Result<ReductoHeader> {
        if data.len() < 4 {
            return Err(ReductoError::InvalidFormat {
                reason: "File too small to contain header".to_string(),
            });
        }

        // Read header length
        let header_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        
        if data.len() < 4 + header_len {
            return Err(ReductoError::InvalidFormat {
                reason: format!("File too small for header length {}", header_len),
            });
        }

        // Deserialize header
        let header_data = &data[4..4 + header_len];
        bincode::deserialize(header_data).map_err(|e| {
            ReductoError::deserialization_error("header", e)
        })
    }

    /// Verify corpus integrity using signature
    fn verify_corpus_integrity(&self, _corpus_data: &[u8], _signature: &[u8]) -> Result<()> {
        // TODO: Implement cryptographic signature verification
        // This would use the security module when available
        Ok(())
    }

    /// Decompress data using corpus
    fn decompress_with_corpus(
        &self,
        compressed_data: &[u8],
        corpus_data: &[u8],
        header: &ReductoHeader,
        metrics: &mut DecompressionMetrics,
    ) -> Result<Vec<u8>> {
        // Parse header length
        let header_len = u32::from_le_bytes([
            compressed_data[0],
            compressed_data[1], 
            compressed_data[2],
            compressed_data[3]
        ]) as usize;

        // Extract compressed instruction stream
        let compressed_instructions = &compressed_data[4 + header_len..];
        
        // Decompress instruction stream
        let instruction_data = zstd::decode_all(compressed_instructions).map_err(|e| {
            ReductoError::DecompressionFailed {
                algorithm: "zstd".to_string(),
                cause: e.to_string(),
            }
        })?;

        // Deserialize instructions
        let instructions: Vec<ReductoInstruction> = bincode::deserialize(&instruction_data)
            .map_err(|e| ReductoError::deserialization_error("instructions", e))?;

        // Reconstruct data
        let mut output = Vec::new();
        
        for instruction in instructions {
            match instruction {
                ReductoInstruction::Reference { offset, size } => {
                    metrics.corpus_access_count += 1;
                    
                    let end_offset = offset + size as u64;
                    if end_offset > corpus_data.len() as u64 {
                        return Err(ReductoError::InvalidBlockReference {
                            offset,
                            max_offset: corpus_data.len() as u64,
                        });
                    }
                    
                    output.extend_from_slice(&corpus_data[offset as usize..end_offset as usize]);
                }
                ReductoInstruction::Residual(data) => {
                    output.extend_from_slice(&data);
                }
            }
        }

        Ok(output)
    }

    /// Fallback decompression using standard compression
    async fn fallback_decompress(
        &self,
        compressed_data: &[u8],
        output_path: &Path,
        mut metrics: DecompressionMetrics,
        mut warnings: Vec<String>,
    ) -> Result<DecompressionResult> {
        if let Some(ref compressor) = self.fallback_compressor {
            warnings.push(format!("Using fallback compression: {}", compressor.name()));
            
            let decompressed_data = compressor.decompress(compressed_data)?;
            
            std::fs::write(output_path, &decompressed_data).map_err(|e| {
                ReductoError::io_error("write fallback decompressed file", e)
            })?;

            metrics.output_size = decompressed_data.len() as u64;
            
            Ok(DecompressionResult {
                success: true,
                corpus_source: CorpusSource::NotFound,
                fallback_used: true,
                data: decompressed_data,
                metrics,
                warnings,
            })
        } else {
            Err(ReductoError::CorpusNotFound {
                corpus_id: "unknown".to_string(),
            })
        }
    }

    /// Clear cache entries older than TTL
    pub async fn cleanup_cache(&mut self) -> Result<u64> {
        let mut cleaned_bytes = 0u64;
        let now = Utc::now();

        // Clean memory cache
        #[cfg(feature = "sdk")]
        {
            if let Ok(mut cache) = self.memory_cache.lock() {
                let mut _to_remove: Vec<String> = Vec::new();
                
                // Note: LruCache doesn't provide iteration, so we can't clean expired entries
                // This is a limitation of the current implementation
                // In a production system, we might use a different cache implementation
            }
        }

        // Clean file cache
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        let age = modified.elapsed().unwrap_or(Duration::ZERO);
                        if age > self.config.cache_ttl {
                            cleaned_bytes += metadata.len();
                            let _ = std::fs::remove_file(entry.path());
                        }
                    }
                }
            }
        }

        Ok(cleaned_bytes)
    }

    /// Get repository configuration (for testing)
    pub fn repositories(&self) -> &[CorpusRepository] {
        &self.repositories
    }

    /// Get configuration (for testing)
    pub fn config(&self) -> &EcosystemConfig {
        &self.config
    }

    /// Get corpus for testing
    pub async fn get_corpus_for_test(&mut self, corpus_id: &str) -> Result<(Arc<Vec<u8>>, CorpusSource)> {
        self.get_corpus(corpus_id).await
    }

    /// Check if fallback compressor is available (for testing)
    pub fn has_fallback_compressor(&self) -> bool {
        self.fallback_compressor.is_some()
    }

    /// Decompress using pre-parsed instructions (for SDK integration)
    pub async fn decompress_instructions(&mut self, instructions: &[ReductoInstruction]) -> Result<DecompressionResult> {
        let start_time = Instant::now();
        let mut metrics = DecompressionMetrics {
            input_size: 0,
            output_size: 0,
            processing_time: Duration::ZERO,
            corpus_fetch_time: Duration::ZERO,
            corpus_access_count: 0,
            memory_usage_bytes: 0,
            cache_hit_rate: 0.0,
        };
        let warnings = Vec::new();

        // For now, we'll assume we have access to the corpus
        // In a real implementation, we'd need to determine which corpus to use
        // based on the instructions or context
        let mut output = Vec::new();
        
        for instruction in instructions {
            match instruction {
                ReductoInstruction::Reference { offset: _, size: _ } => {
                    // TODO: This needs access to the actual corpus data
                    // For now, we'll return an error indicating corpus is needed
                    return Err(ReductoError::CorpusNotFound {
                        corpus_id: "unknown".to_string(),
                    });
                }
                ReductoInstruction::Residual(data) => {
                    output.extend_from_slice(data);
                }
            }
        }

        metrics.output_size = output.len() as u64;
        metrics.processing_time = start_time.elapsed();

        Ok(DecompressionResult {
            success: true,
            corpus_source: CorpusSource::NotFound,
            fallback_used: false,
            data: output,
            metrics,
            warnings,
        })
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let mut stats = CacheStats {
            memory_entries: 0,
            memory_size_bytes: 0,
            file_entries: 0,
            file_size_bytes: 0,
            hit_rate: 0.0,
        };

        // Memory cache stats
        #[cfg(feature = "sdk")]
        {
            if let Ok(cache) = self.memory_cache.lock() {
                stats.memory_entries = cache.len();
                // Note: LruCache doesn't provide size information
                // In production, we'd use a cache that tracks memory usage
            }
        }

        // File cache stats
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().and_then(|s| s.to_str()) == Some("corpus") {
                    stats.file_entries += 1;
                    if let Ok(metadata) = entry.metadata() {
                        stats.file_size_bytes += metadata.len();
                    }
                }
            }
        }

        stats
    }
}

/// Cache statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of entries in memory cache
    pub memory_entries: usize,
    /// Memory cache size in bytes
    pub memory_size_bytes: u64,
    /// Number of entries in file cache
    pub file_entries: usize,
    /// File cache size in bytes
    pub file_size_bytes: u64,
    /// Cache hit rate
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_corpus_repository_creation() {
        let repo = CorpusRepository::new("https://example.com/corpus")
            .with_auth_token("secret-token")
            .with_priority(10)
            .with_timeout(Duration::from_secs(60))
            .with_retries(5, 2000);

        assert_eq!(repo.url, "https://example.com/corpus");
        assert_eq!(repo.auth_token, Some("secret-token".to_string()));
        assert_eq!(repo.priority, 10);
        assert_eq!(repo.timeout, Duration::from_secs(60));
        assert_eq!(repo.max_retries, 5);
        assert_eq!(repo.retry_backoff_ms, 2000);
    }

    #[test]
    fn test_ecosystem_config_defaults() {
        let config = EcosystemConfig::default();
        assert_eq!(config.max_cache_size_bytes, 1024 * 1024 * 1024);
        assert_eq!(config.max_cached_corpora, 100);
        assert_eq!(config.cache_ttl, Duration::from_secs(24 * 60 * 60));
        assert!(config.enable_fallback);
        assert!(config.verify_integrity);
        assert!(!config.parallel_fetch);
    }

    #[tokio::test]
    async fn test_ecosystem_decompressor_creation() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![
            CorpusRepository::new("https://repo1.example.com"),
            CorpusRepository::new("https://repo2.example.com").with_priority(50),
        ];

        let decompressor = EcosystemDecompressor::new(repositories, temp_dir.path()).unwrap();
        
        // Repositories should be sorted by priority
        assert_eq!(decompressor.repositories[0].priority, 50);
        assert_eq!(decompressor.repositories[1].priority, 100);
    }

    #[test]
    fn test_zstd_fallback_compressor() {
        let compressor = ZstdFallbackCompressor::default();
        assert_eq!(compressor.name(), "zstd");

        // Test with valid zstd data
        let original_data = b"Hello, world! This is test data for compression.";
        let compressed = zstd::encode_all(&original_data[..], 3).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, original_data);
    }

    #[tokio::test]
    async fn test_cache_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![CorpusRepository::new("https://example.com")];
        
        let mut config = EcosystemConfig::default();
        config.cache_ttl = Duration::from_millis(100); // Very short TTL for testing
        
        let mut decompressor = EcosystemDecompressor::with_config(
            repositories,
            temp_dir.path(),
            config,
        ).unwrap();

        // Create a test cache file
        let cache_file = temp_dir.path().join("test-corpus.corpus");
        std::fs::write(&cache_file, b"test data").unwrap();

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Cleanup should remove the expired file
        let cleaned_bytes = decompressor.cleanup_cache().await.unwrap();
        assert!(cleaned_bytes > 0);
        assert!(!cache_file.exists());
    }

    #[test]
    fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![CorpusRepository::new("https://example.com")];
        
        let decompressor = EcosystemDecompressor::new(repositories, temp_dir.path()).unwrap();

        // Create some test cache files
        std::fs::write(temp_dir.path().join("corpus1.corpus"), b"data1").unwrap();
        std::fs::write(temp_dir.path().join("corpus2.corpus"), b"data2").unwrap();
        std::fs::write(temp_dir.path().join("not-corpus.txt"), b"other").unwrap();

        let stats = decompressor.cache_stats();
        assert_eq!(stats.file_entries, 2); // Only .corpus files counted
        assert!(stats.file_size_bytes > 0);
    }

    #[test]
    fn test_decompression_metrics_serialization() {
        let metrics = DecompressionMetrics {
            input_size: 1000,
            output_size: 2000,
            processing_time: Duration::from_millis(500),
            corpus_fetch_time: Duration::from_millis(100),
            corpus_access_count: 10,
            memory_usage_bytes: 1024 * 1024,
            cache_hit_rate: 0.85,
        };

        let serialized = serde_json::to_string(&metrics).unwrap();
        let deserialized: DecompressionMetrics = serde_json::from_str(&serialized).unwrap();
        assert_eq!(metrics, deserialized);
    }

    #[test]
    fn test_corpus_source_serialization() {
        let sources = vec![
            CorpusSource::LocalCache,
            CorpusSource::Repository("https://example.com".to_string()),
            CorpusSource::NotFound,
        ];

        for source in sources {
            let serialized = serde_json::to_string(&source).unwrap();
            let deserialized: CorpusSource = serde_json::from_str(&serialized).unwrap();
            assert_eq!(source, deserialized);
        }
    }
}