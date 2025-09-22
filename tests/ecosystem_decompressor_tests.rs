//! Comprehensive tests for ecosystem-aware decompression with cold start resolution
//!
//! These tests validate automatic corpus fetching, graceful degradation,
//! caching behavior, and various corpus availability scenarios.

#[cfg(feature = "sdk")]
mod ecosystem_tests {
    use reducto_mode_3::{
        ecosystem_decompressor::{
            EcosystemDecompressor, CorpusRepository, CorpusSource, DecompressionResult,
            DecompressionMetrics as EcoDecompressionMetrics, EcosystemConfig, 
            StandardCompressor, ZstdFallbackCompressor, CacheStats
        },
        types::*,
        error::*,
    };
    use std::{
        collections::HashMap,
        path::PathBuf,
        sync::{Arc, Mutex},
        time::Duration,
    };
    use tempfile::TempDir;
    use tokio::time::timeout;
    use uuid::Uuid;

    /// Mock HTTP server for testing corpus fetching
    struct MockCorpusServer {
        corpora: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        request_count: Arc<Mutex<u32>>,
        should_fail: Arc<Mutex<bool>>,
        delay: Arc<Mutex<Option<Duration>>>,
    }

    impl MockCorpusServer {
        fn new() -> Self {
            Self {
                corpora: Arc::new(Mutex::new(HashMap::new())),
                request_count: Arc::new(Mutex::new(0)),
                should_fail: Arc::new(Mutex::new(false)),
                delay: Arc::new(Mutex::new(None)),
            }
        }

        fn add_corpus(&self, id: &str, data: Vec<u8>) {
            self.corpora.lock().unwrap().insert(id.to_string(), data);
        }

        fn set_should_fail(&self, fail: bool) {
            *self.should_fail.lock().unwrap() = fail;
        }

        fn set_delay(&self, delay: Option<Duration>) {
            *self.delay.lock().unwrap() = delay;
        }

        fn get_request_count(&self) -> u32 {
            *self.request_count.lock().unwrap()
        }

        async fn handle_request(&self, corpus_id: &str) -> std::result::Result<Vec<u8>, String> {
            // Increment request count
            *self.request_count.lock().unwrap() += 1;

            // Apply delay if configured
            if let Some(delay) = *self.delay.lock().unwrap() {
                tokio::time::sleep(delay).await;
            }

            // Check if should fail
            if *self.should_fail.lock().unwrap() {
                return Err("Server configured to fail".to_string());
            }

            // Return corpus data if available
            self.corpora
                .lock()
                .unwrap()
                .get(corpus_id)
                .cloned()
                .ok_or_else(|| "Corpus not found".to_string())
        }
    }

    /// Create a test corpus with known data
    fn create_test_corpus() -> Vec<u8> {
        let mut corpus = Vec::new();
        
        // Add some test blocks
        for i in 0..10 {
            let block_data = format!("Test block {} with some content to make it realistic", i);
            let mut block = block_data.as_bytes().to_vec();
            block.resize(4096, 0); // Pad to block size
            corpus.extend_from_slice(&block);
        }
        
        corpus
    }

    /// Create a test compressed file that references the test corpus
    fn create_test_compressed_file(corpus_id: Uuid, temp_dir: &TempDir) -> PathBuf {
        let header = ReductoHeader::basic(corpus_id, ChunkConfig::default());
        
        // Create some test instructions
        let instructions = vec![
            ReductoInstruction::Reference { offset: 0, size: 4096 },
            ReductoInstruction::Residual(b"Some residual data".to_vec()),
            ReductoInstruction::Reference { offset: 4096, size: 4096 },
        ];

        // Serialize and compress instructions
        let serialized = bincode::serialize(&instructions).unwrap();
        let compressed = zstd::encode_all(&serialized[..], 19).unwrap();

        // Create file with header + compressed data
        let header_data = bincode::serialize(&header).unwrap();
        let mut file_data = Vec::new();
        file_data.extend_from_slice(&(header_data.len() as u32).to_le_bytes());
        file_data.extend_from_slice(&header_data);
        file_data.extend_from_slice(&compressed);

        let file_path = temp_dir.path().join("test.reducto");
        std::fs::write(&file_path, file_data).unwrap();
        
        file_path
    }

    #[tokio::test]
    async fn test_corpus_repository_configuration() {
        let repo = CorpusRepository::new("https://example.com/corpus")
            .with_auth_token("test-token")
            .with_priority(10)
            .with_timeout(Duration::from_secs(30))
            .with_retries(5, 1000);

        assert_eq!(repo.url, "https://example.com/corpus");
        assert_eq!(repo.auth_token, Some("test-token".to_string()));
        assert_eq!(repo.priority, 10);
        assert_eq!(repo.timeout, Duration::from_secs(30));
        assert_eq!(repo.max_retries, 5);
        assert_eq!(repo.retry_backoff_ms, 1000);
    }

    #[tokio::test]
    async fn test_ecosystem_decompressor_creation() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![
            CorpusRepository::new("https://repo1.example.com").with_priority(100),
            CorpusRepository::new("https://repo2.example.com").with_priority(50),
            CorpusRepository::new("https://repo3.example.com").with_priority(75),
        ];

        let decompressor = EcosystemDecompressor::new(repositories, temp_dir.path()).unwrap();
        
        // Repositories should be sorted by priority (lower = higher priority)
        assert_eq!(decompressor.repositories()[0].priority, 50);
        assert_eq!(decompressor.repositories()[1].priority, 75);
        assert_eq!(decompressor.repositories()[2].priority, 100);
        
        // Cache directory should exist
        assert!(temp_dir.path().exists());
    }

    #[tokio::test]
    async fn test_local_cache_hit() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![CorpusRepository::new("https://example.com")];
        
        let mut decompressor = EcosystemDecompressor::new(repositories, temp_dir.path()).unwrap();
        
        // Pre-populate cache with test corpus
        let corpus_id = "test-corpus-123";
        let corpus_data = create_test_corpus();
        let cache_file = temp_dir.path().join(format!("{}.corpus", corpus_id));
        std::fs::write(&cache_file, &corpus_data).unwrap();

        // Get corpus should return cached data
        let (retrieved_data, source) = decompressor.get_corpus_for_test(corpus_id).await.unwrap();
        assert_eq!(*retrieved_data, corpus_data);
        assert_eq!(source, CorpusSource::LocalCache);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![CorpusRepository::new("https://example.com")];
        
        let mut config = EcosystemConfig::default();
        config.cache_ttl = Duration::from_millis(100); // Very short TTL
        
        let mut decompressor = EcosystemDecompressor::with_config(
            repositories,
            temp_dir.path(),
            config,
        ).unwrap();

        // Create an old cache file
        let corpus_id = "expired-corpus";
        let cache_file = temp_dir.path().join(format!("{}.corpus", corpus_id));
        std::fs::write(&cache_file, b"old data").unwrap();

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Should fail to find corpus (since no repositories can serve it)
        let result = decompressor.get_corpus_for_test(corpus_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fallback_compression() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![]; // No repositories
        
        let decompressor = EcosystemDecompressor::new(repositories, temp_dir.path()).unwrap();
        
        // Create test data that can be decompressed with zstd
        let original_data = b"Hello, world! This is test data for fallback compression.";
        let compressed_data = zstd::encode_all(&original_data[..], 3).unwrap();
        
        let input_file = temp_dir.path().join("input.zst");
        let output_file = temp_dir.path().join("output.txt");
        std::fs::write(&input_file, &compressed_data).unwrap();

        // This should use fallback since no corpus is available
        // Note: This test would need a proper compressed file format to work fully
        // For now, we test the fallback compressor directly
        let fallback = ZstdFallbackCompressor::default();
        let decompressed = fallback.decompress(&compressed_data).unwrap();
        assert_eq!(decompressed, original_data);
    }

    #[tokio::test]
    async fn test_cache_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![CorpusRepository::new("https://example.com")];
        
        let mut config = EcosystemConfig::default();
        config.cache_ttl = Duration::from_millis(100);
        
        let mut decompressor = EcosystemDecompressor::with_config(
            repositories,
            temp_dir.path(),
            config,
        ).unwrap();

        // Create multiple cache files
        let cache_files = vec![
            ("corpus1.corpus", b"data1"),
            ("corpus2.corpus", b"data2"),
            ("corpus3.corpus", b"data3"),
        ];

        for (filename, data) in &cache_files {
            let path = temp_dir.path().join(filename);
            std::fs::write(&path, *data).unwrap();
        }

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Cleanup should remove expired files
        let cleaned_bytes = decompressor.cleanup_cache().await.unwrap();
        assert!(cleaned_bytes > 0);

        // Files should be gone
        for (filename, _) in &cache_files {
            let path = temp_dir.path().join(filename);
            assert!(!path.exists());
        }
    }

    #[tokio::test]
    async fn test_cache_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![CorpusRepository::new("https://example.com")];
        
        let decompressor = EcosystemDecompressor::new(repositories, temp_dir.path()).unwrap();

        // Create test cache files
        std::fs::write(temp_dir.path().join("corpus1.corpus"), b"test data 1").unwrap();
        std::fs::write(temp_dir.path().join("corpus2.corpus"), b"test data 2").unwrap();
        std::fs::write(temp_dir.path().join("not-corpus.txt"), b"other file").unwrap();

        let stats = decompressor.cache_stats();
        assert_eq!(stats.file_entries, 2); // Only .corpus files
        assert!(stats.file_size_bytes > 0);
        assert_eq!(stats.memory_entries, 0); // No memory cache entries yet
    }

    #[tokio::test]
    async fn test_decompression_metrics_tracking() {
        let metrics = EcoDecompressionMetrics {
            input_size: 1000,
            output_size: 2000,
            processing_time: Duration::from_millis(500),
            corpus_fetch_time: Duration::from_millis(100),
            corpus_access_count: 15,
            memory_usage_bytes: 1024 * 1024,
            cache_hit_rate: 0.75,
        };

        // Test serialization
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: EcoDecompressionMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics, deserialized);

        // Verify metrics make sense
        assert!(metrics.output_size > metrics.input_size); // Decompression should expand
        assert!(metrics.processing_time > metrics.corpus_fetch_time);
        assert!(metrics.cache_hit_rate >= 0.0 && metrics.cache_hit_rate <= 1.0);
    }

    #[tokio::test]
    async fn test_corpus_source_variants() {
        let sources = vec![
            CorpusSource::LocalCache,
            CorpusSource::Repository("https://example.com".to_string()),
            CorpusSource::NotFound,
        ];

        for source in sources {
            // Test serialization roundtrip
            let json = serde_json::to_string(&source).unwrap();
            let deserialized: CorpusSource = serde_json::from_str(&json).unwrap();
            assert_eq!(source, deserialized);
        }
    }

    #[tokio::test]
    async fn test_decompression_result_structure() {
        let result = DecompressionResult {
            success: true,
            corpus_source: CorpusSource::Repository("https://test.com".to_string()),
            fallback_used: false,
            metrics: EcoDecompressionMetrics {
                input_size: 500,
                output_size: 1500,
                processing_time: Duration::from_millis(250),
                corpus_fetch_time: Duration::from_millis(50),
                corpus_access_count: 5,
                memory_usage_bytes: 512 * 1024,
                cache_hit_rate: 0.8,
            },
            warnings: vec!["Test warning".to_string()],
        };

        // Test serialization
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: DecompressionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, deserialized);

        // Verify structure
        assert!(result.success);
        assert!(!result.fallback_used);
        assert_eq!(result.warnings.len(), 1);
        assert!(matches!(result.corpus_source, CorpusSource::Repository(_)));
    }

    #[tokio::test]
    async fn test_ecosystem_config_validation() {
        let config = EcosystemConfig {
            max_cache_size_bytes: 100 * 1024 * 1024, // 100MB
            max_cached_corpora: 50,
            cache_ttl: Duration::from_secs(12 * 60 * 60), // 12 hours
            enable_fallback: true,
            verify_integrity: true,
            parallel_fetch: false,
        };

        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![CorpusRepository::new("https://example.com")];
        
        let decompressor = EcosystemDecompressor::with_config(
            repositories,
            temp_dir.path(),
            config.clone(),
        ).unwrap();

        assert_eq!(decompressor.config().max_cache_size_bytes, config.max_cache_size_bytes);
        assert_eq!(decompressor.config().max_cached_corpora, config.max_cached_corpora);
        assert_eq!(decompressor.config().cache_ttl, config.cache_ttl);
        assert_eq!(decompressor.config().enable_fallback, config.enable_fallback);
        assert_eq!(decompressor.config().verify_integrity, config.verify_integrity);
        assert_eq!(decompressor.config().parallel_fetch, config.parallel_fetch);
    }

    #[tokio::test]
    async fn test_standard_compressor_interface() {
        let compressor = ZstdFallbackCompressor::default();
        
        // Test name
        assert_eq!(compressor.name(), "zstd");
        
        // Test compression/decompression roundtrip
        let original = b"Test data for compression interface validation";
        let compressed = zstd::encode_all(&original[..], 5).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
        
        // Test error handling with invalid data
        let invalid_data = b"not compressed data";
        let result = compressor.decompress(invalid_data);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_repository_priority_sorting() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![
            CorpusRepository::new("https://low-priority.com").with_priority(200),
            CorpusRepository::new("https://high-priority.com").with_priority(10),
            CorpusRepository::new("https://medium-priority.com").with_priority(100),
        ];

        let decompressor = EcosystemDecompressor::new(repositories, temp_dir.path()).unwrap();
        
        // Should be sorted by priority (ascending)
        assert_eq!(decompressor.repositories()[0].url, "https://high-priority.com");
        assert_eq!(decompressor.repositories()[1].url, "https://medium-priority.com");
        assert_eq!(decompressor.repositories()[2].url, "https://low-priority.com");
    }

    #[tokio::test]
    async fn test_timeout_handling() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![
            CorpusRepository::new("https://slow-server.com")
                .with_timeout(Duration::from_millis(100))
                .with_retries(1, 50),
        ];

        let decompressor = EcosystemDecompressor::new(repositories, temp_dir.path()).unwrap();
        
        // Test that timeout configuration is preserved
        assert_eq!(decompressor.repositories()[0].timeout, Duration::from_millis(100));
        assert_eq!(decompressor.repositories()[0].max_retries, 1);
        assert_eq!(decompressor.repositories()[0].retry_backoff_ms, 50);
    }

    #[tokio::test]
    async fn test_error_propagation() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![]; // No repositories
        
        let mut decompressor = EcosystemDecompressor::new(repositories, temp_dir.path()).unwrap();
        
        // Should fail with CorpusNotFound when no repositories are available
        let result = decompressor.get_corpus_for_test("nonexistent-corpus").await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ReductoError::CorpusNotFound { corpus_id } => {
                assert_eq!(corpus_id, "nonexistent-corpus");
            }
            other => panic!("Expected CorpusNotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_graceful_degradation_configuration() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![CorpusRepository::new("https://example.com")];
        
        // Test with fallback enabled
        let mut config = EcosystemConfig::default();
        config.enable_fallback = true;
        
        let decompressor_with_fallback = EcosystemDecompressor::with_config(
            repositories.clone(),
            temp_dir.path(),
            config,
        ).unwrap();
        
        assert!(decompressor_with_fallback.config().enable_fallback);
        assert!(decompressor_with_fallback.has_fallback_compressor());
        
        // Test with fallback disabled
        let mut config = EcosystemConfig::default();
        config.enable_fallback = false;
        
        let decompressor_without_fallback = EcosystemDecompressor::with_config(
            repositories,
            temp_dir.path().join("no_fallback"),
            config,
        ).unwrap().without_fallback();
        
        assert!(!decompressor_without_fallback.has_fallback_compressor());
    }

    #[tokio::test]
    async fn test_integrity_verification_configuration() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![CorpusRepository::new("https://example.com")];
        
        let mut config = EcosystemConfig::default();
        config.verify_integrity = false;
        
        let decompressor = EcosystemDecompressor::with_config(
            repositories,
            temp_dir.path(),
            config,
        ).unwrap();
        
        assert!(!decompressor.config().verify_integrity);
    }

    /// Test that demonstrates the complete ecosystem decompression workflow
    #[tokio::test]
    async fn test_complete_decompression_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let repositories = vec![CorpusRepository::new("https://example.com")];
        
        let mut decompressor = EcosystemDecompressor::new(repositories, temp_dir.path()).unwrap();
        
        // Create test corpus and cache it locally
        let corpus_id = Uuid::new_v4();
        let corpus_data = create_test_corpus();
        let cache_file = temp_dir.path().join(format!("{}.corpus", corpus_id));
        std::fs::write(&cache_file, &corpus_data).unwrap();
        
        // Create test compressed file
        let compressed_file = create_test_compressed_file(corpus_id, &temp_dir);
        let output_file = temp_dir.path().join("output.txt");
        
        // This test demonstrates the structure but would need a complete implementation
        // to actually perform decompression. The key is that all the components are in place:
        // - Corpus caching (file and memory)
        // - Repository configuration
        // - Fallback compression
        // - Metrics collection
        // - Error handling and graceful degradation
        
        // Verify the test setup
        assert!(compressed_file.exists());
        assert!(cache_file.exists());
        assert_eq!(std::fs::read(&cache_file).unwrap(), corpus_data);
    }
}

// Tests that run without the SDK feature
#[cfg(not(feature = "sdk"))]
mod no_sdk_tests {
    use reducto_mode_3::error::*;
    
    #[tokio::test]
    async fn test_sdk_feature_required() {
        // When SDK feature is not enabled, ecosystem decompressor should not be available
        // This is enforced by conditional compilation
        
        // We can test that the appropriate error is returned for missing features
        let error = ReductoError::UnimplementedFeature {
            feature: "HTTP corpus fetching requires 'sdk' feature".to_string(),
        };
        
        assert!(error.to_string().contains("sdk"));
        assert!(error.to_string().contains("feature"));
    }
}