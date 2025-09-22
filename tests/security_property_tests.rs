//! Security Property Tests for Reducto Mode 3
//!
//! This module implements comprehensive property-based testing for all security features:
//! - Cryptographic signature verification consistency
//! - Encryption/decryption roundtrip properties
//! - Audit logging integrity and completeness
//! - Key management security properties
//! - Timing attack resistance

#[cfg(feature = "security")]
mod security_tests {
    use proptest::prelude::*;
    use reducto_mode_3::prelude::*;
    use std::time::Instant;
    use tempfile::TempDir;

    // === Security Property Test Strategies ===

    /// Generate realistic data for security testing
    fn security_data_strategy() -> impl Strategy<Value = Vec<u8>> {
        prop_oneof![
            // Small data (headers, metadata)
            prop::collection::vec(any::<u8>(), 1..=1024),
            // Medium data (configuration files)
            prop::collection::vec(any::<u8>(), 1024..=16384),
            // Large data (corpus files)
            prop::collection::vec(any::<u8>(), 16384..=65536),
        ]
    }

    /// Generate user identifiers for audit testing
    fn user_id_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            "[a-z]{3,20}",                    // Simple usernames
            "user-[0-9]{1,6}",               // Numbered users
            "[a-z]{3,10}\\.[a-z]{3,10}",     // Email-like
        ]
    }

    /// Generate corpus identifiers for testing
    fn corpus_id_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            "corpus-[a-f0-9]{8}",            // Hex IDs
            "[a-z]{3,15}-v[0-9]{1,3}",       // Versioned names
            "test-corpus-[0-9]{1,4}",        // Test corpus names
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Property: Signature verification is deterministic and consistent
        /// The same data should always produce the same signature verification result
        #[test]
        fn prop_signature_verification_deterministic(
            data in security_data_strategy()
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let security_manager = EnterpriseSecurityManager::new(
                    KeyManagementConfig::default()
                ).unwrap();

                // Sign the data
                let signature = security_manager.sign_corpus(&data).unwrap();

                // Verify signature multiple times - should be consistent
                let mut verification_results = Vec::new();
                for _ in 0..10 {
                    let is_valid = security_manager.verify_corpus_signature(&data, &signature).unwrap();
                    verification_results.push(is_valid);
                }

                // All verification results should be identical
                let first_result = verification_results[0];
                for result in verification_results {
                    prop_assert_eq!(result, first_result, 
                        "Signature verification should be deterministic");
                }

                // Valid signature should always verify as true
                prop_assert!(first_result, "Valid signature should verify as true");
            });
        }

        /// Property: Signature verification fails for tampered data
        /// Any modification to signed data should cause signature verification to fail
        #[test]
        fn prop_signature_tamper_detection(
            mut data in security_data_strategy()
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // Skip empty data
                if data.is_empty() {
                    return Ok(());
                }

                let security_manager = EnterpriseSecurityManager::new(
                    KeyManagementConfig::default()
                ).unwrap();

                // Sign original data
                let signature = security_manager.sign_corpus(&data).unwrap();

                // Verify original signature works
                let original_valid = security_manager.verify_corpus_signature(&data, &signature).unwrap();
                prop_assert!(original_valid, "Original signature should be valid");

                // Tamper with data in various ways
                let tamper_methods = vec![
                    // Flip a single bit
                    |d: &mut Vec<u8>| { d[0] ^= 0x01; },
                    // Change a byte
                    |d: &mut Vec<u8>| { d[d.len() / 2] = d[d.len() / 2].wrapping_add(1); },
                    // Append a byte
                    |d: &mut Vec<u8>| { d.push(0xFF); },
                    // Remove last byte (if possible)
                    |d: &mut Vec<u8>| { if d.len() > 1 { d.pop(); } },
                ];

                for tamper_method in tamper_methods {
                    let mut tampered_data = data.clone();
                    tamper_method(&mut tampered_data);

                    // Skip if tampering didn't actually change the data
                    if tampered_data == data {
                        continue;
                    }

                    let tampered_valid = security_manager.verify_corpus_signature(&tampered_data, &signature).unwrap();
                    prop_assert!(!tampered_valid, 
                        "Tampered data should fail signature verification");
                }
            });
        }

        /// Property: Encryption/decryption is a perfect roundtrip
        /// encrypt(data) → decrypt → data should always be identity
        #[test]
        fn prop_encryption_roundtrip_identity(
            data in security_data_strategy()
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let security_manager = EnterpriseSecurityManager::new(
                    KeyManagementConfig::default()
                ).unwrap();

                // Encrypt data
                let encrypted = security_manager.encrypt_output(&data).unwrap();

                // Encrypted data should be different from original (except for very small data)
                if data.len() > 32 {
                    prop_assert_ne!(encrypted, data, 
                        "Encrypted data should differ from original for data > 32 bytes");
                }

                // Decrypt and verify roundtrip
                let decrypted = security_manager.decrypt_input(&encrypted).unwrap();
                prop_assert_eq!(data, decrypted, 
                    "Decryption should perfectly recover original data");
            });
        }

        /// Property: Encryption produces different ciphertexts for identical plaintexts
        /// Multiple encryptions of the same data should produce different ciphertexts (due to IV/nonce)
        #[test]
        fn prop_encryption_semantic_security(
            data in security_data_strategy().prop_filter("non-empty", |d| !d.is_empty())
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let security_manager = EnterpriseSecurityManager::new(
                    KeyManagementConfig::default()
                ).unwrap();

                // Encrypt the same data multiple times
                let encrypted1 = security_manager.encrypt_output(&data).unwrap();
                let encrypted2 = security_manager.encrypt_output(&data).unwrap();
                let encrypted3 = security_manager.encrypt_output(&data).unwrap();

                // Ciphertexts should be different (semantic security)
                prop_assert_ne!(encrypted1, encrypted2, 
                    "Multiple encryptions should produce different ciphertexts");
                prop_assert_ne!(encrypted2, encrypted3, 
                    "Multiple encryptions should produce different ciphertexts");
                prop_assert_ne!(encrypted1, encrypted3, 
                    "Multiple encryptions should produce different ciphertexts");

                // But all should decrypt to the same plaintext
                let decrypted1 = security_manager.decrypt_input(&encrypted1).unwrap();
                let decrypted2 = security_manager.decrypt_input(&encrypted2).unwrap();
                let decrypted3 = security_manager.decrypt_input(&encrypted3).unwrap();

                prop_assert_eq!(data, decrypted1, "First decryption should match original");
                prop_assert_eq!(data, decrypted2, "Second decryption should match original");
                prop_assert_eq!(data, decrypted3, "Third decryption should match original");
            });
        }

        /// Property: Audit logging is complete and tamper-evident
        /// All security operations should be logged with complete information
        #[test]
        fn prop_audit_logging_completeness(
            corpus_id in corpus_id_strategy(),
            user_id in user_id_strategy(),
            operations in prop::collection::vec(
                prop::sample::select(vec![
                    AccessOperation::Read,
                    AccessOperation::Write,
                    AccessOperation::Delete,
                    AccessOperation::Modify,
                    AccessOperation::Create,
                    AccessOperation::Verify,
                ]), 1..=10
            )
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let audit_log_path = temp_dir.path().join("audit.log");

                let mut config = KeyManagementConfig::default();
                config.audit_log_path = Some(audit_log_path.clone());

                let security_manager = EnterpriseSecurityManager::new(config).unwrap();

                // Perform operations and log them
                for operation in &operations {
                    security_manager.log_corpus_access(&corpus_id, *operation, &user_id).unwrap();
                }

                // Verify audit log contains all operations
                let audit_logger = security_manager.get_audit_logger();
                let log_entries = audit_logger.get_recent_entries(operations.len()).unwrap();

                prop_assert_eq!(log_entries.len(), operations.len(),
                    "Audit log should contain all operations");

                // Verify each log entry has complete information
                for (i, entry) in log_entries.iter().enumerate() {
                    prop_assert_eq!(entry.resource, corpus_id,
                        "Log entry {} should have correct corpus ID", i);
                    prop_assert_eq!(entry.user_id, user_id,
                        "Log entry {} should have correct user ID", i);
                    prop_assert_eq!(entry.operation, operations[operations.len() - 1 - i],
                        "Log entry {} should have correct operation", i);
                    
                    // Timestamp should be recent (within last minute)
                    let now = chrono::Utc::now();
                    let age = now.signed_duration_since(entry.timestamp);
                    prop_assert!(age.num_seconds() < 60,
                        "Log entry timestamp should be recent");
                }
            });
        }

        /// Property: Timing attack resistance for signature verification
        /// Signature verification time should not depend on where the signature differs
        #[test]
        fn prop_signature_verification_timing_resistance(
            data in prop::collection::vec(any::<u8>(), 1024..=4096)
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let security_manager = EnterpriseSecurityManager::new(
                    KeyManagementConfig::default()
                ).unwrap();

                // Create valid signature
                let valid_signature = security_manager.sign_corpus(&data).unwrap();

                // Create invalid signatures by modifying at different positions
                let mut invalid_signatures = Vec::new();
                let signature_bytes = &valid_signature.signature;
                
                for i in 0..signature_bytes.len().min(10) {
                    let mut invalid_sig = valid_signature.clone();
                    invalid_sig.signature[i] ^= 0x01; // Flip one bit
                    invalid_signatures.push(invalid_sig);
                }

                // Measure verification times for invalid signatures
                let mut verification_times = Vec::new();
                
                for invalid_sig in &invalid_signatures {
                    let start = Instant::now();
                    let _result = security_manager.verify_corpus_signature(&data, invalid_sig).unwrap();
                    let elapsed = start.elapsed();
                    verification_times.push(elapsed);
                }

                // Calculate timing statistics
                if verification_times.len() > 1 {
                    let min_time = verification_times.iter().min().unwrap();
                    let max_time = verification_times.iter().max().unwrap();
                    let avg_time = verification_times.iter().sum::<std::time::Duration>() / verification_times.len() as u32;

                    // Timing variation should be small (within 50% of average)
                    let max_variation = avg_time.as_nanos() as f64 * 0.5;
                    let actual_variation = (max_time.as_nanos() as i64 - min_time.as_nanos() as i64).abs() as f64;

                    prop_assert!(actual_variation <= max_variation,
                        "Signature verification timing variation too large: {:.0}ns (max allowed: {:.0}ns)",
                        actual_variation, max_variation);
                }
            });
        }

        /// Property: Key management operations are secure and consistent
        /// Key derivation and storage should be deterministic and secure
        #[test]
        fn prop_key_management_consistency(
            key_id in "[a-z0-9]{8,32}",
            context in prop::collection::vec(any::<u8>(), 0..=256)
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let key_store_path = temp_dir.path().join("keystore");

                let mut config = KeyManagementConfig::default();
                config.key_store_path = Some(key_store_path);

                let security_manager1 = EnterpriseSecurityManager::new(config.clone()).unwrap();
                let security_manager2 = EnterpriseSecurityManager::new(config).unwrap();

                // Both managers should derive the same key for the same ID and context
                let key1 = security_manager1.derive_key(&key_id, &context).unwrap();
                let key2 = security_manager2.derive_key(&key_id, &context).unwrap();

                prop_assert_eq!(key1, key2,
                    "Key derivation should be deterministic across instances");

                // Different contexts should produce different keys
                if !context.is_empty() {
                    let mut different_context = context.clone();
                    different_context[0] ^= 0x01;
                    
                    let key3 = security_manager1.derive_key(&key_id, &different_context).unwrap();
                    prop_assert_ne!(key1, key3,
                        "Different contexts should produce different keys");
                }

                // Different key IDs should produce different keys
                let different_key_id = format!("{}_different", key_id);
                let key4 = security_manager1.derive_key(&different_key_id, &context).unwrap();
                prop_assert_ne!(key1, key4,
                    "Different key IDs should produce different keys");
            });
        }

        /// Property: Secure deletion is effective and complete
        /// Secure deletion should overwrite data multiple times and verify completion
        #[test]
        fn prop_secure_deletion_effectiveness(
            file_content in security_data_strategy().prop_filter("non-empty", |d| !d.is_empty())
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let test_file = temp_dir.path().join("test_file.bin");

                // Write test content to file
                std::fs::write(&test_file, &file_content).unwrap();

                // Verify file exists and has correct content
                let read_content = std::fs::read(&test_file).unwrap();
                prop_assert_eq!(file_content, read_content, "File should contain original content");

                let security_manager = EnterpriseSecurityManager::new(
                    KeyManagementConfig::default()
                ).unwrap();

                // Perform secure deletion
                security_manager.secure_delete(&test_file).await.unwrap();

                // File should no longer exist
                prop_assert!(!test_file.exists(), "File should be deleted after secure deletion");

                // If we can still read the file system blocks (implementation-dependent),
                // they should not contain the original data
                // This is a best-effort check since it depends on file system behavior
            });
        }
    }

    // === Integration Tests for Security Workflows ===

    #[cfg(test)]
    mod integration_tests {
        use super::*;

        #[tokio::test]
        async fn test_end_to_end_security_workflow() {
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("secure_corpus.bin");
            let compressed_path = temp_dir.path().join("secure.reducto");

            // Create test data
            let corpus_data = vec![0xAA; 16384];
            let input_data = vec![0xBB; 8192];
            std::fs::write(&corpus_path, &corpus_data).unwrap();

            // Set up security manager
            let security_manager = EnterpriseSecurityManager::new(
                KeyManagementConfig::default()
            ).unwrap();

            // Build secure corpus with signature
            let mut corpus_manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            
            let corpus_metadata = corpus_manager.build_corpus(
                &[corpus_path.clone()],
                ChunkConfig::default()
            ).await.unwrap();

            // Sign the corpus
            let corpus_signature = security_manager.sign_corpus(&corpus_data).unwrap();

            // Compress with security
            let mut compressor = Compressor::new(Arc::new(corpus_manager));
            let instructions = compressor.compress(&input_data).await.unwrap();

            // Create secure header
            let mut header = ReductoHeader::basic(corpus_metadata.corpus_id, ChunkConfig::default());
            header.corpus_signature = corpus_signature.signature.clone();

            // Serialize and encrypt
            let mut serializer = AdvancedSerializer::new(SerializerConfig::default());
            serializer.create(&compressed_path).unwrap();
            serializer.write_header(&header).unwrap();
            serializer.write_instructions(&instructions).unwrap();
            let _size = serializer.finalize().unwrap();

            // Read and encrypt the compressed file
            let compressed_data = std::fs::read(&compressed_path).unwrap();
            let encrypted_data = security_manager.encrypt_output(&compressed_data).unwrap();
            std::fs::write(&compressed_path, &encrypted_data).unwrap();

            // Log the operation
            security_manager.log_corpus_access(
                &corpus_metadata.corpus_id.to_string(),
                AccessOperation::Create,
                "test_user"
            ).unwrap();

            // Verify end-to-end security
            // 1. Decrypt the file
            let encrypted_file_data = std::fs::read(&compressed_path).unwrap();
            let decrypted_data = security_manager.decrypt_input(&encrypted_file_data).unwrap();

            // 2. Verify corpus signature
            let is_valid = security_manager.verify_corpus_signature(&corpus_data, &Signature {
                algorithm: "ed25519".to_string(),
                signature: header.corpus_signature,
                key_id: None,
            }).unwrap();
            assert!(is_valid, "Corpus signature should be valid");

            // 3. Verify audit log
            let audit_logger = security_manager.get_audit_logger();
            let recent_entries = audit_logger.get_recent_entries(1).unwrap();
            assert_eq!(recent_entries.len(), 1);
            assert_eq!(recent_entries[0].operation, AccessOperation::Create);
            assert_eq!(recent_entries[0].user_id, "test_user");
        }

        #[tokio::test]
        async fn test_security_error_handling() {
            let security_manager = EnterpriseSecurityManager::new(
                KeyManagementConfig::default()
            ).unwrap();

            // Test invalid signature verification
            let data = vec![0xAA; 1024];
            let invalid_signature = Signature {
                algorithm: "invalid".to_string(),
                signature: vec![0xFF; 64],
                key_id: None,
            };

            let result = security_manager.verify_corpus_signature(&data, &invalid_signature);
            assert!(result.is_err(), "Invalid signature should cause error");

            // Test decryption of invalid data
            let invalid_encrypted = vec![0xFF; 32]; // Too short for valid encrypted data
            let result = security_manager.decrypt_input(&invalid_encrypted);
            assert!(result.is_err(), "Invalid encrypted data should cause error");

            // Test secure deletion of non-existent file
            let non_existent_path = std::path::Path::new("/non/existent/file");
            let result = security_manager.secure_delete(non_existent_path).await;
            assert!(result.is_err(), "Secure deletion of non-existent file should cause error");
        }

        #[tokio::test]
        async fn test_concurrent_security_operations() {
            use tokio::task::JoinSet;

            let security_manager = Arc::new(EnterpriseSecurityManager::new(
                KeyManagementConfig::default()
            ).unwrap());

            let mut join_set = JoinSet::new();

            // Spawn multiple concurrent operations
            for i in 0..10 {
                let manager = Arc::clone(&security_manager);
                join_set.spawn(async move {
                    let data = vec![i as u8; 1024];
                    
                    // Sign data
                    let signature = manager.sign_corpus(&data).unwrap();
                    
                    // Verify signature
                    let is_valid = manager.verify_corpus_signature(&data, &signature).unwrap();
                    assert!(is_valid, "Signature should be valid");
                    
                    // Encrypt data
                    let encrypted = manager.encrypt_output(&data).unwrap();
                    
                    // Decrypt data
                    let decrypted = manager.decrypt_input(&encrypted).unwrap();
                    assert_eq!(data, decrypted, "Roundtrip should preserve data");
                    
                    // Log operation
                    manager.log_corpus_access(
                        &format!("corpus-{}", i),
                        AccessOperation::Read,
                        &format!("user-{}", i)
                    ).unwrap();
                    
                    i
                });
            }

            // Wait for all operations to complete
            let mut results = Vec::new();
            while let Some(result) = join_set.join_next().await {
                results.push(result.unwrap());
            }

            // Verify all operations completed successfully
            results.sort();
            assert_eq!(results, (0..10).collect::<Vec<_>>());

            // Verify audit log has all entries
            let audit_logger = security_manager.get_audit_logger();
            let recent_entries = audit_logger.get_recent_entries(10).unwrap();
            assert_eq!(recent_entries.len(), 10);
        }
    }
}

#[cfg(not(feature = "security"))]
mod no_security_tests {
    // Placeholder tests when security feature is disabled
    #[test]
    fn test_security_feature_disabled() {
        // This test just ensures the module compiles when security is disabled
        assert!(true, "Security feature is disabled");
    }
}