#!/usr/bin/env rust-script
//! Security Audit and Dependency Vulnerability Scanner
//!
//! This script performs comprehensive security auditing including:
//! - Dependency vulnerability scanning with cargo-audit
//! - Static code analysis for security issues
//! - Cryptographic implementation validation
//! - Memory safety verification
//! - Enterprise security compliance checks

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use serde::{Deserialize, Serialize};

/// Security audit result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditResult {
    pub audit_type: String,
    pub severity: SecuritySeverity,
    pub title: String,
    pub description: String,
    pub affected_files: Vec<String>,
    pub recommendation: String,
    pub cve_id: Option<String>,
    pub cvss_score: Option<f64>,
}

/// Security severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for SecuritySeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecuritySeverity::Info => write!(f, "INFO"),
            SecuritySeverity::Low => write!(f, "LOW"),
            SecuritySeverity::Medium => write!(f, "MEDIUM"),
            SecuritySeverity::High => write!(f, "HIGH"),
            SecuritySeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Security audit configuration
#[derive(Debug, Clone)]
pub struct SecurityAuditConfig {
    pub enable_dependency_scan: bool,
    pub enable_static_analysis: bool,
    pub enable_crypto_validation: bool,
    pub enable_memory_safety_check: bool,
    pub enable_compliance_check: bool,
    pub fail_on_high_severity: bool,
    pub fail_on_medium_severity: bool,
}

impl Default for SecurityAuditConfig {
    fn default() -> Self {
        Self {
            enable_dependency_scan: true,
            enable_static_analysis: true,
            enable_crypto_validation: true,
            enable_memory_safety_check: true,
            enable_compliance_check: true,
            fail_on_high_severity: true,
            fail_on_medium_severity: false,
        }
    }
}

/// Main security auditor
pub struct SecurityAuditor {
    workspace_root: PathBuf,
    config: SecurityAuditConfig,
}

impl SecurityAuditor {
    /// Create a new security auditor
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        Self {
            workspace_root: workspace_root.as_ref().to_path_buf(),
            config: SecurityAuditConfig::default(),
        }
    }

    /// Run comprehensive security audit
    pub fn run_security_audit(&self) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        println!("🔒 Starting comprehensive security audit...");
        
        let mut results = Vec::new();
        
        // 1. Dependency vulnerability scanning
        if self.config.enable_dependency_scan {
            println!("🔍 Scanning dependencies for vulnerabilities...");
            let mut dep_results = self.scan_dependencies()?;
            results.append(&mut dep_results);
        }
        
        // 2. Static code analysis for security issues
        if self.config.enable_static_analysis {
            println!("🔍 Performing static security analysis...");
            let mut static_results = self.static_security_analysis()?;
            results.append(&mut static_results);
        }
        
        // 3. Cryptographic implementation validation
        if self.config.enable_crypto_validation {
            println!("🔍 Validating cryptographic implementations...");
            let mut crypto_results = self.validate_cryptography()?;
            results.append(&mut crypto_results);
        }
        
        // 4. Memory safety verification
        if self.config.enable_memory_safety_check {
            println!("🔍 Checking memory safety patterns...");
            let mut memory_results = self.check_memory_safety()?;
            results.append(&mut memory_results);
        }
        
        // 5. Enterprise security compliance
        if self.config.enable_compliance_check {
            println!("🔍 Checking enterprise security compliance...");
            let mut compliance_results = self.check_compliance()?;
            results.append(&mut compliance_results);
        }
        
        // Sort results by severity
        results.sort_by(|a, b| b.severity.cmp(&a.severity));
        
        // Report results
        self.report_security_results(&results);
        
        Ok(results)
    }

    /// Scan dependencies for known vulnerabilities
    fn scan_dependencies(&self) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Check if cargo-audit is installed
        if !self.is_cargo_audit_installed() {
            println!("⚠️  cargo-audit not found, installing...");
            self.install_cargo_audit()?;
        }
        
        // Run cargo audit
        let output = Command::new("cargo")
            .args(&["audit", "--json"])
            .current_dir(&self.workspace_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;
        
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            results.extend(self.parse_cargo_audit_output(&stdout)?);
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("no vulnerabilities found") {
                results.push(SecurityAuditResult {
                    audit_type: "dependency_scan".to_string(),
                    severity: SecuritySeverity::Medium,
                    title: "Dependency scan failed".to_string(),
                    description: format!("cargo-audit failed: {}", stderr),
                    affected_files: vec!["Cargo.toml".to_string()],
                    recommendation: "Check cargo-audit installation and network connectivity".to_string(),
                    cve_id: None,
                    cvss_score: None,
                });
            }
        }
        
        // Additional dependency checks
        results.extend(self.check_dependency_policies()?);
        
        Ok(results)
    }

    /// Parse cargo-audit JSON output
    fn parse_cargo_audit_output(&self, output: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Parse JSON output (simplified - in practice you'd parse the actual cargo-audit JSON format)
        for line in output.lines() {
            if line.contains("\"type\":\"vulnerability\"") {
                // This is a simplified parser - in practice you'd use proper JSON parsing
                if let Some(result) = self.parse_vulnerability_line(line)? {
                    results.push(result);
                }
            }
        }
        
        Ok(results)
    }

    /// Parse a single vulnerability line from cargo-audit output
    fn parse_vulnerability_line(&self, line: &str) -> Result<Option<SecurityAuditResult>, Box<dyn std::error::Error>> {
        // Simplified parsing - in practice you'd use serde_json
        if line.contains("RUSTSEC-") {
            let result = SecurityAuditResult {
                audit_type: "dependency_vulnerability".to_string(),
                severity: SecuritySeverity::High, // Would be parsed from actual data
                title: "Dependency vulnerability detected".to_string(),
                description: "A known vulnerability was found in a dependency".to_string(),
                affected_files: vec!["Cargo.toml".to_string()],
                recommendation: "Update the affected dependency to a patched version".to_string(),
                cve_id: Some("CVE-2023-XXXX".to_string()), // Would be parsed
                cvss_score: Some(7.5), // Would be parsed
            };
            return Ok(Some(result));
        }
        Ok(None)
    }

    /// Check dependency policies (versions, sources, etc.)
    fn check_dependency_policies(&self) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let cargo_toml_path = self.workspace_root.join("Cargo.toml");
        if !cargo_toml_path.exists() {
            return Ok(results);
        }
        
        let cargo_content = fs::read_to_string(&cargo_toml_path)?;
        
        // Check for git dependencies (potential security risk)
        if cargo_content.contains("git = ") {
            results.push(SecurityAuditResult {
                audit_type: "dependency_policy".to_string(),
                severity: SecuritySeverity::Medium,
                title: "Git dependencies detected".to_string(),
                description: "Dependencies from git repositories may pose security risks".to_string(),
                affected_files: vec!["Cargo.toml".to_string()],
                recommendation: "Use published crates from crates.io when possible".to_string(),
                cve_id: None,
                cvss_score: None,
            });
        }
        
        // Check for path dependencies in production
        if cargo_content.contains("path = ") {
            results.push(SecurityAuditResult {
                audit_type: "dependency_policy".to_string(),
                severity: SecuritySeverity::Low,
                title: "Path dependencies detected".to_string(),
                description: "Path dependencies should be reviewed for production use".to_string(),
                affected_files: vec!["Cargo.toml".to_string()],
                recommendation: "Ensure path dependencies are appropriate for deployment".to_string(),
                cve_id: None,
                cvss_score: None,
            });
        }
        
        // Check for wildcard version requirements
        if cargo_content.contains("\"*\"") {
            results.push(SecurityAuditResult {
                audit_type: "dependency_policy".to_string(),
                severity: SecuritySeverity::Medium,
                title: "Wildcard version dependencies".to_string(),
                description: "Wildcard version requirements can introduce unexpected changes".to_string(),
                affected_files: vec!["Cargo.toml".to_string()],
                recommendation: "Use specific version ranges instead of wildcards".to_string(),
                cve_id: None,
                cvss_score: None,
            });
        }
        
        Ok(results)
    }

    /// Perform static security analysis on source code
    fn static_security_analysis(&self) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Find all Rust source files
        let rust_files = self.find_rust_files()?;
        
        for file_path in rust_files {
            let content = fs::read_to_string(&file_path)?;
            let relative_path = file_path.strip_prefix(&self.workspace_root)
                .unwrap_or(&file_path)
                .to_string_lossy()
                .to_string();
            
            // Check for unsafe code blocks
            results.extend(self.check_unsafe_code(&content, &relative_path)?);
            
            // Check for potential timing attacks
            results.extend(self.check_timing_attacks(&content, &relative_path)?);
            
            // Check for hardcoded secrets
            results.extend(self.check_hardcoded_secrets(&content, &relative_path)?);
            
            // Check for insecure random number generation
            results.extend(self.check_insecure_random(&content, &relative_path)?);
            
            // Check for SQL injection vulnerabilities
            results.extend(self.check_sql_injection(&content, &relative_path)?);
            
            // Check for path traversal vulnerabilities
            results.extend(self.check_path_traversal(&content, &relative_path)?);
        }
        
        Ok(results)
    }

    /// Check for unsafe code usage
    fn check_unsafe_code(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        for (line_num, line) in content.lines().enumerate() {
            if line.trim().starts_with("unsafe") {
                // Check if it's properly documented
                let lines: Vec<&str> = content.lines().collect();
                let has_safety_comment = if line_num > 0 {
                    lines[line_num - 1].contains("Safety:") || 
                    lines[line_num - 1].contains("SAFETY:") ||
                    lines[line_num - 1].contains("safe:")
                } else {
                    false
                };
                
                if !has_safety_comment {
                    results.push(SecurityAuditResult {
                        audit_type: "unsafe_code".to_string(),
                        severity: SecuritySeverity::Medium,
                        title: "Undocumented unsafe code".to_string(),
                        description: format!("Unsafe code at line {} lacks safety documentation", line_num + 1),
                        affected_files: vec![file_path.to_string()],
                        recommendation: "Add safety comments explaining why the unsafe code is safe".to_string(),
                        cve_id: None,
                        cvss_score: None,
                    });
                }
            }
        }
        
        Ok(results)
    }

    /// Check for potential timing attack vulnerabilities
    fn check_timing_attacks(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Look for string/byte comparisons that might be vulnerable to timing attacks
        let timing_vulnerable_patterns = [
            "==", "!=", ".eq(", ".ne(",
        ];
        
        for (line_num, line) in content.lines().enumerate() {
            // Skip if line uses constant-time comparison
            if line.contains("subtle::") || line.contains("constant_time") {
                continue;
            }
            
            // Check for comparisons in security-sensitive contexts
            if (line.contains("password") || line.contains("secret") || line.contains("token") || 
                line.contains("hash") || line.contains("signature")) &&
               timing_vulnerable_patterns.iter().any(|pattern| line.contains(pattern)) {
                
                results.push(SecurityAuditResult {
                    audit_type: "timing_attack".to_string(),
                    severity: SecuritySeverity::High,
                    title: "Potential timing attack vulnerability".to_string(),
                    description: format!("Line {} may be vulnerable to timing attacks in cryptographic comparison", line_num + 1),
                    affected_files: vec![file_path.to_string()],
                    recommendation: "Use constant-time comparison functions for cryptographic values".to_string(),
                    cve_id: None,
                    cvss_score: Some(6.5),
                });
            }
        }
        
        Ok(results)
    }

    /// Check for hardcoded secrets
    fn check_hardcoded_secrets(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let secret_patterns = [
            (r"password\s*=\s*[\"'][^\"']+[\"']", "password"),
            (r"secret\s*=\s*[\"'][^\"']+[\"']", "secret"),
            (r"api_key\s*=\s*[\"'][^\"']+[\"']", "API key"),
            (r"token\s*=\s*[\"'][^\"']+[\"']", "token"),
            (r"private_key\s*=\s*[\"'][^\"']+[\"']", "private key"),
        ];
        
        for (line_num, line) in content.lines().enumerate() {
            for (pattern, secret_type) in &secret_patterns {
                if regex::Regex::new(pattern).unwrap().is_match(&line.to_lowercase()) {
                    // Skip test files and examples
                    if file_path.contains("test") || file_path.contains("example") {
                        continue;
                    }
                    
                    results.push(SecurityAuditResult {
                        audit_type: "hardcoded_secret".to_string(),
                        severity: SecuritySeverity::High,
                        title: format!("Hardcoded {} detected", secret_type),
                        description: format!("Line {} contains a hardcoded {}", line_num + 1, secret_type),
                        affected_files: vec![file_path.to_string()],
                        recommendation: "Use environment variables or secure configuration management".to_string(),
                        cve_id: None,
                        cvss_score: Some(7.0),
                    });
                }
            }
        }
        
        Ok(results)
    }

    /// Check for insecure random number generation
    fn check_insecure_random(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Look for usage of insecure random number generators
        let insecure_patterns = [
            "rand::random()",
            "thread_rng()",
            "std::collections::hash_map::DefaultHasher",
        ];
        
        for (line_num, line) in content.lines().enumerate() {
            for pattern in &insecure_patterns {
                if line.contains(pattern) {
                    // Check if it's in a cryptographic context
                    if line.contains("key") || line.contains("nonce") || line.contains("salt") ||
                       line.contains("iv") || line.contains("crypto") {
                        
                        results.push(SecurityAuditResult {
                            audit_type: "insecure_random".to_string(),
                            severity: SecuritySeverity::High,
                            title: "Insecure random number generation".to_string(),
                            description: format!("Line {} uses insecure RNG in cryptographic context", line_num + 1),
                            affected_files: vec![file_path.to_string()],
                            recommendation: "Use cryptographically secure random number generators".to_string(),
                            cve_id: None,
                            cvss_score: Some(6.0),
                        });
                    }
                }
            }
        }
        
        Ok(results)
    }

    /// Check for SQL injection vulnerabilities
    fn check_sql_injection(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Look for string concatenation in SQL queries
        for (line_num, line) in content.lines().enumerate() {
            if (line.contains("SELECT") || line.contains("INSERT") || line.contains("UPDATE") || line.contains("DELETE")) &&
               (line.contains("format!") || line.contains(" + ") || line.contains("&")) {
                
                results.push(SecurityAuditResult {
                    audit_type: "sql_injection".to_string(),
                    severity: SecuritySeverity::High,
                    title: "Potential SQL injection vulnerability".to_string(),
                    description: format!("Line {} may be vulnerable to SQL injection", line_num + 1),
                    affected_files: vec![file_path.to_string()],
                    recommendation: "Use parameterized queries or prepared statements".to_string(),
                    cve_id: None,
                    cvss_score: Some(8.0),
                });
            }
        }
        
        Ok(results)
    }

    /// Check for path traversal vulnerabilities
    fn check_path_traversal(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Look for file operations with user input
        for (line_num, line) in content.lines().enumerate() {
            if (line.contains("File::open") || line.contains("fs::read") || line.contains("fs::write")) &&
               (line.contains("&") || line.contains("format!")) {
                
                results.push(SecurityAuditResult {
                    audit_type: "path_traversal".to_string(),
                    severity: SecuritySeverity::Medium,
                    title: "Potential path traversal vulnerability".to_string(),
                    description: format!("Line {} may be vulnerable to path traversal", line_num + 1),
                    affected_files: vec![file_path.to_string()],
                    recommendation: "Validate and sanitize file paths, use Path::canonicalize()".to_string(),
                    cve_id: None,
                    cvss_score: Some(5.5),
                });
            }
        }
        
        Ok(results)
    }

    /// Validate cryptographic implementations
    fn validate_cryptography(&self) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let rust_files = self.find_rust_files()?;
        
        for file_path in rust_files {
            let content = fs::read_to_string(&file_path)?;
            let relative_path = file_path.strip_prefix(&self.workspace_root)
                .unwrap_or(&file_path)
                .to_string_lossy()
                .to_string();
            
            // Check for weak cryptographic algorithms
            results.extend(self.check_weak_crypto(&content, &relative_path)?);
            
            // Check for proper key management
            results.extend(self.check_key_management(&content, &relative_path)?);
            
            // Check for proper IV/nonce usage
            results.extend(self.check_iv_nonce_usage(&content, &relative_path)?);
        }
        
        Ok(results)
    }

    /// Check for weak cryptographic algorithms
    fn check_weak_crypto(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let weak_algorithms = [
            ("md5", "MD5 is cryptographically broken"),
            ("sha1", "SHA-1 is deprecated for cryptographic use"),
            ("des", "DES has insufficient key length"),
            ("rc4", "RC4 has known vulnerabilities"),
        ];
        
        for (line_num, line) in content.lines().enumerate() {
            for (algorithm, reason) in &weak_algorithms {
                if line.to_lowercase().contains(algorithm) {
                    results.push(SecurityAuditResult {
                        audit_type: "weak_crypto".to_string(),
                        severity: SecuritySeverity::High,
                        title: format!("Weak cryptographic algorithm: {}", algorithm.to_uppercase()),
                        description: format!("Line {} uses {}: {}", line_num + 1, algorithm.to_uppercase(), reason),
                        affected_files: vec![file_path.to_string()],
                        recommendation: "Use modern, secure cryptographic algorithms".to_string(),
                        cve_id: None,
                        cvss_score: Some(7.5),
                    });
                }
            }
        }
        
        Ok(results)
    }

    /// Check key management practices
    fn check_key_management(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Check for hardcoded keys
        if content.contains("let key = ") && (content.contains("[") || content.contains("\"")) {
            results.push(SecurityAuditResult {
                audit_type: "key_management".to_string(),
                severity: SecuritySeverity::High,
                title: "Potential hardcoded cryptographic key".to_string(),
                description: "Cryptographic keys should not be hardcoded".to_string(),
                affected_files: vec![file_path.to_string()],
                recommendation: "Use secure key derivation or key management systems".to_string(),
                cve_id: None,
                cvss_score: Some(8.0),
            });
        }
        
        Ok(results)
    }

    /// Check IV/nonce usage
    fn check_iv_nonce_usage(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Check for reused IVs/nonces
        if (content.contains("iv") || content.contains("nonce")) && content.contains("= [0") {
            results.push(SecurityAuditResult {
                audit_type: "iv_nonce_reuse".to_string(),
                severity: SecuritySeverity::High,
                title: "Potential IV/nonce reuse".to_string(),
                description: "IVs and nonces should be unique for each encryption operation".to_string(),
                affected_files: vec![file_path.to_string()],
                recommendation: "Generate random IVs/nonces for each encryption".to_string(),
                cve_id: None,
                cvss_score: Some(7.0),
            });
        }
        
        Ok(results)
    }

    /// Check memory safety patterns
    fn check_memory_safety(&self) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let rust_files = self.find_rust_files()?;
        
        for file_path in rust_files {
            let content = fs::read_to_string(&file_path)?;
            let relative_path = file_path.strip_prefix(&self.workspace_root)
                .unwrap_or(&file_path)
                .to_string_lossy()
                .to_string();
            
            // Check for potential buffer overflows in unsafe code
            results.extend(self.check_buffer_overflows(&content, &relative_path)?);
            
            // Check for memory leaks
            results.extend(self.check_memory_leaks(&content, &relative_path)?);
            
            // Check for use-after-free patterns
            results.extend(self.check_use_after_free(&content, &relative_path)?);
        }
        
        Ok(results)
    }

    /// Check for buffer overflow vulnerabilities
    fn check_buffer_overflows(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Look for unchecked array access in unsafe blocks
        let mut in_unsafe_block = false;
        for (line_num, line) in content.lines().enumerate() {
            if line.trim().starts_with("unsafe") {
                in_unsafe_block = true;
            } else if line.trim() == "}" && in_unsafe_block {
                in_unsafe_block = false;
            }
            
            if in_unsafe_block && (line.contains(".add(") || line.contains(".offset(")) {
                results.push(SecurityAuditResult {
                    audit_type: "buffer_overflow".to_string(),
                    severity: SecuritySeverity::High,
                    title: "Potential buffer overflow in unsafe code".to_string(),
                    description: format!("Line {} performs pointer arithmetic without bounds checking", line_num + 1),
                    affected_files: vec![file_path.to_string()],
                    recommendation: "Ensure bounds checking before pointer arithmetic".to_string(),
                    cve_id: None,
                    cvss_score: Some(8.5),
                });
            }
        }
        
        Ok(results)
    }

    /// Check for memory leaks
    fn check_memory_leaks(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Look for manual memory management without proper cleanup
        if content.contains("Box::into_raw") && !content.contains("Box::from_raw") {
            results.push(SecurityAuditResult {
                audit_type: "memory_leak".to_string(),
                severity: SecuritySeverity::Medium,
                title: "Potential memory leak".to_string(),
                description: "Box::into_raw without corresponding Box::from_raw".to_string(),
                affected_files: vec![file_path.to_string()],
                recommendation: "Ensure proper cleanup of raw pointers".to_string(),
                cve_id: None,
                cvss_score: Some(4.0),
            });
        }
        
        Ok(results)
    }

    /// Check for use-after-free patterns
    fn check_use_after_free(&self, content: &str, file_path: &str) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // This is a simplified check - real analysis would be much more complex
        if content.contains("drop(") && content.contains("unsafe") {
            results.push(SecurityAuditResult {
                audit_type: "use_after_free".to_string(),
                severity: SecuritySeverity::Info,
                title: "Manual drop in unsafe context".to_string(),
                description: "Manual drop operations in unsafe code require careful review".to_string(),
                affected_files: vec![file_path.to_string()],
                recommendation: "Review for potential use-after-free vulnerabilities".to_string(),
                cve_id: None,
                cvss_score: None,
            });
        }
        
        Ok(results)
    }

    /// Check enterprise security compliance
    fn check_compliance(&self) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Check for required security features
        results.extend(self.check_required_security_features()?);
        
        // Check for proper error handling
        results.extend(self.check_error_handling()?);
        
        // Check for logging and auditing
        results.extend(self.check_logging_auditing()?);
        
        Ok(results)
    }

    /// Check for required security features
    fn check_required_security_features(&self) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let cargo_toml_path = self.workspace_root.join("Cargo.toml");
        if cargo_toml_path.exists() {
            let content = fs::read_to_string(&cargo_toml_path)?;
            
            // Check for security-related dependencies
            let required_security_deps = [
                ("ed25519-dalek", "Digital signatures"),
                ("aes-gcm", "Encryption"),
                ("blake3", "Cryptographic hashing"),
                ("subtle", "Constant-time operations"),
            ];
            
            for (dep, purpose) in &required_security_deps {
                if !content.contains(dep) {
                    results.push(SecurityAuditResult {
                        audit_type: "compliance".to_string(),
                        severity: SecuritySeverity::Medium,
                        title: format!("Missing security dependency: {}", dep),
                        description: format!("Required for {}", purpose),
                        affected_files: vec!["Cargo.toml".to_string()],
                        recommendation: format!("Add {} dependency for {}", dep, purpose),
                        cve_id: None,
                        cvss_score: None,
                    });
                }
            }
        }
        
        Ok(results)
    }

    /// Check error handling patterns
    fn check_error_handling(&self) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let rust_files = self.find_rust_files()?;
        
        for file_path in rust_files {
            let content = fs::read_to_string(&file_path)?;
            let relative_path = file_path.strip_prefix(&self.workspace_root)
                .unwrap_or(&file_path)
                .to_string_lossy()
                .to_string();
            
            // Check for unwrap() usage
            let unwrap_count = content.matches(".unwrap()").count();
            if unwrap_count > 5 {
                results.push(SecurityAuditResult {
                    audit_type: "error_handling".to_string(),
                    severity: SecuritySeverity::Low,
                    title: "Excessive unwrap() usage".to_string(),
                    description: format!("File contains {} unwrap() calls", unwrap_count),
                    affected_files: vec![relative_path],
                    recommendation: "Use proper error handling instead of unwrap()".to_string(),
                    cve_id: None,
                    cvss_score: None,
                });
            }
        }
        
        Ok(results)
    }

    /// Check logging and auditing
    fn check_logging_auditing(&self) -> Result<Vec<SecurityAuditResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Check if audit logging is implemented
        let audit_files = self.find_files_containing("audit")?;
        if audit_files.is_empty() {
            results.push(SecurityAuditResult {
                audit_type: "compliance".to_string(),
                severity: SecuritySeverity::Medium,
                title: "Missing audit logging".to_string(),
                description: "No audit logging implementation found".to_string(),
                affected_files: vec![],
                recommendation: "Implement comprehensive audit logging for security events".to_string(),
                cve_id: None,
                cvss_score: None,
            });
        }
        
        Ok(results)
    }

    /// Find all Rust source files
    fn find_rust_files(&self) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let mut rust_files = Vec::new();
        self.find_files_recursive(&self.workspace_root, &mut rust_files, |path| {
            path.extension().map_or(false, |ext| ext == "rs")
        })?;
        Ok(rust_files)
    }

    /// Find files containing a specific string
    fn find_files_containing(&self, search_term: &str) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let mut matching_files = Vec::new();
        let rust_files = self.find_rust_files()?;
        
        for file_path in rust_files {
            let content = fs::read_to_string(&file_path)?;
            if content.contains(search_term) {
                matching_files.push(file_path);
            }
        }
        
        Ok(matching_files)
    }

    /// Recursively find files matching a predicate
    fn find_files_recursive<F>(&self, dir: &Path, files: &mut Vec<PathBuf>, predicate: F) -> Result<(), Box<dyn std::error::Error>>
    where
        F: Fn(&Path) -> bool + Copy,
    {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    // Skip target and hidden directories
                    if let Some(name) = path.file_name() {
                        if name == "target" || name.to_string_lossy().starts_with('.') {
                            continue;
                        }
                    }
                    self.find_files_recursive(&path, files, predicate)?;
                } else if predicate(&path) {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    /// Check if cargo-audit is installed
    fn is_cargo_audit_installed(&self) -> bool {
        Command::new("cargo")
            .args(&["audit", "--version"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_or(false, |status| status.success())
    }

    /// Install cargo-audit
    fn install_cargo_audit(&self) -> Result<(), Box<dyn std::error::Error>> {
        let output = Command::new("cargo")
            .args(&["install", "cargo-audit"])
            .output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to install cargo-audit: {}", stderr).into());
        }
        
        Ok(())
    }

    /// Report security audit results
    fn report_security_results(&self, results: &[SecurityAuditResult]) {
        println!("\n🔒 Security Audit Results");
        println!("========================");
        
        let mut critical_count = 0;
        let mut high_count = 0;
        let mut medium_count = 0;
        let mut low_count = 0;
        let mut info_count = 0;
        
        for result in results {
            let icon = match result.severity {
                SecuritySeverity::Critical => { critical_count += 1; "🚨" },
                SecuritySeverity::High => { high_count += 1; "🔴" },
                SecuritySeverity::Medium => { medium_count += 1; "🟡" },
                SecuritySeverity::Low => { low_count += 1; "🟢" },
                SecuritySeverity::Info => { info_count += 1; "ℹ️" },
            };
            
            println!("{} [{}] {}", icon, result.severity, result.title);
            println!("   📄 Files: {}", result.affected_files.join(", "));
            println!("   📝 {}", result.description);
            println!("   💡 {}", result.recommendation);
            
            if let Some(cve) = &result.cve_id {
                println!("   🆔 CVE: {}", cve);
            }
            if let Some(score) = result.cvss_score {
                println!("   📊 CVSS Score: {:.1}", score);
            }
            println!();
        }
        
        println!("📊 Summary:");
        println!("  Critical: {}", critical_count);
        println!("  High: {}", high_count);
        println!("  Medium: {}", medium_count);
        println!("  Low: {}", low_count);
        println!("  Info: {}", info_count);
        println!("  Total: {}", results.len());
        
        if critical_count > 0 || high_count > 0 {
            println!("\n🚨 Security issues found that require immediate attention!");
        } else if medium_count > 0 {
            println!("\n⚠️  Medium severity security issues found - review recommended");
        } else {
            println!("\n✅ No critical security issues detected");
        }
    }
}

/// Main entry point
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = std::env::current_dir()?;
    let auditor = SecurityAuditor::new(workspace_root);
    
    let results = auditor.run_security_audit()?;
    
    // Exit with error code if high or critical severity issues are found
    let has_critical_issues = results.iter().any(|r| {
        matches!(r.severity, SecuritySeverity::Critical | SecuritySeverity::High)
    });
    
    if has_critical_issues {
        std::process::exit(1);
    }
    
    Ok(())
}