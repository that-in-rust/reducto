use crate::errors::BenchmarkError;
use std::fs;
use std::path::{Path, PathBuf};
use std::io::Read;

/// Maximum total data size to load from directory (100MB)
const MAX_TOTAL_DATA_SIZE: usize = 100 * 1024 * 1024; // 100MB

/// Size of generated test data (20MB)
const GENERATED_DATA_SIZE: usize = 20 * 1024 * 1024; // 20MB

/// Maximum individual file size to consider (10MB)
const MAX_FILE_SIZE: usize = 10 * 1024 * 1024; // 10MB

/// Loaded data with metadata
#[derive(Debug, Clone)]
pub struct LoadedData {
    pub data: Vec<u8>,
    pub source: DataSource,
    pub total_size: usize,
    pub file_count: usize,
}

/// Source of the loaded data
#[derive(Debug, Clone)]
pub enum DataSource {
    Directory(PathBuf),
    Generated,
}

/// Load data from a directory or generate test data if no path provided
/// 
/// # Arguments
/// * `data_path` - Optional path to directory containing data files
/// 
/// # Returns
/// * `LoadedData` containing the data and metadata
/// 
/// # Requirements
/// - REQ-1.1: Analyze all files in directory up to 100MB total
/// - REQ-1.2: Generate 20MB test data if no path given
pub fn load_data(data_path: Option<&Path>) -> Result<LoadedData, BenchmarkError> {
    match data_path {
        Some(path) => load_from_directory(path),
        None => generate_test_data(),
    }
}

/// Load files from a directory up to the maximum size limit
/// 
/// # Arguments
/// * `dir_path` - Path to directory containing files
/// 
/// # Returns
/// * `LoadedData` with files concatenated together
fn load_from_directory(dir_path: &Path) -> Result<LoadedData, BenchmarkError> {
    if !dir_path.exists() {
        return Err(BenchmarkError::DataLoading(format!(
            "Directory does not exist: {}", 
            dir_path.display()
        )));
    }
    
    if !dir_path.is_dir() {
        return Err(BenchmarkError::DataLoading(format!(
            "Path is not a directory: {}", 
            dir_path.display()
        )));
    }
    
    let mut all_data = Vec::new();
    let mut total_size = 0;
    let mut file_count = 0;
    
    // Collect all files recursively
    let files = collect_files_recursive(dir_path)?;
    
    for file_path in files {
        if total_size >= MAX_TOTAL_DATA_SIZE {
            println!("Reached maximum data size limit ({}MB), stopping file collection", 
                     MAX_TOTAL_DATA_SIZE / (1024 * 1024));
            break;
        }
        
        match load_single_file(&file_path, MAX_TOTAL_DATA_SIZE - total_size) {
            Ok(file_data) => {
                if !file_data.is_empty() {
                    total_size += file_data.len();
                    all_data.extend_from_slice(&file_data);
                    file_count += 1;
                    
                    if file_count % 100 == 0 {
                        println!("Loaded {} files, {} bytes total", file_count, total_size);
                    }
                }
            }
            Err(e) => {
                // Log warning but continue with other files
                eprintln!("Warning: Failed to load file {}: {}", file_path.display(), e);
            }
        }
    }
    
    if all_data.is_empty() {
        return Err(BenchmarkError::DataLoading(
            "No readable files found in directory".to_string()
        ));
    }
    
    println!("Loaded {} files, total size: {} bytes ({:.1}MB)", 
             file_count, total_size, total_size as f64 / (1024.0 * 1024.0));
    
    Ok(LoadedData {
        data: all_data,
        source: DataSource::Directory(dir_path.to_path_buf()),
        total_size,
        file_count,
    })
}

/// Recursively collect all files in a directory
fn collect_files_recursive(dir_path: &Path) -> Result<Vec<PathBuf>, BenchmarkError> {
    let mut files = Vec::new();
    collect_files_recursive_impl(dir_path, &mut files)?;
    
    // Sort files for deterministic order
    files.sort();
    
    Ok(files)
}

/// Implementation of recursive file collection
fn collect_files_recursive_impl(dir_path: &Path, files: &mut Vec<PathBuf>) -> Result<(), BenchmarkError> {
    let entries = fs::read_dir(dir_path)
        .map_err(|e| BenchmarkError::DataLoading(format!(
            "Failed to read directory {}: {}", 
            dir_path.display(), e
        )))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| BenchmarkError::DataLoading(format!(
            "Failed to read directory entry: {}", e
        )))?;
        
        let path = entry.path();
        
        if path.is_file() {
            // Skip hidden files and common non-data files
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if should_include_file(file_name) {
                    files.push(path);
                }
            }
        } else if path.is_dir() {
            // Skip hidden directories and common build/cache directories
            if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                if should_include_directory(dir_name) {
                    collect_files_recursive_impl(&path, files)?;
                }
            }
        }
    }
    
    Ok(())
}

/// Determine if a file should be included in the data loading
fn should_include_file(file_name: &str) -> bool {
    // Skip hidden files
    if file_name.starts_with('.') {
        return false;
    }
    
    // Skip common binary/cache files
    let skip_extensions = [
        ".exe", ".dll", ".so", ".dylib", ".a", ".lib",
        ".obj", ".o", ".pyc", ".class", ".jar",
        ".zip", ".tar", ".gz", ".bz2", ".xz",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp",
        ".mp3", ".mp4", ".avi", ".mov", ".wav",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ];
    
    let file_name_lower = file_name.to_lowercase();
    for ext in &skip_extensions {
        if file_name_lower.ends_with(ext) {
            return false;
        }
    }
    
    true
}

/// Determine if a directory should be included in recursive traversal
fn should_include_directory(dir_name: &str) -> bool {
    // Skip hidden directories
    if dir_name.starts_with('.') {
        return false;
    }
    
    // Skip common build/cache directories
    let skip_dirs = [
        "target", "build", "dist", "node_modules", "__pycache__",
        "cache", "tmp", "temp", ".git", ".svn", ".hg",
        "vendor", "deps", "Debug", "Release",
    ];
    
    !skip_dirs.contains(&dir_name)
}

/// Load a single file with size limits
fn load_single_file(file_path: &Path, max_size: usize) -> Result<Vec<u8>, BenchmarkError> {
    let metadata = fs::metadata(file_path)
        .map_err(|e| BenchmarkError::DataLoading(format!(
            "Failed to get metadata for {}: {}", 
            file_path.display(), e
        )))?;
    
    let file_size = metadata.len() as usize;
    
    // Skip files that are too large
    if file_size > MAX_FILE_SIZE {
        return Ok(Vec::new()); // Return empty data, don't error
    }
    
    // Skip files that would exceed our remaining budget
    if file_size > max_size {
        return Ok(Vec::new()); // Return empty data, don't error
    }
    
    let mut file = fs::File::open(file_path)
        .map_err(|e| BenchmarkError::DataLoading(format!(
            "Failed to open file {}: {}", 
            file_path.display(), e
        )))?;
    
    let mut buffer = Vec::with_capacity(file_size);
    file.read_to_end(&mut buffer)
        .map_err(|e| BenchmarkError::DataLoading(format!(
            "Failed to read file {}: {}", 
            file_path.display(), e
        )))?;
    
    Ok(buffer)
}

/// Generate realistic test data representing common scenarios
/// 
/// # Returns
/// * `LoadedData` with generated test data (20MB total)
/// 
/// # Requirements
/// - REQ-1.2: Generate 20MB test data with source code, JSON logs, and documentation
fn generate_test_data() -> Result<LoadedData, BenchmarkError> {
    println!("Generating 20MB of realistic test data...");
    
    let mut all_data = Vec::with_capacity(GENERATED_DATA_SIZE);
    
    // Generate different types of data to simulate real-world scenarios
    // 40% source code, 30% JSON logs, 30% documentation
    let source_code_size = (GENERATED_DATA_SIZE as f64 * 0.4) as usize;
    let json_logs_size = (GENERATED_DATA_SIZE as f64 * 0.3) as usize;
    let documentation_size = GENERATED_DATA_SIZE - source_code_size - json_logs_size;
    
    // Generate source code data
    let source_code = generate_source_code_data(source_code_size);
    all_data.extend_from_slice(&source_code);
    
    // Generate JSON logs data
    let json_logs = generate_json_logs_data(json_logs_size);
    all_data.extend_from_slice(&json_logs);
    
    // Generate documentation data
    let documentation = generate_documentation_data(documentation_size);
    all_data.extend_from_slice(&documentation);
    
    let total_size = all_data.len();
    
    println!("Generated test data: {} bytes total ({:.1}MB)", 
             total_size, total_size as f64 / (1024.0 * 1024.0));
    
    Ok(LoadedData {
        data: all_data,
        source: DataSource::Generated,
        total_size,
        file_count: 3, // Simulated as 3 "files" (source, logs, docs)
    })
}

/// Generate realistic source code data with patterns common in software projects
fn generate_source_code_data(target_size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(target_size);
    
    // Common source code patterns
    let patterns = [
        "use std::collections::HashMap;\nuse std::path::PathBuf;\n\n",
        "fn main() {\n    println!(\"Hello, world!\");\n}\n\n",
        "pub struct Config {\n    pub host: String,\n    pub port: u16,\n}\n\n",
        "impl Default for Config {\n    fn default() -> Self {\n        Self {\n            host: \"localhost\".to_string(),\n            port: 8080,\n        }\n    }\n}\n\n",
        "#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct User {\n    pub id: u64,\n    pub name: String,\n    pub email: String,\n}\n\n",
        "async fn handle_request(req: Request) -> Result<Response, Error> {\n    let user = authenticate(&req).await?;\n    process_request(user, req).await\n}\n\n",
        "// TODO: Implement proper error handling\n// FIXME: This is a temporary workaround\n",
        "const MAX_CONNECTIONS: usize = 1000;\nconst TIMEOUT_SECONDS: u64 = 30;\n\n",
    ];
    
    let mut pattern_index = 0;
    while data.len() < target_size {
        let pattern = patterns[pattern_index % patterns.len()];
        let remaining = target_size - data.len();
        
        if pattern.len() <= remaining {
            data.extend_from_slice(pattern.as_bytes());
        } else {
            data.extend_from_slice(&pattern.as_bytes()[..remaining]);
            break;
        }
        
        pattern_index += 1;
    }
    
    data
}

/// Generate realistic JSON log data with common log patterns
fn generate_json_logs_data(target_size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(target_size);
    
    let log_templates = [
        r#"{"timestamp":"2024-01-15T10:30:45Z","level":"INFO","message":"Request processed successfully","request_id":"req_123456","duration_ms":45}"#,
        r#"{"timestamp":"2024-01-15T10:30:46Z","level":"ERROR","message":"Database connection failed","error":"Connection timeout","retry_count":3}"#,
        r#"{"timestamp":"2024-01-15T10:30:47Z","level":"DEBUG","message":"Cache hit","key":"user:12345","ttl_seconds":300}"#,
        r#"{"timestamp":"2024-01-15T10:30:48Z","level":"WARN","message":"High memory usage detected","memory_usage_mb":1024,"threshold_mb":800}"#,
        r#"{"timestamp":"2024-01-15T10:30:49Z","level":"INFO","message":"User authenticated","user_id":12345,"session_id":"sess_abcdef"}"#,
    ];
    
    let mut template_index = 0;
    while data.len() < target_size {
        let template = log_templates[template_index % log_templates.len()];
        let log_line = format!("{}\n", template);
        let remaining = target_size - data.len();
        
        if log_line.len() <= remaining {
            data.extend_from_slice(log_line.as_bytes());
        } else {
            data.extend_from_slice(&log_line.as_bytes()[..remaining]);
            break;
        }
        
        template_index += 1;
    }
    
    data
}

/// Generate realistic documentation data with common markdown patterns
fn generate_documentation_data(target_size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(target_size);
    
    let doc_patterns = [
        "# Project Documentation\n\nThis is a comprehensive guide to using our application.\n\n",
        "## Installation\n\nTo install the application, run the following command:\n\n```bash\ncargo install my-app\n```\n\n",
        "## Configuration\n\nThe application can be configured using a TOML file:\n\n```toml\n[server]\nhost = \"localhost\"\nport = 8080\n```\n\n",
        "## API Reference\n\n### GET /api/users\n\nReturns a list of all users in the system.\n\n**Response:**\n```json\n{\n  \"users\": [\n    {\"id\": 1, \"name\": \"John Doe\"}\n  ]\n}\n```\n\n",
        "## Troubleshooting\n\n### Common Issues\n\n- **Connection refused**: Check that the server is running\n- **Permission denied**: Ensure proper file permissions\n- **Out of memory**: Increase heap size\n\n",
        "## Contributing\n\nWe welcome contributions! Please follow these guidelines:\n\n1. Fork the repository\n2. Create a feature branch\n3. Make your changes\n4. Submit a pull request\n\n",
    ];
    
    let mut pattern_index = 0;
    while data.len() < target_size {
        let pattern = doc_patterns[pattern_index % doc_patterns.len()];
        let remaining = target_size - data.len();
        
        if pattern.len() <= remaining {
            data.extend_from_slice(pattern.as_bytes());
        } else {
            data.extend_from_slice(&pattern.as_bytes()[..remaining]);
            break;
        }
        
        pattern_index += 1;
    }
    
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    
    #[test]
    fn test_generate_test_data() {
        let result = generate_test_data().unwrap();
        
        assert_eq!(result.total_size, GENERATED_DATA_SIZE);
        assert_eq!(result.data.len(), GENERATED_DATA_SIZE);
        assert_eq!(result.file_count, 3);
        assert!(matches!(result.source, DataSource::Generated));
        
        // Verify data is not empty and contains expected patterns
        let data_str = String::from_utf8_lossy(&result.data);
        assert!(data_str.contains("fn main()"));
        assert!(data_str.contains("timestamp"));
        assert!(data_str.contains("# Project Documentation"));
    }
    
    #[test]
    fn test_load_from_nonexistent_directory() {
        let result = load_from_directory(Path::new("/nonexistent/path"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }
    
    #[test]
    fn test_load_from_directory_with_files() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();
        
        // Create test files
        fs::write(dir_path.join("test1.txt"), "Hello, world!").unwrap();
        fs::write(dir_path.join("test2.rs"), "fn main() {}").unwrap();
        fs::create_dir(dir_path.join("subdir")).unwrap();
        fs::write(dir_path.join("subdir/test3.md"), "# Documentation").unwrap();
        
        let result = load_from_directory(dir_path).unwrap();
        
        assert!(result.total_size > 0);
        assert!(result.file_count >= 3);
        assert!(matches!(result.source, DataSource::Directory(_)));
        
        // Verify data contains content from files
        let data_str = String::from_utf8_lossy(&result.data);
        assert!(data_str.contains("Hello, world!"));
        assert!(data_str.contains("fn main()"));
        assert!(data_str.contains("# Documentation"));
    }
    
    #[test]
    fn test_should_include_file() {
        assert!(should_include_file("test.txt"));
        assert!(should_include_file("main.rs"));
        assert!(should_include_file("README.md"));
        
        assert!(!should_include_file(".hidden"));
        assert!(!should_include_file("binary.exe"));
        assert!(!should_include_file("image.jpg"));
        assert!(!should_include_file("archive.zip"));
    }
    
    #[test]
    fn test_should_include_directory() {
        assert!(should_include_directory("src"));
        assert!(should_include_directory("docs"));
        assert!(should_include_directory("examples"));
        
        assert!(!should_include_directory(".git"));
        assert!(!should_include_directory("target"));
        assert!(!should_include_directory("node_modules"));
        assert!(!should_include_directory("__pycache__"));
    }
    
    #[test]
    fn test_load_data_with_path() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();
        
        fs::write(dir_path.join("test.txt"), "Test content").unwrap();
        
        let result = load_data(Some(dir_path)).unwrap();
        
        assert!(result.total_size > 0);
        assert!(result.file_count > 0);
        assert!(matches!(result.source, DataSource::Directory(_)));
    }
    
    #[test]
    fn test_load_data_without_path() {
        let result = load_data(None).unwrap();
        
        assert_eq!(result.total_size, GENERATED_DATA_SIZE);
        assert_eq!(result.file_count, 3);
        assert!(matches!(result.source, DataSource::Generated));
    }
}