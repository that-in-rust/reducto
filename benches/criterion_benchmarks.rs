//! Criterion Benchmarks for Reducto Mode 3
//!
//! This module implements comprehensive benchmarking using the Criterion framework
//! to profile critical paths and validate enterprise performance claims.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use reducto_mode_3::prelude::*;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

// === Benchmark Data Generation ===

/// Generate benchmark data with specific patterns
fn generate_benchmark_data(size: usize, pattern: BenchmarkPattern) -> Vec<u8> {
    let mut data = vec![0u8; size];
    
    match pattern {
        BenchmarkPattern::Random => {
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = ((i * 17 + i / 256) % 256) as u8;
            }
        }
        BenchmarkPattern::Repetitive => {
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = (i % 64) as u8;
            }
        }
        BenchmarkPattern::Structured => {
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = match i % 1024 {
                    0..=511 => 0xAA,
                    512..=767 => (i / 1024) as u8,
                    _ => 0x00,
                };
            }
        }
        BenchmarkPattern::Sparse => {
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = if i % 128 == 0 { 0xFF } else { 0x00 };
            }
        }
    }
    
    data
}

#[derive(Clone, Copy)]
enum BenchmarkPattern {
    Random,
    Repetitive,
    Structured,
    Sparse,
}

impl std::fmt::Display for BenchmarkPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchmarkPattern::Random => write!(f, "random"),
            BenchmarkPattern::Repetitive => write!(f, "repetitive"),
            BenchmarkPattern::Structured => write!(f, "structured"),
            BenchmarkPattern::Sparse => write!(f, "sparse"),
        }
    }
}

// === Rolling Hash Benchmarks ===

fn bench_rolling_hash_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_hash_init");
    
    let data_sizes = [1024, 4096, 8192, 16384, 32768];
    let window_sizes = [1024, 2048, 4096, 8192];
    
    for &data_size in &data_sizes {
        for &window_size in &window_sizes {
            if window_size <= data_size {
                let data = generate_benchmark_data(data_size, BenchmarkPattern::Random);
                
                group.throughput(Throughput::Bytes(window_size as u64));
                group.bench_with_input(
                    BenchmarkId::new("init", format!("data_{}_window_{}", data_size, window_size)),
                    &(data, window_size),
                    |b, (data, window_size)| {
                        b.iter(|| {
                            let mut hasher = RollingHasher::new(HASH_BASE, *window_size);
                            black_box(hasher.init(&data[0..*window_size]).unwrap());
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

fn bench_rolling_hash_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_hash_update");
    group.measurement_time(Duration::from_secs(10));
    
    let data_sizes = [8192, 32768, 131072, 524288]; // 8KB to 512KB
    let patterns = [
        BenchmarkPattern::Random,
        BenchmarkPattern::Repetitive,
        BenchmarkPattern::Structured,
        BenchmarkPattern::Sparse,
    ];
    
    for &data_size in &data_sizes {
        for pattern in &patterns {
            let data = generate_benchmark_data(data_size, *pattern);
            let window_size = 4096;
            
            group.throughput(Throughput::Bytes((data_size - window_size) as u64));
            group.bench_with_input(
                BenchmarkId::new("update", format!("{}_{}KB_{}", pattern, data_size / 1024, pattern)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut hasher = RollingHasher::new(HASH_BASE, window_size);
                        hasher.init(&data[0..window_size]).unwrap();
                        
                        for i in window_size..data.len() {
                            let exiting_byte = data[i - window_size];
                            let entering_byte = data[i];
                            black_box(hasher.roll(exiting_byte, entering_byte).unwrap());
                        }
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_dual_hash_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_hash");
    
    let chunk_sizes = [1024, 2048, 4096, 8192, 16384];
    let patterns = [BenchmarkPattern::Random, BenchmarkPattern::Structured];
    
    for &chunk_size in &chunk_sizes {
        for pattern in &patterns {
            let data = generate_benchmark_data(chunk_size, *pattern);
            
            group.throughput(Throughput::Bytes(chunk_size as u64));
            group.bench_with_input(
                BenchmarkId::new("dual_hash", format!("{}_{}_bytes", pattern, chunk_size)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut dual_hasher = DualHasher::new(HASH_BASE, data.len());
                        black_box(dual_hasher.hash_chunk(data).unwrap());
                    });
                },
            );
        }
    }
    
    group.finish();
}

// === CDC Chunking Benchmarks ===

fn bench_cdc_chunking_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("cdc_chunking");
    group.measurement_time(Duration::from_secs(15));
    
    let data_sizes = [65536, 262144, 1048576, 4194304]; // 64KB to 4MB
    let chunk_configs = [
        ChunkConfig::new(4096).unwrap(),
        ChunkConfig::new(8192).unwrap(),
        ChunkConfig::new(16384).unwrap(),
    ];
    let patterns = [BenchmarkPattern::Random, BenchmarkPattern::Structured];
    
    for &data_size in &data_sizes {
        for config in &chunk_configs {
            for pattern in &patterns {
                let data = generate_benchmark_data(data_size, *pattern);
                
                group.throughput(Throughput::Bytes(data_size as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        "chunking",
                        format!("{}MB_{}KB_chunks_{}", 
                               data_size / 1048576, 
                               config.target_size / 1024, 
                               pattern)
                    ),
                    &(data, *config),
                    |b, (data, config)| {
                        b.iter(|| {
                            let mut chunker = FastCDCChunker::new(*config).unwrap();
                            let chunks = black_box(chunker.chunk_data(data).unwrap());
                            let _final_chunk = black_box(chunker.finalize().unwrap());
                            chunks
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

fn bench_cdc_boundary_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cdc_boundary_detection");
    
    let data_sizes = [32768, 131072, 524288]; // 32KB to 512KB
    let mask_sizes = [0x1FFF, 0x3FFF, 0x7FFF]; // Different boundary probabilities
    
    for &data_size in &data_sizes {
        for &mask in &mask_sizes {
            let data = generate_benchmark_data(data_size, BenchmarkPattern::Random);
            let config = ChunkConfig {
                target_size: 8192,
                min_size: 4096,
                max_size: 16384,
                hash_mask: mask,
                hash_base: HASH_BASE,
            };
            
            group.throughput(Throughput::Bytes(data_size as u64));
            group.bench_with_input(
                BenchmarkId::new("boundary", format!("{}KB_mask_{:X}", data_size / 1024, mask)),
                &(data, config),
                |b, (data, config)| {
                    b.iter(|| {
                        let mut gear_hasher = GearHasher::new();
                        let mut boundaries = Vec::new();
                        
                        for (i, &byte) in data.iter().enumerate() {
                            let hash = gear_hasher.update(byte);
                            if i >= config.min_size && (hash & config.hash_mask) == 0 {
                                boundaries.push(i);
                                if i >= config.max_size {
                                    break;
                                }
                            }
                        }
                        
                        black_box(boundaries)
                    });
                },
            );
        }
    }
    
    group.finish();
}

// === Compression Benchmarks ===

fn bench_compression_pipeline(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("compression_pipeline");
    group.measurement_time(Duration::from_secs(20));
    
    let data_sizes = [262144, 1048576, 4194304]; // 256KB to 4MB
    let patterns = [BenchmarkPattern::Random, BenchmarkPattern::Structured];
    
    for &data_size in &data_sizes {
        for pattern in &patterns {
            let test_data = generate_benchmark_data(data_size, *pattern);
            let corpus_data = generate_benchmark_data(data_size / 2, *pattern);
            
            // Setup corpus for each benchmark
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("bench_corpus.bin");
            std::fs::write(&corpus_path, &corpus_data).unwrap();
            
            let config = ChunkConfig::default();
            let corpus_manager = rt.block_on(async {
                let mut manager = EnterpriseCorpusManager::new(
                    Box::new(InMemoryStorage::new())
                );
                manager.build_corpus(&[corpus_path], config).await.unwrap();
                Arc::new(manager)
            });
            
            group.throughput(Throughput::Bytes(data_size as u64));
            group.bench_with_input(
                BenchmarkId::new("pipeline", format!("{}MB_{}", data_size / 1048576, pattern)),
                &(test_data, corpus_manager),
                |b, (data, corpus)| {
                    b.to_async(&rt).iter(|| async {
                        let mut compressor = Compressor::new(Arc::clone(corpus));
                        black_box(compressor.compress(data).await.unwrap())
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_corpus_lookup_performance(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("corpus_lookup");
    
    let corpus_sizes = [1048576, 4194304, 16777216]; // 1MB to 16MB
    let lookup_counts = [1000, 5000, 10000];
    
    for &corpus_size in &corpus_sizes {
        for &lookup_count in &lookup_counts {
            let corpus_data = generate_benchmark_data(corpus_size, BenchmarkPattern::Structured);
            
            let temp_dir = TempDir::new().unwrap();
            let corpus_path = temp_dir.path().join("lookup_corpus.bin");
            std::fs::write(&corpus_path, &corpus_data).unwrap();
            
            let config = ChunkConfig::default();
            let corpus_manager = rt.block_on(async {
                let mut manager = EnterpriseCorpusManager::new(
                    Box::new(InMemoryStorage::new())
                );
                manager.build_corpus(&[corpus_path], config).await.unwrap();
                Arc::new(manager)
            });
            
            // Generate lookup hashes
            let mut lookup_hashes = Vec::new();
            let mut hasher = RollingHasher::new(HASH_BASE, config.target_size);
            for i in 0..lookup_count {
                let start = (i * corpus_size / lookup_count) % (corpus_size - config.target_size);
                hasher.init(&corpus_data[start..start + config.target_size]).unwrap();
                lookup_hashes.push(hasher.current_weak_hash().unwrap());
            }
            
            group.throughput(Throughput::Elements(lookup_count as u64));
            group.bench_with_input(
                BenchmarkId::new("lookup", format!("{}MB_corpus_{}lookups", 
                                 corpus_size / 1048576, lookup_count)),
                &(corpus_manager, lookup_hashes),
                |b, (corpus, hashes)| {
                    b.iter(|| {
                        for &hash in hashes {
                            black_box(corpus.get_candidates(hash));
                        }
                    });
                },
            );
        }
    }
    
    group.finish();
}

// === Memory Layout Benchmarks ===

fn bench_memory_layout_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout");
    
    let chunk_counts = [1000, 5000, 10000, 50000];
    
    for &chunk_count in &chunk_counts {
        // Benchmark different storage layouts
        group.bench_with_input(
            BenchmarkId::new("vec_storage", chunk_count),
            &chunk_count,
            |b, &count| {
                b.iter(|| {
                    let mut chunks = Vec::with_capacity(count);
                    for i in 0..count {
                        chunks.push(CorpusBlock {
                            offset: BlockOffset::new(i as u64 * 4096),
                            size: 4096,
                            strong_hash: blake3::hash(&[i as u8; 32]),
                        });
                    }
                    black_box(chunks)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("hashmap_storage", chunk_count),
            &chunk_count,
            |b, &count| {
                b.iter(|| {
                    let mut map = hashbrown::HashMap::with_capacity(count);
                    for i in 0..count {
                        let weak_hash = WeakHash::new((i * 17 + i / 256) as u64);
                        let block = CorpusBlock {
                            offset: BlockOffset::new(i as u64 * 4096),
                            size: 4096,
                            strong_hash: blake3::hash(&[i as u8; 32]),
                        };
                        map.entry(weak_hash).or_insert_with(Vec::new).push(block);
                    }
                    black_box(map)
                });
            },
        );
    }
    
    group.finish();
}

// === Serialization Benchmarks ===

fn bench_serialization_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    
    let instruction_counts = [100, 1000, 10000, 100000];
    let compression_levels = [1, 6, 19]; // Zstd compression levels
    
    for &count in &instruction_counts {
        // Generate test instructions
        let mut instructions = Vec::new();
        for i in 0..count {
            if i % 3 == 0 {
                instructions.push(ReductoInstruction::Reference { 
                    offset: (i as u64) * 4096, 
                    size: 4096 
                });
            } else {
                let residual_size = 64 + (i % 256);
                instructions.push(ReductoInstruction::Residual(vec![i as u8; residual_size]));
            }
        }
        
        for &level in &compression_levels {
            let config = SerializerConfig {
                compression_profile: CompressionProfile::Custom { level },
                enable_streaming: false,
                buffer_size: 65536,
            };
            
            group.throughput(Throughput::Elements(count as u64));
            group.bench_with_input(
                BenchmarkId::new("serialize", format!("{}instr_level{}", count, level)),
                &(instructions.clone(), config),
                |b, (instructions, config)| {
                    b.iter(|| {
                        let serializer = AdvancedSerializer::new(*config);
                        black_box(serializer.serialize_instructions(instructions).unwrap())
                    });
                },
            );
        }
    }
    
    group.finish();
}

// === Concurrent Performance Benchmarks ===

fn bench_concurrent_compression(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_compression");
    group.measurement_time(Duration::from_secs(30));
    
    let thread_counts = [1, 2, 4, 8];
    let data_size = 1048576; // 1MB per thread
    
    for &thread_count in &thread_counts {
        let test_data = generate_benchmark_data(data_size, BenchmarkPattern::Structured);
        let corpus_data = generate_benchmark_data(data_size / 2, BenchmarkPattern::Structured);
        
        let temp_dir = TempDir::new().unwrap();
        let corpus_path = temp_dir.path().join("concurrent_corpus.bin");
        std::fs::write(&corpus_path, &corpus_data).unwrap();
        
        let config = ChunkConfig::default();
        let corpus_manager = rt.block_on(async {
            let mut manager = EnterpriseCorpusManager::new(
                Box::new(InMemoryStorage::new())
            );
            manager.build_corpus(&[corpus_path], config).await.unwrap();
            Arc::new(manager)
        });
        
        group.throughput(Throughput::Bytes((data_size * thread_count) as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrent", format!("{}threads_{}MB", thread_count, 
                           (data_size * thread_count) / 1048576)),
            &(test_data, corpus_manager, thread_count),
            |b, (data, corpus, &threads)| {
                b.to_async(&rt).iter(|| async {
                    let mut join_set = tokio::task::JoinSet::new();
                    
                    for i in 0..threads {
                        let corpus_clone = Arc::clone(corpus);
                        let mut data_clone = data.clone();
                        
                        // Modify data slightly for each thread
                        for byte in data_clone.iter_mut().take(i * 100) {
                            *byte = byte.wrapping_add(i as u8);
                        }
                        
                        join_set.spawn(async move {
                            let mut compressor = Compressor::new(corpus_clone);
                            compressor.compress(&data_clone).await.unwrap()
                        });
                    }
                    
                    let mut results = Vec::new();
                    while let Some(result) = join_set.join_next().await {
                        results.push(result.unwrap());
                    }
                    
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

// === Performance Regression Detection ===

fn bench_performance_regression_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_baseline");
    group.measurement_time(Duration::from_secs(10));
    
    // Standard benchmark scenarios for regression detection
    let scenarios = [
        ("small_random", 65536, BenchmarkPattern::Random),
        ("medium_structured", 1048576, BenchmarkPattern::Structured),
        ("large_repetitive", 4194304, BenchmarkPattern::Repetitive),
    ];
    
    for (name, size, pattern) in &scenarios {
        let data = generate_benchmark_data(*size, *pattern);
        let config = ChunkConfig::default();
        
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("baseline", name),
            &(data, config),
            |b, (data, config)| {
                b.iter(|| {
                    let mut chunker = FastCDCChunker::new(*config).unwrap();
                    let chunks = black_box(chunker.chunk_data(data).unwrap());
                    let _final_chunk = black_box(chunker.finalize().unwrap());
                    chunks
                });
            },
        );
    }
    
    group.finish();
}

// === Benchmark Groups ===

criterion_group!(
    rolling_hash_benches,
    bench_rolling_hash_initialization,
    bench_rolling_hash_updates,
    bench_dual_hash_performance
);

criterion_group!(
    cdc_benches,
    bench_cdc_chunking_throughput,
    bench_cdc_boundary_detection
);

criterion_group!(
    compression_benches,
    bench_compression_pipeline,
    bench_corpus_lookup_performance
);

criterion_group!(
    memory_benches,
    bench_memory_layout_efficiency
);

criterion_group!(
    serialization_benches,
    bench_serialization_performance
);

criterion_group!(
    concurrent_benches,
    bench_concurrent_compression
);

criterion_group!(
    regression_benches,
    bench_performance_regression_baseline
);

criterion_main!(
    rolling_hash_benches,
    cdc_benches,
    compression_benches,
    memory_benches,
    serialization_benches,
    concurrent_benches,
    regression_benches
);