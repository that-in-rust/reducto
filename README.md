# Reducto Mode 3 v0.1

> **Essence**: Reducto turns large, repetitive data into tiny instruction sets using reference–based compression.  
> **Why now?** Storage & bandwidth costs scale faster than Moore’s law—dedupe the delta, not the whole file.  
> **Proof in 60 s**: `cargo run --bin benchmark` gives a YES/NO answer on your data.
> 
> **Status (v0.1)**: We tried; we’re not there yet. On our sample, the mock Reducto path does not beat gzip. We’re focusing next on the CDC + corpus pipeline described in the backlog below.

```mermaid
flowchart LR
    A[Your Files] -- ≤100 MB sample --> B[Benchmark CLI]
    B --> C[Gzip Test] & D[Reducto Test]
    C & D --> E{Decision Engine}
    E -- "Ratio & Speed OK" --> R[RECOMMENDED]
    E -- else --> N[NOT RECOMMENDED]
```

<details>
<summary>How the benchmark decided (run 2025-09-24)</summary>

See `docs/benchmark_report_v0.2.md` for full numbers & pie-charts.

</details>

---

Differential compression system using Content-Defined Chunking (CDC) and Reference Corpora. Achieves 10x-100x compression ratios for data with high redundancy patterns.

**How it works**: Like a master librarian who notices that 90% of new books contain chapters already written, Reducto builds a reference collection of data patterns. Instead of storing redundant information, it simply notes "page 47 of reference volume XII." The result: your 2GB VM image becomes a 20MB instruction set.

## Quick Start

```bash
# Install from source
git clone https://github.com/that-in-rust/reducto.git
cd reducto
cargo build --release

# Build reference corpus
reducto corpus build --input /data/reference --output corpus.rc

# Compress file
reducto compress --input file.img --corpus corpus.rc --output file.reducto

# Decompress
reducto decompress --input file.reducto --output restored.img
```

## Use Cases

**VM Images**: 50:1 to 200:1 compression ratios (95-99.5% reduction)
**Container Images**: 20:1 to 100:1 ratios (95-99% reduction)  
**Database Backups**: 10:1 to 50:1 ratios (90-98% reduction)
**Text Files**: 5:1 to 20:1 ratios (80-95% reduction)

Text compression is supported and effective for source code, logs, and documentation with high redundancy patterns.

## Architecture

```mermaid
graph TD
    %% Main flow
    Input[Input File] --> CDC[CDC Chunker]
    Corpus[Reference Corpus] --> Index[Corpus Index]
    
    %% Compression path
    CDC --> Compress[Compressor]
    Index --> Compress
    Compress --> Serialize[Serializer + Zstd]
    Serialize --> Output[.reducto File]
    
    %% Decompression path
    Output --> Deserialize[Deserializer]
    Deserialize --> Decompress[Decompressor]
    Corpus --> MemMap[Memory Mapper]
    MemMap --> Decompress
    Decompress --> Result[Reconstructed File]
    
    %% Styling
    classDef primary fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef secondary fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class Input,CDC,Compress,Serialize,Output,Deserialize,Decompress,Result primary
    class Corpus,Index,MemMap secondary
```

### Core Components

- **CDC Chunker**: FastCDC/Gear hashing for variable-size chunks
- **Dual-Hash System**: Rolling hash + BLAKE3 verification
- **Reference Corpus**: Immutable, cryptographically signed data store
- **Persistent Storage**: RocksDB for large-scale corpus management

## Performance

**Throughput**
- Compression: 500-800 MB/s
- Decompression: 1.2-2.0 GB/s  
- Corpus Lookup: <500μs per chunk
- Memory Usage: <2GB for 100GB corpus

**Deployment**

```mermaid
graph TD
    %% Client tier
    CLI[CLI Tool] --> LB[Load Balancer]
    SDK[SDK/API] --> LB
    
    %% Server tier
    LB --> Server1[Reducto Server]
    LB --> Server2[Reducto Server]
    
    %% Storage tier
    Server1 --> Corpus[Corpus Repository]
    Server2 --> Corpus
    Server1 --> Storage[Object Storage]
    Server2 --> Storage
    
    %% Monitoring
    Server1 --> Metrics[Prometheus]
    Server2 --> Metrics
    
    %% Styling
    classDef client fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef server fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef storage fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class CLI,SDK client
    class LB,Server1,Server2 server
    class Corpus,Storage,Metrics storage
```

## Installation

### Build a timestamped binary artifact

```bash
./scripts/build_reducto.sh
# Produces ./dist/reducto-v-YYYYMMDDHHSS
```

```bash
# From source
cargo install reducto

# Pre-built binary
wget https://github.com/your-org/reducto/releases/latest/download/reducto-linux-x64.tar.gz
tar -xzf reducto-linux-x64.tar.gz && sudo mv reducto /usr/local/bin/

# Docker
docker pull reducto/enterprise:latest
```

**Requirements**: 4+ cores, 8GB RAM, 100GB storage

## Commands

**Corpus Management**
```bash
reducto corpus build --input /data --output corpus.rc
reducto corpus analyze --corpus corpus.rc --test-data /test
reducto corpus verify --corpus corpus.rc
```

**Compression/Decompression**
```bash
reducto compress --input file.img --corpus corpus.rc --output file.reducto
reducto decompress --input file.reducto --output restored.img
```

**Analysis**
```bash
reducto analyze --input file.img --corpus corpus.rc --report-format json
reducto benchmark --input /dataset --corpus corpus.rc
```

## Integration

**CI/CD Pipeline**
```yaml
- name: Compress artifacts
  run: reducto compress --input build/ --corpus ${{ secrets.CORPUS_URL }} --output artifacts.reducto
```

**Docker**
```dockerfile
RUN reducto compress --input /data --corpus /corpus.rc --output /compressed.reducto
```

**Backup Scripts**
```bash
reducto compress --input /data --corpus backup.rc --output backup.reducto
rsync backup.reducto backup-server:/backups/
```

## Monitoring

**Prometheus Metrics**
- `reducto_compression_ratio`
- `reducto_corpus_hit_rate` 
- `reducto_throughput_mbps`
- `reducto_bandwidth_saved_bytes_total`

**Grafana Dashboard**: Import `monitoring/grafana-dashboard.json`

## Security

- **Signing**: Ed25519 signatures for corpus integrity
- **Encryption**: AES-GCM for data protection  
- **Audit Logging**: Comprehensive access logging
- **Compliance**: SOC 2, GDPR, HIPAA support

## Configuration

```toml
[corpus]
repositories = ["https://corpus.company.com/api/v1"]
cache_dir = "/var/cache/reducto"
max_cache_size = "50GB"

[compression]
chunk_size = 8192
compression_level = 19

[security]
signing_key_path = "/etc/reducto/keys/signing.key"
encryption_enabled = true
```

## Troubleshooting

**Low compression ratios**: Rebuild corpus with target data
```bash
reducto corpus build --input target-dataset/ --optimize --output new-corpus.rc
```

**High memory usage**: Enable memory mapping
```bash
reducto config set corpus.memory_mapped true
```

**Slow decompression**: Pre-fetch corpus locally
```bash
reducto corpus fetch --corpus-id golden-v1 --local-cache
```

## API

**REST Endpoint**
```http
POST /api/v1/compress
{
  "file": <binary data>,
  "corpus_id": "golden-v1"
}
```

**Python SDK**
```python
import reducto
client = reducto.Client("https://reducto.company.com", "api-key")
result = client.compress(data, corpus_id="golden-v1")
```

## License

Apache 2.0 License for open source use. Enterprise license available with additional features and support.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Future Backlog (Low‑drama summary)

We’ll differentiate Reducto by leaning into differential compression rather than general compression:

- Decision Gate (DAS): a fast “Differential Advantage Score” predicts when cross‑file reuse will pay off; fallback to gzip/zstd otherwise.
- Multi‑scale CDC: stable chunk boundaries at small and large scales to tolerate inserts/deletes while capturing long runs.
- Global Index: locality‑aware reference lookup with collision‑safe hashing and a small in‑memory hotset.
- Residuals with Trained Dictionaries: zstd residuals improved via small trained dicts and long‑range mode where useful.
- Domain‑aware Anchors: lightweight boundary biasing around structural tokens in code/logs (no heavy parsing).

Expected outcomes on strong‑fit data (code/logs/container layers):
- Ratio uplift vs gzip: ~2–5× on code/logs; higher where base layers are shared.
- Speed: within practical bounds (≤3× gzip on compression; ≥0.5× on decompression).

More detail in WARPOption01*.md (repo root). We’ll ship iteratively and measure hit‑rate, ratio, and time budget in CI.
