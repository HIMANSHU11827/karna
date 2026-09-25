# Storage & Deployment Survey for Real-Time AGI Assistant

**Date:** 2026-09-23  
**Author:** performance-fixer  
**Scope:** Model weight storage, real-time inference deployment, I/O bottlenecks, custom format design

---

## 1. Model Weight Storage Formats

### 1.1 Pickle (.pt, .bin, .pkl) — Legacy, Unsafe

**How it works:** Serializes Python object trees into a bytecode stream. Loading executes opcodes to reconstruct objects.

**Critical flaws:**
- **Remote Code Execution by design.** Pickle is a stack-based virtual machine. `torch.load()` executes `__reduce__` callbacks that can run arbitrary code (e.g., `os.system('rm -rf /')`).
- **CVE-2025-32434:** `torch.load(weights_only=True)` was exploitable even with the safety flag on PyTorch 2.5.1 and earlier.
- **CVE-2026-12484:** Keras 3.15.0 deserialized attacker-controlled PyTorch pickle data through public API.
- **No partial loading.** Must parse entire stream to discover tensor layout.
- **No zero-copy.** Data copied through Python heap → tensor → device (3 traversals).
- **Slow:** ~172ms for 500MB weights (vs 3.5ms for safetensors — **50x slower**).

**Verdict:** Never for untrusted checkpoints. Convert to safetensors at ingest, then discard original.

### 1.2 Safetensors — Recommended Default

**Format:**
```
[8 bytes: header length N][N bytes: JSON header][raw tensor bytes]
```
Header maps tensor names → {dtype, shape, data_offsets}. No opcodes, no callables.

**Advantages:**
- **Zero-copy loading:** `mmap()` the file, tensors point directly into mapped region. One traversal (device transfer only).
- **Lazy/partial tensor loading:** Read header → load only needed tensors.
- **Safe:** No code execution possible. Only risk is index manipulation (path traversal in shard index — CVE-2026-65920).
- **Fast:** 50x faster than pickle on CPU, 2x faster on GPU.
- **Framework-agnostic:** PyTorch, TF, JAX, Flax, NumPy.
- **Now under PyTorch Foundation** (April 2026), industry standard.

**For our use case:** Ideal for weight storage. Zero-copy mmap means fast save/load, essential for real-time continuous learning.

### 1.3 GGUF — Edge/CPU Optimized

**Format:** Single-file container with KV metadata, per-tensor quantization, tokenizer, weights.

**Best for:** Quantized inference on CPU/edge (llama.cpp). Mixed per-tensor quantization (Q4_K_M, Q8_0, etc.).

**Not for us:** Our architecture doesn't use transformer blocks that GGUF is optimized for. Single-file convenience doesn't outweigh lack of sparse tensor support.

### 1.4 ONNX — Cross-Framework Graph Format

**Format:** Protobuf computation graph + external data files.

**Best for:** Cross-framework deployment. Training in PyTorch → export to ONNX → run on TensorRT/OpenVINO/ONNX Runtime.

**Limitations:**
- Dynamic control flow is hard to express.
- Protobuf 2GB limit requires external data files for large models.
- Sidecar consistency problem (graph + data must stay in sync).

### 1.5 TensorRT Engine (.engine/.plan) — Hardware-Specific

**Best for:** Fixed model, fixed GPU, latency-critical inference.

**Not for us:** Not portable across hardware. Built for specific GPU architecture. Doesn't support training or continuous learning.

---

## 2. Real-Time Inference Deployment

### 2.1 Benchmark Results (from literature)

| Framework | ResNet152 Latency | MobileNet Latency | SqueezeNet Throughput |
|-----------|------------------|-------------------|----------------------|
| TensorRT FP16 | 2.28ms | ~1ms | 3197 req/s |
| ONNX Runtime CUDA | 283ms | ~40ms | 106 req/s |
| PyTorch eager | ~30ms | ~5ms | 1887 req/s |
| TVM | ~15ms | ~3ms | 456 req/s |

**TensorRT is 10-100x faster than ONNX Runtime** due to:
- Static graph optimization
- Layer/tensor fusion
- Kernel auto-tuning
- FP16/INT8 quantization
- CUDA graph capture

### 2.2 For Our Custom Architecture

Since we're NOT using standard transformers, we can't use TensorRT-LLM or vLLM out of the box. Options:

1. **Custom CUDA kernels** — fastest, most work
2. **TVM** — auto-tune for our ops, good middle ground
3. **ONNX export + ONNX Runtime** — portable, moderate speed
4. **PyTorch eager → torch.compile** — easiest, decent speed

**Recommendation:** Start with PyTorch eager + `torch.compile` for prototyping. Profile hot paths, then write custom CUDA kernels for the bottleneck ops (sparse coding inference, attractor dynamics, fusion).

### 2.3 Latency-Throughput Tradeoffs

| Scenario | Best Choice |
|----------|-------------|
| Single request, low latency | TensorRT/CUDA |
| Batch throughput | TensorRT with large batches |
| Irregular batch sizes | ONNX Runtime or PyTorch |
| Cross-platform | ONNX |

**For real-time AGI:** We need **bounded single-request latency**, not throughput. Custom CUDA kernels with CUDA graphs for our specific ops.

---

## 3. I/O Bottlenecks for Real-Time Systems

### 3.1 Storage Hierarchy Latency

| Layer | Latency | Notes |
|-------|---------|-------|
| L1 cache | ~1ns | 32-64KB per core |
| L2 cache | ~3-10ns | 256KB-1MB per core |
| L3 cache | ~10-50ns | Shared, 6-30MB |
| RAM (DDR4/DDR5) | ~50-100ns | 16-64GB typical |
| NVMe SSD (read) | ~50µs | 7100 MB/s sequential |
| NVMe SSD (write) | ~12µs | 6000 MB/s sequential |
| SATA SSD | ~100µs | 550 MB/s |
| Network (datacenter) | ~100µs-1ms | 10-100 Gbps |

**Key insight:** RAM is 1000x faster than NVMe. NVMe is 1000x faster than network.

### 3.2 Real-Time I/O Bottlenecks

**For our AGI assistant:**

1. **Model weight loading at startup:** If weights are on disk, loading 1GB takes ~150ms (NVMe) to seconds (SATA). Solution: mmap + lazy loading (safetensors).

2. **Continuous learning weight updates:** Writing weights after every interaction. If synchronous, adds disk latency to response time. Solution: async write-back, or keep weights in RAM and snapshot periodically.

3. **Memory-mapped I/O:** The OS page cache means repeated reads are near-RAM speed if data is hot. First access pays disk cost.

4. **Write amplification:** Small writes (e.g., single synapse update) cause read-modify-write cycles on SSD. Solution: batch updates, log-structured merge.

### 3.3 CPU vs I/O Bottleneck Trend

Modern NVMe SSDs are so fast that **CPU is now the bottleneck**, not I/O. A single core can saturate a Gen4 NVMe drive at queue depth 32. For real-time systems:
- Use polling (not interrupts) to avoid context switch overhead
- Dedicate CPU cores to I/O if needed
- Minimize copies (zero-copy everywhere)

---

## 4. Custom Storage Format Design

### 4.1 Requirements for Our AGI Assistant

Based on the architecture:

1. **Fast save/load:** Continuous learning requires frequent weight persistence. Target: <10ms for full checkpoint.

2. **Sparse representation:** Our architecture uses sparse codes. Format should natively support sparse tensors (CSR/CSC format) without dense storage.

3. **Incremental updates:** Don't rewrite entire model for a single synapse update.

4. **Memory-mappable:** Zero-copy loading for real-time performance.

5. **Versioned:** Keep history for rollback (important for continual learning).

6. **Safe:** No code execution on load.

### 4.2 Proposed Format: SparseTensors Real-Time (STR)

**File layout:**
```
[Magic: "STR\0"][Version: u32][Header size: u64]
[JSON Header: tensor metadata]
[Data Section: raw bytes]
[Footer: checksum (SHA-256)]
```

**JSON Header:**
```json
{
  "model_id": "agi-assistant-v1",
  "timestamp": 1727000000,
  "layers": [
    {
      "name": "input_sparse.dictionary",
      "dtype": "f32",
      "shape": [784, 64],
      "format": "dense",
      "offset": 0,
      "size": 200704
    },
    {
      "name": "catch_layer_0.W_ff",
      "dtype": "f32",
      "shape": [64, 32],
      "format": "csr_sparse",
      "sparsity": 0.95,
      "offset": 200704,
      "size": 409600
    }
  ],
  "metadata": {
    "total_parameters": 100000,
    "sparse_ratio": 0.92,
    "learning_steps": 50000
  }
}
```

**Sparse tensor encoding (CSR):**
```
[u32: num_rows][u32: nnz][row_indices: u32*nnz][col_ptr: u32*(rows+1)][values: f32*nnz]
```

### 4.3 Performance Characteristics

| Operation | Expected | Notes |
|-----------|----------|-------|
| Full model load (100MB sparse) | <5ms | mmap + zero-copy |
| Single tensor load | <0.1ms | Direct offset |
| Full checkpoint save | <10ms | Sequential write |
| Incremental update (1 tensor) | <0.1ms | Seek + write sparse block |
| Memory overhead | ~0% | No copies, mmap only |

### 4.4 Comparison with Existing Formats

| Format | Sparse Support | Zero-Copy | Incremental | Safe |
|--------|---------------|-----------|-------------|------|
| STR (proposed) | Yes (native) | Yes | Yes | Yes |
| Safetensors | No | Yes | No | Yes |
| GGUF | No | Yes | No | Yes |
| ONNX | No | Partial | No | Yes |
| Pickle | Yes (Python) | No | No | **No** |

### 4.5 Implementation Plan

**Phase 1 (Now):**
- Use safetensors for standard dense tensors
- Simple numpy `.npz` for sparse matrices

**Phase 2 (After architecture stabilizes):**
- Implement STR format
- Benchmark against safetensors for our workload
- Add incremental write support (log-structured)

**Phase 3 (Production):**
- Memory-mapped incremental updates
- Versioned checkpoints (copy-on-write)
- Compression for sparse tensors

---

## 5. Recommendations Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Weight storage format | **Safetensors** (now), custom STR (later) | Safe, zero-copy, fast |
| Sparse tensor handling | CSR encoding in custom format | 95%+ sparsity in our architecture |
| Inference deployment | **PyTorch eager + torch.compile** now, custom CUDA later | Fastest to prototype, then optimize hot paths |
| I/O strategy | **Async write-back, mmap load** | Don't block response on persistence |
| Incremental updates | **Log-structured** append | Batch synapse updates, avoid read-modify-write |
| Safety | **Never pickle for untrusted data** | Use safetensors, validate shard indices |

---

## References

1. Safetensors security audit (Trail of Bits, 2026-06-25) — Hugging Face
2. PyTorch Foundation Safetensors announcement (2026-04-08)
3. CVE-2025-32434 — PyTorch pickle RCE
4. MDPI Electronics 14(15):2977 — Inference framework benchmark (Jetson Orin)
5. TensorRT 11.3.0 Performance Benchmarking — NVIDIA docs
6. USENIX OSDI'18 — FLASHSHARE: ultra-low latency SSD stack
7. USENIX FAST'21 — D2FQ: device-direct fair queueing for NVMe
8. arXiv:2210.04323 — Deep learning inference frameworks benchmark
