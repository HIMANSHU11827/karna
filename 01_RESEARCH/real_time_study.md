# Real-Time Systems: Streaming ML, Online Learning, and Deployment

**Date:** 2026-09-23  
**Author:** performance-fixer  
**Scope:** Streaming machine learning, online learning algorithms, latency/throughput tradeoffs, real-time inference deployment

---

## 1. Streaming Machine Learning

### 1.1 Definition

Streaming ML processes data **one sample at a time** (or in micro-batches) as it arrives, rather than waiting for a full dataset. Key properties:

- **Single-pass:** Each sample seen once (or buffered briefly)
- **Bounded memory:** Cannot store all historical data
- **Adaptive:** Model updates with each new sample
- **Low latency:** Prediction available immediately

### 1.2 Streaming Architectures

| Architecture | Strengths | Weaknesses | Use Case |
|-------------|-----------|------------|----------|
| **River** (Python) | Online algorithms, drift detection, metrics | Single-threaded | Real-time classification, regression |
| **Apache Flink** | Stateful stream processing, exactly-once | JVM overhead, complex setup | Large-scale event processing |
| **Spark Structured Streaming** | Unified batch/streaming API | Micro-batch latency (100ms+) | ETL + ML pipelines |
| **Kafka Streams** | Lightweight, embedded in Kafka apps | Kafka-coupled | Event-driven microservices |
| **FogML** | Edge/embedded, microcontroller | Limited algorithms | IoT sensor processing |

### 1.3 Concept Drift

Data distributions change over time. Types:

| Drift Type | Description | Example |
|-----------|-------------|---------|
| **Sudden** | Immediate change | Sensor failure, system upgrade |
| **Gradual** | Slow shift over time | Seasonal trends |
| **Incremental** | Continuous evolution | User preference changes |
| **Recurring** | Previously seen patterns return | Seasonal behavior |

**Detection methods:**
- **ADWIN (Adaptive Windowing):** Monitors sliding window statistics, detects mean changes
- **Page-Hinkley:** Cumulative sum-based, good for gradual drift
- **KS-test:** Kolmogorov-Smirnov test on feature distributions
- **DDM (Drift Detection Method):** Monitors error rate, warns at threshold

**Our approach:** Since we learn from every interaction, drift is implicit — the model adapts continuously. No explicit drift detection needed for our scale.

---

## 2. Online Learning Algorithms

### 2.1 Definition

Online learning updates model parameters **after each labeled example**, without batching. Key properties:

- **Regret minimization:** Goal is to match the best fixed predictor in hindsight
- **No storage:** Does not retain historical data (only model state)
- **Immediate feedback:** Label available instantly (or within delay T)
- **Stochastic updates:** SGD and its variants are the backbone

### 2.2 Algorithm Taxonomy

| Category | Algorithm | Update Rule | Complexity | Best For |
|----------|-----------|-------------|------------|----------|
| **First-order** | Online SGD | θ ← θ - η∇f_t(θ) | O(d) | Simple, convex |
| **First-order** | AdaGrad | η_t = η/√(Σg²) | O(d) | Sparse features |
| **First-order** | Adam | Adaptive moments | O(d) | General nonconvex |
| **Second-order** | Online Newton | Hessian inverse | O(d²) | Small d, ill-conditioned |
| **Online learning** | FTRL | Regularized leader | O(d) | L1 regularization |
| **Bandit** | UCB | Confidence bound | O(k arms) | Discrete choices |
| **Bandit** | Thompson Sampling | Bayesian posterior | O(d²) | Exploration/exploitation |

### 2.3 Adaptive Learning Rates (2024-2025)

Recent advances in learning rate adaptation:

**GALA (Gradient Alignment-based Learning Rate Adaptation):**
- Tracks alignment between consecutive gradients
- Increases η when gradients align (consistent direction)
- Decreases η when gradients oscillate
- Uses FTRL for one-dimensional online learning on η
- Provable convergence for nonconvex smooth objectives
- **Relevance to us:** UALR in our design uses a similar principle

**CLARA (Cumulative Learning Rate Adaptation):**
- Cumulative path length vs expected random walk
- η increases when path is longer than expected (good progress)
- η decreases when path is shorter (oscillating)
- Works with Adam (corrected for preconditioning)
- **Relevance to us:** Could replace UALR with CLARA for stability

**Online Learning-guided (Coin-betting):**
- Formulates SGD as a coin-betting game
- No hyperparameter tuning
- Competitive with tuned schedules
- **Relevance to us:** Theoretical basis for parameter-free learning

### 2.4 Key Tradeoffs

| Tradeoff | Option A | Option B | Our Choice |
|----------|----------|----------|------------|
| **Adaptivity vs stability** | High learning rate (fast adapt) | Low learning rate (stable) | Multi-timescale (both) |
| **Per-parameter vs global** | Adam (per-param) | SGD (global) | Per-parameter for sparse, global for time constant |
| **Forgetting vs retaining** | Decay (forget old) | No decay (remember all) | Slow decay (CLS theory) |
| **Compute vs accuracy** | More iterations | Fewer iterations | Fixed iteration cap (bounded) |
| **Memory vs capacity** | Small SDR (compact) | Large SDR (high capacity) | N=1024, k=20 |

### 2.5 Catastrophic Forgetting

Online learning tends to forget old tasks when learning new ones. Mitigations:

| Method | Mechanism | Cost |
|--------|-----------|------|
| **Elastic Weight Consolidation (EWC)** | Penalize changes to important weights | O(d²) Fisher matrix |
| **Progressive Networks** | Freeze old columns, add new | Linear growth |
| **Memory Replay** | Replay old examples | Storage + compute |
| **Sparse Representations** | Different neurons for different tasks | Higher dimensionality |
| **Complementary Learning Systems** | Fast (episodic) + slow (semantic) | Two networks |

**Our choice:** Sparse representations + complementary memory. Different neurons naturally handle different tasks. Slow consolidation prevents overwriting.

---

## 3. Latency/Throughput Tradeoffs

### 3.1 Definitions

| Metric | Definition | Unit |
|--------|-----------|------|
| **Latency (response time)** | Time from request to response | ms |
| **Throughput** | Requests per second | req/s |
| **Tail latency (p99)** | 99th percentile latency | ms |
| **Utilization** | Fraction of time busy | % |
| **Queue depth** | Waiting requests | count |

### 3.2 Little's Law

```
L = λW
```
Where:
- L = average number of items in system
- λ = arrival rate (req/s)
- W = average time in system (seconds)

**Implication:** At 1000 req/s and 10ms latency, average queue depth = 10.

### 3.3 Latency Components

```
Total latency = Queue time + Processing time + Network time + I/O time
```

For our real-time AGI:

| Component | Typical | Target |
|-----------|---------|--------|
| Queue time | 0-100ms (depends on load) | <1ms |
| Processing (inference) | 1-50ms | <10ms |
| Processing (learning) | 0.1-1ms | <1ms |
| I/O (memory) | 0.05ms | <0.1ms |
| I/O (disk, if any) | 10-100ms | 0 (async) |
| **Total end-to-end** | **11-161ms** | **<15ms** |

### 3.4 Throughput Optimization

**Batching:** Process multiple samples together

| Batch Size | Latency | Throughput | Efficiency |
|-----------|---------|------------|------------|
| 1 | 1ms | 1000 req/s | Low |
| 8 | 3ms | 2667 req/s | Medium |
| 32 | 8ms | 4000 req/s | High |
| 128 | 25ms | 5120 req/s | Peak |

**Problem:** Real-time single-request latency **increases** with batching. Solution: **micro-batching** (batch=1-4) or **continuous batching** (variable batch size based on queue).

**Our approach:** No batching for inference (batch=1). Continuous batching only for background consolidation.

### 3.5 Tail Latency

Average latency is useless. P99 matters.

**Causes of tail latency:**
1. **GC pauses** (Python, Java): 10-100ms spikes
2. **OS scheduling:** Context switches, interrupts
3. **Disk I/O:** SSD garbage collection, wear leveling
4. **Network:** TCP retransmission, buffer bloat
5. **Queueing:** Bursty arrivals cause queue buildup

**Mitigations:**
- Pre-allocate memory (avoid GC)
- Pin threads to CPU cores (avoid context switches)
- Use polling instead of interrupts
- Separate I/O thread from processing thread
- Admission control (reject if queue depth > threshold)

**Our approach:**
- NumPy pre-allocates arrays (minimal GC)
- All operations are O(1) or O(n) bounded
- No disk I/O on critical path
- Async write-back for persistence

### 3.6 Queuing Theory Basics

**M/M/1 queue (single server, Poisson arrivals):**
- Average queue depth: L = ρ/(1-ρ) where ρ = λ/μ (utilization)
- Average wait time: W = 1/(μ-λ)

**At ρ=0.5:** W = 2/μ (double the processing time)  
**At ρ=0.9:** W = 10/μ (10x processing time)  
**At ρ=0.99:** W = 100/μ (unusable)

**Rule of thumb:** Keep utilization <70% for predictable latency.

**Our utilization budget:**
- Processing time: 10ms per request
- Target throughput: 100 req/s per core
- Required service rate: 1000 req/s
- Utilization: 100/1000 = 10% ✓ (excellent)

---

## 4. Real-Time Inference Deployment

### 4.1 Deployment Patterns

| Pattern | Description | Latency | Throughput |
|---------|-------------|---------|------------|
| **Embedded** | Model runs on edge device | <1ms | Low |
| **On-premise server** | Local GPU/CPU server | 1-10ms | Medium |
| **Cloud server** | Remote data center | 10-100ms | High |
| **CDN edge** | Cloudflare Workers, etc. | 5-50ms | Very high |

**Our choice:** On-premise server (local CPU). No network latency, no egress costs.

### 4.2 Inference Optimization Techniques

| Technique | Speedup | Cost | Our Applicability |
|-----------|---------|------|-------------------|
| **Operator fusion** | 2-5x | None | Yes (NumPy already fuses) |
| **Quantization (INT8)** | 2-4x | Small accuracy loss | Future (current: FP32) |
| **Pruning** | 2-10x | Accuracy loss | Already sparse (95%) |
| **Batching** | 2-8x | Latency increase | No (batch=1) |
| **CUDA Graphs** | 1.5-2x | GPU only | N/A (CPU) |
| **JIT compilation** | 1.5-3x | Compile time | Yes (NumPy is already compiled) |
| **Memory pooling** | 1.2-1.5x | Code complexity | Yes (pre-allocate) |

### 4.3 Serving Infrastructure

**Single-process (simplest):**
```
[Client] → [Python process] → [NumPy] → [Response]
```
- Pros: Simple, no IPC overhead
- Cons: Single-threaded, GIL-bound

**Multi-process (production):**
```
[Load balancer] → [Worker 1] → [Model]
                → [Worker 2] → [Model]
                → [Worker 3] → [Model]
```
- Pros: Parallelism, fault isolation
- Cons: Memory per worker, serialization overhead

**Our architecture:** Single-process with NumPy threading. For production: multi-process with shared memory.

### 4.4 Monitoring

Key metrics to track:

| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| P50 latency | >10ms | Profile hot path |
| P99 latency | >50ms | Scale up, optimize |
| Throughput | <100 req/s | Add workers |
| Memory growth | >10MB/hour | Check for leaks |
| Error rate | >0.1% | Investigate |
| Queue depth | >5 | Admission control |

---

## 5. Streaming RAG (Retrieval-Augmented Generation)

### 5.1 What is Streaming RAG?

Traditional RAG uses a static index. Streaming RAG continuously updates the index as new documents arrive.

**Key properties:**
- **Incremental updates:** New docs added without full reindex
- **Temporal awareness:** Recent docs weighted higher
- **Low-latency retrieval:** <10ms query time
- **Scalable:** Handles millions of docs

### 5.2 Architecture

```
[Document stream] → [Ingestion] → [Embedding] → [Vector store]
                                               ↓
[Query] → [Encoder] → [ANN search] → [Context assembly] → [LLM]
```

**ANN (Approximate Nearest Neighbor) algorithms:**

| Algorithm | Build Time | Query Time | Memory | Accuracy |
|-----------|------------|------------|--------|----------|
| **Brute force** | O(1) | O(n) | O(nd) | 100% |
| **LSH** | O(n) | O(1) | O(n) | 80-90% |
| **HNSW** | O(n log n) | O(log n) | O(n) | 95-99% |
| **IVF** | O(n) | O(√n) | O(n) | 90-95% |
| **PQ** | O(n) | O(√n) | O(n/4) | 85-95% |

**For our scale:**
- <10k episodes: Brute force (5ms, 100% accuracy)
- 10k-100k episodes: LSH (1ms, 90% accuracy)
- >100k episodes: HNSW (0.5ms, 99% accuracy)

### 5.3 Temporal Knowledge Graphs

For reasoning about time:
- Store events with timestamps
- Query: "What happened at time T?"
- Supports temporal logic: before, after, during

**For us:** Episodic memory with timestamp indexing. Can query "what happened at time T" or "what was the context for event X".

---

## 6. Real-Time Constraints for Our AGI

### 6.1 Latency Budget (from design)

| Operation | Measured | Budget | Margin |
|-----------|----------|--------|--------|
| Text encode | 0.7ms | 10ms | 9.3ms |
| Attractor converge | 2-4ms | 10ms | 6ms |
| Predict (forward) | 0.1ms | 5ms | 4.9ms |
| Predict (learn) | 0.3ms | 1ms | 0.7ms |
| Retrieve | 5ms | 15ms | 10ms |
| Response assembly | 0.1ms | 5ms | 4.9ms |
| **Total** | **8-10ms** | **50ms** | **40ms** |

### 6.2 Throughput Budget

| Resource | Capacity | Utilization |
|----------|----------|-------------|
| CPU cores | 6 (i3-1315U) | 1 core @ 10% for 100 req/s |
| Memory | 15GB | <50MB for model |
| Disk I/O | 7000 MB/s (NVMe) | 0 (async) |

### 6.3 Scaling Limits

| Bottleneck | Limit | Solution |
|------------|-------|----------|
| Single-threaded Python | ~1000 req/s | Multi-process |
| Brute-force retrieval | ~20k episodes | Switch to HNSW |
| Memory for episodes | ~100k episodes | Consolidation + semantic |
| NumPy overhead per call | ~0.05ms | Cython for hot paths |
| Disk I/O (checkpoint) | ~10ms | Async, mmap |

---

## 7. Key Insights for Implementation

### 7.1 Online Learning Is NOT Batch Learning

- **Do not** accumulate gradients and apply in batch
- **Do not** store replay buffers for training
- **Do not** use learning rate schedules designed for epochs
- **Do** update weights after every interaction
- **Do** use adaptive learning rates (GALA, CLARA, or our UALR)
- **Do** monitor gradient alignment for convergence signals

### 7.2 Latency Is a System Property

- **Do not** optimize algorithms in isolation
- **Do not** assume faster algorithm = faster response
- **Do** profile end-to-end, including Python overhead
- **Do** measure P99, not average
- **Do** design for bounded worst-case, not average case

### 7.3 Storage Matters

- **Do not** use pickle (unsafe, slow)
- **Do not** rewrite entire model per update
- **Do** use safetensors + mmap for weights
- **Do** use async write-back for persistence
- **Do** use log-structured merge for incremental updates

### 7.4 Streaming Requires New Design Patterns

- **Do not** store all data (memory explosion)
- **Do not** retrain from scratch (latency spike)
- **Do** use fixed-size buffers with eviction
- **Do** consolidate periodically (sleep phase)
- **Do** use approximate algorithms for scale

---

## 8. Summary Table

| Area | Key Finding | Our Approach |
|------|-------------|--------------|
| **Streaming ML** | Single-pass, bounded memory, drift detection | Implicit drift handling via continuous learning |
| **Online learning** | SGD variants, adaptive rates, regret minimization | Multi-timescale Hebbian + adaptive rates |
| **Latency/throughput** | Little's law, tail latency, batching tradeoffs | Batch=1, bounded operations, async I/O |
| **Inference deployment** | On-premise, quantization, fusion | Local CPU, NumPy, sparse ops |
| **Streaming RAG** | Incremental index, ANN search, temporal graphs | Episodic memory with timestamped retrieval |
| **Real-time constraints** | <50ms, P99, bounded worst-case | 8-10ms measured, 40ms margin |

---

## References

1. Adaptive AI for Real-Time and Streaming Data Processing (arXiv, 2025-09)
2. MODL: Multilearner Online Deep Learning (PMLR, 2025)
3. Supervised Learning from Data Streams (arXiv:2212.14720)
4. Ferret: Efficient Online Continual Learning (arXiv:2503.12053)
5. GALA: Gradient Alignment-based LR Adaptation (arXiv:2506.08419)
6. CLARA: Cumulative LR Adaptation (arXiv:2508.05408)
7. ELITE: Device-Cloud Collaborative Online CL (WWW 2025)
8. River ML library (online-ml/river)
9. Apache Flink streaming engine
10. HNSW ANN algorithm (Malkov & Yashunin, 2018)
