# RALM Design Performance Profile

**Date:** 2026-09-23  
**Analyst:** performance-fixer  
**Design:** `01_RESEARCH/synthesis.md` + `01_RESEARCH/storage_deployment_survey.md`

---

## Executive Summary

| Category | Verdict | Confidence |
|----------|---------|------------|
| **Real-time inference** | **AT RISK** | Medium |
| **Learning latency** | **PASS** | High |
| **Memory usage** | **PASS** | High |
| **I/O bottlenecks** | **PASS** | High |
| **Throughput** | **LIMITED** | Medium |

**Bottom line:** The design CAN meet real-time targets IF the attractor convergence is bounded and retrieval is accelerated. As specified, it risks blowing the 50ms budget.

---

## Component-by-Component Analysis

### 1. Multimodal Encoder

| Sub-component | Complexity | Est. Time | Risk |
|---------------|-----------|-----------|------|
| Text n-gram hashing | O(n), n~100 chars | **<0.1ms** | LOW |
| Image random projection | O(768×64) = 50k ops | **~0.05ms** | LOW |
| Audio FFT | O(N log N), N=1600 | **~0.05ms** | LOW |
| Sparse code encoding (ISTA) | O(k×d×steps), k=10, d=64, steps=10 | **~0.5ms** | MEDIUM |

**Concern:** ISTA encoding has a fixed 10-step loop. If steps increase for quality, latency scales linearly. For production: bound at 5 steps.

**Total encoder: ~0.7ms (well within budget)**

---

### 2. Attractor Memory

| Sub-component | Complexity | Est. Time | Risk |
|---------------|-----------|-----------|------|
| Pattern completion | O(d²) for d=64 | **~0.4ms per step** | HIGH |
| Convergence check | O(d) | **~0.01ms** | LOW |

**CRITICAL ISSUE:** The synthesis says "iterative attractor networks" but doesn't specify a fixed iteration count. Hopfield networks can take 10-100+ iterations to converge. At 0.4ms/step:
- 10 steps = 4ms ✓
- 50 steps = 20ms ⚠️
- 100 steps = 40ms ✗ (blows budget)

**Mitigation:** Use **fixed 5-10 iterations**. Don't wait for convergence. This sacrifices some quality for bounded latency. The energy function will still decrease significantly in 5 steps.

**Total attractor: ~2-4ms (bounded)**

---

### 3. Real-Time Predictor

| Sub-component | Complexity | Est. Time | Risk |
|---------------|-----------|-----------|------|
| Sparse matmul (forward) | O(nnz), 95% sparse, d=1000 | **~0.1ms** | LOW |
| Eligibility trace update | O(nnz) | **~0.1ms** | LOW |
| Prediction error compute | O(d) | **~0.01ms** | LOW |
| Weight update | O(nnz) | **~0.1ms** | LOW |

**Total predictor: ~0.3ms (excellent)**

This is the strongest part of the design. Sparse operations are inherently fast.

---

### 4. Response Generator

| Sub-component | Complexity | Est. Time | Risk |
|---------------|-----------|-----------|------|
| Retrieve top-k (brute force) | O(n×d), n=10k, d=64 | **~5ms** | MEDIUM |
| Retrieve top-k (brute force) | O(n×d), n=100k, d=64 | **~50ms** | HIGH |
| Perturbation | O(d) | **<0.01ms** | LOW |

**CRITICAL SCALABILITY ISSUE:** Brute-force retrieval doesn't scale.

| Memory Size | Retrieval Time |
|-------------|----------------|
| 1,000 | 0.6ms |
| 10,000 | 5ms |
| 100,000 | 50ms ✗ |
| 1,000,000 | 500ms ✗ |

**Mitigations:**
1. **For prototype (<10k episodes):** Brute force is fine. 5ms is acceptable.
2. **For production (>10k episodes):** Need approximate nearest neighbor (ANN):
   - **Locality-Sensitive Hashing (LSH):** O(1) lookup, slight accuracy loss
   - **Hierarchical Navigable Small World (HNSW):** O(log n), high accuracy
   - **Product Quantization:** 4-8x memory reduction + faster search

**Total retrieval: ~5ms (acceptable for prototype, needs ANN for scale)**

---

## Total Latency Budget

| Component | Estimated | Budget | Status |
|-----------|-----------|--------|--------|
| Encoder | 0.7ms | 10ms | ✓ |
| Attractor | 2-4ms | 10ms | ✓ |
| Predictor (forward) | 0.1ms | 5ms | ✓ |
| Predictor (learn) | 0.3ms | 1ms | ✓ |
| Retrieval | 5ms | 15ms | ✓ |
| Response assembly | 0.1ms | 5ms | ✓ |
| **TOTAL** | **~8-10ms** | **50ms** | **✓** |

**With margin: 40ms remaining for I/O overhead, Python overhead, OS jitter.**

---

## Memory Usage Estimate

| Component | Size | Notes |
|-----------|------|-------|
| Sparse matrices (95% sparse) | ~200KB per 1000×1000 layer | 3 layers = 600KB |
| Fast buffer (20 turns × 64D) | ~5KB | Volatile |
| Episodic memory (10k × 64D) | ~2.56MB | Grows with use |
| Semantic memory (1k facts) | ~256KB | Stable |
| Model weights | ~1MB | All dense parameters |
| **Total (10k episodes)** | **~4.5MB** | **Excellent** |
| **Total (100k episodes)** | **~26MB** | **Acceptable** |
| **Total (1M episodes)** | **~260MB** | **Needs management** |

**Concern:** Unbounded episodic growth. Need a capacity policy:
- **Option A:** LRU eviction when capacity reached
- **Option B:** Consolidation (merge similar episodes)
- **Option C:** Semantic extraction (replace raw episodes with extracted facts)

---

## I/O Bottlenecks

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Load model weights (safetensors) | <5ms | — |
| Load episodic memory (mmap) | <10ms | — |
| Save checkpoint (async) | <15ms | — |
| Incremental update (1 episode) | <0.1ms | 10k updates/sec |

**All I/O is async and non-blocking. No response-time impact.**

---

## Throughput Analysis

| Mode | Single Request | Batch-10 | Batch-100 |
|------|---------------|----------|-----------|
| Encode | 0.7ms | 0.7ms | 0.7ms |
| Attractor | 2ms | 2ms | 2ms |
| Predict | 0.4ms | 0.4ms | 0.4ms |
| Retrieve | 5ms | 5ms | 50ms |
| **Total** | **8ms** | **8ms** | **53ms** |

**Throughput:**
- Single-threaded: ~120 req/s
- Batched encode+predict: ~1200 req/s
- **BUT retrieval doesn't batch well** (each request queries different context)

**Real-world throughput:** ~100-150 concurrent conversations per core.

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Attractor doesn't converge | MEDIUM | HIGH | Fixed 5-iteration cap |
| Retrieval too slow at scale | HIGH | MEDIUM | Add ANN (LSH/HNSW) after 10k episodes |
| Memory leak (episodic growth) | MEDIUM | MEDIUM | LRU eviction + consolidation |
| I/O sync blocking response | LOW | HIGH | Always async write-back |
| Python overhead | HIGH | LOW | NumPy vectorization, Cython for hot paths |

---

## Recommendations

### Must-Fix Before Build
1. **Cap attractor iterations at 5.** Don't wait for convergence. Bounded latency > perfect convergence.
2. **Define episodic capacity policy.** LRU at 10k, or consolidation-based.

### Should-Fix for Production
3. **Add approximate nearest neighbor.** LSH for >10k episodes.
4. **Add memory-mapped episodic store.** From survey: STR format with sparse encoding.

### Nice-to-Have
5. **Cython for hot paths.** Attractor dynamics, retrieval loop.
6. **GPU for batch retrieval.** If 100k+ episodes, brute-force GPU matmul is ~0.5ms.

---

## Conclusion

**The design is viable for real-time.** With bounded attractor iterations, the worst-case latency is ~15ms (well within 50ms budget). The learning latency is excellent (<1ms). Memory and I/O are non-issues for the prototype scale.

**The main risks are:**
1. **Attractor convergence time** — must be bounded
2. **Retrieval scalability** — needs ANN for >10k episodes

**For the prototype (Phase 3):** Build as specified with a fixed 5-iteration cap and brute-force retrieval. Profile at 1k, 10k, 100k episodes to find the actual crossover point.

**For production (Phase 5):** Add LRU eviction, LSH-based ANN, and memory-mapped storage.

---

*"Don't guess. Measure. But design for the budget you have, not the budget you want."*
