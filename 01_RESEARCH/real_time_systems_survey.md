# Real-Time Systems Survey
===========================

**Date:** 2026-09-23
**Author:** coder-3
**Purpose:** Study current real-time ML systems for the AGI Research Lab

---

## 1. Streaming ML / Online Learning

### 1.1 Core Concept

Streaming ML processes data sequentially, updating the model one sample (or mini-batch) at a time. Unlike batch training, the model never sees the full dataset — it learns from each example once and discards it.

**Key properties:**
- Single pass over data (or few passes)
- Constant memory (doesn't store training data)
- Adapts to concept drift (changing data distributions)
- Latency per sample matters (must keep up with data rate)

### 1.2 Stochastic Update Rules

**Stochastic Gradient Descent (SGD):**
```
w ← w - η * ∇L(x_i, y_i, w)
```
- Update per sample: O(1) memory, O(d) compute per step
- Learning rate η controls stability vs. adaptivity tradeoff
- Variants: momentum, RMSProp, Adam (all maintain running statistics)

**Online Newton Step:**
```
w ← w - η * H^{-1} * ∇L(x_i, y_i, w)
```
- Uses curvature information for faster convergence
- O(d²) memory for Hessian approximation
- Better for well-conditioned problems

**Follow-the-Regularized-Leader (FTRL):**
```
w_t = argmin_w Σ_{s=1}^{t-1} L_s(w) + λ * R(w)
```
- Strong theoretical guarantees (regret bounds)
- Used in Google's ad prediction systems
- Handles L1 regularization naturally (sparsity)

### 1.3 Concept Drift Detection

Real-time systems must detect when the data distribution changes:

- **DDM (Drift Detection Method):** Monitors error rate; triggers when error exceeds threshold
- **ADWIN (Adaptive Windowing):** Maintains a sliding window; detects change when two sub-windows differ statistically
- **Page-Hinkley Test:** Cumulative sum of deviations from mean

### 1.4 Real-World Streaming Systems

| System | Approach | Scale |
|--------|----------|-------|
| Google FTRL | Online logistic regression | Billions of features |
| Twitter Heron | Stream processing | Millions of events/sec |
| Apache Flink ML | Distributed streaming | Cluster-scale |
| River (Python) | Online learning library | Research/small-scale |

---

## 2. Real-Time Inference

### 2.1 Batching Strategies

**Static Batching:**
- Accumulate N requests, process together
- Higher throughput, higher latency
- Good for offline or near-real-time

**Dynamic Batching:**
- Wait up to T milliseconds OR until batch size B
- Balances latency and throughput
- Used in TensorFlow Serving, Triton Inference Server

**Continuous Batching (iteration-level):**
- Process multiple sequences in parallel
- New sequences join mid-iteration
- Used in vLLM, TensorRT-LLM for LLM serving
- Maximizes GPU utilization

### 2.2 Quantization

Reduce model size and latency by using lower-precision arithmetic:

| Precision | Size | Speed | Accuracy Impact |
|-----------|------|-------|-----------------|
| FP32 | 4 bytes/baseline | Baseline | None |
| FP16 | 2 bytes | 2x faster | Minimal |
| INT8 | 1 byte | 4x faster | ~1-2% drop |
| INT4 | 0.5 bytes | 8x faster | ~3-5% drop |
| Binary | 1 bit | 16x+ faster | ~10-20% drop |

**Quantization-aware training (QAT):** Train with quantization in the loop — best accuracy.
**Post-training quantization (PTQ):** Quantize after training — faster to deploy.

### 2.3 Pruning

Remove weights that contribute little to output:

- **Unstructured pruning:** Zero out individual weights — sparse matrices
- **Structured pruning:** Remove entire neurons/channels — dense but smaller
- **Lottery ticket hypothesis:** Train, prune, retrain — find sparse subnetworks

### 2.4 Knowledge Distillation

Train a small "student" model to mimic a large "teacher" model:
- Student runs in real-time
- Teacher runs offline or in batch
- Student achieves ~95% of teacher accuracy at 10x speed

### 2.5 Caching and Speculation

- **KV Cache:** Store key-value pairs from previous tokens (LLMs)
- **Speculative decoding:** Draft model proposes tokens, target model verifies
- **Prefix caching:** Reuse computations for common prefixes

---

## 3. Adaptive Systems

### 3.1 Meta-Learning

**Learning to learn:** Train a model that can quickly adapt to new tasks with few examples.

**MAML (Model-Agnostic Meta-Learning):**
```
θ' = θ - α * ∇L_T(θ)  # Inner loop: adapt to task T
θ ← θ - β * ∇L_T'(θ')  # Outer loop: update meta-parameters
```
- Find initialization that adapts quickly
- O(d²) compute per task (second-order)
- First-order approximation (FOMAML): O(d), nearly as good

**Prototypical Networks:**
- Learn an embedding space where classes cluster
- New class: compute prototype (mean of support examples)
- Classify by nearest prototype
- O(d) per new class, no gradient updates needed

### 3.2 Few-Shot Learning

**N-way K-shot:** Classify into N classes with K examples each.

| Method | 5-way 1-shot | 5-way 5-shot |
|--------|-------------|-------------|
| MAML | ~49% | ~65% |
| Prototypical Networks | ~49% | ~68% |
| Matching Networks | ~44% | ~55% |
| Relation Networks | ~51% | ~65% |

### 3.3 Continual Learning

Learn new tasks without forgetting old ones:

**Elastic Weight Consolidation (EWC):**
```
L_total = L_new + λ * Σ_i F_i * (w_i - w_old_i)²
```
- F_i: Fisher information (importance of weight i)
- Penalizes changes to important weights
- O(d) memory for Fisher matrix

**Progressive Neural Networks:**
- Add new columns for new tasks
- Old columns frozen
- Lateral connections allow knowledge transfer
- O(d * T) memory for T tasks

**Replay-based:**
- Store exemplars from old tasks
- Interleave old and new data during training
- O(M) memory for M exemplars

### 3.4 Self-Supervised Adaptation

- **Masked autoencoding:** Reconstruct masked input — learns representations without labels
- **Contrastive learning:** Similar inputs close, dissimilar far apart
- **BYOL/SimCLR:** No negative samples needed

---

## 4. Latency/Throughput Tradeoffs

### 4.1 The Fundamental Tradeoff

```
Latency ∝ 1 / Throughput
```

- **High throughput, low latency:** Requires parallelism (batching, pipelining)
- **Low latency, low throughput:** Process each request immediately
- **High throughput, high latency:** Batch aggressively

### 4.2 Amdahl's Law

Speedup is limited by the serial portion:
```
Speedup = 1 / (S + (1-S)/N)
```
- S: serial fraction
- N: number of processors
- If 10% is serial, max speedup is 10x regardless of N

### 4.3 Little's Law

```
L = λ * W
```
- L: average number of requests in system
- λ: arrival rate
- W: average time in system

To handle higher arrival rates, either:
- Reduce latency (W)
- Increase parallelism (more servers)

### 4.4 Quantization vs. Accuracy

| Model | FP32 Acc | INT8 Acc | INT4 Acc | Speedup |
|-------|----------|----------|----------|---------|
| ResNet-50 | 76.1% | 75.5% | 73.2% | 4x / 8x |
| BERT-base | 82.3% | 81.8% | 80.1% | 3x / 6x |
| GPT-2 | — | — | — | 4x / 8x |

### 4.5 Memory vs. Compute

- **Memory-bound:** Small model, large batch — limited by memory bandwidth
- **Compute-bound:** Large model, small batch — limited by FLOPS
- **Roofline model:** Performance = min(compute_peak, memory_bandwidth * arithmetic_intensity)

---

## 5. Implications for AGI Research Lab

### 5.1 Architecture Decisions

For real-time multimodal + continual learning:

1. **Streaming updates:** Use SGD-style updates (not batch) — O(1) memory per step
2. **Quantization:** INT8 for inference, FP16 for training — 4x speedup with minimal accuracy loss
3. **Meta-learning:** Prototypical networks for few-shot adaptation — O(d) per new class
4. **Continual learning:** EWC or replay — prevent catastrophic forgetting
5. **Batching:** Dynamic batching for inference — balance latency and throughput

### 5.2 Latency Budget

For real-time perception (target: <100ms end-to-end):

| Component | Budget | Approach |
|-----------|--------|----------|
| Preprocessing | 10ms | Resize, normalize, quantize |
| Feature extraction | 30ms | Quantized CNN/Transformer |
| Memory retrieval | 20ms | IAC attractor (5 iterations) |
| Classification | 10ms | Logistic regression (closed-form) |
| Post-processing | 10ms | Threshold, filter |
| **Total** | **80ms** | **<100ms target** |

### 5.3 Throughput Target

For continuous learning (target: 1000 samples/sec):

- Batch size: 32 (dynamic batching)
- Model size: <10M parameters (INT8: <10MB)
- Hardware: Single CPU core (neuromorphic-ready)
- Update frequency: Every sample (online) or every 32 samples (mini-batch)

---

## 6. References

1. **Online Learning:** McMahan et al., "Ad Click Prediction: a View from the Trenches" (FTRL)
2. **Streaming ML:** Bifet et al., "Machine Learning for Data Streams" (River library)
3. **Real-Time Inference:** NVIDIA Triton Inference Server, TensorFlow Serving
4. **Quantization:** Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
5. **Meta-Learning:** Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (MAML)
6. **Continual Learning:** Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural Networks" (EWC)
7. **Few-Shot Learning:** Snell et al., "Prototypical Networks for Few-shot Learning"
8. **Knowledge Distillation:** Hinton et al., "Distilling the Knowledge in a Neural Network"
