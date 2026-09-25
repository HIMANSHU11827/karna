# Continual Learning Study

**Date:** 2026-09-23
**Author:** bug-finder
**Scope:** Technical survey of continual learning — what's known, what's used, what's open

---

## 1. Catastrophic Forgetting: The Core Problem

### 1.1 Definition

Catastrophic forgetting (CF) is the tendency of a neural network to abruptly lose previously learned information when trained on new information. First documented by McCloskey & Cohen (1989) and Ratcliff (1990).

**Mechanism:** When a network learns task B after task A, the weight updates that minimize loss on B overwrite the weights that were important for A. Formally:

```
θ_B = argmin_θ L_B(θ)
```

But θ_B ≠ θ_A, so performance on A degrades: L_A(θ_B) >> L_A(θ_A).

### 1.2 Why It's Catastrophic (Not Gradual)

Forgetting in neural networks is not gradual like human memory. It's abrupt because:

1. **Shared representations:** The same weights serve multiple tasks. Changing them for B directly impacts A.
2. **Convex optimization:** SGD finds a single parameter configuration that minimizes current loss, with no mechanism to preserve old configurations.
3. **Interference:** New gradients point in directions that conflict with old task gradients.

**Empirical evidence:** Kirkpatrick et al. (2017) showed that training a network on MNIST permutation A then permutation B drops accuracy on A from 95% to ~10% (random) within a single epoch of B training.

### 1.3 Forgetting in Different Architectures

| Architecture | Forgetting Severity | Why |
|-------------|---------------------|-----|
| MLP | Severe | Fully shared weights, dense representations |
| CNN | Moderate | Convolutional structure provides some locality |
| Transformer | Severe | Attention weights are fully shared across tasks |
| SNN (Spiking) | Moderate | Event-driven updates are naturally sparse |
| Our RAIE/SHD | Severe | Dense weight matrix, no sparsity enforcement on weights |

### 1.4 Forgetting in Our Codebase

**`raie_network.py` IACMemory.store() (line 237-252):**
```python
self.W += np.outer(p, p)
self.W /= self.n_stored
```

This is the worst possible anti-forgetting mechanism. Each new pattern dilutes ALL previous patterns equally. After N patterns, each contributes 1/N to the weight matrix. The network has no mechanism to protect important patterns.

**`raie_unified.py` ComplementaryLearning (line 340):**
```python
self.replay_buffer = deque(maxlen=100)
```

A 100-item FIFO buffer is insufficient. After 100 new experiences, the oldest is permanently lost. No prioritization, no importance weighting.

---

## 2. Elastic Weight Consolidation (EWC)

### 2.1 Core Idea

Kirkpatrick et al. (2017) proposed EWC: penalize changes to weights that are important for previous tasks.

**Loss function:**
```
L_total = L_B(θ) + (λ/2) * Σ_i F_i * (θ_i - θ_A,i)²
```

Where:
- `F_i` is the Fisher Information Matrix (FIM) diagonal for parameter i
- `θ_A,i` is the optimal value for task A
- `λ` is a regularization strength

### 2.2 Fisher Information as Importance

The Fisher Information Matrix measures how much each parameter affects the output:

```
F_i = E[(∂log p(y|x,θ)/∂θ_i)²]
```

High F_i means the parameter is important for the current task. EWC freezes high-F parameters, allowing only low-F parameters to change.

### 2.3 Why EWC Needs Batch Training

**Problem:** Computing the Fisher Information Matrix requires:
1. A batch of examples from task A
2. Forward + backward pass on each example
3. Accumulating squared gradients

This is inherently a batch operation. You cannot compute F_i from a single example because:
- F_i is an expectation over the data distribution
- A single example gives a noisy, biased estimate
- The diagonal approximation requires many examples to be stable

**For real-time learning:** EWC is incompatible with single-example online updates. You would need to:
1. Accumulate a batch of examples
2. Compute FIM on the batch
3. Update the penalty term
4. Then update weights

This introduces latency proportional to batch size.

### 2.4 EWC Variants and Limitations

| Variant | Improvement | Limitation |
|---------|-------------|------------|
| EWC++ (Huszár, 2018) | Online FIM update | Still needs batches |
| RWalk (Chaudhry et al., 2018) | Uses path integral | More compute |
| MAS (Aljundi et al., 2018) | Uses output sensitivity | Different importance measure |
| KFAC (Ritter et al., 2018) | Kronecker-factored FIM | More accurate but slower |

**Fundamental limitation:** All EWC variants require knowing which task you're on. In a continual conversation, there are no clear task boundaries.

---

## 3. Replay Buffers

### 3.1 Core Idea

Store a subset of old examples and replay them during new task training. This is the simplest and most effective continual learning method.

**Training loop:**
```
For each new example (x_new, y_new):
    1. Sample K examples from replay buffer: {(x_i, y_i)}
    2. Train on batch: {(x_new, y_new)} ∪ {(x_i, y_i)}
    3. Add (x_new, y_new) to buffer
    4. If buffer full, evict oldest or least important
```

### 3.2 Buffer Management Strategies

| Strategy | Eviction Policy | Pros | Cons |
|----------|----------------|------|------|
| FIFO | Oldest first | Simple, O(1) | Loses diverse examples |
| Random | Random eviction | Unbiased | May lose rare examples |
| Reservoir sampling | Uniform random | Unbiased, streaming | Same as random |
| Importance-based | Lowest importance first | Keeps important examples | Needs importance metric |
| Clustering-based | Cluster centroids | Preserves diversity | Expensive to maintain |
| Gradient-based | Lowest gradient norm | Keeps hard examples | Needs gradient computation |

### 3.3 Why Replay Needs Batch Training

**The fundamental issue:** Replay works by interleaving old and new examples in a batch. This requires:
1. Storing old examples (memory cost)
2. Sampling from the buffer (compute cost)
3. Training on a batch (not single examples)

**For real-time learning:** You could replay one old example per new example, but:
- The ratio of old:new examples matters (too much old = slow learning, too little = forgetting)
- The buffer size determines how far back you can remember
- There's no principled way to set the replay ratio without validation

### 3.4 Replay in Our Codebase

**`raie_unified.py` (line 340):**
```python
self.replay_buffer = deque(maxlen=100)
```

**Problems:**
1. **Too small:** 100 examples is nothing. MNIST has 60,000 training examples. A 100-item buffer retains 0.17% of the data.
2. **No prioritization:** All examples treated equally. Rare but important examples are evicted first.
3. **No replay during training:** The buffer is stored but never actually replayed during learning. The `_consolidate()` method just does Hebbian updates on the buffer contents, not true replay.

---

## 4. Why Current Methods Need Batch Training

### 4.1 Statistical Reasons

Most continual learning methods rely on statistical quantities that require batches:

| Method | Batch-Dependent Quantity | Why |
|--------|-------------------------|-----|
| EWC | Fisher Information Matrix | Expectation over data distribution |
| Replay | Mini-batch sampling | Need multiple examples for stable gradient |
| iCaRL | Class mean vectors | Need examples per mean |
| GEM | Gradient projection | Need constraint set from old tasks |
| A-GEM | Averaged gradient | Need reference gradient from buffer |

### 4.2 Optimization Reasons

**SGD is inherently batch-based:**
- Single-example gradients are high-variance
- Batch gradients are lower-variance, more stable
- Learning rate scheduling assumes batch training

**For online learning:** You can use SGD with a single example, but:
- Learning rate must be much smaller (to avoid instability)
- Convergence is much slower
- No guarantee of convergence to a good solution

### 4.3 Memory Reasons

**Backpropagation requires storing activations:**
- For a batch of size B, store B × L × H activations (L layers, H hidden dim)
- For B=1, this is minimal, but you lose batch statistics
- For B=32, this is 32x more memory

**For real-time learning:** Single-example updates use minimal memory but sacrifice the benefits of batch normalization, gradient averaging, and stable optimization.

### 4.4 The Online Learning Tradeoff

| Aspect | Batch Training | Online Training |
|--------|---------------|-----------------|
| Sample efficiency | High | Low |
| Compute per update | High | Low |
| Memory per update | High | Low |
| Stability | High | Low |
| Forgetting | Mitigated by replay | Severe |
| Latency | Seconds to hours | Milliseconds |

**The fundamental tension:** You can have low latency OR low forgetting, but not both with current methods.

---

## 5. Open Problems

### 5.1 Single-Example Continual Learning

**The holy grail:** Learn from one example, immediately, without forgetting anything.

**Why it's hard:**
- No batch statistics for stable optimization
- No replay possible (only one example)
- No Fisher information (need distribution)
- Credit assignment is ambiguous

**Current best attempts:**
- **MAML (Finn et al., 2017):** Learn to learn from few examples. But requires meta-training on many tasks.
- **Reptile (Nichol et al., 2018):** Simpler meta-learning. Still needs meta-training.
- **Online EWC (Huszár, 2018):** Approximate FIM online. Still needs batches for stability.

**Our SHD rule** attempts this but fails because:
- The eligibility trace is a single-example approximation
- The decay term causes forgetting
- There's no mechanism to protect important weights

### 5.2 Task-Free Continual Learning

**The problem:** In real-world continual learning, there are no clear task boundaries. The system must:
1. Detect when it's seeing a new task
2. Allocate new resources for the new task
3. Preserve old task performance
4. Do all this without explicit task labels

**Current approaches:**
- **DNN with dynamic architecture:** Add neurons for new tasks (Progressive Networks, Rusu et al., 2016)
- **Clustering-based:** Cluster inputs, allocate resources per cluster
- **Attention-based:** Use attention to select relevant parameters

**Open question:** How to detect task boundaries without explicit labels?

### 5.3 Catastrophic Forgetting in the Wild

**The problem:** Most continual learning research uses toy benchmarks (permuted MNIST, split CIFAR). Real-world continual learning involves:
- Non-stationary data distributions
- Varying task similarity
- Varying task importance
- No clear task boundaries
- No validation set for hyperparameter tuning

**Open question:** How to evaluate continual learning in realistic settings?

### 5.4 Scaling Continual Learning

**The problem:** Current methods work for 5-10 tasks. Real-world systems need to handle:
- Thousands of tasks
- Millions of examples
- Arbitrary task similarity
- No task labels

**Open question:** How to scale continual learning to realistic scales?

### 5.5 Continual Learning + Real-Time Inference

**The problem:** Most continual learning methods focus on training. Real-time systems need:
- Low-latency inference (<50ms)
- Low-latency learning (<1ms per example)
- No separation between training and inference
- Graceful degradation under load

**Open question:** How to do continual learning without sacrificing inference speed?

---

## 6. What This Means for Our Project

### 6.1 Our Current Approach

**SHD (Sparse Hebbian with Decay):**
```
ΔW = η * (pre * post - decay * W)
```

This is Oja's rule with an eligibility trace. It:
- ✅ Works online (single example)
- ✅ Is local (no backprop)
- ❌ Causes catastrophic forgetting (decay term)
- ❌ Is not discriminative (no error signal)
- ❌ Has no mechanism to protect important weights

### 6.2 What We'd Need for Real Continual Learning

| Requirement | Current Status | Gap |
|-------------|---------------|-----|
| Online updates | ✅ SHD works online | None |
| No catastrophic forgetting | ❌ Decay causes forgetting | Need EWC-like protection |
| Discriminative learning | ❌ No error signal | Need error-driven updates |
| Real-time inference | ⚠️ Unmeasured | Need profiling |
| Single-example learning | ✅ SHD works | None |
| Task-free operation | ❌ No task detection | Need task boundary detection |

### 6.3 Honest Assessment

**What we can build:**
- A system that learns online from single examples
- A system that does Hebbian learning with local updates
- A system that stores memories in weights

**What we cannot build (yet):**
- A system that doesn't forget
- A system that matches batch-trained performance
- A system that works without task boundaries
- A system that scales to realistic tasks

**The path forward:**
1. **Accept forgetting as a feature:** Design for "graceful degradation" rather than "no forgetting"
2. **Use sparse representations:** Sparse codes naturally reduce interference
3. **Add importance weighting:** Track which weights are important and protect them
4. **Measure honestly:** Don't claim "no forgetting" without evidence

---

## 7. Key References

1. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks. *Psychology of Learning and Motivation, 24*, 109-165.
2. Ratcliff, R. (1990). Connectionist models of recognition memory: constraints imposed by learning and forgetting functions. *Psychological Review, 97*(2), 285.
3. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS, 114*(13), 3521-3526.
4. Huszár, F. (2018). Note on the quadratic penalties in elastic weight consolidation. *PNAS, 115*(11), E2496.
5. Aljundi, R., et al. (2018). Memory aware synapses: Learning what (not) to forget. *ECCV 2018*.
6. Chaudhry, A., et al. (2018). Efficient lifelong learning with A-GEM. *ICLR 2019*.
7. Rusu, A. A., et al. (2016). Progressive neural networks. *arXiv:1606.04671*.
8. Finn, C., et al. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML 2017*.
9. van de Ven, G. M., & Tolias, A. S. (2019). Three scenarios for continual learning. *arXiv:1904.07734*.
10. De Lange, M., et al. (2022). A continual learning survey: Defying forgetting in classification tasks. *TPAMI, 44*(7), 3366-3385*.

---

## 8. Conclusion

Continual learning is an open problem. Current methods (EWC, replay, progressive networks) all require batch training and have significant limitations. Our SHD rule works online but causes catastrophic forgetting. Building a system that learns continuously without forgetting is a research problem, not an engineering problem.

**Recommendation:** Focus on building a system that learns online and degrades gracefully. Don't claim "no forgetting" without evidence. Measure forgetting honestly and report it.
