# TRANSFORMERS: HOW THEY WORK AND WHY THEY CAN'T LEARN IN REAL-TIME
## Deep Study for AGI Research Lab
## Compiled by: researcher
## Date: 2026-06-28

---

## 1. HOW SELF-ATTENTION WORKS

### 1.1 The Query-Key-Value Mechanism

Self-attention is the core innovation of Transformers. Every token in a sequence is projected into three vectors:

- **Query (Q):** "What am I looking for?" — the current token's question
- **Key (K):** "What do I contain?" — what this token offers to others
- **Value (V):** "What information do I provide?" — the actual content to retrieve

**Mathematical formulation:**

For input sequence X, learned weight matrices W_Q, W_K, W_V:

```
Q = X · W_Q    (queries)
K = X · W_K    (keys)  
V = X · W_V    (values)
```

**Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

Where:
- Q·K^T measures similarity between all query-key pairs
- Division by √d_k prevents large dot products from pushing softmax into flat regions
- Softmax normalizes weights to sum to 1
- The result is a weighted sum of values

**Multi-head attention:** h parallel attention heads with different learned projections, concatenated and projected again. Each head can attend to different relationship types (syntactic, semantic, positional).

### 1.2 Causal Masking

For autoregressive (decoder-only) models like GPT, a causal mask prevents attending to future tokens:

```
Attention(Q, K, V) = softmax((Q·K^T + M) / √d_k) · V
```

Where M is a mask: M_ij = -∞ if i < j, else 0.

This ensures token t can only attend to tokens 1...t, preserving the autoregressive property.

### 1.3 Full Transformer Block

Each layer contains:
1. Multi-head self-attention
2. Residual connection: x + Sublayer(x)
3. Layer normalization
4. Position-wise feed-forward network (two linear layers with ReLU/GELU)
5. Another residual connection

Stacked 12-96+ layers deep.

### 1.4 What Self-Attention Actually Does

**Geometric interpretation (Sanger, 2026):**
> "Scaled dot-product attention projects each query input onto a linear combination of the key vectors with projection weights determined by a Gaussian probability distribution... In other words, the input sequence is constrained to lie closer to a low-dimensional surface within the embedding space."

**Key insight:** Attention is a **content-addressable memory** (Vaswani et al., 2017). Given a query, it retrieves relevant information from a key-value store. This is why it works for language — you can directly retrieve context from anywhere in the input.

---

## 2. PRE-TRAINING PROCESS

### 2.1 Next-Token Prediction

The pre-training objective is simple: given a sequence, predict the next token.

```
L = -Σ log P(x_t | x_{<t}; θ)
```

For a sequence x_1, ..., x_T, the model is evaluated at every position predicting the next token. This is self-supervised — the data provides its own labels.

**Why this works:** To predict the next word in a medical textbook, the model must learn medicine. To predict code, it must learn programming. The objective forces the model to absorb the structure and content of human knowledge.

### 2.2 Training Loop

1. **Sample batch** of text sequences (2048-4096 tokens each)
2. **Forward pass** through transformer → probability distributions over vocabulary
3. **Compute cross-entropy loss** between predictions and actual next tokens
4. **Backpropagation** to compute gradients
5. **AdamW update** to adjust parameters
6. **Repeat** for trillions of tokens

### 2.3 Data Mix

| Source | Proportion | Purpose |
|--------|-----------|---------|
| Web crawl (Common Crawl) | 50-70% | Broad language coverage |
| Books | 5-15% | Long-form reasoning |
| Code (GitHub) | 5-20% | Logical structure |
| Academic papers | 5-10% | Scientific reasoning |
| Wikipedia | 3-5% | Factual density |
| Synthetic data | Variable | Targeted capabilities |

### 2.4 Infrastructure and Cost

| Model | Parameters | Tokens | Compute | Cost |
|-------|-----------|--------|---------|------|
| GPT-2 | 1.5B | ~40GB text | 256 TPU v3 cores | ~$50K |
| GPT-3 | 175B | ~570GB | Thousands of V100s | ~$12M |
| GPT-4 (est.) | ~1.8T (MoE) | ~13T | 16K H100s, 3-6 months | ~$100M+ |

Training is one of the most expensive computational endeavors in human history.

### 2.5 What the Model Learns

Through next-token prediction at scale, the model implicitly learns:
- Syntax and grammar at every level
- World knowledge (facts, dates, relationships)
- Reasoning patterns (logic, math, causality)
- Code structure and algorithms
- Multilingual patterns
- Style and tone variation

---

## 3. WHY TRANSFORMERS CAN'T LEARN IN REAL-TIME

### 3.1 Frozen Weights After Training

**The fundamental problem:** Transformers are trained once on a fixed corpus. After deployment, weights are static. Updating requires full backpropagation.

**Why backpropagation prevents real-time learning:**

1. **Requires storing computation graph:** All activations from forward pass must be kept for gradient computation. For a 70B parameter model, this requires ~100GB+ of memory.

2. **Requires a loss function defined over entire output:** You can't compute gradients from a single token prediction — you need a full sequence and a defined loss.

3. **Requires multiple passes:** SGD/Adam requires multiple epochs over data to converge. A single interaction is not enough.

4. **Global gradient flow:** Backpropagation computes gradients for ALL parameters, not just the ones relevant to the current input. This is computationally wasteful and causes interference.

**Consequence:** Every user interaction is a learning opportunity wasted. The model doesn't remember you misspell "the" as "teh" after 1000 interactions.

### 3.2 Catastrophic Forgetting

Even when transformers are fine-tuned, they forget previously learned information.

**Root cause:** Neural networks have distributed representations. New gradients overwrite the weights that encoded old knowledge.

**Empirical evidence:**
- Pure Hebbian on MNIST: 11.6% (Hebbian does PCA, not classification)
- Krotov-Hopfield + Logistic: 97.8% (but only with batch training)
- Fine-tuning GPT-4 on medical text degrades its coding ability

**Current mitigations (all imperfect):**
- **EWC (Elastic Weight Consolidation):** Penalizes changes to important weights — O(d) memory, limited capacity
- **Replay buffers:** Replays old data — requires storing data, unbounded growth
- **Progressive networks:** Adds new columns for each task — O(T²) parameters
- **LoRA:** Low-rank updates — but still requires batch training

### 3.3 No Persistent Memory

The context window is not memory — it's a fixed-size buffer. Information is lost when the window slides. There's no mechanism to selectively maintain, update, or retrieve information across sessions.

**Test-Time Training (TTT)** is a recent attempt to address this (Feng et al., 2026): updates a small set of "fast weights" during inference using next-token prediction gradients. But this still uses backpropagation — it's faster but not truly local.

### 3.4 Attention Bottleneck

All information must flow through fixed-size hidden states. For a sequence of length n, attention computes an n×n matrix — O(n²) time and memory. This means:
- Long contexts are expensive
- Retrieval quality degrades with distance
- No content-based addressing of past states (only positional)

### 3.5 No Temporal Dynamics

Transformers are feedforward functions — they compute an output from an input and stop. They're not dynamical systems. They have no:
- Attractor states that persist over time
- Continuous state evolution
- Predictive dynamics (predict next state from current)
- Multiple timescales (fast perception, slow concepts)

---

## 4. KEY LIMITATIONS

### 4.1 Computational Limits

| Limitation | Impact |
|-----------|--------|
| O(n²) attention | Limits context length, expensive for long sequences |
| Batch normalization dependency | Requires batch statistics; problematic for online learning |
| Fixed computation graph | Cannot adapt architecture dynamically |
| Dense activations | Every neuron computes for every input; wasteful |
| No sparse computation | Cannot selectively activate relevant circuits |

### 4.2 Learning Limits

| Limitation | Impact |
|-----------|--------|
| Batch-only training | Cannot learn from single examples |
| Backpropagation requirement | Prevents local learning |
| Catastrophic forgetting | Cannot accumulate knowledge over time |
| No credit assignment through time | Struggles with long-horizon dependencies |
| Fixed learning rate schedule | Cannot adapt learning speed per parameter |

### 4.3 Architectural Limits

| Limitation | Impact |
|-----------|--------|
| No world model | Cannot predict outcomes or plan |
| No episodic memory | Cannot recall specific experiences |
| No working memory | Cannot hold and manipulate information |
| No compositional reasoning | Struggles with novel combinations |
| No causal reasoning | Learns correlation, not causation |
| No active learning | Cannot ask questions or explore |

### 4.4 Catastrophic Forgetting — Detailed Analysis

**Recent research (2025) confirms the severity:**

1. **ZeroFlow (arXiv:2501.01045):** Forward-pass-only methods can mitigate forgetting without backpropagation — but still not real-time.

2. **ReCL (arXiv:2411.06916):** Uses model weights as memory buffers via margin maximization — requires gradient-based reconstruction.

3. **FAPM (EMNLP 2025):** Foraging-aware pruning limits forgetting to 0.25% but requires task vectors.

4. **DOC (arXiv:2509.23893):** Dynamic orthogonal fine-tuning tracks drifting functional directions with Online PCA — but still uses gradients.

**The hard truth:** No existing solution provides real-time, single-example learning without catastrophic forgetting. All current methods require either:
- Batch processing
- Gradient computation
- Storing old data
- Adding new parameters

---

## 5. WHAT WE LEARN FROM THIS

### 5.1 Transformers Are Not the End

Transformers are an engineering triumph but a cognitive dead end for real-time adaptation. They optimize for batch throughput, not continual learning.

### 5.2 What a Real-Time Adaptive System Needs

| Property | Transformer Status | Required |
|----------|-------------------|----------|
| Real-time learning | ✗ | ✓ |
| Persistent memory | ✗ | ✓ |
| No catastrophic forgetting | ✗ | ✓ |
| Local learning | ✗ | ✓ |
| Temporal dynamics | ✗ | ✓ |
| Sparse computation | ✗ | ✓ |
| Content-based addressing | Partial | ✓ |

### 5.3 The Gap

No existing architecture integrates all required properties. Our project's contribution is designing a system that does.

---

## 6. KEY REFERENCES

1. **Vaswani et al. (2017)** — "Attention Is All You Need" — Original Transformer
2. **Sanger (2026)** — "Scaled Dot-Product Attention implements projection" — Geometric interpretation
3. **Brown et al. (2020)** — "Language Models are Few-Shot Learners" — GPT-3
4. **Feng et al. (2026)** — TTT-NTP: Test-Time Training with Next-Token Prediction
5. **Kirkpatrick et al. (2017)** — EWC: Overcoming Catastrophic Forgetting
6. **arXiv:2501.01045 (2025)** — ZeroFlow: Forward-pass methods for forgetting
7. **arXiv:2411.06916 (2025)** — ReCL: Reconstruction for Continual Learning
8. **EMNLP 2025** — FAPM: Forgetting-Aware Pruning Metric
9. **arXiv:2509.23893** — DOC: Dynamic Orthogonal Continual Fine-tuning

---

*Study complete. Next: synthesize findings into RTALM architecture design.*
