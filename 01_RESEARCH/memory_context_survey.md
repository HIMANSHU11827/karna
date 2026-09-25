# Memory & Context Systems Survey: Limits of Current Approaches

**Author:** Anvil (Coder 2) — AGI Research Lab
**Date:** 2026-09-23
**Phase:** 1 — Study
**Scope:** How current systems handle memory/context, and why they fail for real-time AGI

---

## Executive Summary

Current AI systems handle memory and context through three dominant paradigms:
1. **Attention-based context windows** (transformers, long-context LLMs)
2. **Retrieval-augmented generation** (RAG, vector databases)
3. **Memory-augmented neural networks** (NTM, DNC, MANNs)

All three have fundamental limitations that prevent real-time adaptive intelligence:
- **Attention windows** are finite, compute-intensive, and passive
- **RAG systems** require pre-indexing, chunking heuristics, and external embedding models
- **MANNs** are difficult to train and have limited capacity

This survey covers each approach, cites key papers and implementations, and identifies the gaps our RAIE architecture must fill.

---

## 1. LLM Context Window Handling

### 1.1 How It Works

Modern LLMs process context through self-attention mechanisms with fixed or sliding windows:

| Model | Context Window | Key Innovation |
|-------|---------------|----------------|
| GPT-4 | 128K tokens | Sparse attention, multi-query attention |
| Claude-3 | 200K tokens | Sliding window + attention sink |
| Gemini-1.5 | 1M tokens | Mixture of experts, optimized KV cache |
| Llama3 | 128K tokens | Grouped query attention, RoPE scaling |
| Qwen2.5 | 1M tokens | Dynamic KV cache compression |

### 1.2 Key Techniques

**Attention Sink (StreamingLLM, 2023):**
- Observation: First few tokens act as "attention sinks" — they receive disproportionate attention
- Trick: Keep first N tokens + sliding window → infinite context without retraining
- Limitation: Only works for generation, not for tasks requiring full context recall

**KV Cache Compression (PyramidKV, DynamicKV, CAKE):**
- PyramidKV: Allocates more KV cache to early layers, fewer to later layers
- DynamicKV: Dynamically prunes low-attention tokens per layer
- CAKE: Uses cumulative attention scores to retain important tokens
- Limitation: Compression is lossy — fine-grained details are lost

**Long-Context vs RAG Tradeoffs (Li et al., 2025, arXiv:2501.018880):**
- Long Context (LC) generally outperforms RAG for QA benchmarks
- RAG advantages: dialogue tasks, general queries, lower compute cost
- Chunk size sweet spot: 40-50% of context window for 512-token chunks
- RAPTOR (2024): Recursive tree-structured summaries improve multi-hop reasoning
- DOS RAG: "Document Original Structure RAG" — simple retrieve-then-read outperforms complex pipelines

### 1.3 Fundamental Limits

1. **Quadratic compute cost:** Self-attention scales O(n²) with sequence length
2. **Finite window:** Even "infinite" methods (StreamingLLM) are approximations
3. **Passive memory:** The model doesn't *choose* what to remember — it's determined by attention weights
4. **No consolidation:** Information fades as it leaves the window — no long-term memory formation
5. **Pre-training dependency:** World knowledge is frozen at training time

---

## 2. Memory-Augmented Neural Networks (MANNs)

### 2.1 Neural Turing Machines (NTM)

**Original Paper:** Graves, Wayne, Danihelka (2014) — arXiv:1410.5401

**Architecture:**
- Neural controller (LSTM or feedforward) + external memory matrix
- Read/Write heads with differentiable addressing
- Two addressing modes:
  - Content-based: cosine similarity between key and memory rows
  - Location-based: circular convolution shift (allows iteration)

**Addressing Pipeline:**
1. Content addressing: w_c = softmax(β · cosine(key, memory))
2. Interpolation: w_g = g · w_c + (1-g) · w_prev
3. Circular convolution shift: w_s = conv(w_g, shift_kernel)
4. Sharpening: w = w_s^γ / sum(w_s^γ)

**Training Insights (Collier & Beel, 2018):**
- Memory initialization to small constant (1e-6) is critical for convergence
- Random initialization causes slow/unstable training
- Controller should receive previous read vector as input (closes the loop)
- RMSProp outperforms Adam for NTM training

**Results:**
- Copy task: NTM (67K params) outperforms LSTM (1.35M params)
- Repeat Copy: Learns iterative behavior (for-loop emulation)
- Associative Recall: Learns key-value associations

**Limitations:**
- Difficult to train (vanishing gradients, memory instability)
- Limited capacity (~200-500 patterns for typical sizes)
- Requires backpropagation through time
- Not suitable for online/continual learning

### 2.2 Differentiable Neural Computer (DNC)

**Original Paper:** Graves et al. (2016) — Nature

**Improvements over NTM:**
- Temporal link matrix: tracks write order (enables sequential traversal)
- Usage vectors: tracks which memory locations are free
- Dynamic allocation: can free and reallocate memory
- More stable training

**Key Innovation: Link Matrix**
- L[i,j] = 1 if location i was written after location j
- Enables forward/backward traversal of stored sequences
- Critical for tasks requiring temporal reasoning

**Limitations:**
- Even harder to train than NTM
- Link matrix maintenance adds significant compute
- Still requires full backpropagation
- Not biologically plausible (no local learning rules)

### 2.3 Modern MANN Variants (2024-2025)

**SANTM (Sparse Access Neural Turing Machine, 2025):**
- Adds local multi-head self-attention for memory access
- ℓ0-constrained sparse mask reduces memory access by 37-54%
- Three-level controller: segmentation → integration → selective access
- Results: 95.7% permuted MNIST (vs 94.0% NTM), 24.2 WikiText-103 perplexity (vs 27.0 Transformer-XL)

**Neural Stored-Program Memory (NSM, 2019):**
- Stores *weights* in memory, not just data
- Enables program switching (like stored-program computers)
- Combined with NTM → Neural Universal Turing Machine (NUTM)
- Supports few-shot learning, compositional tasks, continual learning

**Key Insight:** Modern MANNs combine attention mechanisms with external memory, achieving transformer-like performance with explicit memory operations.

### 2.4 Fundamental Limits

1. **Training difficulty:** Differentiable memory requires backpropagation through time
2. **Capacity:** Pattern capacity scales as O(n) for n neurons — limited vs billions of parameters
3. **No online learning:** All current MANNs require batch training
4. **No consolidation:** No mechanism for transferring memory to long-term storage
5. **No biological plausibility:** Backthrough is not how brains learn

---

## 3. Continual Learning

### 3.1 The Problem: Catastrophic Forgetting

When a neural network learns a new task, gradients overwrite parameters critical for previous tasks. This is catastrophic forgetting (McCloskey & Cohen, 1989).

**Stability-Plasticity Dilemma:**
- Too stable: Can't learn new tasks (high bias)
- Too plastic: Forgets old tasks (high variance)
- Need: Balanced adaptation

### 3.2 Taxonomy of Solutions

**Three families:**

1. **Replay-based:** Store/replay old examples during new-task training
2. **Regularization-based:** Penalize changes to important parameters
3. **Architecture-based:** Isolate parameters per task

### 3.3 Elastic Weight Consolidation (EWC)

**Original Paper:** Kirkpatrick et al. (2017)

**Key Idea:**
After training on task t, estimate importance of each parameter using Fisher Information Matrix (FIM). Penalize deviations from important parameters when learning new tasks.

**Mathematical Formulation:**
```
L(θ) = L_new(θ) + (λ/2) Σ F_i · (θ_i - θ*_i)²
```
Where:
- F_i = Fisher information (diagonal approximation) for parameter i
- θ*_i = parameter value after previous task
- λ = regularization strength

**Fisher Information Matrix:**
```
F_i = E[(∂ log p(y|x,θ) / ∂θ_i)²]
```
Measures how much each parameter affects the loss. Important parameters get high F_i, and their changes are penalized.

**Interpretation (Bayesian):**
EWC is an online Laplace approximation of the posterior p(θ|D₁:ₖ). The Fisher information serves as the precision matrix.

**Extensions:**
- **Synaptic Intelligence (SI, Zenke et al. 2017):** Online importance estimation (no separate Fisher computation)
- **Memory Aware Synapses (MAS, Aljundi et al. 2018):** Uses gradient magnitude as importance proxy
- **EWC-DR (2026):** Fixes EWC's gradient vanishing problem via logits reversal

**Results:**
- Split-MNIST: 95%+ retention across 5 tasks
- Permuted-MNIST: 80%+ retention across 20 tasks
- Medical imaging: DDPM+EWC achieves 0.851 AUROC (vs 0.869 joint training upper bound)

**Limitations:**
1. **Fisher matrix diagonal approximation** ignores parameter correlations
2. **Gradient vanishing:** When model is confident, gradients → 0, so Fisher → 0, and EWC fails to protect
3. **Redundant protection:** MAS protects parameters irrelevant to prior tasks
4. **Scalability:** Computing Fisher for large models is expensive
5. **Task boundary detection:** EWC requires knowing when tasks change

### 3.4 Replay-Based Methods

**Experience Replay:**
- Store subset of old examples in buffer
- Mix with new-task data during training
- Reservoir sampling: each example has equal probability of being retained

**DER++ (2020):**
- Dark Experience Replay: stores logits (not just labels) for better distillation
- Achieves strong results on Split-CIFAR100

**Generative Replay:**
- Train generative model (VAE, GAN, or Diffusion) to produce old-task data
- No need to store raw examples (privacy-preserving)
- DDPM+EWC (2025): Combines diffusion replay with EWC regularization
  - MedMNIST: 78.1% accuracy, 10.5% forgetting
  - CheXpert: 0.851 AUROC, 30% less forgetting than DER++

**ContinualTCR (2026):**
- Combines reservoir replay with EWC for TCR-peptide binding prediction
- Replay alone resolves 30.6% of forgetting
- EWC alone resolves 57.3%
- Combined: 62.9% reduction in forgetting

**Limitations:**
- Buffer size vs performance tradeoff
- Privacy concerns (storing old data)
- Generative models add complexity
- Catastrophic forgetting of the generative model itself

### 3.5 Architecture-Based Methods

**Progressive Networks:**
- New column of neurons for each task
- Lateral connections to previous columns
- Old parameters are frozen
- Problem: Parameter count grows linearly with tasks

**PackNet / Hard Attention:**
- Prune and reallocate parameters per task
- Each task gets a subset of parameters
- Problem: Requires task identity at inference

**HiCL (Hippocampal-Inspired CL, 2025):**
- Grid-cell-like encoding → Dentate Gyrus pattern separation → CA3 autoassociative memory
- DG-gated Mixture-of-Experts for task routing
- Top-k sparsity for pattern separation
- Results: Near state-of-the-art with lower compute cost
- Robust to buffer size reduction (92.8% → 90.1% when buffer shrinks from 500 to 100)

### 3.6 Fundamental Limits of Continual Learning

1. **Task boundary assumption:** Most methods assume discrete tasks with clear boundaries
2. **Static architectures:** Most methods assume fixed network capacity
3. **No meta-learning:** Methods don't learn *how* to learn continuously
4. **No real-time adaptation:** All methods require training epochs, not instant updates
5. **No cross-task transfer:** Most methods prevent forgetting but don't enable positive transfer

---

## 4. Key Gaps & Opportunities for RAIE

### 4.1 What Current Systems CAN'T Do

| Capability | LLM | MANN | Continual Learning | RAIE Target |
|-----------|-----|------|--------------------|-------------|
| Real-time incremental learning | ❌ | ❌ | ❌ | ✅ |
| No task boundaries | ❌ | ❌ | ❌ | ✅ |
| Multimodal binding | ❌ | ❌ | ❌ | ✅ |
| Self-directed memory formation | ❌ | ❌ | ❌ | ✅ |
| Forgetting gracefully | ❌ | ❌ | ⚠️ | ✅ |
| Cross-task transfer | ❌ | ❌ | ❌ | ✅ |
| No pre-training required | ❌ | ❌ | ❌ | ✅ |
| Biological plausibility | ❌ | ⚠️ | ❌ | ✅ |

### 4.2 Design Principles for RAIE (Derived from Limits)

1. **Local learning rules only:** No backpropagation through time (biological constraint)
2. **Online adaptation:** Learn from every example, one-shot, no epochs
3. **Sparse representations:** k-WTA for efficient coding, minimal interference
4. **Multiple memory systems:** Fast episodic + slow semantic (Complementary Learning)
5. **Self-directed consolidation:** Automatic transfer from episodic to semantic
6. **Multimodal binding:** XOR-style binding for cross-modal associations
7. **Graceful forgetting:** Weight decay + consolidation, not catastrophic interference
8. **Real-time operation:** Process input incrementally, no batch training

### 4.3 Open Questions

1. **Can we get 95%+ MNIST without backprop?** Krotov & Hopfield (2019) claim yes. Can our RAIE architecture match it?
2. **How to handle continuous (non-discretized) learning?** No task boundaries in real life.
3. **How to balance multiple modalities?** Text, image, audio arriving simultaneously.
4. **How to prevent representational collapse?** Replay and sparsity help, but are they sufficient?
5. **How to measure "understanding" vs pattern matching?** Need better benchmarks.

---

## 5. References

### LLM Context & RAG
- Xiao et al. (2023). "StreamingLLMs: Efficient Streaming for Large Language Models." arXiv:2309.17453
- Li et al. (2025). "Long Context vs. RAG for LLMs: An Evaluation." arXiv:2501.018880
- Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."
- Laitenberger et al. (2025). "Stronger Baselines for RAG with Long-Context LMs." EMNLP 2025.
- Juvekar & Purwar (2024). "Introducing a new hyper-parameter for RAG: Context Window Utilization." arXiv:2407.19794

### Memory-Augmented Neural Networks
- Graves, Wayne, Danihelka (2014). "Neural Turing Machines." arXiv:1410.5401
- Graves et al. (2016). "Hybrid computing using a neural network with dynamic external memory." Nature
- Collier & Beel (2018). "Implementing Neural Turing Machines." ICANN 2018
- Shan et al. (2025). "SANTM: Sparse Access Neural Turing Machine." Applied Soft Computing
- Neural Stored-Program Memory (2019). "Neural Stored-program Memory." arXiv:1906.08862

### Continual Learning
- Kirkpatrick et al. (2017). "Overcoming catastrophic forgetting in neural networks." PNAS (EWC)
- Zenke et al. (2017). "Continual Learning Through Synaptic Intelligence." ICML (SI)
- Aljundi et al. (2018). "Memory Aware Synapses." ECCV (MAS)
- McClelland, McNaughton, O'Reilly (1995). "Why there are complementary learning systems in the hippocampus and neocortex." Psychological Review
- Li et al. (2026). "EWC Done Right for Continual Learning." arXiv:2603.18596
- HiCL (2025). "Hippocampal-Inspired Continual Learning." arXiv:2508.16651
- ContinualTCR (2026). "Continual Learning for Emerging Epitope Landscapes." bioRxiv

### Key Surveys
- Kirkpatrick et al. (2017). EWC paper includes comprehensive forgetting review
- Sha et al. (2024). Learning-forgetting tradeoff dynamics
- Lee et al. (2022). Catastrophic forgetting in neural networks: A review

---

## 6. Conclusion

Current approaches to memory and context in AI systems are fundamentally limited:
- **LLMs** have finite, passive context windows
- **MANNs** are hard to train and lack online learning
- **Continual learning** assumes discrete task boundaries and batch training

The RAIE architecture aims to combine the best of all three while addressing their limitations:
- Sparse, local learning rules (from predictive coding)
- Multiple memory systems (from complementary learning theory)
- Online, incremental adaptation (from streaming algorithms)
- Multimodal binding (from vector symbolic architectures)

The path to real-time adaptive AGI requires abandoning the batch-training paradigm entirely and building systems that learn continuously from every interaction, just like biological brains do.

---

*End of survey. Next phase: Implement RAIE with SPU, ATM, UALR and test on real MNIST.*
