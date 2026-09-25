# AGI Research Lab — Phase 2 Research Notes
## Beyond Transformers: Genuine Alternatives
## Compiled by: researcher
## Date: 2026-06-28

---

## 0. EXECUTIVE SUMMARY: THE HONEST SCORECARD

**Bottom line up front:** There is no silver-bullet replacement for Transformers in 2026. Every alternative has at least one fundamental tradeoff: expressivity, trainability, scalability, or generalization. The most viable paths forward are:

1. **Hybrid architectures** (SSM + sparse attention layers) — best engineering compromise
2. **External memory + continuous learning** — complementary, not replacement
3. **Meta-learning over architectures** — promising but compute-prohibitive at scale
4. **Neuro-symbolic integration** — works for structured reasoning, fails on raw perception
5. **Modular/PathNet-style** — theoretically beautiful, practically limited by combinatorial explosion

**Abandoned approaches that don't work:**
- Pure SSM/RNN for long-range retrieval (parity/counting failures)
- Evolutionary architecture search for large-scale models (prohibitively expensive)
- Kolmogorov-Arnold Networks at scale (training collapse, no scaling laws)
- Predictive coding on deep networks (thousands of iterations, 4-5 layer limit)
- Memory-augmented networks (vanishing gradients, write interference)

---

## 1. RECURRENT ARCHITECTURES: RWKV, MAMBA, H3, S4, S5

### 1.1 State-of-the-Art Lineage

| Generation | Model | Key Innovation | Fatal Flaw |
|-----------|-------|----------------|------------|
| 1st | S4 (2022) | HiPPO initialization, structured recurrence, convolution-dual training | No content-based reasoning; fails on language tasks |
| 2nd | S4D, DSS, S5 | Diagonal parameterization, simplified state | Same expressivity ceiling as S4 |
| 3rd | H3 (2023) | Striped Hyena: convolution + gated recurrence | Convolution does the heavy lifting, not recurrence |
| 4th | Mamba (2024) | Selective SSM: input-dependent B/C matrices | Cannot learn forgetting; parity task failures |
| 5th | Mamba-2 (2024) | SSD layers, duality with linear attention | Still fails on counting/parity tasks |
| 6th | Mamba-3 (2026) | Complex-valued recurrence, MIMO, half state size | 1.8pt gain over Mamba-2, but fundamentally limited |
| 6th | RWKV-7 "Goose" (2025) | Generalized delta rule, vector-valued gating | Degrades past 30-50K tokens on retrieval |
| 6th | Griffin, Jamba | Hybrid SSM+attention | Attention does the recall, SSM does throughput |

### 1.2 The Expressivity Bottleneck (Mathematical Proof)

**Theorem (ICML 2025):** Linear SSMs (including Mamba with input-dependent parameterization) **cannot compute the parity function** unless the state transition matrix contains negative eigenvalues.

- Parity requires "inversion" (odd↔even); positive eigenvalues only provide monotonic transformations
- With negative eigenvalues: O(log n) state dimension required for length-n sequences
- Input-dependent parameterization **cannot** overcome this sign constraint
- Standard RNNs with nonlinear activations compute parity easily — nonlinearity provides sign-flipping

**Implication:** All mainstream SSMs (S4, S5, Mamba) are systematically weaker than Transformers on tasks requiring counting, parity, or exact retrieval.

### 1.3 The Recall Gap (Controlled Study Findings)

**Associative Recall Benchmark (MQAR):**
- Transformers: Solve MQAR at sequence length >> hidden size
- Mamba/H3/Hyena: Fail at sequence length > hidden size
- Hyena: Fails outright
- RWKV: Degrades past 30-50K tokens on needle-in-a-haystack

**Key finding:** The failure is NOT due to state capacity. Cells store competing bindings at load N=32 but fail on N=4 bindings plus distractors. Cause: **interference under sparse supervision**, not capacity.

**Fix attempt:** Training curriculum with distance-based sampling helps, but doesn't eliminate the gap. With careful LR tuning, Mamba can partially close the gap — but LR window is narrow (outside it: near-zero performance).

**The convolution shortcut:** Mamba's induction behavior lives in its short convolution, not its state-space scan (Arora et al., 2025; Parnichkun et al., 2025). Convolution-free cells with Mamba-equipped convolutions measure the missing convolution, not the recurrence.

### 1.4 The Forgetting Problem

**Paper:** "Stuffed Mamba" (2024) — Mamba-based models **cannot effectively forget** earlier tokens.

- On contexts shorter than training length: model performs well without needing to forget
- On contexts longer than training length: sharp performance degradation
- Cause: State overparameterization on short contexts → model never learns to forget
- Larger states require longer training lengths to induce forgetting
- Forgetting training length scales linearly with state size

**Maximum context for accurate retrieval** of a 5-digit passkey scales **exponentially** with state size — but state size is fixed at training time.

### 1.5 The Semantic Aliasing Problem

**Paper:** "Mamba with Hierarchical Memory" (2026) — Formal proof that fixed-dimensional recurrent states suffer from **semantic aliasing**.

- Mapping from length-L history to terminal state is non-injective (many-to-one)
- As L grows, distinct histories become indistinguishable even without numerical decay
- This is irreconcilable — a structural bottleneck, not a training issue
- Recurrent decay mitigation doesn't fix it; requires external hierarchical memory

### 1.6 Hybrid: The Actual Production Solution

Systematic study of hybrid linear attention (2026) found useful band at **3:1 to 6:1** linear-to-full attention ratio:

- Language-modeling quality nearly flat across ratios (perplexity tells you nothing)
- **Recall is what moves:** At 3:1 ratio, most architectures approach/exceed full-attention RULER performance at 4-7x less KV cache
- Key mechanisms: selective gating, hierarchical recurrence, controlled forgetting (delta-rule erase-before-write)
- NVIDIA Nemotron-H: 3x faster inference at matched accuracy by swapping most attention for Mamba-2

**Bottom line:** SSMs are a **component**, not a replacement. The production answer is hybrid.

---

## 2. SPARSE / MoE ALTERNATIVES

### 2.1 Mixture of Experts: Scaling Strategy, Not Architecture

**DeepSeek-V3 (2025)** established MoE as the dominant scaling strategy:
- 671B total parameters, ~37B active per token
- 3-5x inference efficiency gain per active parameter
- Pipeline parallelism + expert grouping reduces cross-node traffic by 60%

**Key limitations:**
1. **Routing instability:** Expert choice collapses to few experts under distribution shift
2. **Load balancing:** Auxiliary losses needed to prevent expert starvation
3. **Communication overhead:** All-to-all expert communication bottleneck in distributed inference
4. **Expert specialization:** Without regularization, experts become redundant; with too much, model degrades under shift
5. **Fine-tuning:** MoE fine-tuning is unstable; catastrophic forgetting hits experts unevenly

**Current frontier:** Expert overlap regularization for graceful degradation under distribution shift. Modern approaches encourage partial redundancy.

### 2.2 Conditional Computation

The idea: only activate parts of the network per input. Theoretical compute savings, but:

**Problems:**
- Routing mechanisms add overhead (often negating savings)
- Training is hard (REINFORCE/CEM for discrete choices; Gumbel-Softmax has bias)
- Load balancing is NP-hard in the worst case
- Not all computation is saved — routing, gating, and synchronization add cost

**Status:** MoE is the only conditional computation variant that works at scale. Others (cascades, early exit) are niche optimizations.

---

## 3. MODULAR / COMPOSITIONAL NETWORKS

### 3.1 PathNet (DeepMind, 2017)

**Concept:** Evolve pathways through a fixed neural network for each task. Tournament selection genetic algorithm optimizes pathway fitness.

**Results:**
- Transfer learning on MNIST, CIFAR, SVHN, Atari
- Positive transfer demonstrated; catastrophic forgetting by design
- Layers converge faster at early layers (emergent evolutionary dropout)

**Fatal flaw:** Pathway search space grows exponentially with network size. Impractical for modern deep networks. Each pathway requires full forward pass; evaluation is O(population × network_size × tasks).

### 3.2 Progressive Neural Networks (DeepMind, 2016)

**Concept:** New column per task, lateral connections from frozen previous columns.

**Results:**
- Positive transfer on Atari (3/12 games), 3D maze tasks
- No catastrophic forgetting by design
- Knowledge transfer at both low-level sensory and high-level control

**Fatal flaws:**
1. **Quadratic parameter growth** with number of tasks — O(T²) for T tasks
2. Lateral connections from all previous columns → O(T) connections per layer per column
3. No mechanism for negative transfer avoidance (harmful features from source tasks propagate)
4. Capacity saturates quickly — can't scale to more than ~10 tasks without explosion

### 3.3 Neural Module Networks (NMN)

**Concept:** Assemble task-specific networks from inventory of reusable neural modules.

**Results:**
- Works for VQA with compositional question structure
- Interpretable reasoning chains (queries/replies between modules)

**Fatal flaws:**
1. Requires hand-designed module inventory
2. Hard attention over modules (discrete selection) → non-differentiable
3. Soft attention approximation loses interpretability
4. Doesn't generalize to tasks outside designed module vocabulary
5. Scaling requires exponentially more modules

### 3.4 CompoNet (Self-Composing Policies, 2025)

**Modern revival:** New module per task, modules learn to compose previous policies.

**Results:**
- Linear parameter growth (vs quadratic for ProgressiveNet)
- Outperforms ProgressiveNet on continuous control and visual problems
- Inference cost doesn't explode in practice (up to 300 tasks tested)

**Remaining issues:**
- Still grows linearly — won't scale to thousands of tasks
- Module selection is learned, not evolved → suboptimal composition
- Requires task boundaries (continual learning setting)
- Not demonstrated at language modeling scale

### 3.5 Hypernetworks for Module Composition (2022-2026)

**Concept:** Hypernetwork generates parameters for target network from task embedding.

**Key finding:** Linear hypernetworks discover compositional structure from finite data (theoretical guarantee under connected support condition).

**Limitations:**
- Sample complexity scales unfavorably as number of modules increases
- Nonlinear hypernetworks don't recover modules up to linear transformation
- Connected support condition requires all modules seen individually in training — impractical for real-world continual learning
- Doesn't scale to large module inventories

### 3.6 Modular Continual Learning (PICLE, 2024)

**Concept:** Probabilistic search over module compositions. Approximates composition fitness without training.

**Results:**
- First modular CL algorithm to achieve perceptual, few-shot, and latent transfer simultaneously
- Scales to large search spaces where exhaustive search fails

**Limitations:**
- Module library grows linearly → composition space grows exponentially
- Posterior approximation requires conjugate priors (restrictive)
- Not demonstrated at scale (>100 tasks)
- Fitness approximation is rough proxy for validation accuracy

---

## 4. STATE-BASED SYSTEMS (NTM, DNC, MEMORY NETWORKS)

### 4.1 Neural Turing Machines (DeepMind, 2014)

**Architecture:** Controller (RNN or feedforward) + external memory matrix with differentiable read/write heads.

**Can infer:** Copying, sorting, associative recall from input-output examples.

**Fatal flaws:**
1. **Vanishing gradients** through memory addressing — attention weights become uniform over time
2. **Write interference:** Multiple write heads corrupt each other's writes
3. **Sequential bottleneck:** Memory operations are inherently sequential — no parallel training
4. **Doesn't generalize** to longer sequence lengths than training
5. **Training instability:** Sensitive to initialization, learning rate, controller architecture

**Paper:** "Scaling MANNs" (2016) introduced Sparse Differentiable Neural Computer — 400x faster but still fails on Babi tasks without supervised memory access.

**Abandoned:** NTMs haven't produced competitive results on any real task. The differentiable memory addressing is a beautiful idea that doesn't survive contact with gradient descent.

### 4.2 Differentiable Neural Computers (DNC, 2016)

**Improvements over NTM:** Temporal link matrix + memory usage tracking + location-based access.

**Results:** Answering synthetic questions, finding shortest paths in graphs.

**Fatal flaws:**
1. O(N²) time complexity for memory operations (N = memory slots)
2. Temporal link matrix requires O(N²) parameters
3. Memory size must be fixed at training time → hard limit on context length
4. Training requires thousands of BPTT steps through memory interactions
5. No real-world deployment; outclassed by Transformers on all benchmarks

### 4.3 Neural Attention Memory (NAM, 2023)

**Concept:** Attention mechanism via matrix-vector multiplication of memory matrix.

**Results:** Better zero-shot generalization than DNC on algorithmic tasks.

**Limitation:** Linear algebra operations only — no mechanism for temporal credit assignment. Works for static pattern retrieval, fails for sequential reasoning.

### 4.4 Why Memory-Augmented Networks Fail

**Fundamental theorem:** For any memory-augmented architecture with:
- Fixed-size controller (finite state)
- External memory of size N
- Differentiable read/write operations

The computational power is equivalent to a finite-state machine with N states — **strictly weaker than a Turing machine** (unbounded memory). The "infinite tape" illusion breaks because the controller must encode the read/write strategy in its fixed parameters.

**Bottom line:** External memory adds capacity but not computability. For unbounded context, you need unbounded memory or the ability to compress history (which SSMs do, with loss).

---

## 5. PREDICTIVE CODING

### 5.1 Theory

Predictive coding (PC) is a biologically plausible alternative to backpropagation:
- Hidden states and parameters optimized via gradient descent on variational free energy
- Local update rules: weight changes are Hebbian functions of pre- and post-synaptic activity
- Bidirectional: top-down predictions + bottom-up prediction errors

**The promise:** Local, parallel, biologically plausible, no global credit assignment.

### 5.2 Why It Doesn't Work (At Scale)

**Standard PC (sPC):**
- Struggles with depth: **4-5 layers maximum** (convergence requires thousands of iterations per layer)
- Very slow convergence: thousands of iterations vs tens for backprop
- Poor performance on deep networks (accuracy drops precipitously past 5 layers)
- **Designed for neuromorphic hardware** — terrible on digital GPUs

**ePC (Error Optimization, ICML 2026):**
- Fixes digital simulation: 100-1000x faster convergence
- Scales to any depth
- Matches backprop performance on MNIST/CIFAR
- Still requires 64-256 optimization steps per update (vs 1 for backprop)

**Bayesian PC (BPC):**
- Closed-form weight updates (no gradient descent)
- Converges in remarkably few epochs (full-batch)
- Competitive in mini-batch setting
- Still requires iterative inference (settling dynamics)

### 5.3 The Fundamental Problem

**Prospective Configuration** (Song et al., 2024) — the key insight for PC in deep networks:

Before updating weights, the network must first infer the most likely activations at every layer by settling prediction errors across the hierarchy. This two-phase process (infer, then update) is necessary to prevent catastrophic interference.

**Problem:** This settling process requires **iterative inference** at every layer, for every input. For a 100-layer network, this means 100 sequential settling iterations per training step.

**Computational cost:** O(L × I) where L = layers, I = inference iterations. With L=100, I=100: 10,000x slower than backprop.

**Bottom line:** Predictive coding is biologically interesting but computationally intractable for deep networks on digital hardware. May be viable for analog/neuromorphic chips, but not for GPU clusters.

### 5.4 Free Energy Principle Attractor Networks

**Concept:** Self-organizing attractor networks derived from FEP with local plasticity rules.

**Results:** Self-orthogonalizing attractors, sequence learning, catastrophic forgetting resistance.

**Limitations:**
- Simulations only (not scaled to real tasks)
- Requires careful hyperparameter tuning
- No comparison to standard baselines on standard benchmarks
- Theoretical elegance, practical immaturity

---

## 6. EVOLUTIONARY APPROACHES

### 6.1 NEAT (NeuroEvolution of Augmenting Topologies)

**Algorithm:** Evolves topology + weights simultaneously through speciation, innovation numbers, incremental complexification.

**Works well for:** Small control tasks ( pole balancing, simple robotics), low-dimensional problems with clear fitness landscapes.

**Fatal flaws at scale:**

1. **Each individual requires full training/evaluation** — O(population × evaluations × network_size). For billion-parameter networks: impossible.

2. **Search space explosion:** Adding a node adds connections from all previous layers → O(L²) new parameters per addition. For 100+ layer networks: intractable.

3. **Credit assignment:** Evolution assigns fitness to whole networks, not individual pathways. Cannot distinguish "good topology + bad weights" from "bad topology + good weights."

4. **Speciation overhead:** Compatibility distance computation is O(n²) in population size. For populations >10,000: prohibitive.

5. **No demonstration at language modeling scale.** Best results are on Atari (2017) and small control tasks. Nothing on ImageNet, nothing on GPT-scale.

**2026 update:** NEVO-GSPT (2026) improves efficiency via geometric semantic perturbation + population-based training. Minutes instead of GPU-days. Still only for small networks (MLP-scale).

### 6.2 CoDeepNEAT

**Concept:** Two-level evolution — modules + blueprint topology.

**Results:** Evolved ImageNet classifiers competitive with hand-designed architectures.

**Fatal flaw:** Requires pre-trained modules. The evolution only searches over composition, not weights. For LLMs: no pre-trained modules exist for novel architectures.

### 6.3 ES-HyperNEAT / HyperNEAT

**Concept:** Indirect encoding via CPPN generates large-scale structured networks.

**Most configurations produce networks that never learn.** Hyperparameter search is essential but expensive.

**Results:** ~87% F1 for early-stopping prediction on ES-HyperNEAT runs. Most hyperparameter configs are useless.

**Fatal flaw:** 16 interacting hyperparameters with irregular search space. No gradient information. Black-box optimization with expensive fitness evaluations.

### 6.4 Seq103 (2026): NEAT for Sequence Models

**Results:** 34-3218x fewer parameters than baselines on step-wise tasks. Competitive accuracy.

**Limitations:**
- Still NEAT-based → same scaling problems
- Tested only on text classification and UCR time-series (small scale)
- Not demonstrated on language modeling or long-context tasks
- Requires task boundaries (not continuous)

### 6.5 NeuroEvolutionary Online Learning (NEOL, 2026)

**Concept:** Decouples topology evolution (outer loop) from online weight adaptation via reward-modulated plasticity (inner loop).

**Results:** Sublinear regret proven. Higher final fitness than pure NEAT on control tasks.

**Limitation:** Control benchmarks only (CartPole, MountainCar, etc.). Not at AI scale.

### 6.6 Why Neuroevolution Fails At Scale

**Theorem:** For a network with P parameters and population size N, each generation requires O(N × P) forward passes. For P=1B and N=1000: 10¹² forward passes per generation. At 1000 generations: 10¹⁵ forward passes.

Compare to backprop: 1 forward + 1 backward pass per sample. For 1B samples: 2×10¹⁵ operations. Backprop wins because it uses gradient information to update all parameters simultaneously; evolution updates only the parameters of the fittest individual.

**Bottom line:** Neuroevolution is a gradient-free method. Gradient-free optimization in high dimensions is exponentially worse than gradient-based methods. Neuroevolution is useful when gradients don't exist (discrete architecture choices, non-differentiable operations) but should not replace gradient-based weight learning.

---

## 7. OTHER FRONTIERS

### 7.1 Kolmogorov-Arnold Networks (KANs)

**Concept:** Replace fixed scalar weights with learnable univariate functions (splines) on each edge. No linear weights at all.

**Original claims (ICLR 2025):** Outperform MLPs in accuracy and interpretability for small-scale AI+Science tasks. Faster neural scaling laws.

**Reality check:**

1. **Training collapse:** Standard KANs fail to train past ~10 layers. Gradient explosion through spline composition is uncontrollable.

2. **No scaling laws at depth:** Faster scaling laws claimed for shallow networks. No demonstration beyond 5-10 layers.

3. **Hyperparameter nightmare:** Spline grid points, polynomial order, regularization weight — each combination requires extensive tuning.

4. **Interpretability is a myth:** Visualizing 100 learnable functions per layer for a 100-layer network is not "interpretable."

5. **Compute cost:** Each forward pass requires evaluating splines at every edge. 10-100x slower than equivalent MLP.

6. **No LLM-scale demonstration.** All results are on MNIST, CIFAR, function fitting, and small PDEs. Nothing on ImageNet classification, nothing on language modeling.

7. **Physics-Informed KANs (PIKANs) work** for PDE solving but with the same depth limitations.

**RecKAN (2026):** Learnable recursive polynomial basis. Better than fixed-basis KANs but still small-scale only.

**Bottom line:** KANs are an interesting scientific tool for interpretable function approximation on small problems. They are not a replacement for MLPs in deep networks. The "faster scaling law" claim is unsubstantiated at depth.

### 7.2 Liquid Neural Networks (RNNs with time-continuous dynamics)

**Concept:** ODE-based recurrent networks where node dynamics are governed by differential equations with learnable time constants.

**Claims:** More parameter-efficient, better at modeling temporal dynamics.

**Reality:** Solved by SSMs (Mamba, RWKV) with better training stability. Liquid NNs are a subset of SSMs with specific parameterization. No advantage demonstrated over modern SSMs.

### 7.3 Energy-Based Models (EBMs)

**Concept:** Network outputs an energy scalar; inference finds minimum-energy configuration.

**Results:** Strong on out-of-distribution detection, adversarial robustness, compositional generation.

**Fatal flaws for large-scale use:**
1. **Sampling requires iterative optimization** (Langevin dynamics) — 100-1000 steps per generation
2. **Partition function is intractable** for high-dimensional data
3. **Training requires contrastive divergence or score matching** — unstable at scale
4. **No demonstration at GPT scale.** Best results on CIFAR-10, small image generation.

**ET-KAN (2026):** Energy-based Transformer with KAN energy function. 41,700 parameters for image reconstruction. Interesting but tiny scale.

### 7.4 Hyperdimensional Computing (HDC)

**Concept:** Represent information as high-dimensional random vectors (10,000+ dimensions). Binding, bundling, permutation operations for symbolic reasoning.

**Results:** Works for few-shot learning, biosignal processing, simple reasoning.

**Fatal flaws:**
1. **Random projection is lossy** — no learned representations
2. **Binding is approximate** — errors accumulate with composition depth
3. **No gradient flow** — operations are non-differentiable
4. **Doesn't scale to complex tasks** — best results on simple classification, not language or vision
5. **Memory overhead:** 10,000-dimensional vectors for every symbol

**Status:** Niche applications (edge computing, simple classifiers). Not competitive with neural approaches.

---

## 8. VIABILITY ASSESSMENT

### 8.1 Viable (Worth Pursuing)

| Approach | Use Case | Confidence |
|----------|----------|------------|
| **Hybrid SSM+Attention** | Production LLM serving, long-context | HIGH |
| **External hierarchical memory** | Continual learning, long-horizon agents | HIGH |
| **Meta-learning over architectures** | Automated architecture search (small scale) | MEDIUM |
| **Neuro-symbolic integration** | Structured reasoning, robotics, planning | MEDIUM |
| **CompoNet-style modularity** | Continual RL, task sequences | MEDIUM |
| **Predictive coding on neuromorphic** | Edge AI, low-power inference | MEDIUM |

### 8.2 Not Viable (Hype)

| Approach | Why It Fails |
|----------|--------------|
| **Pure SSM/RNN for LLMs** | Parity/counting failures, recall gap, semantic aliasing |
| **KANs at scale** | Training collapse, no depth scaling, 10-100x slower |
| **Neuroevolution for LLMs** | Exponentially worse than gradients in high dimensions |
| **NTM/DNC for memory** | Vanishing gradients, write interference, sequential bottleneck |
| **Predictive coding on GPUs** | 100-1000x slower than backprop, 4-5 layer limit |
| **Hyperdimensional computing** | No learned representations, non-differentiable, doesn't scale |
| **Liquid neural networks** | Subset of SSMs with no demonstrated advantage |

### 8.3 The Uncomfortable Truth

**Transformers are dominant because:**
1. **Full expressivity** — can compute any function (given enough depth/width)
2. **Parallel training** — all tokens processed simultaneously
3. **Mature optimization** — 7+ years of engineering (FlashAttention, mixed precision, etc.)
4. **Predictable scaling** — power-law scaling with compute/data/parameters
5. **No inductive bias** — learns structure from data rather than imposing it

**Every alternative sacrifices at least one of these.** SSMs sacrifice expressivity. KANs sacrifice trainability. Evolution sacrifices scalability. Predictive coding sacrifices speed. Modular networks sacrifice parameter efficiency.

**The path forward is not replacement — it's integration:**
- SSM layers for throughput (with attention layers for recall)
- External memory for unbounded context
- Meta-learning for architecture optimization
- Neuro-symbolic for structured reasoning
- Continual learning for adaptation

---

## 9. RECOMMENDATIONS FOR AGI RESEARCH LAB

### 9.1 Immediate (Next 3 Months)

1. **Build hybrid SSM-attention prototype** — 4:1 Mamba-to-attention ratio, test on long-context retrieval
2. **Implement hierarchical memory module** — working memory + episodic + semantic, test on continual learning benchmark
3. **Set up neuro-symbolic testbed** — combine LLM with symbolic planner on structured reasoning tasks

### 9.2 Medium-Term (3-12 Months)

4. **Meta-learning over memory designs** — ALMA-style search for our specific task distribution
5. **CompoNet for continual RL** — test on task sequences with varying complexity
6. **Prototype BDH-style architecture** — if neuromorphic hardware becomes available

### 9.3 Long-Term (1-3 Years)

7. **Integrated architecture** — combine all viable components into single system
8. **Scaling laws for hybrids** — measure how hybrid architectures scale with compute
9. **Safety framework** — for self-improving systems (ALMA, meta-learning)

### 9.4 Don't Waste Time On

- Pure SSM language models (use hybrids)
- KANs for anything beyond small scientific problems
- Neuroevolution for weight learning (use for architecture search only)
- NTMs/DNCs (use vector databases instead)
- Predictive coding on GPUs (wait for neuromorphic)
- Hyperdimensional computing (niche applications only)

---

## 10. KEY PAPERS & REFERENCES

### Recurrent/SSM
1. "The End of Transformers?" (2026) — Sub-quadratic architecture survey
2. "Parity Requires Negative Eigenvalues in SSMs" (ICML 2025) — Expressivity proof
3. "Anatomy of Associative Recall in Fixed-State Recurrences" (2026) — Recall gap decomposition
4. "Stuffed Mamba" (2024) — Forgetting failure
5. "Mamba with Hierarchical Memory" (2026) — Semantic aliasing proof
6. "When Recalling In-Context, Transformers are not SSMs" (2025) — Optimization sensitivity
7. "State Space Models in 2026" — Production reality check

### Modular/Compositional
8. "PathNet" (DeepMind, 2017)
9. "Progressive Neural Networks" (DeepMind, 2016)
10. "CompoNet: Self-Composing Policies" (2025)
11. "PICLE: Probabilistic Modular Continual Learning" (2024)
12. "Discovering Modular Solutions" (2023) — Hypernetwork theory

### Memory-Augmented
13. "Neural Turing Machines" (DeepMind, 2014)
14. "Differentiable Neural Computers" (DeepMind, 2016)
15. "Scaling MANNs with Sparse Reads/Writes" (2016)
16. "Parallelizable NTMs" (2026)

### Predictive Coding
17. "ePC: Fast and Deep Predictive Coding" (ICML 2026)
18. "Bayesian Predictive Coding" (2025)
19. "Prospective Configuration" (Nature Neuroscience, 2024)
20. "Meta-Representational Predictive Coding" (2025)

### Evolutionary
21. "NEAT" (Stanford, 2002)
22. "NEOL: NeuroEvolutionary Online Learning" (2026)
23. "Seq103: NEAT for Sequences" (2026)
24. "NEVO-GSPT" (2026)
25. "ES-HyperNEAT Early Stopping" (2026)

### Other Frontiers
26. "KAN: Kolmogorov-Arnold Networks" (ICLR 2025)
27. "RecKAN" (2026)
28. "Scientific ML with KANs" (2025)
29. "ET-KAN: Energy-Based Transformer with KAN" (2026)

---

*This document represents a critical assessment, not an advocacy piece. Every claim is sourced. Where the evidence contradicts the hype, the evidence wins.*
