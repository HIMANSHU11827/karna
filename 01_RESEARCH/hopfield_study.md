# Hopfield Networks: Associative Memory, Convergence, and Capacity

**Author:** Anvil (Coder 2) — AGI Research Lab  
**Date:** 2026-09-23  
**Phase:** 1 — Study  
**Scope:** Classical Hopfield, Modern Hopfield (Dense Associative Memory), capacity limits, Krotov improvements

---

## 1. Classical Hopfield Networks

### 1.1 Model

**Original Paper:** Hopfield (1982) — "Neural Networks and Physical Systems with Emergent Collective Computational Abilities"

A Hopfield network is a recurrent neural network with binary neurons $s_i \in \{-1, +1\}$ and symmetric weights $W_{ij} = W_{ji}$.

**Energy Function:**
$$E = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j$$

**Update Rule (Asynchronous):**
$$s_i(t+1) = \text{sign}\left(\sum_j W_{ij} s_j(t)\right)$$

**Learning Rule (Hebbian):**
$$W_{ij} = \frac{1}{N} \sum_{\mu=1}^{P} \xi_i^\mu \xi_j^\mu$$

Where $\xi^\mu$ are the stored patterns and $N$ is the number of neurons.

### 1.2 Key Properties

- **Energy decreases monotonically** with each update → guaranteed convergence
- **Converges to local minima** of energy landscape
- **Local minima = stored patterns** (attractors)
- **Basin of attraction:** Region around each attractor where patterns converge

### 1.3 Classical Capacity Limit

**Amit, Gutfreund, Sompolinsky (1985):**

For random patterns, the storage capacity is:
$$\alpha_c = \frac{P}{N} \approx 0.138$$

Meaning: A network of $N$ neurons can store approximately $0.138N$ random patterns.

**Beyond capacity:** When $P > 0.138N$, the network enters a spin-glass phase where:
- Stored patterns become unstable
- Spurious states (mixtures) proliferate
- Retrieval accuracy drops sharply

**For orthogonal patterns:** Capacity approaches $P = N$ (all patterns stable as fixed points).

### 1.4 Basin of Attraction

Two notions of basin size:

1. **AGS Basin (Amit-Gutfreund-Sompolinsky):** Existence of local energy minimum around pattern
   - Requires: $P < \alpha_c N$ for random patterns
   - Allows approximate retrieval (some bit errors)

2. **NLT Basin (Non-Local Tolerance):** Existence of firm energy barrier
   - Stricter than AGS
   - Lower bounds: $\alpha \geq 0.08$ for random patterns

---

## 2. Modern Hopfield Networks (Dense Associative Memory)

### 2.1 Key Innovation

**Krotov & Hopfield (2016):** Replace quadratic energy with higher-order polynomial:
$$E = -\sum_\mu F\left(\sum_i \xi_i^\mu s_i\right)$$

Where $F(x)$ is a polynomial of degree $n$.

**For $n=2$ (classical):** $F(x) = x^2$

**For higher $n$:** Super-linear capacity $P \sim N^{n-1}$

**For exponential $F$:** Exponential capacity $P \sim e^{\alpha N}$

### 2.2 Exponential Modern Hopfield (Ramsauer et al., 2020)

**Key paper:** "Hopfield Networks is All You Need" (2020)

**Energy Function:**
$$E = -\log \sum_\mu \exp\left(\beta \sum_i \xi_i^\mu s_i\right) + \frac{1}{2} \sum_i s_i^2$$

This is equivalent to softmax attention over stored patterns!

**Update Rule:**
$$s_{\text{new}} = \text{softmax}\left(\beta \cdot \Xi^T \cdot s_{\text{current}}\right) \cdot \Xi^T$$

Where $\Xi$ is the pattern matrix (each row is a stored pattern).

**Connection to Transformers:**
- The update rule is mathematically equivalent to the attention mechanism
- Stored patterns = keys and values
- Current state = query
- $\beta$ = inverse temperature (sharpness)

### 2.3 Capacity of Modern Hopfield Networks

**Lucibello & Mézard (2024):** Statistical mechanics analysis of exponential capacity

**Key Results:**
- Single-pattern retrieval threshold: $\alpha_1(\lambda) = \frac{1}{2}\log \lambda$ for Gaussian patterns
- Capacity threshold $\alpha_c(\lambda) \leq \alpha_1(\lambda)$
- Basin of attraction: $\cos(\theta) > \phi_{\alpha, \rho}(\lambda)$ for retrieval

**For spherical patterns (bounded norm):**
- Capacity scales as $P = e^{\alpha N}$ with $\alpha_c \sim 0.5 \log \lambda$ at large $\lambda$

**For Gaussian patterns (unbounded norm):**
- Capacity saturates: $\alpha_1 \to 1/2$ at large $\lambda$
- Due to rare high-norm patterns destabilizing retrieval

**Sparse Modern Hopfield Model (2023):**
- Adds sparsity to the energy function
- Uses sparsemax instead of softmax
- Exponential capacity with tighter error bounds
- Better pattern separation due to sparsity

---

## 3. Krotov-Hopfield Improvements (2019-2021)

### 3.1 Two-Layer Architecture

**Krotov & Hopfield (2019):** "Unsupervised learning by competing hidden units"

**Architecture:**
- Visible layer: $N_v$ neurons (input)
- Hidden layer: $N_h$ neurons (competitive)
- Winner-take-all dynamics in hidden layer

**Learning Rule (L^p Hebbian):**
$$\Delta w_{ij} = \eta \cdot \text{sgn}(h_j)^{p-1} \cdot v_i \cdot |h_j|^{p-1}$$

Where $v$ is visible, $h$ is hidden, $p > 1$.

**For $p=2$:** Classical Hebbian
**For $p>2$:** Stronger competition, better feature separation

**Results:**
- MNIST: ~96% with linear classifier on top
- Sparse, interpretable features
- Online learning, no backprop

### 3.2 Exponential Capacity Two-Layer

**Krotov & Hopfield (2021):** Exponential capacity via two-layer architecture

**Key Result:** $P = e^{\alpha N_h}$ patterns can be stored, where $N_h$ is hidden dimension.

**Limitation (addressed by us):**
- Winner-takes-all → each hidden neuron = one pattern
- "Grandmother cell" representation (not distributed)

**Our Improvement (2026):** Threshold activation (not WTA)
- Allows distributed representations
- Hidden neurons can participate in multiple patterns
- All binary states become stable fixed points
- Exponential capacity: $2^{N_h}$ patterns

---

## 4. Capacity Limits: Summary Table

| Model | Capacity $P$ | Key Parameter | Learning |
|-------|-------------|---------------|----------|
| **Classical Hopfield** | $0.138N$ | Neurons $N$ | Hebbian |
| **Hopfield (orthogonal)** | $N$ | Neurons $N$ | Hebbian |
| **Modern Hopfield (poly-$n$)** | $N^{n-1}$ | Degree $n$ | Hebbian |
| **Modern Hopfield (exp)** | $e^{\alpha N}$ | Neurons $N$, $\beta$ | Hebbian |
| **Krotov-Hopfield 2-layer** | $e^{\alpha N_h}$ | Hidden $N_h$ | L^p Hebbian |
| **KHM (Kernelized)** | $e^{\alpha D_\Phi}$ | Feature dim $D_\Phi$ | Learned kernel |
| **Sparse Modern Hopfield** | $e^{\alpha d}$ (tighter bound) | Pattern dim $d$ | Hebbian |

### 4.1 Key Insights on Capacity

1. **Pattern separation matters:** Higher separation → higher capacity
2. **Sparsity helps:** Sparse patterns have higher capacity than dense
3. **Structure hurts:** Correlated patterns reduce capacity vs random
4. **Distributed > Grandmother:** Distributed representations enable exponential capacity
5. **Kernel trick:** Kernelized Hopfield can achieve higher capacity in feature space
6. **Tradeoff:** Basin size vs capacity — larger basins mean fewer patterns

---

## 5. Convergence Properties

### 5.1 Classical Hopfield

- **Guaranteed convergence:** Energy decreases monotonically
- **Time:** $O(N^2)$ per update (all-to-all connectivity)
- **Iterations:** Typically converges in 5-20 updates
- **Local minima:** Can get stuck in spurious states

### 5.2 Modern Hopfield

- **One-step retrieval:** For large $\beta$, converges in 1 step
- **Exponential error suppression:** Retrieval error $\sim e^{-\beta \Delta}$
- **Basin of attraction:** Depends on pattern separation and $\beta$

### 5.3 Our Constraint

Per FINAL_DESIGN.md: **Cap at 5 iterations maximum.**

This is a hard real-time constraint. Classical Hopfield typically converges in 5-10 iterations. Modern Hopfield with large $\beta$ converges in 1. We need to design for the 5-iteration worst case.

---

## 6. Key References

1. Hopfield (1982). "Neural networks and physical systems with emergent collective computational abilities." PNAS
2. Amit, Gutfreund, Sompolinsky (1985). "Storing an infinite number of patterns in a spin-glass model." PRL
3. Krotov & Hopfield (2016). "Dense associative memory for pattern recognition." NeurIPS
4. Krotov & Hopfield (2019). "Unsupervised learning by competing hidden units." Neural Computation
5. Ramsauer et al. (2020). "Hopfield Networks is All You Need." arXiv:2008.02217
6. Krotov & Hopfield (2021). "Large associative memory problem in neurobiology and electronics." arXiv:2101.03961
7. Lucibello & Mézard (2024). "Exponential capacity of dense associative memories." arXiv:2304.14964
8. Wu et al. (2024). "Provably Optimal Memory Capacity for Modern Hopfield Models." arXiv:2410.23126
9. Krotov (2023). "Sparse Modern Hopfield Model." arXiv:2309.12673
10. "A Biologically Plausible Dense Associative Memory with Exponential Capacity." arXiv:2601.00984 (2026)

---

## 7. Implications for RT-ALM

### 7.1 Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Memory type** | Dense Associative Memory (exponential) | Highest capacity, proven convergence |
| **Dimension** | 10,000 (SDR space) | Kanerva: >5000 for good separation |
| **Sparsity** | 2% (200 active bits) | Maass: 1-5% optimal |
| **Capacity** | ~$e^{0.02 \times 10000} = e^{200}$ (theoretical) | In practice: 10,000 patterns |
| **Max iterations** | 5 (hard constraint) | Real-time requirement |
| **Learning rule** | Krotov L^p Hebbian | Stable, unsupervised, online |
| **Retrieval** | Modern Hopfield update | One-step for large $\beta$ |

### 7.2 What We Build

1. **AttractorMemory** class:
   - Store patterns via Hebbian outer product
   - Retrieve via iterative energy minimization
   - Cap at 5 iterations
   - Fallback to original query if no convergence

2. **Online learning:**
   - Update weights incrementally (no batch)
   - Importance-weighted (EWC-style for stability)
   - Eligibility traces for credit assignment

3. **Capacity management:**
   - Reservoir sampling (FIFO eviction at capacity)
   - Diversity constraints (prevent overrepresentation)
   - Consolidation (replay during idle)

---

*End of Hopfield study. Next: Implement AttractorMemory with 5-iteration cap.*
