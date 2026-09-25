# First Principles Deduction for AGI Neural Architecture
*Deduced by planner | No existing templates, no copying*

## Preamble

All existing architectures (transformers, Mamba, SSMs, Hopfield networks, etc.) solve specific instances of the credit assignment problem. We are deducing a new architecture from first principles, designed specifically for an AGI assistant that must learn continually, reason hierarchically, and operate without backpropagation.

---

## 1. First Principles

### Principle 1: Bidirectional Information Flow
Perception flows bottom-up; predictions flow top-down. Both are essential. A network that only processes input forward loses the ability to generate, imagine, or plan. A network that only generates loses grounding in reality.

**Deduction:** The architecture must have both feedforward (perceptual) and feedback (generative/predictive) connections, running simultaneously.

### Principle 2: Local Learning with Global Coordination
Backpropagation requires global gradient flow — biologically implausible and computationally expensive. Yet pure local learning (Hebbian) fails at credit assignment across many layers.

**Deduction:** Use **local eligibility traces** modulated by a **global neuromodulatory signal**. The trace records "this synapse was active before an outcome"; the global signal says "the outcome was good/bad." Together they assign credit without backprop.

### Principle 3: Sparse Distributed Representations
Dense representations mix signals, making them hard to disentangle. Sparse representations (few active neurons per pattern) have exponentially higher capacity, better separation, and natural interpretability.

**Deduction:** Each layer uses k-winner-take-all (KWTA) activation. Only the top-$k$ neurons fire; the rest are silent. This creates a sparse binary-like code.

### Principle 4: Energy-Based Dynamics
Stable computation requires the network to converge to consistent states, not oscillate or diverge. An energy function guarantees convergence to a fixed point.

**Deduction:** Define an energy function $E(\mathbf{a}; \mathbf{W})$ over neuron activations $\mathbf{a}$ and weights $\mathbf{W}$. Neuron dynamics minimize this energy. Learning modifies weights to shape the energy landscape.

### Principle 5: Continual Learning Through Complementary Systems
Catastrophic forgetting is the direct result of overwriting weights that store old knowledge. The solution is not regularization but **complementary learning systems**: a fast-learning system for new knowledge and a slow-learning system for consolidated knowledge.

**Deduction:** Two coupled networks — one with fast, sparse learning (episodic), one with slow, distributed learning (semantic). The fast system rehearses/replays to the slow system during consolidation.

### Principle 6: Temporal Coherence
AGI operates in time. The network must maintain stable representations over time (working memory) and predict future states.

**Deduction:** Recurrent connections within each layer create attractor dynamics. Temporal continuity is enforced by slow neuron time constants and predictive learning across time steps.

### Principle 7: Hierarchical Abstraction
Low-level features compose into high-level concepts. Abstraction reduces dimensionality while preserving relevant information.

**Deduction:** Multiple layers, each with decreasing spatial extent and increasing temporal stability. Each level predicts the level below; prediction errors propagate upward.

---

## 2. Mathematical Framework

### 2.1 Neuron Dynamics

For neuron $i$ at level $l$ with activation $a_i^l \in [0, 1]$:

$$\tau_i^l \frac{da_i^l}{dt} = -a_i^l + \sigma\left(\underbrace{\sum_j w_{ij}^{l,\text{ff}} a_j^{l-1}}_{\text{feedforward}} + \underbrace{\sum_k w_{ik}^{l,\text{fb}} a_k^{l+1}}_{\text{feedback}} + \underbrace{\sum_m w_{im}^{l,\text{rec}} a_m^l}_{\text{recurrent}} + b_i^l\right)$$

where:
- $\tau_i^l$ is the neuron time constant (fast for low levels, slow for high levels)
- $w_{ij}^{l,\text{ff}}$ are feedforward weights (from level $l-1$ to $l$)
- $w_{ik}^{l,\text{fb}}$ are feedback weights (from level $l+1$ to $l$)
- $w_{im}^{l,\text{rec}}$ are recurrent weights (within level $l$)
- $b_i^l$ is the bias
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid activation function

**Key insight:** All three connection types are learned, but with different rules. Feedforward and feedback use predictive learning; recurrent uses attractor learning.

### 2.2 Energy Function

Define the energy of the network state:

$$E = \sum_l \left[-\frac{1}{2} \sum_{i,j} w_{ij}^{l,\text{rec}} a_i^l a_j^l - \sum_{i,j} w_{ij}^{l,\text{ff}} a_j^{l-1} a_i^l - \sum_{i,k} w_{ik}^{l,\text{fb}} a_k^{l+1} a_i^l + \sum_i \int_0^{a_i^l} \sigma^{-1}(u) \, du\right]$$

The neuron dynamics naturally minimize this energy. At equilibrium, the network reaches a consistent state where predictions match perceptions.

### 2.3 Sparse Activation (KWTA)

After the dynamics converge, apply k-winner-take-all at each level:

$$a_i^l \leftarrow \begin{cases} a_i^l & \text{if } a_i^l \in \text{top-}k \text{ of level } l \\ 0 & \text{otherwise} \end{cases}$$

This enforces sparse distributed representations. The top-$k$ neurons are the "code" for the current input.

### 2.4 Predictive Learning (No Backprop)

**Prediction:** Each level $l+1$ predicts level $l$:
$$\hat{a}_i^l = \sigma\left(\sum_k w_{ik}^{l+1,\text{fb}} a_k^{l+1} + b_i^l\right)$$

**Prediction error:**
$$\epsilon_i^l = a_i^l - \hat{a}_l^l$$

**Eligibility traces (local memory of recent activity):**
$$e_{ij}^{l,\text{ff}} \leftarrow \gamma_{\text{ff}} e_{ij}^{l,\text{ff}} + a_j^{l-1} a_i^l$$
$$e_{ik}^{l,\text{fb}} \leftarrow \gamma_{\text{fb}} e_{ik}^{l,\text{fb}} + a_k^{l+1} a_i^l$$

where $\gamma_{\text{ff}}, \gamma_{\text{fb}} \in (0, 1)$ are decay constants.

**Global neuromodulatory signal (the "surprise" or "learning gate"):**
$$G = \frac{1}{L} \sum_l \frac{1}{N_l} \sum_i |\epsilon_i^l|$$

This scalar measures total prediction error across the network. When $G$ is high, the network is surprised and should learn. When $G$ is low, the network is accurate and should consolidate.

**Weight updates (the core learning rule):**

Feedforward (perceptual) weights — strengthen connections that reduce prediction error:
$$\Delta w_{ij}^{l,\text{ff}} = \eta_{\text{ff}} \cdot G \cdot \epsilon_i^l \cdot a_j^{l-1}$$

Feedback (generative) weights — reduce prediction error by adjusting predictions:
$$\Delta w_{ik}^{l,\text{fb}} = -\eta_{\text{fb}} \cdot G \cdot \epsilon_i^l \cdot a_k^{l+1}$$

Recurrent (attractor) weights — stabilize the current state:
$$\Delta w_{im}^{l,\text{rec}} = \eta_{\text{rec}} \cdot a_i^l a_m^l \cdot (1 - G)$$

**Key insight:** The feedforward update is Hebbian but gated by prediction error. The feedback update is anti-Hebbian (weakens wrong predictions). The recurrent update is purely Hebbian but gated by low surprise — it consolidates stable patterns.

### 2.5 Continual Learning via Complementary Systems

**Fast-learning system (episodic):**
- Uses high learning rate $\eta_{\text{fast}}$
- Sparse representations
- Encodes specific experiences
- Subject to catastrophic forgetting

**Slow-learning system (semantic):**
- Uses low learning rate $\eta_{\text{slow}} \ll \eta_{\text{fast}}$
- More distributed representations
- Encodes general knowledge
- Resistant to forgetting

**Consolidation (replay):**
Periodically, the fast system replays recent experiences to the slow system:
$$\Delta w_{ij}^{\text{slow}} += \eta_{\text{slow}} \cdot e_{ij}^{\text{fast}}$$

This is NOT experience replay in the RL sense — it's the fast system reactivating its traces, which then modify the slow system.

### 2.6 Temporal Prediction

To handle sequences, the network predicts its own future state:

$$\hat{a}_i^l(t+1) = \sigma\left(\sum_m w_{im}^{l,\text{rec}} a_m^l(t) + \sum_j w_{ij}^{l,\text{ff}} a_j^{l-1}(t+1)\right)$$

Temporal prediction error:
$$\epsilon_{\text{time},i}^l = a_i^l(t+1) - \hat{a}_i^l(t+1)$$

This error also contributes to the global signal $G$, driving temporal learning.

---

## 3. Architecture: The Predictive Equilibrium Network (PEN)

### 3.1 Structure

```
Level 3 (Abstract Concepts, slow timescale)
    ↕ ff/fb/rec
Level 2 (Object Features, medium timescale)
    ↕ ff/fb/rec
Level 1 (Edge/Texture Detectors, fast timescale)
    ↕ ff/rec
Level 0 (Input: pixels, tokens, sensor data)
```

Plus:
- **Episodic Memory:** Fast-learning copy of Level 2-3, stores specific experiences
- **Semantic Memory:** Slow-learning copy of Level 2-3, stores consolidated knowledge
- **Consolidation Controller:** Replays episodic → semantic during low-activity periods
- **Global Neuromodulator:** Computes $G$ from all prediction errors, broadcasts to all layers

### 3.2 Information Flow

**Perception (bottom-up):**
Input → Level 1 → Level 2 → Level 3 → (prediction of Level 2) → ...

**Prediction (top-down):**
Level 3 → Level 2 → Level 1 → (prediction of input) → ...

**Learning:**
All levels compute prediction errors simultaneously. Global signal $G$ gates learning. Local eligibility traces assign credit.

### 3.3 Key Innovations Over Existing Work

| Existing Approach | Our Difference |
|-------------------|----------------|
| Transformers (attention) | No attention mechanism; sparse KWTA replaces softmax attention |
| Predictive Coding (Rao & Ballard) | We add recurrent attractor dynamics + complementary memory systems |
| Equilibrium Propagation | We use continuous dynamics, not two-phase perturbation |
| Boltzmann Machines | We use deterministic KWTA, not stochastic sampling |
| Hebbian Learning (BCM, Oja) | We add global gating + eligibility traces for temporal credit assignment |
| Hopfield Networks | We use hierarchical feedforward/feedback, not single-layer recurrent |
| Neural Turing Machines | Memory is emergent from attractor dynamics, not explicit external memory |

---

## 4. Implementation Priorities

### Priority 1: Single-Level Proof of Concept
- Implement one level with KWTA activation
- Feedforward + feedback connections
- Verify it can learn a simple pattern (e.g., MNIST digit)
- Confirm prediction error decreases over time

### Priority 2: Two-Level Hierarchy
- Add second level
- Implement top-down prediction
- Verify hierarchical abstraction emerges

### Priority 3: Recurrent Dynamics + Temporal Learning
- Add recurrent connections within each level
- Test on sequential data (e.g., moving MNIST)
- Verify temporal prediction works

### Priority 4: Complementary Memory Systems
- Implement fast + slow systems
- Add consolidation (replay) mechanism
- Test on split-MNIST continual learning benchmark

### Priority 5: Scaling
- Extend to CIFAR-10
- Add 2D grid world navigation
- Profile and optimize

---

## 5. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Equilibrium doesn't converge | Medium | Critical | Add leak term to dynamics; use damping |
| KWTA too destructive to learning | Medium | High | Soft KWTA (winners boosted, others suppressed but not zeroed) |
| Global signal too noisy | High | Medium | Smooth $G$ over time window |
| Eligibility traces decay too fast/slow | Medium | High | Make $\gamma$ a learnable parameter |
| Temporal credit assignment fails | Medium | High | Use multi-timescale eligibility traces |
| Scales poorly with NumPy | High | Medium | Use sparse matrices; JIT compile hot loops |

---

*End of First Principles Deduction — planner*
