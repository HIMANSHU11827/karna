# AGI Research Lab - Research Notes: Alternative Architectures

**Date:** 2026-09-23
**Author:** Anvil (Coder 2) — Memory, Skill & Learning Architectures
**Source Papers:** Compiled from arxiv, Frontiers, PLOS Comp Biol

---

## 1. Why Move Beyond Transformers

### Fundamental Limitations
1. **Attention is O(n²)** — limits context length
2. **Backprop is biologically implausible** — brains don't do matrix calculus
3. **No temporal dynamics** — static processing, no world model
4. **Dense computation** — every neuron fires on every input (unlike biology: ~10% active)
5. **Catastrophic forgetting** — no natural continual learning
6. **No episodic memory** — each forward pass is independent

### What Brains Do Differently
- Hierarchical prediction (not next-token prediction)
- Sparse coding (~10% of neurons active at once)
- Local Hebbian learning (no global gradient)
- Dual memory systems (hippocampus + neocortex)
- Event-driven computation (neuromorphic)

---

## 2. Architecture Alternatives — Deep Survey

### 2.1 Predictive Coding Networks (PCN)

**Source:** Rao & Ballard 1999, Friston 2003/2005, Whittington & Bogacz 2017

**Core Idea:** Cortex is a hierarchical prediction machine. Each higher layer predicts the activity of the layer below. Prediction errors flow upward; predictions flow downward.

**Mathematics:**
- Free energy: F = Σ ε²/(2σ²)  (sum of precision-weighted errors)
- Inference: minimize F w.r.t. neural activations
- Learning: minimize F w.r.t. synaptic weights (Hebbian update)

**Key Properties:**
- During inference: equivalent to feedforward network
- During training: biologically plausible (local updates)
- Can be implemented with dedicated error neurons
- Outperforms backprop on few-shot, continual learning tasks

**Recent Results (2024-2025):**
- PCNs trained with "prospective configuration" converge to BP minima
- Close connection to target propagation
- Works with arbitrary graph structures (PC graphs)

### 2.2 Temporal Predictive Coding (tPC)

**Source:** PLOS Computational Biology, 2024

**Extension:** Adds temporal dimension to PC via recurrent connections. Learns to predict future states.

**Key Innovation:** Low-rank transition matrices (neurally plausible)
```
s_{t+1} = tanh(U · V^T · s_t)
```

**Results:**
- Approximates Kalman filter for linear systems
- Develops motion-sensitive Gabor-like filters (like real V1 neurons)
- Supports online learning (unlike Kalman filter)
- Emerges temporal hierarchy (like dorsal visual pathway)

### 2.3 Sparse Predictive Coding (SPC)

**Source:** Multiple 2024 papers

**Core Idea:** Sparse coding + Predictive Coding = biologically plausible attention alternative.

**Sparse Coding:**
- Find sparse representation c: ||x - Dc||² + λ||c||₁
- Only k neurons active (typically k=5-10% of dict size)
- Dictionary D learned via Hebbian outer products

**Predictive Coding Extension:**
- Each sparse coding layer also predicts the next layer's sparse code
- Cascaded SC layers implement PC objective
- Tight mathematical link to Hebbian plasticity

**Results:**
- Drone navigation (real-world robot control)
- People detection in video
- Better OOD generalization than dense networks

### 2.4 Neural Generative Coding (NGC)

**Source:** Nature Communications, 2024

**Key Innovation:** Decouples forward generative weights from feedback weights.

**Biological Motivation:**
- Cortex has separate feedforward and feedback pathways
- Layer 2/3: error neurons, Layer 4/5: value neurons
- No weight symmetry required

**Architecture:**
- Forward pathway: W matrices
- Feedback pathway: E matrices (learned independently)
- Lateral competition: V matrix for sparse dynamics
- Precision-weighting: learned reliability per neuron

**Results:**
- Outperforms VAE on pattern completion
- Better downstream classification than backprop-trained models
- Supports implicit gradient descent

### 2.5 Hebbian Learning Variants

#### Global-guided Hebbian Learning (GHL)
**Source:** arxiv 2601.21367v1 (2025)

- Local: Oja's rule (stable PCA extraction)
- Global: sign of global gradient (dopamine-like)
- Result: competitive with backprop on ImageNet
- Model-agnostic, works with any architecture

#### Spike-Timing-Dependent Plasticity (STDP)
- Δw depends on spike timing difference
- Biologically precise: pre-before-post = LTP, post-before-pre = LTD
- Approximates Hebbian learning in rate-coded networks
- Key for temporal/sequence learning

#### BCM Rule
- Bienenstock-Cooper-Munro
- Sliding threshold: only sufficiently active neurons learn
- Explains orientation selectivity development
- Prevents "dead neuron" problem

### 2.6 Spiking Neural Networks (SNN) + Meta-Learning

**Source:** HLML-SNN, AAAI 2025

**Architecture:**
- Short-term: Hebbian learning (fast synaptic plasticity)
- Long-term: Meta-learning (slow cortical integration)

**Results:**
- State-of-the-art continual learning (split-MNIST/CIFAR)
- No catastrophic forgetting
- Bio-plausible dual memory system

### 2.7 Deep Hebbian Predictive Coding (DHPC)

**Source:** Frontiers Computational Neuroscience, 2021

**Architecture:**
- Stacked gated Hebbian PC layers
- Scales to millions of synapses
- Unsupervised learning

**Emergent Properties:**
- Orientation selectivity (like V1 simple cells)
- Object selectivity (like IT cortex)
- Sparse representations increasing up hierarchy
- Reproduces visual cortical hierarchy

---

## 3. Our Architecture: PCHH (Predictive Coding + Hebbian Hybrid)

### Design Principles

1. **Local Learning Only**
   - Each layer learns from local information
   - No backpropagation through time or layers
   - Weight updates: outer product of pre- and post-synaptic activity

2. **Hierarchical Prediction**
   - Lower layers: sensory/short-term
   - Higher layers: abstract/long-time-scale
   - Prediction errors carry "surprise" upward

3. **Sparse Activity**
   - Only 10-20% of neurons active at any time
   - Lateral competition (WTA-inspired)
   - Energy-efficient computation

4. **Dual Pathways**
   - Forward: generative model (predictions)
   - Feedback: error propagation (corrections)
   - No weight symmetry (NGC design)

5. **Temporal Dynamics**
   - Recurrent connections propagate predictions through time
   - Low-rank transition matrices
   - Learns world model from temporal structure

6. **Memory Integration**
   - Working memory: 7±2 slots, attention-based
   - Episodic memory: consolidation from working memory
   - Semantic memory: graph-structured concept relationships
   - Procedural memory: skill composition + mastery tracking

### Component Map

```
Input → [PC Layer 1] → [PC Layer 2] → [PC Layer 3] → Output
         ↕ tPC          ↕ tPC          ↕ tPC
         Sparse          Sparse          Sparse
         Coding          Coding          Coding

    ↕ Working Memory (7±2 slots)
    ↕ Episodic Memory (consolidated experiences)
    ↕ Semantic Memory (concept graph)
    ↕ Procedural Memory (skill library)

    All connected to: Learning Architecture (Hebbian updates)
```

---

## 4. Key Formulas (Non-Transformer)

### Predictive Coding
```
ε_l = x_l - f(W_l · x_{l-1})        # prediction error
F = Σ ||ε_l||² / (2σ_l²)              # free energy
ΔW_l = η · ε_l · x_{l-1}^T            # Hebbian weight update
Δx_l = -∂F/∂x_l                        # inference (gradient descent on F)
```

### Temporal PC
```
s_{t+1} = tanh(U · V^T · s_t)       # state prediction
L = ||s_{t+1} - s_{t+1}^{obs}||²    # temporal prediction error
```

### Sparse Coding (ISTA)
```
c_{t+1} = prox_λ(c_t + ηD^T(x - Dc_t))  # iterative soft thresholding
x̂ = Dc                                    # reconstruction
```

### Hebbian Variants
```
Oja:       Δw = η·(pre·post - w·post²)
BCM:       Δw = η·pre·post·(post - θ)
STDP:      Δw = A⁺·exp(-Δt/τ⁺) if Δt>0 else -A⁻·exp(Δt/τ⁻)
GHL:       Δw = η·pre·post·sign(∂L/∂w_global)
```

---

## 5. Implementation Roadmap

### Phase 1: Core Architecture (Current)
- ✅ Memory system prototype (4 types, consolidation)
- ✅ Learning architecture prototype (PCN + tPC + SC)
- ✅ Hebbian plasticity (Oja, BCM, STDP, GHL)
- ✅ Integration demo

### Phase 2: Scale & Train
- [ ] Real datasets (MNIST first, then CIFAR)
- [ ] Compare against backprop baseline
- [ ] Continual learning benchmark
- [ ] Measure catastrophic forgetting resistance

### Phase 3: Integration
- [ ] Connect to perception system (coder-3)
- [ ] Connect to evaluation framework (coder-4)
- [ ] Connect to world model/environment
- [ ] End-to-end agent demo

### Phase 4: Advanced
- [ ] Multimodal (vision + language + action)
- [ ] Neuromorphic simulation (event-driven SNN)
- [ ] Self-improvement loop (meta-learning)

---

## 6. Open Research Questions

1. **Credit Assignment:** Can purely local Hebbian learning solve complex tasks?
   - GHL suggests yes (sign-based global guidance)
   - But is sign-only enough for deep hierarchies?

2. **Capacity:** How deep can PCN hierarchy go?
   - DHPC scaled to millions of synapses
   - But ImageNet-scale? Need to verify.

3. **Speed:** How fast is inference vs backprop?
   - PC requires iterative convergence (slow?)
   - But highly parallelizable (neuromorphic)

4. **Memory:** How to integrate working/episodic/semantic seamlessly?
   - Current prototypes are separate systems
   - Need unified consolidation mechanism

5. **Temporal Abstraction:** Can tPC learn hierarchical time scales?
   - Dynamic Predictive Coding (DPC) suggests yes
   - Higher levels modulate lower-level dynamics

6. **Self-Improvement:** Can the architecture modify its own learning rules?
   - Meta-learning on plasticity rules (GHL-style)
   - Evolutionary search over architectures

---

## 7. References

1. Rao, R.P. & Ballard, D.H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
2. Friston, K. (2005). A free energy principle for the brain. *Nature Reviews Neuroscience*.
3. Whittington, J.C. & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network. *Neural Computation*.
4. Millidge, B. et al. (2022). The predictive coding network as a superset of the feedforward network. *Neural Computation*.
5. GHL Paper (2025). Hebbian Learning with Global Direction. *arxiv 2601.21367v1*.
6. tPC Paper (2024). Predictive coding networks for temporal prediction. *PLOS Computational Biology*.
7. NGC Paper (2024). The neural coding framework for learning generative models. *Nature Communications*.
8. HLML-SNN (2025). Fast Continual Learning in SNNs via Hebbian Meta-Learning. *AAAI*.
9. DHPC (2021). Deep Gated Hebbian Predictive Coding. *Frontiers Computational Neuroscience*.
10. SPC Survey (2024). Continual Learning with Hebbian Plasticity in Sparse and Predictive Coding Networks.

---

**Next:** Implement real-data training demo. Show that PCHH learns MNIST without backpropagation. Compare accuracy, speed, and forgetting against a baseline CNN.

**Maintained by:** Anvil (Coder 2) — Memory, Skill & Learning Architectures
**Project:** AGI Research Lab
**Status:** Active development
