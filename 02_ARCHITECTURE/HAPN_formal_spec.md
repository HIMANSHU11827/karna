# FROM FIRST PRINCIPLES: A Novel Neural Network Architecture for AGI
## For the AGI Research Lab
## Researcher: Architecture Deduction
## Date: 2026-06-28

---

## 0. METHODOLOGY

We deduce from **five first principles** that any AGI neural network must satisfy. Each principle yields a mathematical constraint. Combining them gives a unique architecture.

This is not a modification of Transformers, SSMs, Hopfield, BCM, or any prior work. It is a **deductive synthesis** from axioms.

---

## 1. THE FIVE FIRST PRINCIPLES

### Principle 1: Universality (Turing-Completeness)

An AGI must be able to compute any computable function.

**Axiom 1.1:** The architecture must have unbounded memory or a mechanism to grow its memory.

**Axiom 1.2:** The architecture must support sequential state transitions (finite control).

**Implication:** A pure feedforward network cannot be an AGI. Recurrence is necessary. But recurrence alone is not sufficient — we need a memory that can grow without bound.

**Mathematical constraint:** Let M(t) be the memory content at time t. We need:
- M(t+1) = f(M(t), I(t), S(t)) where I(t) is input, S(t) is internal state
- |M(t)| can grow with t

### Principle 2: Local Learning (Biological Plausibility)

An AGI must learn from experience using only locally available information.

**Axiom 2.1:** Weight updates depend only on pre-synaptic activity, post-synaptic activity, and a global modulatory signal.

**Axiom 2.2:** No backpropagation of error through arbitrary depth.

**Mathematical constraint:**
```
ΔW_ij = g(a_i, a_j, m)
```
where a_i is post-synaptic activity, a_j is pre-synaptic activity, m is a global modulatory signal. No dependence on activities of neurons k ≠ i,j.

### Principle 3: Interference Avoidance (Continual Learning)

An AGI must learn new information without catastrophically forgetting old information.

**Axiom 3.1:** Different memories should occupy different representational subspaces.

**Axiom 3.2:** The overlap between unrelated memories should be minimal.

**Mathematical constraint:**
```
⟨S^μ, S^ν⟩ ≈ 0 for μ ≠ ν
```
where S^μ is the SDR for memory μ, and ⟨·,·⟩ is inner product.

This requires **sparsity**: if each memory activates fraction s of N neurons, then for random SDRs:
```
E[⟨S^μ, S^ν⟩] = s²
```
For s = 0.01 and N = 10,000: expected overlap = 0.0001. Near-orthogonal.

### Principle 4: World Modeling (Prediction)

An AGI must be able to predict the consequences of actions.

**Axiom 4.1:** The network maintains an internal model P(s' | s, a) of state transitions.

**Axiom 4.2:** Prediction error drives learning.

**Mathematical constraint:**
```
e(t) = s(t+1) - ŝ(t+1)
```
where ŝ(t+1) = P(s(t), a(t)). Learning minimizes E[||e(t)||²].

### Principle 5: Compositionality

An AGI must be able to combine known concepts to represent novel concepts.

**Axiom 5.1:** There exists a binding operation ⊗ such that:
```
bind(X, Y) = Z
```
where Z represents the composition of X and Y.

**Axiom 5.2:** There exists an unbinding operation ⊘ such that:
```
unbind(Z, X) = Y
```

**Mathematical constraint:** ⊗ must be:
- Distributive over sets: bind(A, B ∪ C) = bind(A, B) ∪ bind(A, C)
- Invertible: unbind(bind(X, Y), X) = Y

---

## 2. MATHEMATICAL FORMULATION

### 2.1 Sparse Distributed Representations (SDR)

**Definition:** An SDR S of dimension N and sparsity s is a binary vector S ∈ {0,1}^N where exactly sN entries are 1.

**Capacity:** The number of nearly-orthogonal SDRs is:
```
C(N, s) ≈ (1/s)^(N·s)
```
For N=10,000 and s=0.01: C ≈ 100^100 ≈ 10^200. Effectively unbounded.

**Similarity measure:** Cosine similarity:
```
sim(S, T) = ⟨S, T⟩ / (||S|| · ||T||) = |S ∩ T| / (sN)
```

**Sparse union:** The union of k SDRs has sparsity ≈ k·s (for small k/s).

### 2.2 Binding Operation: Circular Convolution

**Definition:** For X, Y ∈ {0,1}^N, the circular convolution is:
```
Z = X ⊗ Y, where Z_i = Σ_j X_j · Y_{(i-j) mod N}
```

**Properties:**
1. **Invertible:** Y = Z ⊘ X (deconvolution)
2. **Distributive:** X ⊗ (Y ∪ Z) = (X ⊗ Y) ∪ (X ⊗ Z)
3. **Preserves sparsity:** |Z| ≈ |X| · |Y| / N (if X, Y sparse, Z is approximately sparse)
4. **Near-orthogonal:** ⟨X ⊗ Y, X' ⊗ Y'⟩ ≈ 0 unless X≈X' or Y≈Y'

**Implementation:** For binary SDRs, XOR is a simpler binding:
```
Z = X ⊕ Y
```
XOR is its own inverse: Y = Z ⊕ X.

**We adopt XOR binding** for binary SDRs. For continuous SDRs, circular convolution.

### 2.3 Attractor Dynamics

**Energy function:**
```
E(S) = -½ Σ_{i,j} W_ij · S_i · S_j + Σ_i θ_i · S_i
```

**Dynamics:** Discrete-time async update:
```
S_i(t+1) = H(Σ_j W_ij · S_j(t) - θ_i(t))
```
where H is the Heaviside step function (or sigmoid for continuous).

**Convergence:** The network converges to a local minimum of E(S). Each minimum is a stored memory.

**Memory storage (Hebbian):**
```
W_ij = Σ_μ (2S_i^μ - 1)(2S_j^μ - 1) / (sN)
```
This is the Hopfield storage rule, normalized by sparsity.

### 2.4 Predictive Learning

**World model:** P(s' | s, a)

We implement this as a separate network:
```
ŝ' = W_pred · [s; a]
```
where [s; a] is the concatenation of state and action SDRs.

**Prediction error:**
```
e = s' - ŝ'
```

**Learning rule (predictive coding):**
```
ΔW_pred = η · e · [s; a]^T
```

This is a local rule: the update to W_pred depends only on the local activities s, a, and the global error e.

### 2.5 Adaptive Threshold (BCM-like Stabilization)

To prevent runaway weight growth and maintain stable attractors:

```
θ_i(t+1) = θ_i(t) + η_θ · (a_i(t)² - θ_i(t))
```
This is a running average of squared output. The update rule becomes:
```
ΔW_ij = η · a_i · a_j · (a_i - θ_i) - η_decay · W_ij
```

For a_i > θ_i: potentiation (LTP)
For a_i < θ_i: depression (LTD)

---

## 3. THE ARCHITECTURE: Hierarchical Attractor Predictive Network (HAPN)

### 3.1 Overview

```
┌─────────────────────────────────────────────────┐
│                 HIERARCHY LEVEL 3                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │  Planning    │  │  Abstract   │  │  Meta   │ │
│  │  Module      │  │  Concepts   │  │  Learner│ │
│  └──────┬───────┘  └──────┬──────┘  └────┬────┘ │
│         │ bind/unbind     │              │      │
│  ┌──────┴─────────────────┴──────────────┴────┐ │
│  │            HIERARCHY LEVEL 2               │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
│  │  │ Episodic │  │ Semantic │  │ Procedural│ │ │
│  │  │ Memory   │  │ Memory   │  │ Memory   │ │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘ │ │
│  │       │             │             │        │ │
│  │  ┌────┴─────────────┴─────────────┴─────┐  │ │
│  │  │         HIERARCHY LEVEL 1            │  │ │
│  │  │  ┌────────┐  ┌────────┐  ┌────────┐ │  │ │
│  │  │  │Vision  │  │Language│  │Action  │ │  │ │
│  │  │  │Encoder │  │Encoder │  │Decoder │ │  │ │
│  │  │  └────────┘  └────────┘  └────────┘ │  │ │
│  │  └─────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### 3.2 Module Descriptions

#### Level 1: Perception & Action

**Vision Encoder:**
- Input: x ∈ 784 (MNIST) or x ∈ 3072 (CIFAR-10)
- Output: S_v ∈ {0,1}^N with sparsity s=0.05
- Process: k-WTA on learned random projection

**Language Encoder:**
- Input: text → bag-of-words or simple token embedding
- Output: S_l ∈ {0,1}^N with sparsity s=0.03

**Action Decoder:**
- Input: S_a ∈ {0,1}^N
- Output: discrete action a ∈ {0,...,9}
- Process: linear readout + argmax

#### Level 2: Memory Systems

**Episodic Memory:**
- Stores specific experiences as attractor states
- Retrieval: partial cue → convergence to nearest attractor
- Capacity: C ≈ N/(sN·ln(N)) ≈ 1/(s·ln(N)) for N=10,000, s=0.01: C ≈ 1000

**Semantic Memory:**
- Stores facts as binding: subject ⊗ predicate → object
- Retrieval: (subject ⊗ predicate) ⊘ subject = object
- Compositional queries: (subject ⊗ predicate1) ⊗ predicate2 → object2

**Procedural Memory:**
- Stores action sequences: state ⊗ action → next_state
- Serves as world model

#### Level 3: Planning & Meta-Learning

**Planning Module:**
- Uses world model for internal simulation
- Process: start_state → predict_state_1 → ... → predict_state_k
- Evaluates predicted states against goals
- Generates action sequences

**Meta-Learner:**
- Adjusts learning rates based on prediction error
- High error → increase plasticity
- Low error → decrease plasticity
- Formula: η(t) = η_0 · tanh(||e(t)|| / e_target)

---

## 4. LEARNING RULES SUMMARY

### 4.1 Memory Storage (Unsupervised)

Given input SDR S:
```
For each pair (i,j) where S_i = 1 and S_j = 1:
    W_ij ← W_ij + η_storage

For each i where S_i = 1:
    θ_i ← θ_i + η_BCM · (S_i² - θ_i)

Normalize: W_ij ← W_ij / ||W||_F  (Frobenius norm)
```

### 4.2 Associative Learning (Supervised)

Given (input, target) pair (S_in, S_out):
```
# Phase 1: Encode input
S = attractor_converge(S_in)

# Phase 2: Predict
Ŝ_out = W_pred · [S; A]  (A is action, if any)

# Phase 3: Compute error
e = S_out - Ŝ_out

# Phase 4: Update (local, Hebbian-like)
ΔW_pred = η · e · [S; A]^T
```

### 4.3 Planning (Internal Simulation)

```
S = current_state
for k in range(planning_horizon):
    # Generate candidate action
    A = sample_action(S)
    
    # Predict next state
    Ŝ_next = W_pred · [S; A]
    
    # Evaluate against goal
    score = sim(Ŝ_next, goal_state)
    
    # Store best action
    if score > best_score:
        best_action = A
        best_score = score
    
    S = Ŝ_next

return best_action
```

---

## 5. NOVEL CONTRIBUTIONS

### 5.1 What's New

1. **Binding-based episodic memory:** Unlike standard Hopfield networks, we use XOR binding to compose and decompose episodes. An episode is a binding of its components: episode = time ⊗ location ⊗ action ⊗ outcome.

2. **Hierarchical attractor dynamics:** Each level operates at different sparsity levels. Lower levels: dense, high-resolution. Upper levels: sparse, abstract. This mimics cortical hierarchy.

3. **Dual-threshold neurons:** Each neuron has:
   - θ_recurrent: threshold for attractor recall
   - θ_predictive: threshold for predictive activation
   This separates memory retrieval from prediction.

4. **Meta-learning via prediction error:** Learning rates adapt globally based on prediction uncertainty. This is a simple form of attention — focusing plasticity where it's needed.

5. **Compositional world model:** The world model P(s'|s,a) is not a monolithic function. It's a set of composable transition rules, each stored as a binding: (state ⊗ action) → next_state. New transitions are learned by binding, not by gradient descent on a global function.

### 5.2 Comparison to Prior Work

| Feature | Transformer | SSM | Hopfield | KH2019 | Ours |
|---------|-------------|-----|----------|--------|------|
| Sparse representation | ✗ | ✗ | ✗ | ✓ | ✓ |
| Binding/composition | ✗ | ✗ | ✗ | ✗ | ✓ |
| Attractor dynamics | ✗ | ✓ | ✓ | ✗ | ✓ |
| Local learning | ✗ | ✗ | ✓ | ✓ | ✓ |
| Continual learning | ✗ | ✗ | ✗ | ✗ | ✓ |
| World model | ✗ | ✗ | ✗ | ✗ | ✓ |
| Planning | ✗ | ✗ | ✗ | ✗ | ✓ |
| Meta-learning | ✗ | ✗ | ✗ | ✗ | ✓ |
| Interference avoidance | ✗ | ✗ | ✗ | ✓ | ✓ |
| Turing-complete | ✓ | ✓ | ✗ | ✗ | ✓ |

---

## 6. MNIST EXPERIMENT DESIGN

### 6.1 Setup

- Input: 784 pixels → Level 1 Vision Encoder
- Level 1: 784 → 2000 (s=0.05, 100 active units)
- Level 2 Episodic: 2000 units, attractor dynamics
- Level 2 Procedural: 2000 × 10 (actions) → 2000
- Level 3 Planning: 2000 → 2000 (10-step lookahead)
- Output: argmax over 10 action units

### 6.2 Training Procedure

**Phase 1: Unsupervised Feature Learning**
- Present MNIST images
- Store as attractors in episodic memory
- Learn S_v → action bindings in procedural memory

**Phase 2: Supervised Fine-tuning**
- Present (image, label) pairs
- Label is encoded as a special "goal" SDR
- World model learns: (state, goal) → action
- Meta-learner adjusts learning rate

### 6.3 Predicted Performance

**Hypothesis:** 92-96% MNIST accuracy.

**Rationale:**
- Vision encoder extracts sparse features (similar to KH2019)
- Episodic memory stores ~1000 distinct attractors
- Procedural memory learns (state, action) → next_state
- Planning module uses world model for 10-step lookahead
- Meta-learner stabilizes training

**Risk factors:**
- Binding noise may accumulate over hierarchy levels
- Planning horizon limited by prediction error
- No backprop means slow credit assignment

---

## 7. IMPLEMENTATION ROADMAP

### Week 1: Core SDR Operations
- [ ] SDR data structure (sparse binary vectors)
- [ ] XOR binding/unbinding
- [ ] Circular convolution binding
- [ ] k-WTA activation function
- [ ] Attractor convergence

### Week 2: Single-Level Network
- [ ] Level 1: Vision encoder (random projection + k-WTA)
- [ ] Level 2: Episodic memory (attractor network)
- [ ] Hebbian storage rule
- [ ] Recall by convergence

### Week 3: Two-Level Network with Binding
- [ ] Semantic memory (binding-based)
- [ ] Episode = time ⊗ location ⊗ action
- [ ] Retrieval by unbinding
- [ ] MNIST classification test

### Week 4: World Model + Planning
- [ ] Procedural memory (predictive)
- [ ] Internal simulation
- [ ] Planning horizon = 5
- [ ] Full MNIST + planning benchmark

### Week 5: Meta-Learning + Evaluation
- [ ] Adaptive learning rates
- [ ] Split-MNIST continual learning
- [ ] CIFAR-10 with CNN vision encoder
- [ ] 2D grid world navigation

---

## 8. KEY MATHEMATICAL FORMULAS (Summary)

### SDR Properties
```
Capacity:     C ≈ (1/s)^(N·s)
Overlap:      ⟨S^μ, S^ν⟩ ≈ s² for μ ≠ ν
```

### Binding
```
XOR:          Z_i = X_i ⊕ Y_i
Circular Conv: Z_i = Σ_j X_j · Y_{(i-j) mod N}
```

### Attractor Dynamics
```
E(S) = -½ Σ W_ij S_i S_j + Σ θ_i S_i
S_i(t+1) = H(Σ W_ij S_j(t) - θ_i(t))
```

### Learning Rules
```
Storage:      ΔW_ij = η · S_i · S_j · (1 - S_i)
Forgetting:   ΔW_ij = -η · (S_i · S_j - s²)
Prediction:   ΔW_pred = η · e · [s; a]^T
Threshold:    θ_i = E[S_i²]
Meta:         η(t) = η_0 · tanh(||e(t)|| / e_target)
```

---

## 9. CONCLUSION

This architecture is **deduced from first principles**, not adapted from prior work. It satisfies:

1. **Universality** — recurrent attractors with growing memory
2. **Local learning** — all updates are Hebbian or BCM-like
3. **Interference avoidance** — sparse distributed representations
4. **World modeling** — predictive coding in procedural memory
5. **Compositionality** — XOR binding for hierarchical structure

The architecture is novel in its **integration** of these principles into a unified system. No prior work combines all five.

---

*This is a living document. As we implement and test, the formulas will be refined.*
