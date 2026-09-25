# Original Research — Compositional State Networks (CSN)

## Deduction: What Properties Does a Neural Network Need for AGI?

### Deduction 1: Temporal Depth (State)
Current transformers are stateless — each forward pass is independent. AGI requires **persistent state** that evolves over time, allowing the system to:
- Maintain context across interactions
- Learn from every experience (online learning)
- Develop a sense of identity and continuity

**Mathematical requirement**: The network dynamics must be a function of both input and previous state:
```
s(t+1) = f(s(t), x(t))
```
This is recurrence. But standard RNNs suffer from vanishing gradients and limited memory.

### Deduction 2: Compositional Depth (Systematicity)
AGI must be able to systematically combine known concepts to understand new ones. If the system knows "red," "blue," "car," and "bike," it must understand "red bike" without explicit training.

**Mathematical requirement**: The network must support a **binding operation** that combines representations:
```
bind(a, b) → representation of "a + b"
```
Current solutions (concatenation, attention, tensor products) are either not compositional or require exponentially large vectors.

### Deduction 3: Self-Modification (Structural Plasticity)
AGI must be able to modify its own structure based on experience — not just weights, but topology. The brain grows new synapses, prunes unused ones, and adds new neurons in some regions.

**Mathematical requirement**: The network topology T must be a function of experience:
```
T(t+1) = g(T(t), x(t), error(t))
```

### Deduction 4: Causal Understanding (Intervention)
AGI must understand cause and effect, not just correlation. It must be able to answer "what if" questions through intervention, not just observation.

**Mathematical requirement**: The network must support **do-calculus** operations:
```
P(effect | do(cause)) ≠ P(effect | observe(cause))
```

### Deduction 5: Grounded Perception (World Model)
AGI must have a world model that grounds abstract concepts in sensory experience. The model must predict future states and explain observations.

**Mathematical requirement**: A generative model P(observation | state) and transition model P(next_state | state, action).

---

## Original Architecture: Compositional State Networks (CSN)

### Core Idea

Combine **state**, **compositionality**, and **self-modification** into a unified architecture. The key innovation is a **learnable binding tensor** that allows the network to discover how to compose concepts from data, rather than using a fixed binding operation.

### Mathematical Formulation

#### Network State

The network has three types of neurons:
- **State neurons** s(t): persistent state that evolves over time
- **Composition neurons** c(t): represent bound concepts
- **Memory neurons** m(t): store attractor states (memories)

#### Dynamics

```
s(t+1) = σ(W_s · s(t) + W_x · x(t) + W_c · c(t) + b_s)

c(t) = B(s(t), x(t))  # Binding operation (learned tensor)

m(t+1) = α · m(t) + (1-α) · tanh(W_m · s(t))  # Memory update

output = W_o · [s(t); c(t); m(t)]
```

Where:
- W_s: recurrent state weights
- W_x: input weights
- W_c: composition weights
- B: **third-order binding tensor** (the key innovation)
- W_m: memory weights
- W_o: output weights
- α: memory decay rate (learnable)

#### The Binding Tensor B

The binding tensor is a 3D tensor B ∈ ℝ^(d_c × d_s × d_x) that maps a state-input pair to a composition:

```
c_i = Σ_{j,k} B_{ijk} · s_j · x_k
```

This is a **bilinear map** that is learned from data. Unlike fixed binding operations (XOR, circular convolution), this is discovered from experience.

**Key property**: The binding tensor can learn to implement different types of composition for different contexts:
- Feature binding: "red" + "car" → "red car"
- Relational binding: "A" + "loves" + "B" → "A loves B"
- Temporal binding: "event1" + "then" + "event2" → sequence

### Learning Rules (Derived from First Principles)

#### Rule 1: Predictive Hebbian (for W_s, W_x)

Minimize prediction error:
```
error = x(t+1) - predicted_x(t+1)
ΔW_s = η · error ⊗ s(t)
ΔW_x = η · error ⊗ x(t)
```

This is Hebbian: neurons that fire together, wire together, but only when there's a prediction error.

#### Rule 2: Compositional Hebbian (for B)

The binding tensor is trained to minimize reconstruction error:
```
target_composition = actual_composition(s(t), x(t))
predicted_composition = B(s(t), x(t))
composition_error = target - predicted

ΔB = η · composition_error ⊗ s(t) ⊗ x(t)
```

This is a **third-order Hebbian rule**: the weight change depends on the product of three signals (error, state, input).

#### Rule 3: Memory Hebbian (for W_m)

Memories are stored as attractor states:
```
if prediction_error > threshold:  # Surprising event
    ΔW_m = η · m(t) ⊗ s(t)  # Store current state in memory
```

#### Rule 4: Structural Plasticity (for topology)

```
if sustained_high_error:
    # Add new composition neuron
    new_neuron = initialize_at_high_error_region
    expand B with new slice
    
    # Or add new state neuron
    expand W_s with new row/column
```

This is **genuinely original**: the network grows its own structure based on experience.

---

## Comparison to Existing Work

| Feature | Transformer | RNN | Hopfield | Hyperdimensional | **CSN** |
|---|---|---|---|---|---|
| State | ❌ | ✅ | ✅ | ❌ | ✅ |
| Composition | ❌ | ❌ | ❌ | ✅ (fixed) | ✅ (learned) |
| Self-modification | ❌ | ❌ | ❌ | ❌ | ✅ |
| Online learning | ❌ | ✅ | ✅ | ❌ | ✅ |
| Causal reasoning | ❌ | ❌ | ❌ | ❌ | ✅ (planned) |

---

## Prototype Implementation

### Architecture for MNIST

```
Input: 784 (28×28)
State layer: 256 neurons (persistent state)
Composition layer: 128 neurons (binding tensor B)
Memory layer: 64 neurons (attractor states)
Output: 10 neurons (classification)
```

### Training Procedure

1. **Phase 1: Unsupervised Pre-training**
   - Present images one at a time
   - Train B (binding tensor) with compositional Hebbian
   - Train W_s (recurrent weights) with predictive Hebbian
   - Memory layer stores surprising/important states

2. **Phase 2: Supervised Fine-tuning**
   - Freeze B and W_s
   - Train output layer with logistic regression
   - This is NOT backprop — it's a separate classifier

3. **Phase 3: Online Adaptation**
   - Continue updating B and W_s during inference
   - Network adapts to new examples in real-time
   - Structural plasticity adds neurons if needed

### Code Structure

```
19_PROTOTYPES/csn/
├── __init__.py
├── binding.py          # Binding tensor B
├── state_layer.py      # State neurons with recurrence
├── memory_layer.py     # Attractor memory
├── learning_rules.py   # All 4 learning rules
├── structural.py       # Structural plasticity
├── network.py          # Full CSN assembly
├── experiments/
│   ├── mnist_csn.py    # MNIST experiment
│   └── visualize.py    # Visualization tools
└── tests/
    └── test_binding.py
```

---

## Key Research Questions

1. **Can the binding tensor learn meaningful compositions?**
   - Test: Does bind("edge", "curve") look like a digit part?
   - Metric: Composition reconstruction error

2. **Does structural plasticity improve learning?**
   - Test: Compare fixed vs. growing topology
   - Metric: Accuracy over time, number of neurons

3. **Can the memory layer store and recall patterns?**
   - Test: Store MNIST digits, test recall with partial cues
   - Metric: Recall accuracy

4. **Does online learning work?**
   - Test: Train on 0-4, then 5-9, measure forgetting
   - Metric: Accuracy on both tasks

5. **Can we scale to CIFAR-10?**
   - Test: Same architecture, color images
   - Metric: Classification accuracy

---

## Expected Results

| Experiment | Target | Notes |
|---|---|---|
| MNIST (unsupervised features) | >90% | With logistic regression on top |
| MNIST (online learning) | >85% | No separate classifier |
| Memory recall | >80% | Partial cue completion |
| Structural plasticity | Positive | Growing network outperforms fixed |
| CIFAR-10 | >60% | Scaling test |

---

## Why This Is Original

1. **No fixed binding operation**: Unlike hyperdimensional computing, the binding tensor is learned
2. **Third-order Hebbian learning**: Novel learning rule for the binding tensor
3. **Structural plasticity during inference**: The network grows its own neurons
4. **Unified architecture**: State, composition, and memory in one framework
5. **Derived from first principles**: Each component follows from a clear deduction about AGI requirements

---

*This is ground-up original research. No templates. No copying. Create something genuinely new.*
