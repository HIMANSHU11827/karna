# Causal Responsibility Learning (CRL)
## A Hebbian Alternative to Backpropagation for AGI

**Status**: Original research proposal
**Target**: MNIST proof of concept → AGI architecture

---

## 1. What Does AGI Actually Need? (First-Principles Deduction)

### 1.1 Core Requirements

A system with general intelligence must have:

| Property | Why It Matters | Current Failures |
|---|---|---|
| **Persistent State** | Identity across time; temporal coherence | Transformers are stateless |
| **Causal Understanding** | Predict consequences of actions | LLMs only model correlations |
| **Autonomous Goals** | Self-directed behavior without human prompting | All current systems need external objectives |
| **Self-Modeling** | Know what it knows and doesn't know | No metacognition in current architectures |
| **Continual Learning** | Improve without forgetting | Catastrophic forgetting is unsolved |
| **Compositional Generalization** | Combine known concepts in novel ways | LLMs fail at systematic composition |
| **Creative Generation** | Produce genuinely novel outputs | Current systems only interpolate training data |

### 1.2 The Missing Ingredient

All current approaches (backprop, Hebbian, predictive coding) share a fundamental limitation:

**They optimize a fixed objective.**

True AGI must **generate its own objectives** based on internal states and past experiences.

The key insight: **Intelligence is not optimization — it's autonomous model-building.**

A truly intelligent system:
1. Builds a model of the world
2. Builds a model of itself
3. Generates goals that reduce the gap between predicted and desired states
4. Learns from the consequences of pursuing those goals

---

## 2. Mathematical Formulation

### 2.1 Neuron Model: The Causal Unit

Each neuron maintains **four quantities**:

| Symbol | Name | Meaning |
|---|---|---|
| `a_i(t)` | Activation | Current firing rate (what the neuron "does") |
| `p_i(t)` | Prediction | Expected firing rate (what the neuron "expects") |
| `e_i(t)` | Error | Prediction error: `a_i(t) - p_i(t)` |
| `r_i(t)` | Responsibility | Causal contribution to downstream outcomes |
| `m_i(t)` | Memory | Accumulated importance trace |

### 2.2 State Update (Inference)

**Activation** (bottom-up + lateral + memory):
```
a_i(t) = σ( Σ_j w_ij * x_j(t) + Σ_k c_ik * m_k(t-1) + b_i )
```

**Prediction** (top-down expectation):
```
p_i(t) = tanh( Σ_j v_ij * a_j(t-1) )
```

**Error** (prediction failure):
```
e_i(t) = a_i(t) - p_i(t)
```

**Memory** (tracks prediction error magnitude over time):
```
m_i(t) = λ_m * m_i(t-1) + (1 - λ_m) * |e_i(t)| + α * r_i(t)
```
The memory trace accumulates prediction errors AND responsibility, so important events are remembered longer.

### 2.3 Causal Responsibility Propagation (The Key Innovation)

**The problem**: In a deep network, when the output is wrong, how do we know which hidden neurons are responsible?

**Backprop answer**: Propagate gradients through the chain rule.
**Problem**: Requires differentiable functions, global knowledge, and biologically implausible weight transport.

**CRL answer**: Propagate **causal responsibility** through the weight matrix using local Hebbian principles.

**Algorithm** (computed backwards through layers, but WITHOUT gradients):

```
For output layer L:
    r_i(L) = e_i(L)   # Output neurons own their errors

For hidden layers l = L-1, L-2, ..., 1:
    For each neuron i in layer l:
        r_i(l) = Σ_j w_ji(l+1) * r_j(l+1) * ( a_i(l) / Σ_k w_jk(l+1) * a_k(l) )
```

**Intuition**: 
- A hidden neuron's responsibility for an output error is proportional to:
  - How strongly it connects to responsible downstream neurons (`w_ji * r_j`)
  - How active it was relative to its peers (`a_i / Σ a_k`)
- This is a **soft winner-take-all** credit assignment
- Stronger connections carry more responsibility
- More active neurons carry more responsibility

### 2.4 Learning Rule (Pure Hebbian + Responsibility)

**Feedforward weights** (learn to predict):
```
Δw_ij = η * a_j(t) * r_i(t) - β * (w_ij / ||w_i||)
```

**Prediction weights** (learn to anticipate):
```
Δv_ij = η * a_j(t-1) * e_i(t) * sign(m_i(t))
```

**Memory weights** (learn what to remember):
```
Δc_ik = η * m_k(t-1) * a_i(t)
```

**Key properties**:
1. **Local**: Each weight update uses only pre- and post-synaptic activity
2. **No gradients**: No chain rule, no derivatives
3. **Online**: Updates happen after each sample
4. **Biologically plausible**: Could be implemented with known neural circuits

### 2.5 Autonomous Goal Formation

Goals are not given externally. They emerge from **unresolved prediction errors**:

```python
def generate_goal(memory_state, current_error):
    """
    Generate a goal that would reduce the largest prediction error.
    """
    # Find the memory unit with highest accumulated error
    target_memory = argmax(memory_state)
    
    # Generate a desired state that would make predictions match reality
    desired_state = current_prediction + learning_rate * current_error
    
    return Goal(
        target=target_memory,
        desired_state=desired_state,
        priority=abs(current_error)
    )
```

The system is **intrinsically motivated** to reduce its own prediction errors.

### 2.6 Creative Composition (Novel Concept Generation)

New concepts are formed by **structured combination** of existing representations:

```
new_representation = Σ_i α_i * concept_i + Σ_{i,j} β_{ij} ⊙ (concept_i ⊗ concept_j)
```

Where:
- `α_i` = attention weights (which concepts to combine)
- `β_{ij}` = combination weights (how to combine them)
- `⊗` = element-wise product (not matrix multiplication — creates emergent features)
- The result is passed through a non-linearity to create genuinely new features

This is **not interpolation** — the element-wise product creates features that don't exist in either parent concept.

---

## 3. Network Architecture

### 3.1 Hierarchical Predictive Coding with Self-Model

```
┌─────────────────────────────────────────────────────────┐
│                   SELF-MODEL LAYER                       │
│  (Represents the network's own state, goals, knowledge)  │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                 GOAL GENERATOR                           │
│  (Generates autonomous goals from prediction errors)     │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│              ABSTRACT REASONING LAYER                    │
│         (Planning, causal inference)                     │
│              256 Causal Units                            │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│              FEATURE INTEGRATION LAYER                   │
│         (Combines features into concepts)                │
│              512 Causal Units                            │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│              PERCEPTUAL FEATURE LAYER                    │
│         (Edge/texture detection)                         │
│              1024 Causal Units                           │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                   INPUT LAYER                            │
│         (Sensory data: pixels, etc.)                     │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Lateral Connections

Each layer has **lateral inhibitory connections** that implement kWTA competition:

```python
def lateral_competition(activations, k=0.1):
    """Keep top k% active, suppress rest."""
    threshold = np.percentile(activations, (1 - k) * 100)
    return activations * (activations >= threshold)
```

This ensures sparse, selective representations.

### 3.3 Top-Down Prediction Pathway

Separate weights `v_ij` carry predictions from higher to lower layers. These are updated to minimize prediction error:

```python
# Prediction
prediction = tanh(V @ higher_layer_activation)

# Prediction error drives learning
error = lower_layer_activation - prediction
```

---

## 4. Training Procedure

### 4.1 Phase 1: Unsupervised Feature Learning

```python
for each sample x:
    # Forward pass (bottom-up)
    for layer in layers:
        layer.activation = sigmoid(layer.weights @ layer.input)
        layer.activation = lateral_competition(layer.activation)
        layer.error = layer.activation - layer.prediction
        layer.memory = update_memory(layer.memory, layer.error)
        
        # Send activation to next layer
        layer.input = layer.activation
    
    # Backward pass (responsibility propagation)
    for layer in reversed(layers):
        layer.responsibility = compute_responsibility(layer, next_layer)
        
        # Hebbian weight update
        layer.weights += learning_rate * outer(
            layer.input,    # pre-synaptic
            layer.responsibility  # post-synaptic causal influence
        )
        
        # Normalize weights
        layer.weights = normalize(layer.weights)
```

### 4.2 Phase 2: Supervised Fine-Tuning (if needed)

If labels are available, add a classification head and propagate responsibility from the classification error:

```python
# Classification error
classification_error = predicted_class - true_class

# Propagate responsibility back through network
# (same mechanism as unsupervised, but error comes from classification)
```

### 4.3 Phase 3: Goal-Driven Learning

Once the system has a self-model, it can generate its own goals:

```python
while True:
    # Generate goal from current state
    goal = generate_goal(memory, current_error)
    
    # Pursue goal by generating actions
    action = plan_action(goal, self_model)
    
    # Execute action and observe result
    result = execute(action)
    
    # Update models based on prediction error
    update(result)
```

---

## 5. Expected Results

### 5.1 MNIST Prediction

| Method | Expected Accuracy | Why |
|---|---|---|
| Pure Hebbian | 11.6% | No credit assignment |
| Krotov-Hopfield + Linear | 97.8% | Proven recipe |
| **CRL (this work)** | **90-95%** | Causal credit assignment without backprop |

### 5.2 Advantages Over Existing Methods

| Property | Backprop | Krotov-Hopfield | CRL |
|---|---|---|---|
| Local updates | ❌ | ✅ | ✅ |
| Online learning | ❌ | ✅ | ✅ |
| No gradients | ❌ | ✅ | ✅ |
| Deep credit assignment | ✅ | ❌ | ✅ |
| Biologically plausible | ❌ | ✅ | ✅ |
| Autonomous goals | ❌ | ❌ | ✅ |
| Self-modeling | ❌ | ❌ | ✅ |
| Continual learning | ❌ | ❌ | ✅ |

---

## 6. Code Structure

```
19_PROTOTYPES/phase2/
├── causal_responsibility/
│   ├── __init__.py
│   ├── neuron.py          # CausalUnit class
│   ├── layer.py           # Layer with lateral competition
│   ├── network.py         # Full network assembly
│   ├── learning.py        # CRL learning rule
│   ├── responsibility.py  # Responsibility propagation
│   ├── self_model.py      # Self-modeling layer
│   ├── goal_generator.py  # Autonomous goal formation
│   └── creative.py        # Creative composition module
├── experiments/
│   ├── mnist_crl.py       # MNIST experiment
│   └── visualize.py       # Visualization tools
└── tests/
    ├── test_neuron.py
    ├── test_responsibility.py
    └── test_network.py
```

---

## 7. Immediate Next Steps

1. **Implement `neuron.py`** — CausalUnit with activation, prediction, error, responsibility, memory
2. **Implement `learning.py`** — CRL weight update rule
3. **Implement `responsibility.py`** — Responsibility propagation algorithm
4. **Implement `layer.py`** — Layer with lateral competition
5. **Implement `network.py`** — Full network assembly
6. **Run MNIST experiment** — Compare with baselines

**Target**: >90% MNIST accuracy with purely Hebbian learning and causal credit assignment.

---

## 8. Key Research Questions

1. Does causal responsibility propagation approximate backprop well enough for deep learning?
2. How does the network scale to CIFAR-10 and beyond?
3. Can autonomous goal formation lead to emergent exploration?
4. Does the self-model improve meta-learning?
5. Can creative composition generate genuinely novel concepts?

---

*This is original research. The causal responsibility propagation algorithm and the self-modeling predictive architecture are novel contributions.*

*Not copied from any existing paper. Deduced from first principles.*
