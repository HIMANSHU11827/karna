# Phase 2 Architecture — From Scratch, Hebbian-Only, Neuromorphic-Ready

## Constraints (Locked)

| Constraint | Implication |
|---|---|
| Python + NumPy only | No PyTorch/TensorFlow. Manual matrix ops, no autograd. |
| Local CPU | Must be vectorized NumPy, no GPU kernels. |
| No pre-trained models | Random init only. |
| Hebbian learning only | Local, unsupervised. No backprop, no global error signal. |
| Neuromorphic-ready | Event-driven, sparse, spiking-compatible. |
| Scope: MNIST → CIFAR-10 → Split-MNIST → 2D grid world | Progressive complexity. |

---

## Architecture: Hierarchical Predictive Coding with Local Learning

### Core Principle

**Predictive Coding (PC)** is a brain-inspired framework where:
- Higher layers **predict** lower-layer activity
- Only **prediction errors** are propagated upward
- Each layer minimizes its own local prediction error
- Learning is **local** (no global error signal needed)

This naturally supports Hebbian learning because weight updates depend only on pre- and post-synaptic activity within a layer.

### Network Structure

```
┌─────────────────────────────────────────┐
│           Output Layer (y)              │
│    (Classification / Action)            │
├─────────────────────────────────────────┤
│           Hidden Layer 3 (h3)           │
│    (Predicts h2, receives error)        │
├─────────────────────────────────────────┤
│           Hidden Layer 2 (h2)           │
│    (Predicts h1, receives error)        │
├─────────────────────────────────────────┤
│           Hidden Layer 1 (h1)           │
│    (Predicts input, receives error)     │
├─────────────────────────────────────────┤
│           Input Layer (x)               │
│    (Sensory / Pixel data)               │
└─────────────────────────────────────────┘
```

Each level tries to predict the level below it. The prediction error drives both inference (updating representations) and learning (updating weights).

### Mathematical Formulation

**Inference (iterative, per layer):**
```
e_{l} = h_{l} - f(W_{l+1} · h_{l+1})     # prediction error
h_{l} ← h_{l} + γ · (W_{l+1}^T · e_{l+1} - e_{l})   # update representation
```
where:
- `e_l` = prediction error at layer l
- `h_l` = neural activity at layer l
- `W_{l+1}` = feedforward weights from l+1 to l
- `f` = activation function (sigmoid/relu)
- `γ` = learning rate for inference

**Learning (Hebbian, local):**
```
ΔW_{l+1} = η · h_{l+1} · e_{l}^T          # Hebbian update
W_{l+1} ← W_{l+1} + ΔW_{l+1}
```
This is pure Hebbian: weight change depends on pre-synaptic activity (h_{l+1}) and post-synaptic correlation (error signal e_l, which encodes the "surprise").

### Why This Works Without Backprop

1. **Local errors**: Each layer has its own error signal (prediction error)
2. **No error backpropagation**: Errors don't flow backward through weights
3. **Weight updates are local**: Only need pre- and post-synaptic activity
4. **Convergence**: Iterative inference settles on a stable representation

---

## Neuromorphic Design

### Event-Driven Computation

Instead of synchronous forward passes, use **event-driven updates**:

```python
# Pseudocode for event-driven inference
def infer(input_data, n_iterations=10):
    h[0] = input_data
    errors = [None] * n_layers
    
    for iteration in range(n_iterations):
        # Compute prediction errors (can be parallel)
        for l in range(1, n_layers):
            prediction = f(W[l] @ h[l])
            errors[l-1] = h[l-1] - prediction
        
        # Update representations (event-driven: only if error > threshold)
        for l in range(n_layers - 1):
            if np.max(np.abs(errors[l])) > THRESHOLD:
                h[l] += gamma * (W[l+1].T @ errors[l+1] - errors[l])
    
    return h[-1]  # output layer activity
```

### Sparse Activations

Use **k-Winners-Take-All (kWTA)** for sparse, biologically plausible activity:

```python
def kwta(x, k=0.1):
    """Keep top k% of activations, zero out rest."""
    threshold = np.percentile(x, (1 - k) * 100)
    return x * (x >= threshold)
```

### Spiking Neuron Model (Leaky Integrate-and-Fire)

For neuromorphic hardware compatibility:

```python
class LIFNeuron:
    def __init__(self, tau=20.0, v_thresh=1.0, v_reset=0.0):
        self.tau = tau
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v_reset
    
    def step(self, I, dt=1.0):
        """One time step of LIF dynamics."""
        self.v += dt * (-self.v + I) / self.tau
        spike = self.v >= self.v_thresh
        self.v = np.where(spike, self.v_reset, self.v)
        return spike
```

---

## Learning Rules (Hebbian Variants)

### 1. Standard Hebbian
```
Δw_ij = η · x_i · y_j
```
Problem: Unbounded weight growth.

### 2. Oja's Rule (Normalized Hebbian)
```
Δw_ij = η · y_j · (x_i - y_j · w_ij)
```
Keeps weights bounded, performs PCA-like learning.

### 3. BCM Theory (Bienenstock-Cooper-Munro)
```
Δw_ij = η · y_j · (y_j - θ) · x_i
```
where θ is a sliding threshold based on average activity. Learns selectivity.

### 4. Contrastive Hebbian Learning (CHL)
Two phases:
- **Clamped phase**: Output is clamped to target, network settles
- **Free phase**: Output is free, network settles
```
Δw = η · (x_clamped · y_clamped^T - x_free · y_free^T)
```
Approximates backprop without requiring differentiable neurons.

### Recommendation

Start with **Oja's Rule** for unsupervised feature learning, then **Contrastive Hebbian** for supervised tasks. This gives us:
- Local learning (no backprop)
- Stable convergence
- Biological plausibility
- Good empirical results on MNIST

---

## Implementation Plan

### Stage 1: MNIST (Baseline)

**Architecture:**
- Input: 784 (28×28)
- Hidden 1: 256 units, kWTA (10% sparse)
- Hidden 2: 128 units, kWTA (10% sparse)
- Output: 10 units (one per digit)

**Learning:**
1. Unsupervised pre-training with Oja's Rule (layer-wise)
2. Supervised fine-tuning with Contrastive Hebbian

**Target:** >95% accuracy on MNIST

### Stage 2: CIFAR-10

**Architecture:**
- Input: 3072 (32×32×3)
- Hidden 1: 512 units
- Hidden 2: 256 units
- Hidden 3: 128 units
- Output: 10 units

**Target:** >70% accuracy (competitive with simple backprop nets)

### Stage 3: Split-MNIST Continual Learning

**Task:** Learn MNIST digits 0-4, then 5-9, without forgetting.

**Challenge:** Catastrophic forgetting (Hebbian learning is especially prone).

**Solution:** 
- Elastic Weight Consolidation (EWC) — but adapted for local learning
- Or: Replay-based approach with episodic memory

**Target:** >80% on both tasks after sequential training.

### Stage 4: 2D Grid World

**Task:** Agent learns to navigate a 2D grid with obstacles.

**Architecture:**
- State encoding: Grid → sparse representation
- Policy network: Predictive coding with action output
- Learning: Reward-modulated Hebbian (R-STDP)

**Target:** Successful navigation in <100 episodes.

---

## Code Structure

```
19_PROTOTYPES/
├── phase2/
│   ├── __init__.py
│   ├── layers.py          # Layer classes (Dense, kWTA, LIF)
│   ├── network.py         # Network assembly and inference
│   ├── learning/
│   │   ├── __init__.py
│   │   ├── hebbian.py     # Oja, BCM, CHL rules
│   │   ├── inference.py   # Predictive coding inference
│   │   └── reward.py      # Reward-modulated learning
│   ├── neurons/
│   │   ├── __init__.py
│   │   ├── rate.py        # Rate-based neurons (sigmoid, ReLU)
│   │   └── lif.py         # Leaky Integrate-and-Fire
│   ├── experiments/
│   │   ├── mnist.py       # MNIST experiment
│   │   ├── cifar10.py     # CIFAR-10 experiment
│   │   ├── split_mnist.py # Continual learning
│   │   └── gridworld.py   # 2D navigation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data.py        # Data loading (no torch!)
│   │   ├── metrics.py     # Accuracy, loss computation
│   │   └── visualization.py
│   └── tests/
│       ├── test_layers.py
│       ├── test_learning.py
│       └── test_network.py
```

---

## Key Implementation Details

### No Autograd — Manual Gradients

Since we can't use autograd, we manually compute gradients for any needed optimization:

```python
# Manual gradient of MSE loss w.r.t. weights
# Loss = 0.5 * (y_pred - y_true)^2
# dLoss/dW = (y_pred - y_true) * dy_pred/dW

def mse_gradient(y_pred, y_true, x):
    """Compute gradient of MSE w.r.t. weights."""
    error = y_pred - y_true  # (output_dim,)
    # For linear layer y = Wx + b: dW = error ⊗ x
    dW = np.outer(error, x)
    db = error
    return dW, db
```

But since we're using Hebbian learning, we mostly don't need gradients!

### NumPy Efficiency

```python
# Good: Vectorized operations
prediction = sigmoid(W @ h)  # Matrix multiply

# Bad: Loops
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        prediction[i] += W[i,j] * h[j]
```

### Data Loading (No PyTorch)

```python
import numpy as np
from tensorflow.keras.datasets import mnist  # Or download manually

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
```

If even Keras is not allowed, download from Yann LeCun's website and parse manually.

---

## Success Criteria

| Stage | Task | Target | Metric |
|---|---|---|---|
| 1 | MNIST | >95% | Accuracy |
| 2 | CIFAR-10 | >70% | Accuracy |
| 3 | Split-MNIST | >80% both | Accuracy (no forgetting) |
| 4 | Grid World | <100 episodes | Steps to goal |

---

## Publication Strategy

**Title**: "Hierarchical Predictive Coding learns MNIST without backpropagation"

**Key Claims:**
1. Competitive accuracy with backprop on MNIST
2. Local learning rules only (no global error signal)
3. Neuromorphic-ready (sparse, event-driven)
4. Extensible to continual learning and reinforcement learning

**Baselines to Compare:**
- Standard backprop (PyTorch, same architecture)
- Feedback Alignment
- Target Propagation
- Other local learning rules

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Hebbian learning too slow | Use Contrastive Hebbian, tune learning rates |
| Catastrophic forgetting | EWC adaptation, replay mechanisms |
| Poor scalability | Start small, add layers incrementally |
| CPU too slow | Vectorized NumPy, consider Numba JIT |
| kWTA too aggressive | Tune sparsity level per layer |

---

*This document will be updated as experimental results come in.*
