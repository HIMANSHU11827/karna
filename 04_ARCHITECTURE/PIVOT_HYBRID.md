# Phase 2 Pivot — Hybrid Architecture (Hebbian Features + Linear Classifier)

## Problem Confirmed

Pure Hebbian learning: **11.6%** on MNIST (barely above random baseline of 10%).

**Root cause**: Pure Hebbian learning finds correlated features but doesn't optimize for class discrimination. The features are not discriminative enough for classification.

---

## Solution: Proven Hybrid Approach

Based on Krotov & Hopfield (2019) and subsequent research, the proven recipe is:

```
Unsupervised Hebbian Feature Extraction  →  Linear Classifier (SVM/Logistic)
```

This is NOT backpropagation. The classifier is a separate, simple model on top.

---

## Krotov & Hopfield (2019) — The Reference Implementation

**Paper**: "Unsupervised Learning by Competing Hidden Units" (PNAS 2019)
**Result**: **97.8%** on MNIST with 784-2000-10 architecture
**Code**: https://github.com/DimaKrotov/Biological_Learning

### Key Innovations Over Naive Hebbian

1. **L^p Norm Weight Normalization** (not L^2 like Oja's rule)
   - Generalizes Oja's normalization
   - Better feature diversity

2. **Anti-Hebbian Term for Losers**
   - Winner-takes-all: top neuron gets strengthened
   - Bottom-k neurons get weakened (anti-Hebbian)
   - This creates competition and diversity

3. **Rectified Power (RePU) Activation**
   - `f(x) = max(0, x)^p` where p is optimized (p=3 gives +6% over p=2)
   - Creates sparse, selective features

4. **Hard K-WTA Competition**
   - Only top-k neurons active per sample
   - Forces specialization

### Their Exact Recipe

```python
# Krotov & Hopfield Learning Rule
# For each sample x:
#   1. Compute activations: h = W @ x
#   2. Find winner: j = argmax(h)
#   3. Compute r vector:
#      r[j] = 1 (winner)
#      r[j-k:j-1] = -Δ (anti-Hebbian for losers)
#      r[others] = 0
#   4. Update: ΔW = η * r @ x^T
#   5. Normalize weights: W = normalize(W, p)  # L^p norm
```

---

## Recommended Architecture

### Stage 1: Unsupervised Feature Learning

**Architecture**: 784 → 2000 → 10

**Layer 1 (784 → 2000)**: Krotov-Hopfield learning
- L^p normalized Hebbian with anti-Hebbian
- RePU activation (p=3)
- Hard kWTA (top 10% active)

**Layer 2 (2000 → 10)**: Linear classifier
- Logistic regression or SVM
- Trained on frozen features from Layer 1
- This is NOT backprop — it's a separate classifier

### Why This Works

1. **Hebbian layer** learns feature detectors (edge detectors, stroke detectors)
2. **Competition** forces diverse features (not all neurons detect the same thing)
3. **Anti-Hebbian** prevents "dead" neurons from dominating
4. **Linear classifier** finds the decision boundary in feature space
5. **No backprop** needed — the classifier is convex optimization

---

## Implementation Plan

### Step 1: Implement Krotov-Hopfield Learning Rule

```python
import numpy as np

def krotov_hopfield_update(W, x, learning_rate=0.01, p=3, n_anti=10):
    """
    Krotov & Hopfield learning rule.
    
    Args:
        W: weight matrix (n_hidden, n_input)
        x: input vector (n_input,)
        learning_rate: η
        p: RePU power
        n_anti: number of anti-Hebbian neurons
    
    Returns:
        Updated W
    """
    # Compute activations
    h = W @ x
    
    # Find winner
    winner = np.argmax(h)
    
    # Create r vector
    r = np.zeros_like(h)
    r[winner] =1  # Winner
    
    # Anti-Hebbian for losers (bottom n_anti neurons)
    sorted_indices = np.argsort(h)
    for i in range(n_anti):
        r[sorted_indices[i]] = -0.1  # Anti-Hebbian term
    
    # Hebbian update
    dW = learning_rate * np.outer(r, x)
    W = W + dW
    
    # L^p norm normalization
    norms = np.linalg.norm(W, ord=p, axis=1, keepdims=True)
    W = W / (norms + 1e-8)
    
    return W
```

### Step 2: Train Unsupervised Features

```python
def train_features(X_train, n_hidden=2000, n_epochs=10, learning_rate=0.01):
    n_input = X_train.shape[1]  # 784 for MNIST
    
    # Random init
    W = np.random.randn(n_hidden, n_input) * 0.01
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        
        for idx in indices:
            x = X_train[idx]
            W = krotov_hopfield_update(W, x, learning_rate)
        
        # Evaluate
        features = relu(W @ X_train.T)  # RePU activation
        print(f"Epoch {epoch+1}: feature sparsity = {np.mean(features > 0):.3f}")
    
    return W
```

### Step 3: Train Linear Classifier

```python
from sklearn.linear_model import LogisticRegression

def train_classifier(W, X_train, y_train, X_test, y_test):
    # Extract features
    train_features = relu(W @ X_train.T).T
    test_features = relu(W @ X_test.T).T
    
    # Train logistic regression (convex, no backprop)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(train_features, y_train)
    
    # Evaluate
    accuracy = clf.score(test_features, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    return clf, accuracy
```

---

## Expected Results

| Approach | MNIST Accuracy | Notes |
|---|---|---|
| Pure Hebbian (current) | 11.6% | Not discriminative enough |
| Krotov & Hopfield (2019) | **97.8%** | Proven recipe |
| SoftHebb (2022) | 97.8% | Similar approach |
| Equilibrium Propagation | 97.6% | Alternative |
| Backprop (reference) | 98.5% | Upper bound |

**Target**: 95-98% on MNIST with Krotov-Hopfield + linear classifier.

---

## Code Structure

```
19_PROTOTYPES/phase2/
├── layers/
│   ├── krotov_hopfield.py    # Krotov-Hopfield learning rule
│   ├── oja.py                # Oja's rule (baseline)
│   ├── bcm.py                # BCM rule
│   └── activation.py         # RePU, kWTA, etc.
├── experiments/
│   ├── mnist_hybrid.py       # MNIST with hybrid approach
│   ├── mnist_pure_hebbian.py # Pure Hebbian (baseline)
│   └── compare_rules.py      # Compare different learning rules
├── utils/
│   ├── data.py               # MNIST/CIFAR data loading
│   └── metrics.py            # Accuracy, sparsity metrics
└── main.py                   # Entry point
```

---

## Next Steps

1. **coder-1**: Implement Krotov-Hopfield learning rule
2. **coder-2**: Implement RePU activation and kWTA
3. **coder-3**: Implement training loop and feature extraction
4. **coder-4**: Implement logistic regression classifier and evaluation
5. **researcher**: Track results, compare with baselines
6. **definer**: Refine architecture based on results

---

## Key Insight

**The classifier is NOT the network.** The network learns features. The classifier is a separate, simple model. This is biologically plausible (cortex learns features, downstream areas do classification) and avoids backpropagation entirely.

---

*This is the pivot. Stop pure Hebbian. Start hybrid.*
