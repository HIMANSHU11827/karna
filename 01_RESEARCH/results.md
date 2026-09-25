# Phase 2 Pivot Results
## Researcher: Experiment Tracking
## Date: 2026-06-28

---

## CONFIRMED RESULTS

| Method | Accuracy | Params | Training Time | Notes |
|--------|----------|--------|---------------|-------|
| **Pure Hebbian (Ours)** | **11.6%** | 1.6M | 3.2s | No credit assignment. ≈random baseline. CONFIRMED FAILURE. |
| **Backprop (reference)** | 98.5% | 1.6M | ~60s | Standard 2-layer MLP with Adam optimizer |
| **Krotov-Hopfield (Literature)** | 97.8% | 1.6M | ~120s | 2000 hidden units, p=4, k=3, 1000 epochs |

---

## PIVOT EXPERIMENTS (Pending)

### Experiment 1: Krotov-Hopfield + Logistic Regression
**Coder:** coder-1 (Forge)
**Status:** NOT STARTED
**Target:** ≥95% MNIST

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Test Accuracy | ≥95% | — | — |
| Feature Sparsity (% active) | 10-30% | — | — |
| Feature Diversity (cos sim) | <0.3 | — | — |
| Training Time | <300s | — | — |
| Convergence Epoch | <1000 | — | — |

**Hyperparameters to test:**
- p (Lebesgue norm): {2, 3, 4, 5}
- k (WTA count): {2, 3, 5, 10}
- Learning rate: {0.01, 0.02, 0.04, 0.08}
- Hidden units: {500, 1000, 2000}

---

### Experiment 2: Oja's Rule + Logistic Regression
**Coder:** TBD
**Status:** NOT STARTED
**Target:** ≥90% MNIST

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Test Accuracy | ≥90% | — | — |
| Feature Sparsity | 20-50% | — | — |
| Feature Diversity | <0.4 | — | — |
| Training Time | <120s | — | — |

---

### Experiment 3: BCM Rule + Logistic Regression
**Coder:** TBD
**Status:** NOT STARTED
**Target:** ≥85% MNIST

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Test Accuracy | ≥85% | — | — |
| Feature Sparsity | 15-40% | — | — |
| Feature Diversity | <0.35 | — | — |
| Training Time | <180s | — | — |

---

### Experiment 4: Krotov-Hopfield Alone (No Classifier)
**Coder:** TBD
**Status:** NOT STARTED
**Target:** ≥80% MNIST (using raw hidden unit activations as classifier)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Test Accuracy | ≥80% | — | — |
| Linear Separability | High | — | — |

---

## COMPARISON TABLE (To Be Updated)

| Method | Accuracy | Sparsity | Diversity | Time (s) | Status |
|--------|----------|----------|-----------|----------|--------|
| Pure Hebbian | **11.6%** | ? | ? | 3.2 | ❌ CONFIRMED FAILURE |
| Oja + Logistic | — | — | — | — | ⏳ PENDING |
| BCM + Logistic | — | — | — | — | ⏳ PENDING |
| Krotov-Hopfield + Logistic | — | — | — | — | ⏳ PENDING |
| Krotov-Hopfield (alone) | — | — | — | — | ⏳ PENDING |
| Backprop (reference) | **98.5%** | ? | ? | ~60 | ✅ UPPER BOUND |
| Krotov-Hopfield (Literature) | **97.8%** | ? | ? | ~120 | 📋 REFERENCE |

---

## RESEARCH QUESTIONS

### Q1: Does Krotov-Hopfield actually work in our implementation?
**Hypothesis:** Yes, if weight normalization and anti-Hebbian term are correctly implemented.

### Q2: How sensitive is it to hyperparameters (p, k, learning rate)?
**Hypothesis:** p=4, k=3, lr=0.04 should work (from literature). But may need tuning for our NumPy implementation.

### Q3: How does sparsity affect accuracy?
**Hypothesis:** Moderate sparsity (10-30% active) gives highest accuracy. Too sparse = not enough features. Too dense = redundant features.

### Q4: Can we scale to CIFAR-10 with the same approach?
**Hypothesis:** Literature says 50.75% with MLP on grayscale CIFAR-10. We'll likely need BCM-CNN approach for ≥70%.

---

## EXPERIMENT LOG

### 2026-06-28: Initial Setup
- Pure Hebbian confirmed failure: 11.6%
- Krotov-Hopfield selected as primary pivot target
- BCM-CNN identified as CIFAR-10 fallback

---

## METRICS DEFINITIONS

**Feature Sparsity:** For each input, count fraction of hidden units with activation > 0.1. Average over 1000 test samples.

**Feature Diversity:** For each input, compute pairwise cosine similarity between all hidden unit activation vectors. Average across pairs, then across 1000 test samples.

**Training Time:** Wall-clock time for unsupervised phase only (before adding linear classifier).

**Convergence Rate:** Epoch at which validation accuracy stops improving (measured every 10 epochs).

---

*Results file — update as experiments complete.*
