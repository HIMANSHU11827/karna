# Hebbian Learning: Study Report

**Date:** 2026-01-15
**Researcher:** coder-1
**Status:** Phase 1 Study Complete

---

## 1. Oja's Rule

### Definition

Oja's rule (1982) modifies basic Hebbian learning by adding a weight decay term proportional to the square of the output.[11]

### Formula

$$w_i(t+1) = w_i(t) + \eta \cdot y \cdot (x_i - y \cdot w_i(t))$$

Where:
- $y = \mathbf{w}^T \mathbf{x}$ (linear output)
- $\eta$ = learning rate
- $x_i$ = input $i$
- $w_i$ = weight $i$

### Derivation

Start with Hebbian update: $\mathbf{w}' = \mathbf{w} + \eta \cdot y \cdot \mathbf{x}$

Normalize: $\mathbf{w}_{new} = \frac{\mathbf{w}'}{|\mathbf{w}'|}$

For small $\eta$, Taylor expansion gives:
$$\mathbf{w}_{new} \approx \mathbf{w} + \eta(y\mathbf{x} - y^2\mathbf{w})$$

### Key Properties

- **Stable**: Weights converge to unit norm (prevents unbounded growth)[11]
- **PCA**: Weight vector converges to first principal component of input data[11]
- **Unsupervised**: No label information required[11]

### Limitations

- Only captures first principal component
- Not discriminative (no class information)
- Single-layer only (without extensions)

---

## 2. BCM Rule (Bienenstock-Cooper-Munro)

### Definition

The BCM rule (1982) adds a **sliding threshold** to Hebbian learning that distinguishes between LTP (long-term potentiation) and LTD (long-term depression).[12][14]

### Formula

$$\Delta w_i = \eta \cdot x_i \cdot y \cdot (y - \theta)$$

Where:
- $y$ = post-synaptic activity
- $\theta$ = sliding threshold (function of average past activity)
- $\eta$ = learning rate

### Sliding Threshold

The threshold $\theta$ is not fixed — it slides based on the time-average of post-synaptic activity:[12][15]

$$\theta(t) = \alpha \cdot \langle y^2 \rangle_t$$

- **High activity** → threshold increases → LTD dominates (depression)
- **Low activity** → threshold decreases → LTP dominates (potentiation)

### Key Properties

- **Stable**: Prevents unbounded weight growth[12]
- **Selective**: Neurons become selective to specific input features[14]
- **Homeostatic**: Maintains dynamic range of synaptic weights[15]

### Biological Plausibility

- Matches LTP/LTD experiments in visual cortex[14]
- Calcium-dependent plasticity mechanism[14]
- Explains orientation selectivity development[15]

---

## 3. Why Pure Hebbian Learning Isn't Enough

### 3.1 Instability (Solved by Oja/BCM)

Pure Hebbian rule: $\Delta \mathbf{w} = \eta \cdot y \cdot \mathbf{x}$

The norm $|\mathbf{w}|$ grows without bound — positive feedback loop.[11]

**Fix**: Oja's rule adds $-y^2\mathbf{w}$ decay term.[11]

### 3.2 No Discriminative Signal

Hebbian learning is **unsupervised** — it learns correlations in input data, not class boundaries.[13][15]

For MNIST:
- Pure Hebbian: ~12% accuracy (barely above random)[13]
- Oja's rule: ~20% (PCA is not discriminative)[13]

**Why**: Hebbian rules detect first- and second-order statistics. Class discrimination requires higher-order structure.[13]

### 3.3 Linear Separability Problem

Oja's rule converges to the **first principal component** — the direction of maximum variance.[11]

But classification often requires finding decision boundaries that are NOT aligned with principal components.[13]

### 3.4 The Parity Problem

Hebbian rules fail completely on parity/XOR-type problems:[13]

- Parity bits have **zero linear correlation** with the label
- PCA finds no useful structure
- BCM selectivity cannot help (still linear)[13]

**Result**: Simple Hebbian, Oja's rule, and BCM all produce chance-level accuracy on parity.[13]

### 3.5 What Hebbian Learning CAN Do

- Feature extraction (PCA, clustering)[11]
- Self-organizing maps[15]
- Unsupervised pretraining[15]
- Pattern completion (Hopfield networks)[15]

### 3.6 What Hebbian Learning CANNOT Do

- Supervised classification (without augmentation)[13]
- Learning higher-order interactions[13]
- Discriminative feature learning[13]
- Credit assignment across layers (no backprop)[15]

---

## 4. What Problems Remain

### 4.1 Online Discriminative Learning

**Problem**: No existing rule combines:
1. Online (per-example) updates
2. Discriminative (label-aware) signal
3. Stable (no unbounded growth)
4. Multi-layer (deep credit assignment)

**Our contribution**: Eligibility traces + error-driven updates + neuromodulatory signal.

### 4.2 Multi-Layer Credit

**Problem**: Hebbian rules are local — they don't propagate error signals across layers.[15]

**Possible solutions**:
- Eligibility traces (temporal credit)[15]
- Feedback alignment (random feedback)[15]
- Predictive coding (layer-wise error signals)[15]

### 4.3 Catastrophic Forgetting

**Problem**: Learning new patterns overwrites old weights.[15]

**Possible solutions**:
- Complementary Learning Systems (episodic + semantic)[15]
- Elastic Weight Consolidation (importance-weighted anchoring)[15]
- Replay mechanisms[15]

### 4.4 Compositional Binding

**Problem**: How to bind features into structured representations.

**Possible solutions**:
- XOR binding (Vector Symbolic Architectures)[15]
- Circular convolution for sequences[15]

### 4.5 Real-Time Inference

**Problem**: Most Hebbian systems are slow (convergence iterations).

**Target**: <50ms inference, <1ms learning per example.

---

## 5. Summary Table

| Rule | Type | Stable | Discriminative | Online | Multi-layer |
|------|------|--------|----------------|--------|-------------|
| Pure Hebbian | Unsupervised | ❌ | ❌ | ✅ | ❌ |
| Oja's Rule | Unsupervised | ✅ | ❌ | ✅ | ❌ |
| BCM Rule | Unsupervised | ✅ | ❌ | ✅ | ❌ |
| Krotov-Hopfield | Unsupervised | ✅ | ~✅ | ✅ | ❌ |
| Backprop | Supervised | ✅ | ✅ | ❌ | ✅ |
| **Our Rule** | **Both** | **✅** | **✅** | **✅** | **✅** |

---

## Sources

[11] https://en.wikipedia.org/wiki/Oja%27s_rule
[12] https://pmc.ncbi.nlm.nih.gov/articles/PMC5318375
[13] https://cybertronai.github.io/SutroYaro/findings/exp_hebbian/
[14] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6622209
[15] https://web.stanford.edu/~jlmcc/papers/McClellandIPHowFar.pdf
