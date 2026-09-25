# AGI Research Lab — Phase 2 Baseline Tracking
## Researcher: Baseline Metrics & Targets
## Date: 2026-06-28

---

## PHASE 2 CONSTRAINTS (LOCKED)

| Constraint | Value |
|------------|-------|
| Language | Python 3.10+ |
| Dependencies | NumPy ONLY (no PyTorch, no TensorFlow, no JAX, no autograd) |
| Compute | Local CPU (no GPU) |
| Hardware target | Neuromorphic-ready (event-driven, sparse) |
| Base models | NONE — from scratch, random init |
| Learning rule | Hebbian / local plasticity ONLY — no backpropagation |
| License | MIT |
| Publication target | arXiv |

---

## BASELINE METRICS (Literature)

### 1. MNIST (70K samples, 10 classes, 28x28 grayscale)

| Method | Accuracy | Params | Notes |
|--------|----------|--------|-------|
| **Standard MLP (backprop, 2-layer)** | 97.5% | 600K | Upper bound for "simple" MLP |
| **Standard MLP (backprop, 3-layer)** | 98.0% | 1M | With batch norm |
| **CNN (LeNet, backprop)** | 99.0% | 60K | Gold standard |
| **KNN (k=3, Euclidean)** | 97.2% | 0 | Non-parametric, no training |
| **Random Forest (100 trees)** | 96.7% | 0 | Ensemble |
| **SVM (RBF kernel)** | 98.6% | 0 | Kernel method |
| **Hebbian MLP (Oja's rule)** | 94.5% | 600K | Classic Hebbian, 2-layer |
| **PCA + linear classifier** | 95.0% | ~1K | 50 PCA components |
| **Predictive Coding (sPC, 5-layer)** | 95.8% | 500K | From Whittington & Bogacz 2017 |
| **ePC (fast PC, 10-layer)** | 97.0% | 800K | 2026 ICML |
| **KAN (5-layer, B-spline)** | 97.5% | 200K | ICLR 2025 (small grid) |
| **Krotov-Hopfield 2019** | **98.48%** | 1.6M | Local unsupervised + linear classifier |
| **Pure Hebbian (Ours)** | **11.6%** | N/A | CONFIRMED FAILURE — no credit assignment |

**PIVOTED Target for MNIST:** ≥95% (stretch ≥98% — KH2019 proven)
**APPROACH:** Krotov-Hopfield competing hidden units + linear classifier on top

### 2. CIFAR-10 (60K samples, 10 classes, 32x32 RGB)

| Method | Accuracy | Params | Notes |
|--------|----------|--------|-------|
| **Standard CNN (backprop)** | 95.0% | 1.3M | ResNet-18 level |
| **Simple CNN (backprop)** | 88.0% | 500K | 3 conv layers |
| **MLP (backprop, 3-layer)** | 55.0% | 2M | MLP fails on images |
| **KNN (k=3)** | 42.0% | 0 | Raw pixels |
| **Random Forest** | 46.0% | 0 | Raw pixels |
| **SVM (RBF)** | 58.0% | 0 | Raw pixels, slow |
| **Hebbian CNN (none exist)** | N/A | N/A | No published baseline |
| **Predictive Coding** | N/A | N/A | Not demonstrated on CIFAR-10 |

**Target for Phase 2:** ≥70% (non-backprop), ≥85% (approaching simple CNN)
**Stretch target:** ≥90% (approaching backprop ResNet-18)

### 3. Split-MNIST Continual Learning

**Setup:** 5 tasks, 2 classes each (0/1, 2/3, 4/5, 6/7, 8/9). Train sequentially. Evaluate on all seen tasks.

| Method | Avg Accuracy (after all 5) | Forgetting | Notes |
|--------|---------------------------|------------|-------|
| **Backprop (naive SGD)** | 45.0% | Severe | Classic catastrophic forgetting |
| **Backprop + EWC** | 72.0% | Moderate | Elastic Weight Consolidation |
| **Backprop + replay buffer** | 85.0% | Low | Experience replay |
| **Progressive Nets** | 95.0% | None | Quadratic parameter growth |
| **CompoNet** | 90.0% | Low | Linear growth, 2025 |
| **Hebbian (Oja, no CL)** | 35.0% | Severe | Forgets immediately |
| **Hebbian + EWC-like** | 55.0% | Moderate | Hebbian-EWC hybrid |

**Target for Phase 2:** ≥70% average accuracy, ≤20% forgetting
**Stretch target:** ≥80% average accuracy, ≤10% forgetting

### 4. 2D Grid World (Navigation Task)

**Setup:** 8x8 grid, 4 actions (up/down/left/right), sparse reward at goal.

| Method | Avg Steps to Goal | Success Rate | Notes |
|--------|-------------------|--------------|-------|
| **Q-learning (tabular)** | 15 | 98% | Tabular, discrete states |
| **DQN (backprop)** | 12 | 99% | Small CNN |
| **PPO (backprop)** | 10 | 99% | Policy gradient |
| **NEAT (neuroevolution)** | 20 | 95% | Evolved policy |
| **Random policy** | 50+ | 30% | Baseline |
| **Hebbian (none)** | N/A | N/A | No published baseline |

**Target for Phase 2:** ≤30 steps average, ≥90% success
**Stretch target:** ≤15 steps, ≥98% success

---

## EXPERIMENTAL LOG TEMPLATE

Each experiment logs:
```
Date:
Experiment ID:
Architecture:
Hyperparameters:
Dataset:
Accuracy/Performance:
Training time:
Convergence speed:
Stability:
Notes:
```

## CODER COORDINATION

### For coder-1 (Forge) — MNIST prototype
- Priority: Build predictive coding network with Hebbian updates
- Baseline to beat: 94.5% (classic Hebbian MLP)
- Target: 96%+ on MNIST

### For coder-2 (Anvil) — Memory system
- Priority: Build hierarchical memory module
- Test on Split-MNIST continual learning
- Baseline to beat: 72% (backprop + EWC)
- Target: 70%+ without backprop

### For coder-3 — Perception/CIFAR
- Priority: Build feature extractor for CIFAR-10
- Baseline to beat: 58% (SVM on raw pixels)
- Target: 70%+ without backprop

### For coder-4 — Evaluation
- Priority: Build benchmarking harness
- Track all metrics above
- Automated comparison tables

## GAPS TO INVESTIGATE

1. **Hebbian convolutional networks** — almost no published work. This is a major gap.
2. **Predictive coding on CIFAR-10** — nobody has done it. Why?
3. **Hebbian continual learning** — sparse literature, mostly theoretical.
4. **Neuromorphic grid world agents** — spiking RL is niche but growing.

## RISKS & MITIGATIONS

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Predictive coding too slow | HIGH | Can't finish experiments | Limit to ≤10 layers, use ePC |
| Hebbian unstable | MEDIUM | Divergent training | Add weight normalization, Oja's rule |
| MNIST ceiling at 95% | MEDIUM | Miss target | Try hierarchical PC, add lateral connections |
| CIFAR-10 too hard | HIGH | Can't reach 70% | Start with 32x32 features, not raw pixels |
| Split-MNIST forgetting | HIGH | Catastrophic forgetting | Use EWC-like regularization, task boundaries |
| No backprop = no transfer | MEDIUM | Each task from scratch | Meta-learning initial weights |

## KEY REFERENCES FOR BASELINES

1. Whittington & Bogacz (2017) — "Theories of Error Backpropagation"
2. Song et al. (2024) — "Prospective Configuration" (Nature Neuroscience)
3. ePC (ICML 2026) — Fast predictive coding
4. KAN (ICLR 2025) — Kolmogorov-Arnold Networks
5. CompoNet (2025) — Modular continual learning
6. PICLE (2024) — Probabilistic modular CL
7. NEAT (2002) — Neuroevolution baseline
8. Oja (1982) — Hebbian learning rule

---

*Tracking file — update as experiments complete.*
