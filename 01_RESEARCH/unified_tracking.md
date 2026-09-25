# UNIFIED ARCHITECTURE: RAIE + PEN + HAPN
## Researcher: Phase 2 Final Tracking (CORRECTED)
## Date: 2026-06-28

---

## 0. CORRECTIONS FROM AUDIT

1. **REAL MNIST ONLY** — Must use the actual MNIST dataset (70K samples, 28x28 grayscale), not synthetic or toy data.
2. **NOVELTY HONESTY** — Stop claiming "from scratch" for renamed existing work. Our genuine novelty is the **integration** of: SDR hierarchy, XOR binding between levels, compositional world model, and UALR meta-learning. Individual components (Hebbian updates, BCM thresholds, attractor convergence) are known.
3. **PRIORITY: SPU + ATM + UALR** — These three mechanisms are our unique contributions. Implement and test first.
4. **BUGS FIRST** — Fix critical bugs before next benchmark. Known issues listed in Section 8.

---

## 1. UNIFIED ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAIE (Reasoning AI Engine)                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     PEN (Perception Engine)               │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │                  HAPN Core                         │  │  │
│  │  │  Level 3: Planning + Meta-learner                  │  │  │
│  │  │  Level 2: Episodic / Semantic / Procedural Memory  │  │  │
│  │  │  Level 1: Vision + Language Encoders (SDRs)        │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**HAPN** — The neural substrate (sparse distributed representations, attractor dynamics, local learning)
**PEN** — The perception layer (encoding raw inputs into SDRs, decoding SDRs into actions)
**RAIE** — The reasoning layer (compositional queries, planning, meta-cognition)

---

## 2. TARGETS (LOCKED)

### MNIST (Primary Benchmark)

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| **Test Accuracy** | **≥95%** | **≥97%** | **≥98.5%** |
| **KWTA Sparsity** | 3-7% | 5% | 2-10% |
| **Attractor Convergence** | <50 iterations | <20 iterations | <10 iterations |
| **XOR Binding Accuracy** | >99% | >99.5% | >99.9% |
| **Training Time** | <600s | <300s | <120s |
| **Memory Capacity** | >500 memories | >1000 memories | >2000 memories |

### CIFAR-10 (Secondary)

| Metric | Minimum | Target |
|--------|---------|--------|
| **Test Accuracy** | **≥70%** | **≥75%** |

### Split-MNIST Continual

| Metric | Minimum | Target |
|--------|---------|--------|
| **Average Accuracy** | **≥70%** | **≥80%** |
| **Forgetting** | ≤20% | ≤10% |

### 2D Grid World

| Metric | Minimum | Target |
|--------|---------|--------|
| **Avg Steps to Goal** | ≤30 | ≤15 |
| **Success Rate** | ≥90% | ≥98% |

---

## 3. KEY METRICS TO MONITOR

### 3.1 KWTA Sparsity

**What:** % of hidden units active per input.
**Why:** Too dense → interference, catastrophic forgetting. Too sparse → not enough features.
**Expected:** 3-7% (s=0.05 target).
**Flag if:** <1% or >15% consistently.

```python
def compute_sdr_sparsity(sdr):
    """Fraction of active bits in SDR."""
    return np.mean(sdr > 0)

def monitor_sparsity(model, dataloader, n_samples=1000):
    """Track sparsity distribution."""
    sparsities = []
    for i, (x, _) in enumerate(dataloader):
        if i >= n_samples:
            break
        sdr = model.encode(x)
        sparsities.append(compute_sdr_sparsity(sdr))
    
    return {
        'mean': np.mean(sparsities),
        'std': np.std(sparsities),
        'min': np.min(sparsities),
        'max': np.max(sparsities)
    }
```

### 3.2 Attractor Convergence

**What:** Number of iterations for attractor dynamics to settle.
**Why:** Too slow → unusable for real-time. Too fast → shallow attractors, poor recall.
**Expected:** <20 iterations.
**Flag if:** >50 iterations or doesn't converge.

```python
def measure_convergence(model, cue_sdr, max_iter=100):
    """Measure iterations to convergence."""
    S = cue_sdr.copy()
    for t in range(max_iter):
        S_new = model.attractor_step(S)
        if np.array_equal(S_new, S):
            return t
        S = S_new
    return max_iter  # Did not converge
```

### 3.3 XOR Binding Accuracy

**What:** Fraction of correctly recovered bits after bind→unbind cycle.
**Why:** Binding is our composition mechanism. Any error cascades.
**Expected:** >99%.
**Flag if:** <98%.

```python
def binding_accuracy(model, n_tests=1000):
    """Test XOR bind/unbind cycle."""
    N = model.N
    correct = 0
    total = 0
    
    for _ in range(n_tests):
        # Generate random SDRs
        x = random_sdr(N, model.s)
        y = random_sdr(N, model.s)
        
        # Bind then unbind
        z = model.bind_xor(x, y)
        y_recovered = model.unbind_xor(z, x)
        
        # Count correct bits
        correct += np.sum(y == y_recovered)
        total += N
    
    return correct / total
```

### 3.4 Prediction Error (World Model)

**What:** Mean squared error of next-state prediction.
**Why:** Measures quality of world model.
**Expected:** Should decrease over training.
**Flag if:** Increases or plateaus above threshold.

```python
def world_model_error(model, dataloader, n_samples=1000):
    """Track prediction error."""
    errors = []
    for i, (s, a, s_next) in enumerate(dataloader):
        if i >= n_samples:
            break
        s_pred = model.predict(s, a)
        e = s_next - s_pred
        errors.append(np.mean(e ** 2))
    
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'trend': 'decreasing' if errors[-1] < errors[0] else 'increasing'
    }
```

---

## 4. COMPARISON TABLE (To Be Updated)

| Method | MNIST Acc | Sparsity | Convergence | Bind Acc | Time | Status |
|--------|-----------|----------|-------------|----------|------|--------|
| Pure Hebbian | 11.6% | — | — | — | 3.2s | ❌ FAILURE |
| KH2019 + LogReg | 98.5% (lit) | — | — | — | ~120s | 📋 REFERENCE |
| **HAPN (Our Target)** | **≥95%** | **3-7%** | **<20 iter** | **>99%** | **<300s** | **⏳ CODING** |

---

## 5. RISK MONITORING

### 5.1 Regression Flags

| Symptom | Possible Cause | Action |
|---------|----------------|--------|
| Sparsity >15% | KWTA k too large, threshold too low | Reduce k, increase θ |
| Sparsity <1% | KWTA k too small, threshold too high | Increase k, decrease θ |
| No attractor convergence | W too sparse, θ too high | Increase storage rate |
| Binding errors >1% | Bit flips in SDR storage | Check weight clamping |
| Training diverges | η_storage too high | Reduce learning rate |
| Accuracy <90% after 100 epochs | Encoder too weak | Try GSI instead of SRBI |

### 5.2 Failure Modes

1. **Sparsity collapse:** All units converge to same representation.
   - Fix: Increase kWTA k temporarily, add noise during training.

2. **Attractor merging:** Distinct memories merge into single attractor.
   - Fix: Increase sparsity (reduce s), add more hidden units.

3. **Binding noise accumulation:** Errors cascade through bind/unbind cycles.
   - Fix: Limit hierarchy depth to 3 levels, add error correction.

4. **Threshold runaway:** θ grows unbounded, silencing all units.
   - Fix: Add θ_max cap, reduce η_θ.

---

## 6. EXPERIMENTAL LOG

### 2026-06-28: Architecture Unified
- HAPN formal spec: COMPLETE
- From-scratch implementation spec: COMPLETE
- RAIE + PEN integration: PENDING CODER

---

## 7. CODER COORDINATION

### coder-1 (Forge): Core HAPN
- Implement: SRBI init, SHD update, XOR binding, IAC convergence
- Test: MNIST classification
- Target: 95%+ accuracy

### coder-2 (Anvil): Memory System
- Implement: Episodic, Semantic, Procedural memory modules
- Test: Split-MNIST continual learning
- Target: 70%+ average, ≤20% forgetting

### coder-3: Perception Engine (PEN)
- Implement: Vision encoder, language encoder
- Test: SDR quality, sparsity monitoring
- Target: 3-7% sparsity, >99% binding accuracy

### coder-4: Evaluation & Benchmarks
- Implement: All metric trackers
- Test: Full benchmark suite
- Target: All metrics within spec

---

## 8. KNOWN CRITICAL BUGS (FIX BEFORE NEXT BENCHMARK)

### 8.1 SPU Implementation Bug
**Symptom:** Procedural memory weights diverge after ~50 training steps.
**Root cause:** Sign flip in SPU allows negative weight updates to amplify instead of correct.
**Fix:** Remove sign() term. Use: `ΔW_pred = η · e · [S;A]^T` without sign preservation.

### 8.2 ATM Threshold Runaway
**Symptom:** θ_i exceeds 1000x initial value after 200 epochs. All units silenced.
**Root cause:** Quadratic term `a_i² - s²` has no cap. When a_i is briefly high, θ_i jumps and never recovers.
**Fix:** Add cap: `θ_i = min(max(θ_i, θ_min), θ_max)`. Recommend θ_max = 10 · s.

### 8.3 UALR Sigmoid Saturation
**Symptom:** η(t) saturates at η_max permanently after first large error.
**Root cause:** Sigmoid of large positive input ≈ 1.0, never decreases when error drops.
**Fix:** Use bounded rational function: `η(t) = η_max · E_avg / (E_avg + E_target)`. Smooth, never saturates.

### 8.4 Attractor Convergence Deadlock
**Symptom:** Attractor oscillates between two states (S1 → S2 → S1 → ...) without converging.
**Root cause:** Async update with symmetric weights can produce 2-cycles.
**Fix:** Add small noise to h before k-WTA: `h = h + noise · randn(N)`. Breaks symmetry.

### 8.5 Binding Noise Accumulation
**Symptom:** After 3+ levels of binding, output is random.
**Root cause:** Circular convolution amplifies noise at each level. XOR is lossless but convolution loses precision.
**Fix:** Limit hierarchy depth to 3 levels. Use XOR for level 1→2 binding, convolution for level 2→3 only.

---

## 9. PRIORITY IMPLEMENTATION ORDER

1. **Fix 5 bugs above** — coder-3, coder-4
2. **Implement SPU + ATM + UALR** — coder-1, coder-2 (these are our genuine novelty)
3. **Test on REAL MNIST** — coder-4 (download from torchvision or sklearn)
4. **Track all metrics** — coder-4
5. **Iterate**

---

*Live tracking document. Update as results come in.*
