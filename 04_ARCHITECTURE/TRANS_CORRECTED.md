---
title: Unified Architecture — TRANS (Corrected & Annotated)
date: 2026-04-28
type: architecture_master
status: draft_corrected
tags: [architecture, trans, raie, pen, hapn, unified, corrected, annotated]
---

# Unified Architecture: TRANS — The Real-Time Adaptive Neural System
## Corrected & Annotated Edition

> Merging RAIE + PEN + HAPN into ONE system, with honest prior-work citations,
> novelty assessment, bug-audit response, and corrected claims.

---

## 0. PRIOR WORK CITATIONS & FOUNDATIONS

Every component of this architecture builds on existing work. We acknowledge and
cite our foundations honestly. The novelty lies in the **combination, real-time
deduction, and implementation approach** — not in inventing the underlying
components.

### 0.1 RAIE Prior Work

| Component | Prior Work | Citation |
|-----------|------------|----------|
| Deadline Scheduler (EDF) | Real-time scheduling theory | Liu & Layland (1973), *Journal of the ACM* |
| Latency Monitor (P50-P99) | Distributed tracing | Google Dapper (Sigelman et al., 2010) |
| TextEncoder (n-gram hashing) | Feature hashing | Weinberger et al. (2009), *Hashing Kernels* |
| Spelling correction (edit distance) | Levenshtein distance | Levenshtein (1966) |
| ImageEncoder (random projection) | Random projections for ML | Dasgupta (2000), *Random Projections for Learning* |
| AudioEncoder (FFT + projection) | MFCCs | Davis & Mermelstein (1980) |
| FusionCore (weighted concat) | Multimodal fusion | Baltrusaitis et al. (2019) |
| ContinuousLearner (SGD + EWC) | Elastic Weight consolidation | Kirkpatrick et al. (2017), *PNAS* |
| ResponseGenerator (retrieval) | Nearest-neighbor retrieval | Cover & Hart (1967) |

### 0.2 PEN Prior Work

| Component | Prior Work | Citation |
|-----------|------------|----------|
| k-WTA sparse activation | Competitive learning | Rumelhart & Zipser (1985), *Competitive Learning* |
| Energy-based dynamics | Hopfield networks | Hopfield (1982), *PNAS* |
| Eligibility traces | TD learning / STDP | Sutton & Barto (2018), *Reinforcement Learning* |
| Hierarchical predictive coding | Rao & Ballard predictive coding | Rao & Ballard (1999), *Nature Neuroscience* |
| Complementary learning systems | Complementary learning theory | McClelland et al. (1995), *Psychological Review* |
| KWTA sparse activation | k-Winner-Take-All | Maass (2000), *Neural Networks* |

### 0.3 HAPN Prior Work

| Component | Prior Work | Citation |
|-----------|------------|----------|
| XOR binding | Holographic reduced representations | Plate (1995), *Holographic Reduced Representations* |
| Attractor dynamics | Hopfield attractors | Hopfield (1982), *PNAS* |
| Semantic binding (tensor products) | Tensor product binding | Smolensky (1990), *Tensor Product Variable Binding* |
| Episodic/semantic/procedural | Memory systems taxonomy | Tulving (1972); Squire (2004) |
| Complementary learning systems | Hippocampal-neoclassical consolidation | McClelland et al. (1995) |
| Hierarchical attractors | Attractor networks | Hoshino et al. (1997) |

---

## 1. HONEST NOVELTY ASSESSMENT

### What is NOT Novel

The following components are **re-implementations of existing work** and should
not be claimed as original contributions:

1. **Hebbian learning layers** — Re-implements classic Hebbian/anti-Hebbian
   learning with homeostatic plasticity (already well-studied in the 1980s-90s)

2. **Predictive coding hierarchy** — Directly follows Rao & Ballard (1999);
   the hierarchical extension follows existing hierarchical predictive coding models

3. **KWTA sparse activation** — Well-established competitive learning mechanism

4. **XOR binding** — Re-implements Plate's holographic reduced representations

5. **Attractor dynamics** — Direct application of Hopfield (1982)

6. **Complementary learning systems** — Follows McClelland et al. (1995) closely

7. **EDF scheduling** — Standard real-time systems theory (Liu & Layman, 1973)

8. **Random projection encoders** — Standard dimensionality reduction technique

### What MAY Be Novel

The following are the project's **potential original contributions**, pending
rigorous verification:

1. **Real-time integration of all components** — Combining predictive coding,
   KWTA, Hebbian learning, and XOR binding under a single real-time deadline
   constraint may be a novel integration, though each component is known.

2. **Unified sparse activation across all modalities** — Applying the same
   sparse coding principle consistently across text, image, audio, and video
   encoders is not commonly done in a single system.

3. **Real-time multimodal fusion with deadline scheduling** — The combination of
   modality-aware fusion with admission control and fail-fast rejection for
   real-time processing may be original.

4. **Continuous learning from single examples without replay** — The claim that
   this system learns from individual interactions without replay buffers is
   **unverified** and needs rigorous testing. This would be novel if true, but
   is likely overstated.

### Novelty Verdict

**The project is best described as a novel ENGINEERING INTEGRATION of existing
ideas, not a new theory of intelligence.** The contribution is in the
real-time, from-scratch NumPy implementation — not in new mathematical
foundations.

---

## 2. CORRECTED ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRANS — Real-Time Adaptive Neural System         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    REAL-TIME LAYER (RAIE Core)               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────┐  │   │
│  │  │Deadline │  │ Latency │  │Admission│  │  Modality    │  │   │
│  │  │Schedule │  │ Monitor │  │ Control │  │  Encoders    │  │   │
│  │  │ (EDF)   │  │(P50-99) │  │(fail-fast)│ │(text,image, │  │   │
│  │  │         │  │         │  │         │  │ audio,video) │  │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └──────┬───────┘  │   │
│  │       └────────────┴────────────┴──────────────┘           │   │
│  │                          │                                  │   │
│  │                    ┌─────▼─────┐                            │   │
│  │                    │  FUSION   │                            │   │
│  │                    │  CORE     │                            │   │
│  │                    └─────┬─────┘                            │   │
│  └──────────────────────────┼──────────────────────────────────┘   │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │              NEURAL FOUNDATION (PEN + HAPN)                  │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐ │   │
│  │  │  HIERARCHICAL PREDICTIVE CODING (PEN-derived)          │ │   │
│  │  │  Level 3: Abstract Concepts (KWTA, slow)               │ │   │
│  │  │  Level 2: Object Features (KWTA, medium)               │ │   │
│  │  │  Level 1: Edge/Texture (KWTA, fast)                    │ │   │
│  │  │  Level 0: Raw Input → Sparse Representation            │ │   │
│  │  │                                                         │ │   │
│  │  │  Activation: k-WTA (competitive learning, Rumelhart)    │ │   │
│  │  │  Learning: Local eligibility traces + global signal     │ │   │
│  │  │  Dynamics: Energy-based (Hopfield 1982)                 │ │   │
│  │  └────────────────────────────────────────────────────────┘ │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐ │   │
│  │  │  MEMORY SYSTEM (HAPN-derived)                          │ │   │
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐         │ │   │
│  │  │  │ Episodic  │  │ Semantic  │  │ Procedural │         │ │   │
│  │  │  │ (attractor│  │ (XOR      │  │ (world     │         │ │   │
│  │  │  │ dynamics) │  │ binding)  │  │ model)     │         │ │   │
│  │  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘         │ │   │
│  │  │        │              │              │                 │ │   │
│  │  │  ┌─────▼──────────────▼──────────────▼─────┐          │ │   │
│  │  │  │     COMPLEMENTARY LEARNING SYSTEM       │          │ │   │
│  │  │  │  Fast episodic → Slow semantic          │          │ │   │
│  │  │  │  Consolidation via replay               │          │ │   │
│  │  │  └─────────────────────────────────────────┘          │ │   │
│  │  └────────────────────────────────────────────────────────┘ │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐ │   │
│  │  │  COMPOSITIONAL ENGINE (HAPN-derived)                   │ │   │
│  │  │  XOR binding: bind(X, Y) = X ⊕ Y                      │ │   │
│  │  │  Unbinding: unbind(Z, X) = Z ⊕ X                      │ │   │
│  │  │  Hierarchical composition via sparse union             │ │   │
│  │  └────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              RESPONSE GENERATOR                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │   │
│  │  │  Predictive │  │  Multimodal │  │  User Pattern       │ │   │
│  │  │  Decoder    │  │  Output     │  │  Adapter            │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. COMPONENT INTEGRATION (CORRECTED)

### 3.1 RAIE → Real-Time Infrastructure

**What RAIE provides:**
- DeadlineScheduler (EDF scheduling, admission control, fail-fast)
- LatencyMonitor (P50/P90/P95/P99 tracking)
- TextEncoder (character n-gram hashing with fuzzy spelling correction)
- ImageEncoder (nearest-neighbor downsample + random projection)
- AudioEncoder (FFT + random projection, streaming)
- FusionCore (weighted concatenation + modality attention)
- ResponseGenerator (cache-based retrieval with semantic similarity)
- ContinuousLearner (online SGD with elastic weight consolidation)

**Correction from audit:** The ContinuousLearner uses **backpropagation-based
SGD with EWC**, not local learning. This conflicts with the project's "NO
backpropagation" claim. **This must be resolved** — either the ContinuousLearner
is updated to use only local learning rules, or the project acknowledges that
backprop is used in this component.

**Integration:** RAIE becomes the **real-time shell** around the neural
foundation. All inputs flow through RAIE's encoders, get fused, then processed
by the neural foundation.

### 3.2 PEN → Neural Foundation

**What PEN provides:**
- k-WTA sparse activation (competitive learning, Rumelhart & Zipser 1985)
- Energy-based dynamics (Hopfield 1982)
- Local eligibility traces + global neuromodulatory signal
- Hierarchical predictive coding (Rao & Ballard 1999)
- Complementary learning systems (McClelland et al. 1995)

**Correction from audit:** The "global neuromodulatory signal" in PEN is a
**scalar broadcast signal** — this is essentially a simplified version of the
global error signal in backpropagation. The claim of "local learning only" is
**misleading**. The project must either acknowledge this is a weak form of
global learning, or redesign the signal to be truly local.

**Integration:** PEN's KWTA activation replaces RAIE's dense activations in
the encoders. PEN's energy-based dynamics govern the neural foundation's
state evolution.

### 3.3 HAPN → Memory + Compositionality

**What HAPN provides:**
- XOR binding for compositionality (Plate 1995, holographic reduced representations)
- Attractor dynamics for episodic memory (Hopfield 1982)
- Semantic memory via binding (subject ⊗ predicate → object)
- Procedural memory as world model: state ⊗ action → next_state
- Hierarchical attractor dynamics (different sparsity per level)
- Complementary learning systems (McClelland et al. 1995)

**Correction from audit:** XOR binding via direct XOR is **only valid for
binary vectors**. For continuous-valued representations (which the encoders
produce), XOR binding **does not preserve information** and will corrupt
representations. The project must switch to:
- Circular convolution binding (Plate 1995)
- Tensor product binding (Smolensky 1990)
- Or a custom continuous binding operator with proven invertibility

**Integration:** HAPN's memory systems replace RAIE's cache-based retrieval.
HAPN's binding enables compositional representations.

---

## 4. KEY FORMULAS (CORRECTED)

### 4.1 Sparse Activation (PEN) — CORRECT
$$a_i^l \leftarrow \begin{cases} a_i^l & \text{if } a_i^l \in \text{top-}k \\ 0 & \text{otherwise} \end{cases}$$

This is correct and well-established (k-WTA, Maass 2000).

### 4.2 Energy Function (PEN) — CORRECT
$$E = \sum_l \left[-\frac{1}{2} \sum_{i,j} w_{ij}^{l,\text{rec}} a_i^l a_j^l - \sum_{i,j} w_{ij}^{l,\text{ff}} a_j^{l-1} a_i^l + \sum_i \int_0^{a_i^l} \sigma^{-1}(u) \, du\right]$$

This is the standard energy function for Hopfield-style networks. Correct.

### 4.3 Learning Rule (PEN) — NEEDS CORRECTION
$$\Delta w_{ij}^{l,\text{ff}} = \eta \cdot G \cdot \epsilon_i^l \cdot a_j^{l-1}$$

**Issue:** G is a **global signal** (mean absolute prediction error across all
layers). This is **not local learning**. It is a simplified form of
backpropagation where the error signal is broadcast globally.

**Correction options:**
1. Replace G with a local signal: $\Delta w_{ij}^{l} = \eta \cdot \epsilon_i^l \cdot a_j^{l-1}$ (pure Hebbian with error modulation)
2. Acknowledge that G is a neuromodulatory signal (like dopamine) and accept the "weak global learning" framing

### 4.4 XOR Binding (HAPN) — NEEDS CORRECTION
$$Z = X \oplus Y, \quad Y = Z \oplus X$$

**Issue:** XOR only works for binary vectors. For continuous-valued
representations, XOR is **non-invertible** and will corrupt representations.

**Correction options:**
1. Use circular convolution: $Z = X \circledast Y$ (Plate 1995)
2. Use tensor product: $Z = X \otimes Y$ (Smolensky 1990)
3. Use a custom continuous binding: $Z = \sigma(W_1 X + W_2 Y)$ with learned unbinding

### 4.5 Attractor Dynamics (HAPN) — CORRECT
$$S_i(t+1) = H\left(\sum_j W_{ij} S_j(t) - \theta_i(t)\right)$$

This is the standard Hopfield attractor update rule. Correct.

### 4.6 Complementary Learning (HAPN) — CORRECT
$$\Delta w_{ij}^{\text{slow}} += \eta_{\text{slow}} \cdot e_{ij}^{\text{fast}}$$

This follows McClelland et al. (1995) correctly. However, the "replay-based
consolidation" requires either:
- An explicit replay buffer (not yet implemented)
- Online consolidation (which may not prevent forgetting)

### 4.7 Real-Time Guarantee (RAIE) — CORRECT
$$\text{latency} = t_{\text{encode}} + t_{\text{neural}} + t_{\text{generate}} < \text{deadline}$$

This is a straightforward latency decomposition. Correct.

---

## 5. BUG-FINDER AUDIT RESPONSE

### 5.1 Audit Status

The bug-finder audit found:
- `08_BUG_REPORTS/`: **empty** — no bugs have been filed
- `13_BUG_REPORTS/`: **empty** — no bugs have been filed
- `16_SECURITY_AUDITS/`: **empty** — no security audits have been performed

### 5.2 Known Issues (Pre-Audit)

The following issues were identified during architecture review but have **not
been formally filed** as bug reports:

1. **XOR binding for continuous vectors** — will silently corrupt representations
2. **Global learning signal in PEN** — contradicts "local learning only" claim
3. **ContinuousLearner uses SGD + EWC** — contradicts "no backpropagation" claim
4. **Complementary learning replay buffer not implemented** — consolidation is non-functional
5. **ResponseGenerator cache uses hash of embedding as key** — high collision risk for similar embeddings
6. **FusionCore uses fixed modality weights** — should be learned per-user
7. **No numerical stability guards** in HebbianLayer (weight bounding only)
8. **No convergence guarantees** for HAPN attractor dynamics
9. **Random projection encoders are not trained** — representational quality unknown
10. **No tests** for any component

### 5.3 Recommended Actions

| Priority | Action | Owner |
|----------|--------|-------|
| **P0** | Fix XOR binding for continuous vectors | coder-2 |
| **P0** | Resolve global learning signal contradiction | definer |
| **P0** | File formal bug reports for P0 issues | bug-finder |
| **P1** | Implement replay buffer for complementary learning | coder-2 |
| **P1** | Replace fixed fusion weights with learned weights | coder-1 |
| **P1** | Add numerical stability guards | coder-1 |
| **P2** | Add unit tests for all components | verifier |
| **P2** | Profile random projection encoder quality | researcher |
| **P3** | Formal security audit of model serialization | bug-finder |

---

## 6. EXPERIMENT RESULTS

### 6.1 Current Status

No experiment results have been filed. The following directories are empty:
- `08_EVALUATION/` — no evaluation results
- `15_PERFORMANCE_REPORTS/` — no performance reports
- `12_VERIFICATION/` — no verification reports

### 6.2 Claims vs Reality

| Claimed | Reality | Status |
|---------|---------|--------|
| >95% MNIST accuracy | 9.5% on synthetic data (5 epochs) | ⚠️ Not verified |
| 8% sparsity | Achieved (8% of neurons active) | ✅ Verified |
| <100ms text response | Achieved in benchmark | ✅ Verified |
| Real-time image processing | Not benchmarked with real images | ⚠️ Unknown |
| Real-time learning | ContinuousLearner uses SGD, not local learning | ❌ Contradicts project goal |
| No backpropagation | ContinuousLearner uses SGD (backprop) | ❌ Contradicts project goal |

---

## 7. IMPLEMENTATION PLAN (CORRECTED)

### Phase 1: Fix Contradictions (Week 1)
1. Resolve global learning signal in PEN (local-only or reframe claim)
2. Fix XOR binding for continuous vectors (switch to circular convolution)
3. Replace SGD-based ContinuousLearner with pure local learning
4. File formal bug reports for all P0 issues

### Phase 2: Core Integration (Week 2)
1. Merge RAIE's encoders with PEN's KWTA activation
2. Implement unified energy-based dynamics
3. Add replay buffer for complementary learning
4. Test on MNIST with corrected local learning only

### Phase 3: Memory Integration (Week 3)
1. Add HAPN's episodic memory (attractor dynamics) with corrected binding
2. Add XOR binding for semantic memory (circular convolution)
3. Test compositional queries

### Phase 4: Evaluation (Week 4)
1. Benchmark on MNIST → CIFAR-10 → Split-MNIST
2. Compare against backprop baseline (honest comparison)
3. Measure real-world latency on local CPU
4. Document what works and what doesn't

---

## 8. SUCCESS CRITERIA (CORRECTED)

1. **MNIST:** >95% accuracy with local learning only (verify, don't assume)
2. **Real-time:** <100ms text response, <200ms image processing (verify on CPU)
3. **Continual learning:** No catastrophic forgetting on Split-MNIST (verify)
4. **Compositionality:** XOR binding enables novel combinations (test with correct binding)
5. **User adaptation:** Measurable improvement within a single conversation (define metric)
6. **Honest publication:** Report failures, not just successes

---

## 9. FILE STRUCTURE (CORRECTED)

```
AGI_RESEARCH_LAB/
├── 04_ARCHITECTURE/
│   ├── UNIFIED_ARCHITECTURE.md           # Planner's original (reference only)
│   ├── TRANS_CORRECTED.md                # This document (corrected)
│   ├── RAIE_SPEC.md
│   ├── PEN_SPEC.md
│   └── HAPN_SPEC.md
├── 05_TEAM_DECISIONS/
│   ├── 2026-04-28-final-direction-deduce-from-first-principles.md
│   ├── 2026-04-28-full-vision-realtime-adaptive-agi.md
│   └── 2026-04-28-phase-2-decisions.md
├── 08_BUG_REPORTS/                       # EMPTY — needs filing
├── 12_VERIFICATION/
│   └── phase1_audit.md
├── 15_PERFORMANCE_REPORTS/
│   ├── RAIE_PERFORMANCE_REPORT.md
│   └── benchmarks/
├── 16_SECURITY_AUDITS/                   # EMPTY — needs audit
├── 19_PROTOTYPES/
│   ├── raie_engine.py                    # RAIE implementation (800+ lines)
│   ├── hapn.py                           # HAPN implementation
│   ├── unified_system.py                 # Partial unified implementation
│   └── hebbian_mnist.py                  # Hebbian MNIST experiment
└── 02_ARCHITECTURE/
    ├── HAPN_formal_spec.md               # First-principles deduction
    └── from_scratch_implementation.md    # Implementation spec
```

---

*End of Corrected Architecture — 2026-04-28 — filed by documenter@Chronicler*
*Next review: pending bug-finder audit and verifier sign-off*
