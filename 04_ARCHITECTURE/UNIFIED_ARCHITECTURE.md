# Unified Architecture: The Real-Time Adaptive Neural System (TRANS)
*Unified by planner | Merging RAIE + PEN + HAPN into ONE system*

## Preamble

Three architectures have been developed in parallel:
- **RAIE** — Real-time engine with benchmarked performance (0.19ms text, 45ms image, 5186 req/s)
- **PEN** — Predictive Equilibrium Network with KWTA sparsity, energy-based dynamics, eligibility traces
- **HAPN** — Hierarchical Attractor Predictive Network with XOR binding, complementary learning, world models

This document unifies them into ONE system: **TRANS** (The Real-Time Adaptive Neural System).

Each system contributes its core innovation:
- **RAIE →** Real-time infrastructure, encoders, deadline scheduling, latency monitoring
- **PEN →** KWTA sparse activation, predictive learning with eligibility traces, energy-based dynamics
- **HAPN →** XOR binding for compositionality, attractor-based episodic/semantic memory, complementary learning systems

---

## 1. Unified Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE REAL-TIME ADAPTIVE NEURAL SYSTEM (TRANS)      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    REAL-TIME LAYER (RAIE Core)               │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────┐  │    │
│  │  │Deadline │  │ Latency │  │Admission│  │  Modality    │  │    │
│  │  │Schedule │  │ Monitor │  │ Control │  │  Encoders    │  │    │
│  │  │ (EDF)   │  │(P50-99) │  │(fail-fast)│ │(text,image, │  │    │
│  │  │         │  │         │  │         │  │ audio,video) │  │    │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └──────┬───────┘  │    │
│  │       └────────────┴────────────┴──────────────┘           │    │
│  │                          │                                  │    │
│  │                    ┌─────▼─────┐                            │    │
│  │                    │  FUSION   │                            │    │
│  │                    │  CORE     │                            │    │
│  │                    └─────┬─────┘                            │    │
│  └──────────────────────────┼──────────────────────────────────┘    │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐    │
│  │              NEURAL FOUNDATION (PEN + HAPN)                  │    │
│  │                                                              │    │
│  │  ┌────────────────────────────────────────────────────────┐ │    │
│  │  │  HIERARCHICAL PREDICTIVE CODING (PEN-derived)          │ │    │
│  │  │  Level 3: Abstract Concepts (KWTA, slow)               │ │    │
│  │  │  Level 2: Object Features (KWTA, medium)               │ │    │
│  │  │  Level 1: Edge/Texture (KWTA, fast)                    │ │    │
│  │  │  Level 0: Raw Input → Sparse Representation            │ │    │
│  │  │                                                         │ │    │
│  │  │  Activation: k-Winner-Take-All (KWTA)                  │ │    │
│  │  │  Learning: Local eligibility traces + global signal     │ │    │
│  │  │  Dynamics: Energy-based convergence                     │ │    │
│  │  └────────────────────────────────────────────────────────┘ │    │
│  │                                                              │    │
│  │  ┌────────────────────────────────────────────────────────┐ │    │
│  │  │  MEMORY SYSTEM (HAPN-derived)                          │ │    │
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐         │ │    │
│  │  │  │ Episodic  │  │ Semantic  │  │ Procedural │         │ │    │
│  │  │  │ (attractor│  │ (XOR      │  │ (world     │         │ │    │
│  │  │  │ dynamics) │  │ binding)  │  │ model)     │         │ │    │
│  │  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘         │ │    │
│  │  │        │              │              │                 │ │    │
│  │  │  ┌─────▼──────────────▼──────────────▼─────┐          │ │    │
│  │  │  │     COMPLEMENTARY LEARNING SYSTEM       │          │ │    │
│  │  │  │  Fast episodic → Slow semantic          │          │ │    │
│  │  │  │  Consolidation via replay               │          │ │    │
│  │  │  └─────────────────────────────────────────┘          │ │    │
│  │  └────────────────────────────────────────────────────────┘ │    │
│  │                                                              │    │
│  │  ┌────────────────────────────────────────────────────────┐ │    │
│  │  │  COMPOSITIONAL ENGINE (HAPN-derived)                   │ │    │
│  │  │  XOR binding: bind(X, Y) = X ⊕ Y                      │ │    │
│  │  │  Unbinding: unbind(Z, X) = Z ⊕ X                      │ │    │
│  │  │  Hierarchical composition via sparse union             │ │    │
│  │  └────────────────────────────────────────────────────────┘ │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │              RESPONSE GENERATOR                               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │    │
│  │  │  Predictive │  │  Multimodal │  │  User Pattern       │ │    │
│  │  │  Decoder    │  │  Output     │  │  Adapter            │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │    │
│  └──────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Integration

### 2.1 RAIE → Real-Time Infrastructure

**What RAIE provides:**
- DeadlineScheduler (EDF scheduling, admission control, fail-fast)
- LatencyMonitor (P50/P90/P95/P99 tracking)
- TextEncoder (character n-gram hashing with fuzzy spelling correction)
- ImageEncoder (nearest-neighbor downsample + random projection)
- AudioEncoder (FFT + random projection, streaming)
- FusionCore (weighted concatenation + modality attention)
- ResponseGenerator (cache-based retrieval with semantic similarity)
- ContinuousLearner (online SGD with elastic weight consolidation)

**Integration:** RAIE becomes the **real-time shell** around the neural foundation. All inputs flow through RAIE's encoders, get fused, then processed by the neural foundation. Responses flow back through RAIE's generator.

### 2.2 PEN → Neural Foundation

**What PEN provides:**
- k-WTA sparse activation (replaces dense activations)
- Energy-based dynamics (guaranteed convergence)
- Local eligibility traces + global neuromodulatory signal
- Hierarchical predictive coding (top-down prediction, bottom-up error)
- Complementary learning systems (fast + slow)

**Integration:** PEN's KWTA activation replaces RAIE's dense activations in the encoders. PEN's energy-based dynamics govern the neural foundation's state evolution. PEN's learning rules update weights.

### 2.3 HAPN → Memory + Compositionality

**What HAPN provides:**
- XOR binding for compositionality (bind/unbind operations)
- Attractor dynamics for episodic memory (convergence to stored patterns)
- Semantic memory via binding: subject ⊗ predicate → object
- Procedural memory as world model: state ⊗ action → next_state
- Hierarchical attractor dynamics (different sparsity per level)
- Meta-learning via prediction error

**Integration:** HAPN's memory systems replace RAIE's cache-based retrieval. HAPN's XOR binding enables compositional representations. HAPN's attractor dynamics provide the recall mechanism.

---

## 3. Unified Data Flow

### 3.1 Input Processing (Real-Time)

```
Input (text/image/audio/video)
    │
    ▼
┌──────────────────┐
│  RAIE Encoders   │  ← TextEncoder, ImageEncoder, AudioEncoder
│  (with KWTA)     │  ← PEN's sparse activation applied
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Fusion Core     │  ← Weighted concatenation + modality attention
│  (KWTA-gated)    │  ← Only top-k features survive
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Hierarchical    │  ← PEN's predictive coding hierarchy
│  Predictive      │  ← Level 0 (raw) → Level 1 (edges) → Level 2 (objects) → Level 3 (concepts)
│  Coding          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Memory System   │  ← HAPN's episodic/semantic/procedural
│  (Attractor +    │  ← XOR binding for compositionality
│   Binding)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Response        │  ← Predictive decoder + user pattern adapter
│  Generator       │
└──────────────────┘
```

### 3.2 Learning Flow (Continuous)

```
Every input:
    │
    ▼
┌──────────────────┐
│  Prediction      │  ← Top-down prediction from current state
│  (PEN dynamics)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Error Compute   │  ← Bottom-up prediction error
│  (PEN)           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Global Signal   │  ← G = mean absolute prediction error
│  (PEN)           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Weight Update   │  ← Local eligibility traces × global signal
│  (PEN + HAPN)    │  ← Fast episodic system learns immediately
│                  │  ← Slow semantic system learns via consolidation
└──────────────────┘
```

---

## 4. Key Formulas (Unified)

### 4.1 Sparse Activation (PEN)
$$a_i^l \leftarrow \begin{cases} a_i^l & \text{if } a_i^l \in \text{top-}k \\ 0 & \text{otherwise} \end{cases}$$

### 4.2 Energy Function (PEN)
$$E = \sum_l \left[-\frac{1}{2} \sum_{i,j} w_{ij}^{l,\text{rec}} a_i^l a_j^l - \sum_{i,j} w_{ij}^{l,\text{ff}} a_j^{l-1} a_i^l + \sum_i \int_0^{a_i^l} \sigma^{-1}(u) \, du\right]$$

### 4.3 Learning Rule (PEN)
$$\Delta w_{ij}^{l,\text{ff}} = \eta \cdot G \cdot \epsilon_i^l \cdot a_j^{l-1}$$

### 4.4 XOR Binding (HAPN)
$$Z = X \oplus Y, \quad Y = Z \oplus X$$

### 4.5 Attractor Dynamics (HAPN)
$$S_i(t+1) = H\left(\sum_j W_{ij} S_j(t) - \theta_i(t)\right)$$

### 4.6 Complementary Learning (HAPN)
$$\Delta w_{ij}^{\text{slow}} += \eta_{\text{slow}} \cdot e_{ij}^{\text{fast}}$$

### 4.7 Real-Time Guarantee (RAIE)
$$\text{latency} = t_{\text{encode}} + t_{\text{neural}} + t_{\text{generate}} < \text{deadline}$$

---

## 5. Implementation Plan

### Phase 1: Core Integration (Week 1)
- Merge RAIE's encoders with PEN's KWTA activation
- Implement unified energy-based dynamics
- Test on MNIST with sparse activations

### Phase 2: Memory Integration (Week 2)
- Add HAPN's episodic memory (attractor dynamics)
- Add XOR binding for semantic memory
- Test compositional queries

### Phase 3: Real-Time Learning (Week 3)
- Implement continuous learning pipeline
- Add complementary learning (fast + slow)
- Test on split-MNIST continual learning

### Phase 4: Full System (Week 4)
- Integrate all components
- Benchmark on MNIST → CIFAR-10 → grid world
- Performance optimization

---

## 6. File Structure

```
AGI_RESEARCH_LAB/
├── 04_ARCHITECTURE/
│   └── UNIFIED_ARCHITECTURE.md      ← This document
├── 19_PROTOTYPES/
│   └── unified_system/
│       ├── __init__.py
│       ├── realtime_layer.py        ← RAIE core (encoders, scheduler, monitor)
│       ├── neural_foundation.py     ← PEN core (KWTA, energy, eligibility)
│       ├── memory_system.py         ← HAPN core (attractors, binding, complementary)
│       ├── fusion_core.py           ← Modality fusion with KWTA
│       ├── response_generator.py    ← Predictive decoder + user adapter
│       └── unified_trainer.py       ← Continuous learning pipeline
├── 01_RESEARCH/
│   └── unified_research_notes.md
├── 08_EVALUATION/
│   └── unified_benchmarks.py
└── 00_PROJECT_PLAN/
    └── master_plan.md               ← Updated
```

---

## 7. Success Criteria

1. **MNIST:** > 95% accuracy with sparse activations + no backprop
2. **Real-time:** < 100ms text response, < 200ms image processing
3. **Continual learning:** No catastrophic forgetting on split-MNIST
4. **Compositionality:** XOR binding enables novel concept combinations
5. **User adaptation:** Measurable improvement within a single conversation

---

*End of Unified Architecture — planner*
