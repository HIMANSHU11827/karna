# AGI Research Lab — Architecture Overview

## Project Vision

Build an AI assistant research system to investigate and engineer increasingly general intelligence. The system integrates memory, learning, perception, reasoning, and evaluation into a cohesive architecture that can be iteratively improved.

## Phase 2 Locked Decisions (2026-01-15)

1. **Language:** Python 3.11+, NumPy only — no PyTorch, no TensorFlow, no autograd
2. **Compute:** Local CPU, neuromorphic-ready (event-driven, sparse activity)
3. **Base models:** NONE — all from scratch, random init, Hebbian learning only
4. **Scope:** MNIST → CIFAR-10 → Split-MNIST continual → 2D grid world
5. **Publication:** MIT license, public GitHub, arXiv target ("Hierarchical Predictive Coding learns MNIST without backpropagation")

## Core Architecture

### Design Principles

- **Local learning only** — No global gradient, no backpropagation
- **Sparse activity** — Event-driven updates, energy efficient
- **Modular** — Each layer learns independently via local rules
- **Neuromorphic-ready** — Spiking, event-based computation friendly

### Component Architecture

```
AGI_RESEARCH_LAB/
├── 19_PROTOTYPES/          # Working code (coder-1, coder-2, coder-3, coder-4)
│   ├── hebbian_layers.py   # HebbianLayer, PredictiveCodingLayer, Classifier
│   ├── hpc_network.py      # Hierarchical Predictive Coding Network
│   ├── baseline_agent.py   # Agent loop with memory and tools
│   ├── world_model.py      # Predictive world model for planning
│   ├── memory_system.py    # Multi-tier memory system
│   ├── neural_architecture_search.py  # Evolutionary NAS
│   └── experiment_framework.py         # Experiment tracking
├── 04_ARCHITECTURE/        # Architecture documentation
├── 06_EXPERIMENTS/         # Training scripts and results
├── 06_MEMORY/              # Persisted memory states
├── 16_DATA/                # Training data (MNIST, CIFAR, etc.)
├── 20_OUTPUT/              # Experiment outputs and results
└── 20_RESULTS/             # Analysis reports
```

## Core Components

### 1. Hebbian Learning Layers (`19_PROTOTYPES/hebbian_layers.py`)

**HebbianLayer** — Sparse Hebbian learning with:
- Lateral inhibition (soft competition)
- Homeostatic plasticity (activity regulation)
- Weight bounding (stability)
- Event-driven updates

**PredictiveCodingLayer** — Top-down predictive feedback:
- Feedforward weights: encode input
- Feedback weights: predict input from representation
- Error minimization: local teaching signal
- Weight normalization for stability

**HebbianSoftmaxClassifier** — Prototype-based classifier:
- Running average prototype per class
- Cosine similarity for classification
- No gradient descent

### 2. Hierarchical Predictive Coding Network (`19_PROTOTYPES/hpc_network.py`)

Multi-layer architecture where each layer predicts the layer below.

**Training protocol:**
1. Layer-wise unsupervised pretraining
2. Supervised fine-tuning with slow learning rate

**Architecture (default):**
- Input: 784 (28×28 MNIST)
- Hidden: 256 → 100
- Output: 10 (classifier)

**Status (Phase 2):**
- ✅ Framework complete and stable
- ✅ No overflow/NaN errors
- ✅ Sparse activity (8% sparsity)
- ⚠️ Accuracy low on synthetic data (9.5%, random baseline = 10%)
- 🔄 Needs real MNIST data and longer training

### 3. Baseline Agent (`19_PROTOTYPES/baseline_agent.py`)

Agent loop with:
- Multi-turn conversation
- Sliding-window memory
- Tool registry and execution
- State persistence

### 4. World Model (`19_PROTOTYPES/world_model.py`)

- Simple 2D grid environment
- Model-based planning via learned transitions
- Greedy action selection

### 5. Memory System (`19_PROTOTYPES/memory_system.py`)

Four-tier memory:
- **Working memory:** Short-term, limited capacity (7 items)
- **Semantic memory:** Facts and knowledge
- **Episodic memory:** Experiences and events
- **Procedural memory:** Skills and procedures

Features:
- Consolidation from working to semantic
- Relevance-based retrieval
- Tag-based search

### 6. Experiment Framework (`19_PROTOTYPES/experiment_framework.py`)

- Config-driven experiments
- Fingerprinting for deduplication
- Metric tracking
- Result persistence

## Interface Definitions

### Learning API
```
forward(x) -> representation
update(x, y, reward_signal) -> None
encode(x) -> layer_activations
compute_error(x, y) -> error_vector
```

### Agent API
```
step(user_input) -> response
run_loop(prompt, max_steps) -> final_response
remember(fact) -> None
save(path) / load(path) -> AgentState
```

### Memory API
```
perceive(content, importance, tags) -> None
recall(query, n) -> MemoryItems
remember_fact(key, fact) -> None
record_episode(episode) -> None
register_procedure(name, fn) -> None
execute_procedure(name, **kwargs) -> result
```

### Experiment API
```
create_experiment(name, description, parameters) -> ExperimentConfig
run(config, fn) -> ExperimentResult
list_experiments(tag) -> Results
compare(fingerprints) -> Metrics
```

## Technology Stack (Phase 2 Locked)

- **Language:** Python 3.11+
- **Compute:** NumPy only (no PyTorch/TensorFlow/JAX)
- **Hardware:** Local CPU, neuromorphic-ready
- **Learning:** Hebbian/anti-Hebbian, predictive coding
- **Data:** MNIST, CIFAR-10, Split-MNIST, custom 2D environments

## Performance Targets

| Metric | Target | Current (Phase 2) |
|--------|--------|--------------------|
| MNIST accuracy | >95% | 9.5% (synthetic data, 5 epochs) |
| Training stability | No NaN/overflow | ✅ Achieved |
| Sparse activity | <20% active | 8% (achieved) |
| Memory retrieval | >90% | Not yet tested |
| Forward pass | <100ms | Achieved |

## Success Criteria

1. ✅ Reproducible experiments with seed management
2. ✅ Local learning (no backprop)
3. ✅ Sparse, event-driven activity
4. ⚠️ MNIST accuracy >95% (needs real data + tuning)
5. 🔄 CIFAR-10 and continual learning (next phase)
6. 🔄 Publication-ready results

---

*Last updated: 2026-01-15 by coder-1. Phase 2 architecture locked per hermes decisions.*