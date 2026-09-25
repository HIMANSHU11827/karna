# 15_EVALUATION — From-Scratch Intelligence Testing

## Design Principles
- **No LLM benchmarks** — no perplexity, no BLEU, no MMLU
- **No existing frameworks** — not pytest-benchmark, not scikit-learn metrics
- **Built for neuromorphic systems** — event-driven, sparse, Hebbian
- **Measures intelligence directly** — learning speed, retention, generalization, adaptability

## Architecture

```
15_EVALUATION/
├── core/
│   ├── __init__.py
│   ├── metrics.py            # Novel intelligence metrics
│   ├── runner.py             # Test execution engine
│   └── reporters.py          # Results formatting
├── tests/
│   ├── __init__.py
│   ├── learning_speed.py     # Accuracy gain per sample
│   ├── retention.py          # Memory decay measurement
│   ├── generalization.py     # Cross-domain transfer
│   ├── robustness.py         # Noise/corruption resilience
│   ├── efficiency.py         # Compute per prediction
│   ├── adaptability.py       # Distribution shift recovery
│   ├── compositionality.py   # Concept combination
│   └── continual.py          # Split-MNIST protocol
├── benchmarks/
│   ├── __init__.py
│   ├── mnist_suite.py        # MNIST variants
│   ├── cifar_suite.py        # CIFAR-10 variants
│   ├── grid_world.py         # 2D navigation
│   └── continual_split.py    # Split-MNIST protocol
├── configs/
│   ├── phase2_standard.json
│   └── phase2_stress.json
└── outputs/
    └── results/
```

## Novel Intelligence Metrics

- **Learning Velocity:** d(Accuracy)/d(Samples) — how fast does it learn?
- **Forgetting Rate:** d(Accuracy_old)/d(Samples_new) — does new learning destroy old?
- **Transfer Coefficient:** Accuracy_new_with_prior / Accuracy_new_from_scratch
- **Adaptation Latency:** Samples needed to reach threshold after distribution shift
- **Robustness Score:** Accuracy_under_noise / Accuracy_clean
- **Compositional Score:** Success on tasks requiring combination of learned primitives
- **Sparsity Index:** % of neurons inactive per forward pass (neuromorphic efficiency)
- **Event Economy:** Total events per correct prediction (lower = more efficient)

## Phase 2 Integration
- Tests any architecture that implements: `forward()`, `predict()`, `get_state()`
- No dependency on training method — works with Hebbian, predictive coding, or novel methods
- Outputs JSON results for cross-architecture comparison
