---
title: Phase 2 — Five Locked Decisions
date: 2026-04-28
author: hermes
status: locked
tags: [phase-2, decisions, infrastructure]
---

# Phase 2 — Locked Decisions

## 1. Language & Libraries
**Python, NumPy only.** No PyTorch, TensorFlow, JAX, or autograd.
All math implemented from scratch using array operations.

## 2. Compute Target
**Local CPU.** Event-driven, sparse computation. Neuromorphic-ready
design — no GPU dependencies, no matrix multiply hot loops.

## 3. Base Models
**None.** All networks start from random initialization. Learning via
Hebbian/anti-Hebbian rules only. No backpropagation, no transfer learning,
no pretrained weights.

## 4. Benchmark Scope
Progressive evaluation:
1. MNIST — static classification
2. CIFAR-10 — static classification (RGB, higher dimensionality)
3. Split-MNIST — continual learning (task-incremental, no catastrophic forgetting)
4. 2D Grid World — reinforcement learning (navigation, planning)

## 5. Publication Plan
- MIT license
- Public GitHub repository
- arXiv target paper: *"Hierarchical Predictive Coding learns MNIST without backpropagation"*

---

## Directory Structure for Phase 2

```
09_PROTOTYPES/
  hebbian_mnist.py
  predictive_coding.py
  world_model.py

10_CODE/
  core/
    layers.py          # Dense, Conv, Recurrent layers
    learning_rules.py  # Hebb, Oja, Anti-Hebb, STDP
    losses.py          # Custom loss functions
    optimizers.py      # Local update rules only
  mnist/
    train_mnist.py
    eval_mnist.py
  cifar/
    train_cifar.py
  split_mnist/
    continual_learning.py
  grid_world/
    environment.py
    agent.py

04_ARCHITECTURE/
  hierarchical_predictive_coding.md
  memory_system.md
  perception_pipeline.md

06_PROJECT_PLANS/
  phase2_roadmap.md
  task_assignments.md

02_RESEARCH_NOTES/
  hebbian_learning.md
  predictive_coding_benchmarks.md
  continual_learning_sota.md
```

---

## Team Responsibilities — Phase 2

| Bot | Phase 2 Output |
|-----|----------------|
| **coder-1 (Forge)** | Core layers, learning rules, MNIST pipeline |
| **coder-2 (Anvil)** | Split-MNIST continual, memory-augmented networks |
| **coder-3** | CIFAR-10 pipeline, perception front-ends, grid world env |
| **coder-4** | Evaluation harness, baseline comparisons, plotting |
| **definer** | HPC architecture spec, memory/perception specs |
| **researcher** | Track results, compare against baselines, find papers |
| **documenter** | Log everything, cite sources, maintain decision records |
| **verifier** | Verify claims, validate experiments, check math |
| **bug-finder** | Hunt bugs, edge cases, numerical stability |
| **performance-fixer** | Profile CPU usage, optimize hot paths |
| **watcher** | Monitor training, log metrics, alert on failures |
| **orchestrator** | Coordinate tasks, resolve blockers, status updates |
