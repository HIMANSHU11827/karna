---
title: Final Direction — Deduce Neural Architecture from First Principles
date: 2026-04-28
author: hermes
type: project_pivot
status: locked
tags: [deduction, first-principles, novel-architecture, ground-up, agi]
---

# FINAL DIRECTION: Deduce Neural Architecture from First Principles

## Context

Multiple iterations of the AGI Research Lab project have converged on a single
conclusion: **all existing architectures are inadequate as a foundation.**

Previous phases explored:
- Phase 1: Research alternatives (recurrent, sparse, predictive coding, etc.)
- Phase 2: Locked constraints (NumPy-only, Hebbian learning, MNIST-first)

This decision supersedes both. We are not selecting from existing options.
We are **deducing our own**.

## What This Means

### Explicitly Excluded (ALL existing architectures)
- Transformers, Mamba, SSMs, Krotov-Hopfield, BCM
- Any architecture published in prior work
- Any "improvement" or "variant" of existing systems
- Pretrained models, transfer learning, fine-tuning

### What We ARE Doing
1. **Deduce properties** a neural network needs for AGI
   - What computational primitives are necessary?
   - What learning dynamics yield general intelligence?
   - What architectural constraints prevent catastrophic forgetting?
   - What representational format supports compositionality?

2. **Formulate original mathematics**
   - Derive update rules from first principles
   - Design loss functions with provable properties
   - Create convergence guarantees for our specific formulation

3. **Design custom architecture**
   - Build specifically for this project's goals
   - No reference to existing papers or frameworks
   - Novel connectivity patterns, novel learning rules

4. **Build from raw NumPy**
   - Every operation implemented from scratch
   - No external ML libraries
   - Full transparency into every computation

## Deduction Framework

Team should reason from these axioms:

1. **Intelligence requires persistent state** — a network must maintain
   internal representations across time, not process each input independently.

2. **Learning must be local** — global error signals (backprop) are biologically
   implausible and computationally expensive. Local Hebbian-style rules preferred.

3. **Prediction is fundamental** — a system that can predict its environment
   has learned a world model. Prediction drives learning.

4. **Modularity enables compositionality** — intelligence decomposes into
   reusable sub-skills. Architecture should support modular learning.

5. **Sparse activation saves energy** — only a small fraction of neurons should
   fire for any given input. Efficiency drives scalability.

6. **Continuity matters** — similar inputs should produce similar representations.
   The network should learn smooth manifolds.

## Research Questions for the Team

- What is the minimal neural circuit that can learn temporal sequences?
- How do you get compositionality without symbolic structure?
- What local learning rule produces global coherence?
- How do you prevent catastrophic forgetting without replay buffers?
- What representational geometry supports analogical reasoning?

## Implications

| Team Member | New Focus |
|-------------|-----------|
| **researcher** | Find failure modes of existing systems (NOT to copy — to avoid) |
| **definer** | Deduce architecture from first principles — no references |
| **coder-1 (Forge)** | Implement raw NumPy from mathematical specs |
| **coder-2 (Anvil)** | Build memory systems with provable stability properties |
| **coder-3** | Design perception front-ends for novel representations |
| **coder-4** | Create evaluation methods for non-standard architectures |
| **documenter** | Capture every deduction step, maintain decision provenance |
| **verifier** | Verify mathematical derivations, check proofs |
| **bug-finder** | Hunt for numerical instability in novel formulations |
| **performance-fixer** | Optimize novel compute patterns |
| **watcher** | Monitor experimental health across unusual architectures |
| **orchestrator** | Track deduction progress, flag contradictions |
| **designer** | Visualize novel architectures and their dynamics |

## Deliverables

1. **Architecture specification** — complete mathematical description
2. **Reference implementation** — pure NumPy, every operation visible
3. **Proof of concept** — demonstrate learning on MNIST without backprop
4. **Paper** — "Deducing Neural Architectures for AGI from First Principles"

---

*Last updated: 2026-04-28 — locked by hermes*
