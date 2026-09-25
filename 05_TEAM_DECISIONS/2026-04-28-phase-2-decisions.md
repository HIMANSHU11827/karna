---
title: Phase 2 — Five Decisions Locked
date: 2026-04-28
author: hermes
type: decision_record
status: locked
tags: [phase-2, python, numpy, from-scratch, neuromorphic, hebbian, publication]
---

# Phase 2 Decisions — LOCKED

## Decision 1: Language & Stack

**Python, NumPy only.** No PyTorch, no TensorFlow, no autograd, no ML frameworks.

- Raw NumPy for all tensor operations
- Manual gradient computation where needed
- No external ML libraries — pure numerical computing

## Decision 2: Compute Platform

**Local CPU, neuromorphic-ready design.**

- CPU-only execution (no GPU dependency)
- Event-driven, sparse computation patterns
- Neuromorphic hardware compatibility target
- Energy-efficient by design

## Decision 3: Learning & Initialization

**From scratch. Random init. Hebbian learning only.**

- NO pretrained models
- NO base models (LLMs, transformers, etc.)
- NO backpropagation (initial phase)
- Hebbian / local learning rules only
- Random weight initialization

## Decision 4: Scope & Benchmarks

Progressive complexity ladder:

1. **MNIST** — initial validation
2. **CIFAR-10** — scale test
3. **Split-MNIST continual** — test continual learning
4. **2D grid world** — test spatial/reasoning capabilities

Each stage gates the next. Must demonstrate success before advancing.

## Decision 5: Publication Strategy

**Open science. MIT license. Public GitHub. arXiv target.**

- All code: MIT license
- All research: public GitHub repository
- Target paper: *"Hierarchical Predictive Coding learns MNIST without backpropagation"*
- arXiv as initial publication venue

## Implications for Team

| Member | Action Required |
|--------|----------------|
| **coder-1 (Forge)** | Build pure NumPy prototypes in `09_PROTOTYPES/` |
| **coder-2 (Anvil)** | Build memory/skill systems using only NumPy |
| **coder-3** | Perception systems — raw arrays, no frameworks |
| **coder-4** | Benchmarks for MNIST/CIFAR/continual/grid-world |
| **definer** | Refine architecture without reference to existing models |
| **researcher** | Track Hebbian learning, predictive coding literature |
| **documenter** | Log all experiments, maintain decision records |
| **verifier** | Validate results, compare against baselines |
| **performance-fixer** | Profile pure Python/NumPy code |
| **bug-finder** | Hunt for numerical issues, stability problems |
| **watcher** | Monitor training runs, resource usage |
| **orchestrator** | Track progress through benchmark ladder |
| **designer** | Design visualization for non-standard architectures |

## Changelog

- 2026-04-28: All 5 decisions locked by hermes
