# Karna Research Lab — Karna-1

Self-growing unified brain. One pathway for text/image/audio/video.
NumPy only. No backprop. No torch.

## Project
- Brain: `20_BENCHMARKS/karna_brain.py` (UniBrain)
- Core: `karna_core.py` (SPLRCell + PSM), `karna_uni.py` (encoder),
  `karna_grow.py` (self-growing regions), `karna_psm.py` (weights format)
- Daemon: `karna_daemon.py` runs 24/7 (see `GOAL.md`)
- Docs: `20_BENCHMARKS/KARNA_FULL_DOC.md` (A-to-Z), `00_PAPER/paper_outline.md`
- Model: `20_BENCHMARKS/releases/karna-1/` (excluded from git, rebuild below)

## Quickstart
```
.venv/bin/python 20_BENCHMARKS/karna_brain.py
.venv/bin/python 20_BENCHMARKS/karna_mnist.py   # needs 16_DATA/*.npy
.venv/bin/python -m pytest 20_BENCHMARKS/test_karna.py -q
```

## Data (excluded, rebuild)
- `16_DATA/` MNIST `.gz` + `.npy` excluded from git (431MB).
- Rebuild: `18_SCRIPTS/download_mnist.py` or place `X_train.npy,y_train.npy,X_test.npy,y_test.npy` in `16_DATA/`.
- Releases excluded: rebuild via `karna_brain.py` save path in `KARNA_FULL_DOC.md`.

## Results (measured)
- MNIST 0.824-0.864 single-pass (n1024/k64/3000)
- Split 5x2 current-mean 0.936-0.962; per-task heads 0.92-1.00 retention
- Task-free 0.19-0.257 (oracle needed, documented)
- QA4 4/4, JOINT 8/8, STREAM 0.8ms/step CPU, pytest 5 passed
