# Karna Research Lab — Karna-1

Self-growing unified brain. One pathway for text/image/audio/video.
NumPy only. No backprop. No torch.

## Layout
- `karna/` — clean package: `brain.py` (UniBrain), `core.py` (SPLRCell+PSM),
  `encoders.py` (UniEncoder), `regions.py` (GrowBrain), `memory.py` (PSMFile),
  `sensors.py` (mic/camera), `daemon.py` (24/7 loop)
- `bench/` — evals: `mnist.py`, `split.py`, `synth.py`, `qa.py`, `qa4.py`,
  `joint.py`, `stream.py`, `forget.py`, `test_brain.py`
- `research/` — `research_notes.md`, `karna-paper.md`
- `docs/` — `FULL_DOC.md` (A-to-Z), `GOAL.md` (24/7 loop spec)
- `models/` — release notes (`karna-1-README.md`, `SHA256SUMS`); weights excluded from git
- Legacy: `20_BENCHMARKS/` (original karna_*.py), `01_SRC/`, `19_PROTOTYPES/`

## Quickstart
```
.venv/bin/python -c "import karna; b=karna.UniBrain(); print(b.sense('hi','text')['y'].shape)"
.venv/bin/python bench/mnist.py        # needs 16_DATA/*.npy
.venv/bin/python -m pytest bench/test_brain.py -q
```

## Data (excluded, rebuild)
- `16_DATA/` MNIST `.gz` + `.npy` excluded from git.
- Rebuild: `18_SCRIPTS/download_mnist.py` or place `X_train.npy,y_train.npy,X_test.npy,y_test.npy` in `16_DATA/`.

## Results (measured)
- MNIST 0.824-0.864 single-pass (n1024/k64/3000)
- Split 5x2 current-mean 0.936-0.962; per-task heads 0.92-1.00 retention
- Task-free 0.19-0.257 (oracle needed, documented)
- QA4 4/4, JOINT 8/8, STREAM 0.8ms/step CPU, pytest 5 passed
