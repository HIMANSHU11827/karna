# Karna-4M Model Release (Karna Research Lab)

4.26M params. NumPy only. No backprop. Multimodal text+image trained.

## Contents (`models/karna-4m/`, 9.6MB)
- `brain.psm.npz` (137KB) — Ws,Wf,F + checksum
- `brain.psm.norm.npz` (24KB) — mu,sd
- `brain.psm.grow.npz` (5MB) — regions mus/biases + cell D/h + t
- `encoder_P.npy` (4.7MB) [768,768] + `encoder_bias.npy` (6KB)
- Train: 400 Shakespeare lines + 400 MNIST images, 800 steps, sleep included

## Arch
`bytes->UniEncoder768(k96) -> GrowBrain64(r0=8,rmax=32) -> SPLRCell(n1536/k96) -> PSM(out10)`

## Eval (measured)
- Tokens: unbounded streaming O(1), 6-8ms/step, 129 toks/s CPU, regions=8
- QA4 4/4, SPEECH 2/3, VIDEO 2/2, pytest 5 passed
- Reload t:800 regions:8 infer ok
- MNIST single-pass cap ~0.84 (4M capacity not yet the cap)

## Limits
- Task-free retention needs oracle; generation = word-overlap heuristic
- 400-sample train only (CPU-capped); scale to 2000+ with faster box
