# Karna-1 Model Release (Karna Research Lab)

NumPy-only self-growing unified brain. No backprop. No torch.

## Contents
- `brain.psm.npz` — Ws,Wf,F,dims,clock,checksum (PSMFile)
- `brain.psm.norm.npz` — mu,sd running norm (required)
- `brain.psm.grow.npz` — mus,biases,levels,actives,n,D,h,t,meta (GrowBrain+cell)
- `encoder_P.npy` [512,512] float64, seed0, scale 1/sqrt(dim), center raw-0.5
- `encoder_bias.npy` [512]
- Code: `ajo_brain.py,ajo1.py,ajo_uni.py,ajo_grow.py,ajo_psm.py` (canonical)

## Arch
`bytes->UniEncoder512(k64) -> GrowBrain64(r0=4,rmax=24) -> SPLRCell(n512/k32) -> PSM(out10)`
Formulas: cell `z=D@x,y=kWTA(z+h)`; router `softmax(-err/T-b)`; readout mistake-only
`Ws[true]+=eta*P*|feat|`, `P=1/(1+lF)`; replay-10 stratified; per-task heads optional.

## Install
`.venv` + NumPy 2.5.3. `../.venv/bin/python -c "import numpy;print(numpy.__version__)"`

## Quickstart
```
cd 20_BENCHMARKS
../.venv/bin/python ajo_brain.py        # BRAIN_FINAL_OK
../.venv/bin/python ajo_mnist.py        # AJO1_MNIST ~0.84 (needs ../16_DATA/*.npy)
../.venv/bin/python ajo_qa4.py          # QA4_OK 4/4
../.venv/bin/python -m pytest test_ajo.py -q   # 5 passed
```

## Eval (measured)
- MNIST 3000/500 save→load: 0.824-0.864 (single-pass perceptron cap ~0.86)
- Split 5x2x400 current-mean: 0.936-0.962; retention with task heads 0.92-1.00
- Task-free retention: 0.19-0.257 (routing!=tasking) — needs task oracle
- QA cross 4/4, JOINT 8/8, STREAM 0.8ms/step CPU, compose drift 7.587→0.969 attractor
- Daemon 300k+ steps, ckpt loadable, heartbeat <60s

## Limits (honest)
- Not 95%+ MNIST. Single-pass cap. n2048 worse (0.736).
- Generation = retrieval >> compose. Decode = 16-word overlap heuristic.
- Sensors need arecord/ffmpeg+/dev/video0 or synthetic fallback (live flag).
- Task-free lifelong open. Use task=ID for zero forgetting.
