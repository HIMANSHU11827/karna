"""AJO1 benchmark: 500-step synthetic 8-class task, supervised every 3rd step."""
import time

import numpy as np

import karna_core


def main():
    rng = np.random.default_rng(0)
    protos = rng.normal(0, 1, (8, 32))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
    ajo = karna_core.AJO1(in_dim=32, n=256, k=8, out_dim=8)
    steps = 500
    preds = np.zeros(steps, dtype=int)
    labels = np.zeros(steps, dtype=int)
    lats = np.zeros(steps)
    acts = np.zeros(steps)
    for t in range(steps):
        c = int(rng.integers(0, 8))
        labels[t] = c
        x = protos[c] + 0.5 * rng.normal(0, 1, 32)
        target = np.zeros(8)
        target[c] = 1.0
        sup = target if t % 3 == 0 else None
        s = time.perf_counter()
        r = ajo.step(x, sup)
        lats[t] = (time.perf_counter() - s) * 1000.0
        preds[t] = int(np.argmax(r["y"]))
        acts[t] = float(r["active"])
    acc = float(np.mean(preds[-100:] == labels[-100:]))
    lat = float(np.mean(lats))
    spars = float(np.mean(acts) / 256.0)
    print(f"AJO1_BENCH acc={acc:.4f} latency={lat:.3f}ms sparsity={spars:.4f}")


if __name__ == "__main__":
    main()
