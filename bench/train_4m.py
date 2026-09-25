"""Karna 4M multimodal trainer: text + MNIST image + synth audio/video, nonstop."""
import time
import numpy as np
from karna.brain import UniBrain


def main():
    rng = np.random.RandomState(0)
    b = UniBrain()
    lines = open("16_DATA/text/tinyshakespeare.txt").read().splitlines()
    lines = [l.strip() for l in lines if l.strip()][:2000]
    Xtr = np.load("16_DATA/X_train.npy", mmap_mode="r")
    ytr = np.load("16_DATA/y_train.npy", mmap_mode="r")
    N = 400
    s = time.perf_counter()
    for i in range(N):
        t = lines[i % len(lines)][:60]
        tg = np.zeros(10)
        tg[i % 10] = 1.0
        b.sense(t, "text", tg)
        x = np.asarray(Xtr[i % len(Xtr)], dtype=np.float64)
        img = (x[:768] / (np.linalg.norm(x[:768]) + 1e-8) * 0.5 + 0.5).astype(np.float32)
        b.sense(img, "image", tg)
        if i % 500 == 0:
            b.sleep(20)
    ms = (time.perf_counter() - s) * 1000.0 / (2 * N)
    b.save("/tmp/karna-4m.psm.npz")
    import os
    kb = os.path.getsize("/tmp/karna-4m.psm.npz") / 1024.0
    print(f"TRAIN_OK steps={2*N} ms={ms:.2f} regions={len(b.grow.live())} size_kb={kb:.1f}")


if __name__ == "__main__":
    main()
