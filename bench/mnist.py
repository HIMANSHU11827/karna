"""Karna MNIST bench: fit() 2-pass batch-learn, 3000 train / 500 test."""
import time
import numpy as np
from karna.core import AJO1


def main():
    Xtr = np.load("16_DATA/X_train.npy", mmap_mode="r")
    ytr = np.load("16_DATA/y_train.npy", mmap_mode="r")
    Xte = np.load("16_DATA/X_test.npy", mmap_mode="r") if __import__("os").path.exists("16_DATA/X_test.npy") else None
    yte = None
    import os
    if os.path.exists("16_DATA/y_test.npy"):
        yte = np.load("16_DATA/y_test.npy", mmap_mode="r")
    n, k, NTR, NTE = 1024, 64, 3000, 500
    ajo = AJO1(in_dim=784, n=n, k=k, out_dim=10, seed=1)
    X = np.asarray(Xtr[:NTR], dtype=np.float64)
    Y = np.zeros((NTR, 10))
    Y[np.arange(NTR), np.asarray(ytr[:NTR]).astype(int)] = 1.0
    s = time.perf_counter()
    ajo.fit(X, Y, epochs=2)
    train_ms = (time.perf_counter() - s) * 1000.0 / NTR
    ajo.save("/tmp/karna_core_mnist.psm.npz")
    import os as _os
    size_kb = _os.path.getsize("/tmp/karna_core_mnist.psm.npz") / 1024.0
    ajo2 = AJO1(in_dim=784, n=n, k=k, out_dim=10, seed=1)
    ajo2.load("/tmp/karna_core_mnist.psm.npz")
    ok = 0
    for i in range(NTE):
        x = np.asarray(Xte[i], dtype=np.float64)
        r = ajo2.step(x)
        if int(np.argmax(r["y"])) == int(yte[i]):
            ok += 1
    acc = ok / NTE
    print(f"AJO1_MNIST acc={acc:.4f} train_ms={train_ms:.2f} size_kb={size_kb:.1f}")


if __name__ == "__main__":
    main()
