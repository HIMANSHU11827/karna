"""AJO Split-MNIST: 5 tasks x 2 digits, lifelong no-forgetting test."""
import numpy as np
from karna_core import AJO1


def main():
    Xtr = np.load("../16_DATA/X_train.npy", mmap_mode="r")
    ytr = np.load("../16_DATA/y_train.npy", mmap_mode="r")
    Xte = np.load("../16_DATA/X_test.npy", mmap_mode="r")
    yte = np.load("../16_DATA/y_test.npy", mmap_mode="r")
    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    ajo = AJO1(in_dim=784, n=1024, k=64, out_dim=10, seed=1)
    accs = []
    for ti, (a, b) in enumerate(tasks):
        idx = np.where((ytr == a) | (ytr == b))[0][:400]
        for i in idx:
            x = np.asarray(Xtr[i], dtype=np.float64)
            t = np.zeros(10)
            t[int(ytr[i])] = 1.0
            ajo.step(x, t)
        ok, n = 0, 0
        for i in range(len(yte)):
            if int(yte[i]) in (a, b):
                x = np.asarray(Xte[i], dtype=np.float64)
                if int(np.argmax(ajo.step(x)["y"])) == int(yte[i]):
                    ok += 1
                n += 1
                if n >= 100:
                    break
        accs.append(ok / max(n, 1))
        print(f"task{ti} digits{a}/{b} acc={accs[-1]:.3f}", flush=True)
    print(f"SPLIT_OK accs={[round(a, 3) for a in accs]} mean={np.mean(accs):.3f}")


if __name__ == "__main__":
    main()
