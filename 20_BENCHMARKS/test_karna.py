"""AJO pytest suite: core guarantees (no NaN, shapes, save/load, QA, split)."""
import numpy as np
from karna_brain import UniBrain
from karna_core import AJO1


def test_brain_smoke():
    b = UniBrain()
    r = b.sense("hello brain", "text")
    assert r["y"].shape == (10,)
    assert np.all(np.isfinite(r["y"]))


def test_four_kinds():
    import numpy as np
    rng = np.random.RandomState(0)
    b = UniBrain()
    for d, k in [("hi", "text"), (rng.rand(16, 16), "image"),
                 (rng.randn(1000).astype(np.float32), "audio"),
                 (rng.rand(2, 8, 8).astype(np.float32), "video")]:
        r = b.sense(d, k)
        assert r["regions"] >= 4


def test_save_load():
    b = UniBrain()
    for i in range(5):
        b.sense("t%d" % i, "text")
    b.save("/tmp/karna_pytest.npz")
    c = UniBrain()
    c.load("/tmp/karna_pytest.npz")
    assert c.t == 5


def test_compose_stable():
    b = UniBrain()
    ev = b.enc.encode("moon song", "text")
    seq = b.compose(np.asarray(ev["vector"], dtype=np.float64), steps=5)
    assert float(np.linalg.norm(seq[-1] - seq[0])) < 3.0


def test_mnist_floor():
    Xtr = np.load("../16_DATA/X_train.npy", mmap_mode="r")
    ytr = np.load("../16_DATA/y_train.npy", mmap_mode="r")
    Xte = np.load("../16_DATA/X_test.npy", mmap_mode="r")
    yte = np.load("../16_DATA/y_test.npy", mmap_mode="r")
    ajo = AJO1(in_dim=784, n=512, k=32, out_dim=10, seed=1)
    for i in range(500):
        x = np.asarray(Xtr[i], dtype=np.float64)
        t = np.zeros(10)
        t[int(ytr[i])] = 1.0
        ajo.step(x, t)
    ok = sum(1 for i in range(100) if int(np.argmax(ajo.step(np.asarray(Xte[i], dtype=np.float64))["y"])) == int(yte[i]))
    assert ok / 100 > 0.3
