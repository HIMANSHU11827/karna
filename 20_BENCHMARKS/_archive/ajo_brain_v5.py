"""AJO UniBrain v5: need-driven self-growth. No hand specialists. Brain grows own."""
import time
import numpy as np
from ajo_uni import UniEncoder
from ajo_grow import GrowBrain
from ajo1 import SPLRCell, PSM


class UniBrainV5:
    def __init__(self, dim=512, n=512, k=32, out_dim=10, seed=1):
        self.enc = UniEncoder(dim=dim, k=64, seed=seed)
        self.grow = GrowBrain(dim=64, r0=4, rmax=24, seed=seed)
        self.cell = SPLRCell(n=n, k=k, in_dim=dim, seed=seed)
        self.psm = PSM(out_dim, n)
        self.t = 0

    def sense(self, data, kind, target=None):
        self.t += 1
        ev = self.enc.encode(data, kind)
        uv = np.asarray(ev["vector"], dtype=np.float64)
        key = np.resize(uv, 64)
        key /= np.linalg.norm(key) + 1e-8
        g = self.grow.step(key)
        conf = float(g["w"].max())
        h, err, z = self.cell.step(uv * (0.5 + 0.5 * conf))
        z_g = z * (0.7 + 0.3 * conf)
        y = self.psm.act(h, dense=z_g)
        M = 0.0
        if target is not None:
            M = self.psm.learn(h, (target - y) * (0.5 + conf), target, dense=z_g)
            y = self.psm.act(h, dense=z_g)
        return {"y": y, "M": M, "active": int(h.sum()), "winner": g["winner"],
                "regions": g["regions"], "conf": round(conf, 3), "kind": kind}


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    b = UniBrainV5()
    poems = ["moon soft night song", "river flows silent deep", "stars whisper dreams aloft"]
    others = [rng.rand(48, 48), rng.randn(8000).astype(np.float32)]
    s = time.perf_counter()
    for ep in range(5):
        for i, p in enumerate(poems):
            tg = np.zeros(10)
            tg[i] = 1.0
            b.sense(p, "text", tg)
        for d, k in [(others[0], "image"), (others[1], "audio")]:
            b.sense(d, k)
    ms = (time.perf_counter() - s) * 1000.0 / 25.0
    ok = sum(1 for i, p in enumerate(poems) if int(np.argmax(b.sense(p, "text")["y"])) == i)
    print(f"V5_OK steps=25 regions={b.grow.live().__len__()} poem_recall={ok}/3 ms={ms:.2f}")
