"""AJO self-growing brain: starts uniform, builds own regions+subs while learning."""
import time
import numpy as np


class Region:
    def __init__(self, dim, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        self.mu = rng.normal(0, 0.01, dim)
        self.W1 = rng.normal(0, np.sqrt(1.0 / dim) * 0.3, (dim, hidden))
        self.W2 = rng.normal(0, np.sqrt(1.0 / hidden) * 0.3, (hidden, dim))
        self.b = 0.0
        self.u = 0.0
        self.e = 1.0
        self.age = 0
        self.hot = 0
        self.idle = 0
        self.parent = -1
        self.level = 0
        self.active = True

    def err(self, x):
        h = np.maximum(0, (x - self.mu) @ self.W1)
        p = self.mu + h @ self.W2
        return float(np.sum((x - p) ** 2) + 0.3 * np.sum((x - self.mu) ** 2)), p

    def adapt(self, x, eta):
        h = np.maximum(0, (x - self.mu) @ self.W1)
        p = self.mu + h @ self.W2
        e = x - p
        self.mu += eta * (x - self.mu)
        g = (e @ self.W2.T) * (h > 0)
        self.W2 += eta * np.outer(h, e) * 0.1
        self.W1 += eta * np.outer(x - self.mu, np.zeros_like(h)) * 0.0 + eta * 0.1 * np.outer(x - self.mu, (g > 0).astype(float) * 0.1)


class GrowBrain:
    def __init__(self, dim=64, r0=6, rmax=16, seed=1):
        self.dim = dim
        self.rmax = rmax
        self.regions = [Region(dim, seed=seed + i) for i in range(r0)]
        self.t = 0

    def live(self):
        return [r for r in self.regions if r.active]

    def step(self, x, eta0=0.03, reward=0.0, novelty=0.0):
        self.t += 1
        live = self.live()
        errs = np.array([r.err(x)[0] for r in live])
        surprise = float(errs.min())
        drive = 1.0 + 0.5 * surprise + 0.5 * novelty + 0.5 * max(0.0, reward)
        scores = -errs / 0.5 - np.array([r.b for r in live])
        scores += np.random.randn(len(live)) * 0.02
        w = np.exp(scores - scores.max())
        w /= w.sum()
        for i, r in enumerate(live):
            eta = eta0 * w[i] * drive
            if w[i] < 0.01:
                r.mu -= 0.001 * (x - r.mu)
            else:
                r.adapt(x, eta)
            r.u = 0.99 * r.u + 0.01 * w[i]
            r.e = 0.98 * r.e + 0.02 * errs[i]
            r.b = float(np.clip(r.b + 0.05 * (w[i] - 1.0 / len(live)), -2, 2))
            r.age += 1
        self._grow()
        bi = int(np.argmax(w))
        return {"winner": bi, "regions": len(live), "err": float(errs[bi]), "w": w,
                "drive": round(drive, 3), "surprise": round(surprise, 3)}

    def _grow(self):
        live = self.live()
        if len(live) >= self.rmax:
            return
        es = np.array([r.e for r in live])
        us = np.array([r.u for r in live])
        for r in live:
            hot = (r.e > np.median(es) + 0.5 * es.std()) and (r.u > 1.5 / len(live))
            r.hot = r.hot + 1 if hot else max(0, r.hot - 2)
            if r.hot >= 200 and r.level < 2:
                self._split(r)
                break
        for r in live:
            idle = (r.u < 0.25 / len(live)) and (r.age > 1000)
            r.idle = r.idle + 1 if idle else 0

    def _split(self, r):
        if len(self.regions) >= self.rmax:
            return
        d = 0.05 * np.random.randn(self.dim)
        c = Region(self.dim, seed=self.t)
        c.mu = r.mu - d
        r.mu = r.mu + d
        c.W1, c.W2 = r.W1.copy(), r.W2.copy()
        c.parent = self.regions.index(r)
        c.level = r.level + 1
        c.u, c.e = r.u / 2, r.e
        r.u /= 2
        r.hot = 0
        self.regions.append(c)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    b = GrowBrain(dim=64, r0=6, rmax=16)
    kinds = [rng.normal(2.0, 0.5, 64), rng.normal(-2.0, 0.5, 64),
             rng.normal(0.0, 0.3, 64), rng.normal(0.0, 2.0, 64)]
    s = time.perf_counter()
    wins = np.zeros(16, dtype=int)
    for t in range(3000):
        x = kinds[t % 4] + rng.normal(0, 0.1, 64)
        r = b.step(x / (np.linalg.norm(x) + 1e-8))
        if r["winner"] < 16:
            wins[r["winner"]] += 1
    ms = (time.perf_counter() - s) * 1000.0 / 3000.0
    print(f"GROW_OK regions={r['regions']} ms={ms:.2f} wins={wins[:r['regions']].tolist()} err={r['err']:.3f}")
