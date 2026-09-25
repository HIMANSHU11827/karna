"""AJO UniBrain FINAL: self-grow primary. Specialists removed (scaffold legacy in _archive helpers)."""
import time
import numpy as np
from karna.encoders import UniEncoder
from karna.regions import GrowBrain
from karna.core import SPLRCell, PSM


class UniBrain:
    def __init__(self, dim=1024, n=2496, k=156, out_dim=10, seed=1):
        self.enc = UniEncoder(dim=dim, k=128, seed=seed)
        self.grow = GrowBrain(dim=64, r0=8, rmax=32, seed=seed)
        self.cell = SPLRCell(n=n, k=k, in_dim=dim, seed=seed)
        self.psm = PSM(out_dim, n)
        self.t = 0

    def _bias(self, data, kind):
        return None

    def _infer_task(self, z_g, h, task):
        if task is not None:
            return task
        if not self.psm.heads:
            return None
        if not hasattr(self, "_task_proto"):
            self._task_proto = {}
        zn = z_g / (float(np.linalg.norm(z_g)) + 1e-8)
        best, bi = -1e18, None
        for k, W in self.psm.heads.items():
            p = self._task_proto.get(k)
            if p is None:
                s = float(np.max(W @ zn))
            else:
                pn = p / (float(np.linalg.norm(p)) + 1e-8)
                head_score = float(np.max(W @ zn)) / (float(np.linalg.norm(W)) + 1e-8)
                proto_score = float(zn @ pn)
                s = 0.5 * head_score + 0.5 * proto_score
            if s > best:
                best, bi = s, k
        return bi

    def _update_task_proto(self, z_g, task):
        if task is None:
            return
        if not hasattr(self, "_task_proto"):
            self._task_proto = {}
        zn = z_g / (float(np.linalg.norm(z_g)) + 1e-8)
        if task in self._task_proto:
            self._task_proto[task] = 0.999 * self._task_proto[task] + 0.001 * zn
        else:
            self._task_proto[task] = zn.copy()

    def sense(self, data, kind, target=None, reward=0.0, novelty=0.0, task=None):
        self.t += 1
        ev = self.enc.encode(data, kind)
        uv = np.asarray(ev["vector"], dtype=np.float64)
        key = np.resize(uv, 64)
        key /= np.linalg.norm(key) + 1e-8
        g = self.grow.step(key, reward=reward, novelty=novelty)
        conf = float(g["w"].max())
        h, err, z = self.cell.step(uv * (0.5 + 0.5 * conf))
        z_g = z * (0.7 + 0.3 * conf)
        auto = self._infer_task(z_g, h, task)
        y = self.psm.act(h, dense=z_g, task=auto if self.psm.heads else None)
        M = 0.0
        if target is not None:
            M = self.psm.learn(h, (target - y) * (0.5 + conf), target, dense=z_g, task=auto)
            y = self.psm.act(h, dense=z_g, task=auto if self.psm.heads else None)
            self._update_task_proto(z_g, auto)
        if self.t % 500 == 0:
            self.cell.consolidate()
        return {"y": y, "M": M, "active": int(h.sum()), "winner": g["winner"],
                "regions": g["regions"], "conf": round(conf, 3),
                "drive": g.get("drive", 0.0), "kind": kind}

    def _tok_vec(self, tok_id):
        vec = np.zeros(self.cell.D.shape[1])
        vec[tok_id % self.cell.D.shape[1]] = 1.0
        return vec

    def train_text(self, sentence, vocab, eta=0.02):
        from karna.seq_head import SeqHead
        if not hasattr(self, "seq"):
            self.seq = SeqHead(len(vocab), self.cell.n, seed=1)
            self.seq_vocab = vocab
        ids = vocab.encode(sentence)
        if len(ids) < 2:
            return {"updates": 0}
        ups = 0
        for t in range(len(ids) - 1):
            vec = self._tok_vec(ids[t])
            _, _, z = self.cell.step(vec * 0.3)
            y = np.zeros(len(vocab))
            y[ids[t + 1]] = 1.0
            ups += self.seq.step_train(z, y, eta=eta)
        return {"updates": ups}

    def generate_tokens(self, seed_text, vocab=None, max_len=16, temp=1.0):
        from karna.seq_head import SeqHead
        vocab = vocab or getattr(self, "seq_vocab", None)
        if vocab is None or not hasattr(self, "seq"):
            return self.enc.generate(seed_text, brain=self, steps=4, top=6)
        ids = vocab.encode(seed_text)
        out = []
        for _ in range(max_len):
            tok = ids[-1] if ids else 1
            _, _, z = self.cell.step(self._tok_vec(tok))
            lin = self.seq.logits(z) / max(temp, 1e-6)
            nxt = int(np.argmax(lin))
            if nxt == 2:
                break
            out.append(nxt)
            ids = [nxt]
        return vocab.decode(out)

    def sleep(self, steps=50):
        n = min(len(self.psm.mem_h), steps)
        if n == 0:
            return {"replayed": 0}
        idx = np.random.choice(len(self.psm.mem_h), n, replace=False)
        for i in idx:
            h = self.psm.mem_h[i]
            t = self.psm.mem_y[i]
            z = (h - self.psm.mu) / (self.psm.sd + 1e-8)
            e = np.clip(t - (self.psm.Ws @ z), -2.0, 2.0)
            self.psm.Ws += 0.01 * np.outer(e, z)
        self.psm.Ws /= np.maximum(1, np.linalg.norm(self.psm.Ws, axis=1, keepdims=True) / 2.0)
        self.cell.consolidate(strength=0.05)
        return {"replayed": n}

    def compose(self, seed_vec, steps=8, temp=1.0, alpha=0.2):
        x0 = np.asarray(seed_vec, dtype=np.float64).reshape(-1)
        if x0.shape[0] != self.cell.D.shape[1]:
            x0 = x0[:self.cell.D.shape[1]]
        x0 /= np.linalg.norm(x0) + 1e-8
        W = self.cell.D @ self.cell.D.T
        W = (W + W.T) / 2.0
        np.fill_diagonal(W, 0.0)
        z = self.cell.D @ x0
        best, best_e = z.copy(), 1e18
        out = [np.zeros(self.enc.dim)]
        out[0][:x0.shape[0]] = x0
        for _ in range(steps):
            z_new = (1 - alpha) * z + alpha * np.tanh(W @ z + self.cell.D @ x0)
            z_new /= np.linalg.norm(z_new) + 1e-8
            e = float(np.sum((z_new - self.cell.D @ x0) ** 2))
            if e < best_e:
                best, best_e = z_new.copy(), e
            if float(np.sum((z_new - z) ** 2)) < 1e-6:
                z = z_new
                break
            z = z_new
            xh = np.tanh(self.cell.D.T @ z)
            full = np.zeros(self.enc.dim)
            full[:xh.shape[0]] = xh / (np.linalg.norm(xh) + 1e-8)
            out.append(full)
        return out

    def save(self, path):
        import hashlib
        import json
        import os
        import tempfile
        self.psm.save(path)
        n = len(self.grow.regions)
        pad = max(0, self.grow.rmax - n)
        mus = np.stack([r.mu for r in self.grow.regions] + [np.zeros(self.grow.dim)] * pad)
        meta = {"arch": "UniBrain", "ver": 1, "n": n, "rmax": self.grow.rmax,
                "dim": self.enc.dim, "k": self.enc.k, "cell_n": self.cell.n,
                "cell_k": self.cell.k, "out_dim": self.psm.m, "seed_note": "encoder_P+seed0"}
        blob = json.dumps(meta, sort_keys=True).encode()
        meta_hash = hashlib.sha256(blob).hexdigest()
        tmp = path.replace(".npz", ".grow.tmp.npz")
        np.savez_compressed(tmp,
                            mus=mus,
                            biases=np.array([r.b for r in self.grow.regions] + [0.0] * pad),
                            levels=np.array([r.level for r in self.grow.regions] + [0] * pad),
                            actives=np.array([r.active for r in self.grow.regions] + [False] * pad),
                            n=np.asarray(n),
                            D=self.cell.D, h=self.cell.h, t=np.asarray(self.t),
                            meta_json=np.asarray(list(blob)),
                            meta_hash=np.asarray(meta_hash))
        os.replace(tmp, path.replace(".npz", ".grow.npz"))

    def load(self, path):
        self.psm.load(path)
        d = np.load(path.replace(".npz", ".grow.npz"), allow_pickle=False)
        n = int(d["n"])
        while len(self.grow.regions) < n:
            from karna.regions import Region
            self.grow.regions.append(Region(self.grow.dim, seed=len(self.grow.regions)))
        for r, mu, b, lv, ac in zip(self.grow.regions, d["mus"][:n], d["biases"][:n], d["levels"][:n], d["actives"][:n]):
            if len(mu) == len(r.mu):
                r.mu, r.b, r.level, r.active = np.asarray(mu), float(b), int(lv), bool(ac)
        if d["D"].shape == self.cell.D.shape:
            self.cell.D = np.asarray(d["D"])
            self.cell.h = np.asarray(d["h"])
        self.t = int(d["t"])


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    b = UniBrain()
    kinds = [("hello brain learns", "text"), (rng.rand(48, 48), "image"),
             (rng.randn(8000).astype(np.float32), "audio"),
             (rng.rand(3, 32, 32).astype(np.float32), "video")]
    s = time.perf_counter()
    for data, kind in kinds * 5:
        r = b.sense(data, kind)
    sl = b.sleep()
    ms = (time.perf_counter() - s) * 1000.0 / 20.0
    print(f"BRAIN_FINAL_OK steps=20 kinds=4 regions={r['regions']} winner={r['winner']} conf={r['conf']} drive={r['drive']} ms={ms:.2f} active={r['active']} replayed={sl['replayed']}")
