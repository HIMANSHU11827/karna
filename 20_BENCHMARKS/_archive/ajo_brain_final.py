"""AJO UniBrain FINAL: merged v4/v5/v6. Self-grow primary + specialists bias + sleep replay."""
import time
import numpy as np
from ajo_uni import UniEncoder
from ajo_grow import GrowBrain
from ajo1 import SPLRCell, PSM
from ajo_vision import VisionSpecialist
from ajo_audio import AudioSpecialist
from ajo_text import TextSpecialist
from ajo_video import VideoSpecialist


class UniBrainFinal:
    def __init__(self, dim=512, n=512, k=32, out_dim=10, seed=1):
        self.enc = UniEncoder(dim=dim, k=64, seed=seed)
        self.grow = GrowBrain(dim=64, r0=4, rmax=24, seed=seed)
        self.cell = SPLRCell(n=n, k=k, in_dim=dim, seed=seed)
        self.psm = PSM(out_dim, n)
        self.vision = VisionSpecialist()
        self.audio = AudioSpecialist()
        self.text = TextSpecialist()
        self.video = VideoSpecialist()
        self.t = 0

    def _bias(self, data, kind):
        if kind == "image":
            e = self.vision.encode(data)["embedding"]
        elif kind == "audio":
            e = self.audio.encode(data)["embedding"]
        elif kind == "text":
            e = self.text.encode(data)["embedding"]
        else:
            e = self.video.encode(data)["embedding"]
        v = np.resize(np.asarray(e, dtype=np.float64), 64)
        return v / (np.linalg.norm(v) + 1e-8)

    def sense(self, data, kind, target=None, reward=0.0, novelty=0.0):
        self.t += 1
        ev = self.enc.encode(data, kind)
        uv = np.asarray(ev["vector"], dtype=np.float64)
        bias = self._bias(data, kind)
        key = np.resize(uv, 64)
        key /= np.linalg.norm(key) + 1e-8
        fused = 0.7 * key + 0.3 * bias
        g = self.grow.step(fused, reward=reward, novelty=novelty)
        conf = float(g["w"].max())
        h, err, z = self.cell.step(uv * (0.5 + 0.5 * conf))
        z_g = z * (0.7 + 0.3 * conf)
        y = self.psm.act(h, dense=z_g)
        M = 0.0
        if target is not None:
            M = self.psm.learn(h, (target - y) * (0.5 + conf), target, dense=z_g)
            y = self.psm.act(h, dense=z_g)
        if self.t % 500 == 0:
            self.cell.consolidate()
        return {"y": y, "M": M, "active": int(h.sum()), "winner": g["winner"],
                "regions": g["regions"], "conf": round(conf, 3),
                "drive": g.get("drive", 0.0), "kind": kind}

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


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    b = UniBrainFinal()
    kinds = [("hello brain learns", "text"), (rng.rand(48, 48), "image"),
             (rng.randn(8000).astype(np.float32), "audio"),
             (rng.rand(3, 32, 32).astype(np.float32), "video")]
    s = time.perf_counter()
    for data, kind in kinds * 5:
        r = b.sense(data, kind)
    sl = b.sleep()
    ms = (time.perf_counter() - s) * 1000.0 / 20.0
    print(f"FINAL_OK steps=20 kinds=4 regions={r['regions']} winner={r['winner']} conf={r['conf']} drive={r['drive']} ms={ms:.2f} active={r['active']} replayed={sl['replayed']}")
