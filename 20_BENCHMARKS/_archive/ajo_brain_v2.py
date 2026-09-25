"""AJO UniBrain v2: self-growing regions + 4 specialists + shared understanding."""
import time
import numpy as np
from ajo_uni import UniEncoder
from ajo_grow import GrowBrain
from ajo1 import SPLRCell, PSM
from ajo_vision import VisionSpecialist
from ajo_audio import AudioSpecialist
from ajo_text import TextSpecialist
from ajo_video import VideoSpecialist


class UniBrainV2:
    def __init__(self, dim=512, n=512, k=32, out_dim=10, seed=1):
        self.enc = UniEncoder(dim=dim, k=64, seed=seed)
        self.grow = GrowBrain(dim=64, r0=6, rmax=16, seed=seed)
        self.cell = SPLRCell(n=n, k=k, in_dim=dim, seed=seed)
        self.psm = PSM(out_dim, n)
        self.vision = VisionSpecialist()
        self.audio = AudioSpecialist()
        self.text = TextSpecialist()
        self.video = VideoSpecialist()
        self.t = 0

    def _special(self, data, kind):
        if kind == "image":
            return self.vision.encode(data)["embedding"]
        if kind == "audio":
            return self.audio.encode(data)["embedding"]
        if kind == "text":
            return self.text.encode(data)["embedding"]
        return self.video.encode(data)["embedding"]

    def sense(self, data, kind, target=None):
        self.t += 1
        ev = self.enc.encode(data, kind)
        uv = np.asarray(ev["vector"], dtype=np.float64)
        sv = np.asarray(self._special(data, kind), dtype=np.float64)
        sv = np.resize(sv, 64)
        sv /= np.linalg.norm(sv) + 1e-8
        g = self.grow.step(sv)
        h, err, z = self.cell.step(uv)
        y = self.psm.act(h, dense=z)
        M = 0.0
        if target is not None:
            M = self.psm.learn(h, target - y, target, dense=z)
            y = self.psm.act(h, dense=z)
        return {"y": y, "M": M, "active": int(h.sum()), "winner": g["winner"],
                "regions": g["regions"], "kind": kind, "sal": ev["salience"]}


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    b = UniBrainV2()
    kinds = [("hello brain learns", "text"), (rng.rand(48, 48), "image"),
             (rng.randn(8000).astype(np.float32), "audio"),
             (rng.rand(3, 32, 32).astype(np.float32), "video")]
    s = time.perf_counter()
    for data, kind in kinds * 5:
        r = b.sense(data, kind)
    ms = (time.perf_counter() - s) * 1000.0 / 20.0
    print(f"V2_OK steps=20 kinds=4 regions={r['regions']} winner={r['winner']} ms={ms:.2f} active={r['active']}")
