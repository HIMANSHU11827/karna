"""AJO fusion: bind STE events from text/image/audio/video into objects."""
import time
import numpy as np


class Binder:
    def __init__(self, dim=256, window_ms=80.0, max_obj=128):
        self.dim = dim
        self.window = window_ms / 1000.0
        self.max_obj = max_obj
        self.objs = []

    def _sim(self, a, b):
        na, nb = float(a @ a) ** 0.5 + 1e-8, float(b @ b) ** 0.5 + 1e-8
        return float(a @ b / (na * nb))

    def _proj(self, ev):
        v = np.asarray(ev["vector"], dtype=np.float64)
        if v.shape[0] == self.dim:
            return v / (np.linalg.norm(v) + 1e-8)
        rng = np.random.default_rng(abs(hash(ev["source"])) % (2 ** 31))
        P = rng.normal(0, 1.0 / v.shape[0], (self.dim, v.shape[0]))
        p = P @ v
        return p / (np.linalg.norm(p) + 1e-8)

    def ingest(self, events):
        now = time.time()
        bound = []
        for ev in events:
            p = self._proj(ev)
            sal = float(ev.get("salience", 0.5))
            best, bi = -1.0, -1
            for i, o in enumerate(self.objs):
                if abs(now - o["t"]) > self.window and abs(ev.get("t", now) - o["t"]) > self.window:
                    continue
                s = self._sim(p, o["vec"])
                if s > best:
                    best, bi = s, i
            if best > 0.5:
                o = self.objs[bi]
                o["vec"] = (o["vec"] + sal * p) / 2.0
                o["vec"] /= np.linalg.norm(o["vec"]) + 1e-8
                o["t"] = now
                o["sal"] = max(o["sal"], sal)
                o["mods"].add(ev["source"])
                bound.append(o["id"])
            else:
                oid = len(self.objs)
                self.objs.append({"id": oid, "vec": p, "t": now, "sal": sal, "mods": {ev["source"]}})
                bound.append(oid)
        self.objs.sort(key=lambda o: -o["sal"])
        self.objs = self.objs[:self.max_obj]
        return bound

    def state(self):
        if not self.objs:
            return np.zeros(self.dim)
        w = np.array([o["sal"] for o in self.objs])
        w /= w.sum() + 1e-8
        return sum(wi * o["vec"] for wi, o in zip(w, self.objs))


if __name__ == "__main__":
    from ajo_encoders import encode_text, encode_image, encode_audio, encode_video
    rng = np.random.RandomState(0)
    b = Binder()
    evs = [
        encode_text("buzzing circuit board"),
        encode_image(rng.rand(64, 48)),
        encode_audio(rng.randn(16000).astype(np.float32)),
        encode_video(rng.rand(4, 32, 32).astype(np.float32)),
    ]
    ids = b.ingest(evs)
    print(f"BINDER_OK objects={len(b.objs)} bound={ids} state_norm={np.linalg.norm(b.state()):.3f}")
