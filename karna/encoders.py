"""AJO uni-encoder: one pathway for text/image/audio/video -> shared STE."""
import hashlib
import time
import numpy as np


def _now(t=None):
    return time.time() if t is None else float(t)


def _tok_raw(tok, dim):
    seed = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    return rng.random(dim)


def _to_vec(raw, dim):
    a = np.asarray(raw, dtype=np.float64).reshape(-1)
    if a.shape[0] == dim:
        return a
    if a.shape[0] > dim:
        n = a.shape[0] // dim
        return a[:n * dim].reshape(dim, n).mean(axis=1)
    out = np.zeros(dim)
    out[:a.shape[0]] = a
    return out


def _bytes_of(data, kind):
    if kind == "text":
        return str(data).lower().encode("utf-8")
    a = np.asarray(data)
    if a.dtype != np.uint8:
        b = np.clip(a.reshape(-1), 0, 1) if a.size else np.zeros(1)
        q = (b * 255).astype(np.uint8)
        return q.tobytes()
    return a.tobytes()


class UniEncoder:
    def __init__(self, dim=512, k=64, seed=0):
        self.dim, self.k = dim, k
        rng = np.random.default_rng(seed)
        self.P = rng.normal(0, 1.0 / np.sqrt(dim), (dim, dim))
        self.bias = np.zeros(dim)

    def encode(self, data, kind, t=None):
        if kind == "text":
            words = str(data).lower().split() or [str(data).lower()]
            raw = np.mean([_tok_raw(w, self.dim) for w in words], axis=0)
        else:
            raw = _to_vec(np.frombuffer(_bytes_of(data, kind), dtype=np.uint8).astype(np.float64) / 255.0, self.dim)
        z = self.P @ (raw - 0.5) + self.bias
        idx = np.argpartition(z, -self.k)[-self.k:]
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[idx] = 1.0
        sal = float(min(1.0, np.linalg.norm(raw) + 0.1))
        return {"vector": vec, "t": _now(t), "salience": sal, "source": kind}

    def adapt(self, vec, eta=0.001):
        self.bias += eta * (np.asarray(vec).mean() - self.bias.mean())

    def decode(self, vec, gram=3, top=12, vocab=None, order=2, seed_vec=None, binary=True):
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if binary:
            act = set(np.where(v[:self.dim] > 0.5)[0].tolist())
            if not act:
                act = set(np.argpartition(v[:self.dim], -self.k)[-self.k:].tolist())
        else:
            act = set(np.where(v[:self.dim] > 0.5)[0].tolist())
        if vocab is None:
            vocab = ["the", "and", "moon", "soft", "night", "song", "star", "river",
                     "flow", "silent", "deep", "hello", "world", "brain", "red", "blue",
                     "light", "dark", "wind", "rain", "fire", "earth", "sky", "sea",
                     "dream", "whisper", "loud", "quiet", "fast", "slow", "bright", "wild"]
        tok_sets = {}
        for tok in vocab:
            raw = _tok_raw(tok, self.dim)
            z = self.P @ (raw - 0.5) + self.bias
            tok_sets[tok] = set(np.argpartition(z, -self.k)[-self.k:].tolist())
        scored = [(len(act & tk) / max(len(tk), 1), tok) for tok, tk in tok_sets.items()]
        scored.sort(reverse=True)
        top_toks = [t for _, t in scored[:max(top, order + 1)]]
        if seed_vec is not None:
            sv = np.asarray(seed_vec, dtype=np.float64).reshape(-1)
            sact = set(np.where(sv[:self.dim] > 0.5)[0].tolist())
            top_toks.sort(key=lambda t: -len(sact & tok_sets[t]))
        if order >= 2 and len(top_toks) >= 2:
            best, best_s = None, -1.0
            import itertools
            for perm in itertools.permutations(top_toks[:8], min(order + 1, 3)):
                combo = " ".join(perm)
                raw = _to_vec(np.frombuffer(combo.encode("utf-8"), dtype=np.uint8).astype(np.float64) / 255.0, self.dim)
                z = self.P @ (raw - 0.5) + self.bias
                ck = set(np.argpartition(z, -self.k)[-self.k:].tolist())
                s = len(act & ck) / max(len(ck), 1)
                if s > best_s:
                    best, best_s = combo, s
            return best
        return " ".join(t for _, t in scored[:top])

    def generate(self, seed_text, brain=None, steps=4, top=6):
        seed = self.encode(seed_text, "text")["vector"]
        sv = np.asarray(seed, dtype=np.float64)
        direct = self.decode(sv, top=top, order=2)
        if brain is not None and steps > 0:
            seq = brain.compose(sv, steps=steps)
            vec = seq[-1]
            direct_set = set(direct.split())
            scored = []
            for tok in ["the", "and", "moon", "soft", "night", "song", "star", "river",
                        "flow", "silent", "deep", "hello", "world", "brain", "red", "blue",
                        "light", "dark", "wind", "rain", "fire", "earth", "sky", "sea",
                        "dream", "whisper", "loud", "quiet", "fast", "slow", "bright", "wild"]:
                raw = _tok_raw(tok, self.dim)
                z = self.P @ (raw - 0.5) + self.bias
                tk = set(np.argpartition(z, -self.k)[-self.k:].tolist())
                ck = set(np.argpartition(vec[:self.dim], -self.k)[-self.k:].tolist())
                s = len(ck & tk) / max(len(tk), 1)
                scored.append((s, tok))
            scored.sort(reverse=True)
            out = " ".join(t for _, t in scored[:top])
            if set(out.split()) == direct_set:
                return direct
            return out
        return direct


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    e = UniEncoder()
    r1 = e.encode("buzzing circuit board", "text")
    r2 = e.encode(rng.rand(64, 48), "image")
    r3 = e.encode(rng.randn(16000).astype(np.float32), "audio")
    r4 = e.encode(rng.rand(4, 32, 32).astype(np.float32), "video")
    print(f"UNI_OK text={r1['vector'].sum():.0f} img={r2['vector'].sum():.0f} aud={r3['vector'].sum():.0f} vid={r4['vector'].sum():.0f} dim={r1['vector'].shape}")
