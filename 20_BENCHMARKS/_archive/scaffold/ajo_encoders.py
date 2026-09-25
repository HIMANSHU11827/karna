"""AJO sparse encoders for text, image, audio and video.

Each encoder returns a dict with keys vector, t, salience and source.
Vectors are sparse float32 arrays. t is a unix timestamp. salience is a
float in [0, 1]. source names the modality.
"""
import hashlib
import time

import numpy as np


def _now(t=None):
    return time.time() if t is None else float(t)


def _hash_idx(token, dim):
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dim


def encode_text(text, dim=1024, n=3, t=None):
    s = str(text).lower()
    grams = [s[i:i + n] for i in range(max(len(s) - n + 1, 1))]
    if len(s) < n:
        grams = [s]
    vec = np.zeros(dim, dtype=np.float32)
    for g in grams:
        vec[_hash_idx(g, dim)] = 1.0
    active = float(vec.sum())
    sal = min(1.0, active / 64.0)
    return {"vector": vec, "t": _now(t), "salience": float(sal), "source": "text"}


def _to_gray(img):
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return a


def _block16(gray):
    h, w = gray.shape
    out = np.zeros((16, 16), dtype=np.float32)
    rh = h / 16.0
    cw = w / 16.0
    for i in range(16):
        r0 = int(i * rh)
        r1 = max(r0 + 1, int((i + 1) * rh))
        for j in range(16):
            c0 = int(j * cw)
            c1 = max(c0 + 1, int((j + 1) * cw))
            out[i, j] = gray[r0:r1, c0:c1].mean()
    return out


def encode_image(img, k=64, t=None):
    small = _block16(_to_gray(img))
    flat = small.reshape(-1)
    mx = float(flat.max()) if flat.size else 0.0
    vec = np.zeros(256, dtype=np.float32)
    if mx > 0:
        idx = np.argsort(flat)[-int(max(min(k, 256), 1)):]
        vec[idx] = 1.0
        sal = float(flat[idx].mean() / (mx + 1e-8))
    else:
        sal = 0.0
    return {"vector": vec, "t": _now(t), "salience": float(sal), "source": "image"}


def encode_audio(wave, sr=16000, frame=256, dim=128, t=None):
    x = np.asarray(wave, dtype=np.float32).reshape(-1)
    nfr = int(x.size // frame)
    vec = np.zeros(dim, dtype=np.float32)
    if nfr < 1:
        return {"vector": vec, "t": _now(t), "salience": 0.0, "source": "audio"}
    fr = x[:nfr * frame].reshape(nfr, frame)
    energy = (fr * fr).mean(axis=1)
    zc = (np.abs(np.diff(np.sign(fr), axis=1)) > 0).mean(axis=1)
    en = energy / (energy.max() + 1e-8)
    zn = zc / (zc.max() + 1e-8)
    score = 0.5 * (en + zn)
    thr = float(np.median(score))
    idx = (np.arange(dim) * nfr // dim)
    vec = (score[idx] > thr).astype(np.float32)
    return {"vector": vec, "t": _now(t), "salience": float(score.mean()), "source": "audio"}


def encode_video(frames, k=64, t=None):
    arr = np.asarray(frames, dtype=np.float32)
    vec = np.zeros(256, dtype=np.float32)
    if arr.shape[0] < 2:
        return {"vector": vec, "t": _now(t), "salience": 0.0, "source": "video"}
    small = np.stack([_block16(_to_gray(f)) for f in arr], axis=0)
    diff = np.abs(np.diff(small, axis=0)).mean(axis=0)
    flat = diff.reshape(-1)
    mx = float(flat.max()) if flat.size else 0.0
    if mx > 0:
        idx = np.argsort(flat)[-int(max(min(k, 256), 1)):]
        vec[idx] = 1.0
        sal = float(flat[idx].mean() / (mx + 1e-8))
    else:
        sal = 0.0
    return {"vector": vec, "t": _now(t), "salience": float(sal), "source": "video"}


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    rt = encode_text("hello world ajo encoders")
    ri = encode_image(rng.rand(64, 48))
    ra = encode_audio(rng.randn(16000).astype(np.float32))
    rv = encode_video(rng.rand(4, 32, 32).astype(np.float32))
    print(rt["vector"].shape, rt["t"], rt["salience"], rt["source"])
    print(ri["vector"].shape, ri["t"], ri["salience"], ri["source"])
    print(ra["vector"].shape, ra["t"], ra["salience"], ra["source"])
    print(rv["vector"].shape, rv["t"], rv["salience"], rv["source"])
