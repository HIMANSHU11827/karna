"""Ventral-stream vision specialist (NumPy only): V1 edges -> V2 grouping -> V4 summary -> IT embedding."""

import numpy as np


class VisionSpecialist:
    """Hierarchical visual encoder mimicking V1/V2/V4/IT."""

    def __init__(self, n_filters=8, emb_dim=32, seed=0):
        rng = np.random.default_rng(seed)
        self.nf = n_filters
        self.emb_dim = emb_dim
        self.filters = rng.standard_normal((n_filters, 3, 3))
        self.proj = None
        self.seed = seed

    @staticmethod
    def _gray(img):
        img = np.asarray(img, dtype=np.float32) / 255.0
        return img.mean(axis=2) if img.ndim == 3 else img

    def v1(self, img):
        """Edge responses: depthwise 3x3 via stride trick, vectorized over filters."""
        g = self._gray(img)
        gp = np.pad(g, 1)
        s0, s1 = gp.strides
        H, W = g.shape
        win = np.lib.stride_tricks.as_strided(gp, shape=(H, W, 3, 3), strides=(s0, s1, s0, s1))
        F = np.stack(list(self.filters), axis=-1)
        return np.tanh(np.einsum("hwij,ij f->hwf", win, F))

    @staticmethod
    def v2(x):
        """Grouping: 2x2 max-pool over H,W (per channel)."""
        h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
        return x[:h, :w].reshape(h // 2, 2, w // 2, 2, -1).max(axis=(1, 3))

    @staticmethod
    def v4(img, v1map):
        """Color/curvature summary: channel stats + edge energy."""
        img = np.asarray(img, dtype=np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2)
        m, s = img.mean(axis=(0, 1)), img.std(axis=(0, 1))
        energy = np.abs(v1map).mean(axis=(0, 1))
        return np.concatenate([m, s, energy]).astype(np.float32)

    def it(self, pooled):
        """Object embedding: lazy random projection + tanh + L2 norm."""
        flat = pooled.ravel()
        if self.proj is None or self.proj.shape[0] != flat.shape[0]:
            rng = np.random.default_rng(self.seed)
            self.proj = rng.standard_normal((flat.shape[0], self.emb_dim)) / 32.0
        e = np.tanh(flat @ self.proj)
        return (e / (np.linalg.norm(e) + 1e-8)).astype(np.float32)

    def encode(self, img):
        """Encode image -> dict with embedding, parts, salience."""
        v1m = self.v1(img)
        pool = self.v2(v1m)
        summ = self.v4(img, v1m)
        emb = self.it(pool[:, :, :])
        sal = np.abs(v1m).mean(axis=2)
        sal = (sal - sal.min()) / (np.ptp(sal) + 1e-8)
        parts = {"v1": v1m, "v2": pool, "v4": summ, "it": emb}
        return {"embedding": emb, "parts": parts, "salience": sal.astype(np.float32)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = (rng.random((48, 64, 3)) * 255).astype(np.uint8)  # 64x48 (WxH)
    out = VisionSpecialist().encode(img)
    print("embedding:", out["embedding"].shape)
    for k, v in out["parts"].items():
        print(k, v.shape)
    print("salience:", out["salience"].shape)
