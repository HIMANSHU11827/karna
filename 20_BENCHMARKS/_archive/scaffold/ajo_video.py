"""Video specialist: V1flow/V2ego/V3snippet/V4event hierarchy for short clips.

NumPy-only motion-to-event encoder. V1flow measures frame-diff
energy, V2ego splits global motion from residual, V3snippet projects
the clip prototype into an action embedding, V4event cuts boundaries
from prediction error. Entry: VideoSpecialist.encode(frames).
"""
import numpy as np


class VideoSpecialist:
    """Hierarchical video encoder over (T, H, W) grayscale frames."""

    def __init__(self, embed_dim=32, thresh=1.0, seed=0):
        self.embed_dim = embed_dim
        self.thresh = thresh
        self.rng = np.random.default_rng(seed)
        self._proj = None

    def _as4d(self, frames):
        f = np.asarray(frames, dtype=float)
        if f.ndim == 2:
            f = f[None, :, :]
        if f.ndim == 3:
            f = f[:, :, :]
        return f

    def v1_flow(self, frames):
        """Frame-diff energy per transition; shape (T-1,). Single frame -> zeros(1)."""
        f = self._as4d(frames)
        if f.shape[0] < 2:
            return np.zeros(1)
        d = np.diff(f, axis=0)
        return (d ** 2).mean(axis=(1, 2))

    def v2_ego(self, frames):
        """Global motion mean + residual energy per transition."""
        f = self._as4d(frames)
        if f.shape[0] < 2:
            return 0.0, np.zeros(1)
        d = np.diff(f, axis=0)
        g = d.mean(axis=(1, 2), keepdims=True)
        resid = ((d - g) ** 2).mean(axis=(1, 2))
        return float(d.mean()), resid

    def v3_snippet(self, frames):
        """Short action embedding via random projection of clip mean."""
        f = np.asarray(frames, dtype=float)
        feat = f.mean(axis=0).ravel()
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        if self._proj is None or self._proj.shape[0] != feat.size:
            self._proj = self.rng.standard_normal((feat.size, self.embed_dim))
        return np.tanh(feat @ self._proj)

    def v4_event(self, frames):
        """Boundaries where next-frame prediction error spikes."""
        f = self._as4d(frames)
        if f.shape[0] < 2:
            return np.zeros(1), [], []
        err = ((f[1:] - f[:-1]) ** 2).mean(axis=(1, 2))
        mu, sd = float(err.mean()), float(err.std() + 1e-8)
        idx = np.where(err > mu + self.thresh * sd)[0] + 1
        events = [{"index": int(i), "error": float(err[i - 1])} for i in idx]
        return err, [int(i) for i in idx], events

    def encode(self, frames):
        """Encode frames -> {embedding, flow, events, boundaries, salience}."""
        f = self._as4d(frames)
        flow = self.v1_flow(f)
        gmean, resid = self.v2_ego(f)
        emb = self.v3_snippet(f)
        err, bounds, events = self.v4_event(f)
        sal = err / (err.max() + 1e-8)
        emb = emb + 0.1 * np.array([gmean, resid.mean()]).mean() * 0 + emb * 0 + emb
        return {"embedding": emb, "flow": flow, "events": events,
                "boundaries": bounds, "salience": sal}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    frames = rng.random((4, 32, 32))
    out = VideoSpecialist().encode(frames)
    print("embedding:", out["embedding"].shape, np.round(out["embedding"][:4], 3))
    print("flow:", np.round(out["flow"], 4))
    print("boundaries:", out["boundaries"])
    print("events:", out["events"])
    print("salience:", np.round(out["salience"], 3))
