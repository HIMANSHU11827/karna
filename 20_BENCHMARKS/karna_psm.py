"""PSM file container with int8 per-row quantization helpers."""
import numpy as np


def quantize_int8(W):
    """Quantize 2D array per row to int8 with float scales."""
    W = np.asarray(W, dtype=np.float64)
    peak = np.max(np.abs(W), axis=1)
    scale = np.where(peak == 0, 1.0, peak / 127.0)
    q = np.clip(np.round(W / scale[:, None]), -128, 127).astype(np.int8)
    return q, scale.astype(np.float64)


def dequantize(q, scale):
    """Dequantize int8 array per row using scales."""
    q = np.asarray(q, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    return q * scale[:, None]


class PSMFile:
    """Container for Ws, Wf, F arrays with dims, clock and checksum."""

    def __init__(self, Ws, Wf, F, dims=None, clock=0.0):
        """Store arrays and metadata."""
        self.Ws = np.asarray(Ws)
        self.Wf = np.asarray(Wf)
        self.F = np.asarray(F)
        self.dims = np.asarray(dims if dims is not None else self.Ws.shape)
        self.clock = float(clock)

    def checksum(self):
        """Simple sum checksum over stored arrays."""
        return float(np.sum(self.Ws) + np.sum(self.Wf) + np.sum(self.F))

    def save(self, path):
        """Save arrays and metadata with np.savez_compressed."""
        np.savez_compressed(
            path,
            Ws=self.Ws,
            Wf=self.Wf,
            F=self.F,
            dims=self.dims,
            clock=np.asarray(self.clock),
            checksum=np.asarray(self.checksum()),
        )

    @classmethod
    def load(cls, path):
        """Load arrays, verify simple sum checksum, return instance."""
        d = np.load(path, allow_pickle=False)
        Ws = d["Ws"]
        Wf = d["Wf"]
        F = d["F"]
        dims = d["dims"]
        clock = float(d["clock"])
        saved = float(d["checksum"])
        now = float(np.sum(Ws) + np.sum(Wf) + np.sum(F))
        if not np.isclose(saved, now):
            raise ValueError("checksum mismatch")
        return cls(Ws, Wf, F, dims=dims, clock=clock)


if __name__ == "__main__":
    """Smoke test save/load round trip on random 16x32 arrays."""
    rng = np.random.default_rng(0)
    Ws = rng.standard_normal((16, 32))
    Wf = rng.standard_normal((16, 32))
    F = rng.standard_normal((16, 32))
    psm = PSMFile(Ws, Wf, F, dims=np.asarray(Ws.shape), clock=1.0)
    psm.save("/tmp/test.psm.npz")
    back = PSMFile.load("/tmp/test.psm.npz")
    assert np.allclose(back.Ws, Ws)
    assert np.allclose(back.Wf, Wf)
    assert np.allclose(back.F, F)
    q, s = quantize_int8(Ws)
    assert np.allclose(dequantize(q, s), Ws, atol=0.05)
