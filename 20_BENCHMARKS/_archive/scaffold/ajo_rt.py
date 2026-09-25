"""AJO realtime demo: uni-encoder -> binder -> AJO1, one loop for all kinds."""
import time
import numpy as np
from ajo_uni import UniEncoder
from ajo_fusion import Binder
from ajo1 import AJO1


def main():
    rng = np.random.RandomState(0)
    enc = UniEncoder(dim=512, k=64)
    bind = Binder(dim=256)
    ajo = AJO1(in_dim=256, n=512, k=32, out_dim=10, seed=1)
    inputs = [
        ("hello robot world", "text"),
        (rng.rand(48, 48), "image"),
        (rng.randn(8000).astype(np.float32), "audio"),
        (rng.rand(3, 32, 32).astype(np.float32), "video"),
    ]
    s = time.perf_counter()
    for data, kind in inputs * 5:
        ev = enc.encode(data, kind)
        bind.ingest([ev])
        st = bind.state()
        r = ajo.step(st)
        enc.adapt(ev["vector"])
    ms = (time.perf_counter() - s) * 1000.0 / 20.0
    print(f"RT_OK objects={len(bind.objs)} ms_per_step={ms:.2f} active={r['active']}")


if __name__ == "__main__":
    main()
