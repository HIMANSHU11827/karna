"""AJO live stream bind: audio chunks + video frames bind into events in real time."""
import time
import numpy as np
from karna_brain import UniBrain


def main():
    rng = np.random.RandomState(0)
    b = UniBrain()
    s = time.perf_counter()
    n = 20
    for t in range(n):
        wave = (rng.randn(1600).astype(np.float32) * (0.5 + 0.5 * np.sin(t))).astype(np.float32)
        frame = (rng.rand(32, 32).astype(np.float32) * (0.5 + 0.5 * np.cos(t))).astype(np.float32)
        ra = b.sense(wave, "audio", novelty=0.2)
        rv = b.sense(frame, "video", novelty=0.2)
    ms = (time.perf_counter() - s) * 1000.0 / (2 * n)
    print(f"STREAM_OK steps={2 * n} regions={rv['regions']} ms={ms:.2f} conf={rv['conf']}")


if __name__ == "__main__":
    main()
