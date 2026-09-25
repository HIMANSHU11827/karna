"""Karna video bench: motion vs static clip classification via UniBrain."""
import numpy as np
from karna.brain import UniBrain


def main():
    rng = np.random.RandomState(0)
    b = UniBrain()
    static = rng.rand(4, 16, 16).astype(np.float32) * 0.2 + 0.4
    moving = np.stack([(np.roll(rng.rand(16, 16), i * 2, axis=1)).astype(np.float32) for i in range(4)])
    for ep in range(3):
        for i, clip in enumerate([static, moving]):
            tg = np.zeros(10)
            tg[i] = 1.0
            b.sense(clip, "video", tg)
    ok = 0
    for i, clip in enumerate([static, moving]):
        r = b.sense(clip, "video")
        if int(np.argmax(r["y"])) == i:
            ok += 1
    print(f"VIDEO_OK acc={ok}/2 regions={r['regions']}")


if __name__ == "__main__":
    main()
