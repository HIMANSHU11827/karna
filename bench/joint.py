"""AJO joint bench: image+text query - one brain answers both together."""
import time
import numpy as np
from karna.brain import UniBrain


def main():
    rng = np.random.RandomState(0)
    b = UniBrain()
    texts = ["red circle round", "blue square flat", "loud engine car", "dog bark loud"]
    imgs = [rng.rand(48, 48) for _ in range(4)]
    for ep in range(3):
        for i in range(4):
            tg = np.zeros(10)
            tg[i] = 1.0
            b.sense(texts[i], "text", tg)
            b.sense(imgs[i], "image", tg)
    ok = 0
    s = time.perf_counter()
    for i in range(4):
        rt = b.sense(texts[i], "text")
        ri = b.sense(imgs[i], "image")
        if int(np.argmax(rt["y"])) == i:
            ok += 1
        if int(np.argmax(ri["y"])) == i:
            ok += 1
    ms = (time.perf_counter() - s) * 1000.0 / 8.0
    print(f"JOINT_OK acc={ok}/8 regions={rt['regions']} ms={ms:.2f}")


if __name__ == "__main__":
    main()
