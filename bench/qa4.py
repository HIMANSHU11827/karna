"""AJO 4-way QA: text<->image<->audio<->video cross-understanding matrix."""
import numpy as np
from karna.brain import UniBrain


def main():
    rng = np.random.RandomState(0)
    b = UniBrain()
    evs = [("red round fruit sweet", "text"), (rng.rand(48, 48), "image"),
           (rng.randn(8000).astype(np.float32), "audio"),
           (rng.rand(3, 32, 32).astype(np.float32), "video")]
    tg = np.zeros(10)
    tg[0] = 1.0
    for _ in range(4):
        for d, k in evs:
            b.sense(d, k, tg)
    print("train done regions:", len(b.grow.live()))
    ok = 0
    for d, k in evs:
        r = b.sense(d, k)
        p = int(np.argmax(r["y"]))
        hit = p == 0
        ok += hit
        print(f"{k:6s} -> region{r['winner']} pred{p} {'HIT' if hit else 'miss'}")
    print(f"QA4_OK understood={ok}/4")


if __name__ == "__main__":
    main()
