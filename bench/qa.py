"""AJO cross-region QA: can non-master region answer about master event?"""
import numpy as np
from karna.brain import UniBrain


def main():
    rng = np.random.RandomState(0)
    b = UniBrain()
    items = [("red circle round object", "text", rng.rand(48, 48), "image"),
             ("blue square flat thing", "text", rng.rand(48, 48) + 0.5, "image")]
    for ep in range(3):
        for i, (t, tk, im, ik) in enumerate(items):
            tg = np.zeros(10)
            tg[i] = 1.0
            b.sense(t, tk, tg)
            b.sense(im, ik, tg)
    ok = 0
    for i, (t, tk, im, ik) in enumerate(items):
        rt = b.sense(t, tk)
        ri = b.sense(im, ik)
        pt, pi = int(np.argmax(rt["y"])), int(np.argmax(ri["y"]))
        if pt == i:
            ok += 1
        if pi == i:
            ok += 1
        print(f"item{i} text->region{rt['winner']} pred{pt} | image->region{ri['winner']} pred{pi}")
    print(f"QA_OK understood={ok}/4 cross=text<->image via regions")


if __name__ == "__main__":
    main()
