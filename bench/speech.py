"""Karna speech bench: spoken-digit style energy/ZC classification via UniBrain."""
import numpy as np
from karna.brain import UniBrain


def synth_word(seed, n=1600, kind="tone"):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 8000.0
    if kind == "tone":
        return (np.sin(2 * np.pi * (300 + seed * 50) * t) * 0.5).astype(np.float32)
    if kind == "noise":
        return (rng.standard_normal(n) * 0.3).astype(np.float32)
    return ((np.sin(2 * np.pi * 440 * t) > 0).astype(np.float32) - 0.5)


def main():
    b = UniBrain()
    words = ["tone", "noise", "square"]
    for ep in range(3):
        for i, w in enumerate(words):
            tg = np.zeros(10)
            tg[i] = 1.0
            b.sense(synth_word(i * 7 + ep, kind=w), "audio", tg)
    ok = 0
    for i, w in enumerate(words):
        r = b.sense(synth_word(i * 7 + 99, kind=w), "audio")
        if int(np.argmax(r["y"])) == i:
            ok += 1
    print(f"SPEECH_OK acc={ok}/3 regions={r['regions']}")


if __name__ == "__main__":
    main()
