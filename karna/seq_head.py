"""Karna SeqHead: learned token head + vocab + autoregressive text generation."""
import numpy as np


class Vocab:
    def __init__(self, words=None):
        base = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.itos = list(base) + sorted(set(words or []))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(w.lower(), 3) for w in str(text).split()]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    def __len__(self):
        return len(self.itos)


class SeqHead:
    def __init__(self, vocab_size, d, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1.0 / np.sqrt(d), (vocab_size, d))
        self.b = np.zeros(vocab_size)

    def logits(self, h):
        return self.W @ np.asarray(h, dtype=np.float64) + self.b

    def step_train(self, h, y_true, eta=0.02, margin=0.2):
        lin = self.logits(h)
        order = np.argsort(lin)
        pred, second = int(order[-1]), int(order[-2])
        true = int(np.argmax(y_true))
        gap = float(lin[true] - lin[pred]) if pred != true else float(lin[pred] - lin[second])
        if pred != true or gap < margin:
            self.W[true] += eta * h
            self.b[true] += eta
            self.W[pred] -= eta * h
            self.b[pred] -= eta
            self.W /= np.maximum(1, np.linalg.norm(self.W, axis=1, keepdims=True) / 2.0)
            return 1.0
        return 0.0


def build_vocab(corpus_paths, top=4000):
    from collections import Counter
    c = Counter()
    for p in corpus_paths:
        try:
            for line in open(p, encoding="utf-8", errors="ignore"):
                c.update(w.lower() for w in line.split())
        except Exception:
            pass
    words = [w for w, _ in c.most_common(top)]
    return Vocab(words)
