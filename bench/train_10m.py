"""Karna 10M nonstop multimodal trainer: text-understand + text-gen + image + audio + video."""
import os
import time
import traceback
import numpy as np
from karna.brain import UniBrain
from karna.seq_head import build_vocab

STATE = "models/karna-10m"
CKPT = os.path.join(STATE, "brain.psm.npz")


def main():
    os.makedirs(STATE, exist_ok=True)
    rng = np.random.RandomState(0)
    b = UniBrain()
    if os.path.exists(CKPT):
        try:
            b.load(CKPT)
            print("resumed t=%d" % b.t)
        except Exception:
            pass
    vocab = build_vocab(["16_DATA/text/tinyshakespeare.txt"], top=2000)
    lines = [l.strip() for l in open("16_DATA/text/tinyshakespeare.txt") if l.strip()]
    X = np.load("16_DATA/X_train.npy", mmap_mode="r")
    y = np.load("16_DATA/y_train.npy", mmap_mode="r")
    step = b.t
    while True:
        try:
            for i in range(200):
                step += 1
                t = lines[step % len(lines)][:60]
                tg = np.zeros(10)
                tg[step % 10] = 1.0
                b.sense(t, "text", tg, task="text")
                b.train_text(t, vocab)
                if step % 20 == 0:
                    x = np.asarray(X[step % len(X)], dtype=np.float64)
                    img = (x[:1024] / (np.linalg.norm(x[:1024]) + 1e-8) * 0.5 + 0.5).astype(np.float32)
                    lab = int(y[step % len(y)] % 10)
                    ig = np.zeros(10)
                    ig[lab] = 1.0
                    b.sense(img, "image", ig, task="image")
                    f = 110 * (1 + (step % 10) / 10)
                    tt = np.arange(8000) / 8000.0
                    aud = (0.5 * np.sin(2 * np.pi * f * tt) + 0.1 * rng.randn(8000)).astype(np.float32)
                    b.sense(aud, "audio", tg, task="audio", novelty=0.1)
                    vid = np.zeros((3, 32, 32), dtype=np.float32)
                    vid[:, :, (step % 32)] = 1.0
                    b.sense(vid, "video", tg, task="video", novelty=0.1)
                if step % 500 == 0:
                    b.sleep(30)
                if step % 1000 == 0:
                    b.save(CKPT)
            print("step=%d regions=%d" % (step, len(b.grow.live())), flush=True)
        except KeyboardInterrupt:
            b.save(CKPT)
            raise
        except Exception:
            traceback.print_exc()
            try:
                b.save(CKPT)
            except Exception:
                pass
            time.sleep(1.0)


if __name__ == "__main__":
    main()
