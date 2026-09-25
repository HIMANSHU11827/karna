"""AJO 24/7 daemon: forever sense-learn-grow-sleep-checkpoint loop."""
import json
import os
import time
import traceback
import numpy as np
from karna.brain import UniBrain


STATE = "/tmp/karna_24_7"
CKPT = os.path.join(STATE, "brain.psm.npz")
HEART = os.path.join(STATE, "heartbeat.json")


def heartbeat(step, regions, conf):
    with open(HEART, "w") as f:
        json.dump({"t": time.time(), "step": step, "regions": regions, "conf": conf}, f)


def main():
    import fcntl
    os.makedirs(STATE, exist_ok=True)
    lockf = open(os.path.join(STATE, "daemon.lock"), "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        print("another daemon holds lock, exiting")
        return
    rng = np.random.RandomState(0)
    b = UniBrain()
    if os.path.exists(CKPT):
        b.load(CKPT)
        print("resumed t=%d regions=%d" % (b.t, len(b.grow.live())))
    kinds = [("hello brain learns forever", "text"), (rng.rand(48, 48), "image"),
             (rng.randn(8000).astype(np.float32), "audio"),
             (rng.rand(3, 32, 32).astype(np.float32), "video")]
    step = b.t
    while True:
        try:
            for data, kind in kinds:
                step += 1
                r = b.sense(data, kind, novelty=0.1)
                if step % 500 == 0:
                    b.sleep(50)
                if step % 1000 == 0:
                    b.save(CKPT)
                    heartbeat(step, r["regions"], r["conf"])
            heartbeat(step, r["regions"], r["conf"])
            time.sleep(0.05)
        except KeyboardInterrupt:
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
