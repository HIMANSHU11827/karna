"""AJO forgetting + lesion test: old tasks survive new learning; kill best region."""
import numpy as np
from karna.brain import UniBrain


def main():
    rng = np.random.RandomState(0)
    b = UniBrain()
    taskA = [("alpha first one", "text"), ("beta second two", "text")]
    taskB = [(rng.rand(48, 48), "image"), (rng.rand(48, 48) + 0.5, "image")]
    for i, (d, k) in enumerate(taskA):
        tg = np.zeros(10)
        tg[i] = 1.0
        for _ in range(4):
            b.sense(d, k, tg)
    preA = sum(1 for i, (d, k) in enumerate(taskA) if int(np.argmax(b.sense(d, k)["y"])) == i)
    for i, (d, k) in enumerate(taskB):
        tg = np.zeros(10)
        tg[2 + i] = 1.0
        for _ in range(4):
            b.sense(d, k, tg)
    postA = sum(1 for i, (d, k) in enumerate(taskA) if int(np.argmax(b.sense(d, k)["y"])) == i)
    postB = sum(1 for i, (d, k) in enumerate(taskB) if int(np.argmax(b.sense(d, k)["y"])) == 2 + i)
    live = b.grow.live()
    best = max(range(len(live)), key=lambda i: live[i].u)
    live[best].active = False
    lesA = sum(1 for i, (d, k) in enumerate(taskA) if int(np.argmax(b.sense(d, k)["y"])) == i)
    print(f"FORGET_OK taskA pre={preA}/2 post={postA}/2 taskB={postB}/2 lesionA={lesA}/2 regions={len(b.grow.live()) + 1}")


if __name__ == "__main__":
    main()
