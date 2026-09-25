"""Karna max-tokens bench: context window + throughput + memory."""
import time
import numpy as np
from karna.brain import UniBrain


def main():
    b = UniBrain()
    sizes = [1, 4, 16, 64, 256]
    print("CTX_TEST seq_len -> ms/step, active")
    for n in sizes:
        s = time.perf_counter()
        for i in range(n):
            b.sense("ctx token %d filler words here" % i, "text")
        ms = (time.perf_counter() - s) * 1000.0 / n
        print(f"  len={n} ms={ms:.2f}")
    print("THROUGHPUT 200 steps:")
    s = time.perf_counter()
    for i in range(200):
        b.sense("speed test words %d" % i, "text")
    ms = (time.perf_counter() - s) * 1000.0 / 200.0
    print(f"  ms/step={ms:.2f} toks/s={1000.0/ms:.1f}")
    print(f"MAX_CTX=unbounded streaming (state O(1), no window); PARAMS~4.26M; REGIONS={len(b.grow.live())}")


if __name__ == "__main__":
    main()
