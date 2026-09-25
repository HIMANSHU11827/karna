#!/usr/bin/env python3
"""Benchmark suite for AGI Research Lab. Provides timing, profiling, and comparison utilities."""

import time
import json
import os
import sys
import functools
import statistics
import tracemalloc
import cProfile
import pstats
import io
from datetime import datetime
from pathlib import Path
from typing import Callable, Any
from contextlib import contextmanager

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class BenchmarkResult:
    """Stores benchmark results for a single function."""

    def __init__(self, name: str, timings: list[float], memory_peak: float = 0, memory_current: float = 0):
        self.name = name
        self.timings = timings
        self.memory_peak_mb = memory_peak
        self.memory_current_mb = memory_current
        self.mean = statistics.mean(timings) if timings else 0
        self.median = statistics.median(timings) if timings else 0
        self.stdev = statistics.stdev(timings) if len(timings) > 1 else 0
        self.min_val = min(timings) if timings else 0
        self.max_val = max(timings) if timings else 0
        self.total = sum(timings) if timings else 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "runs": len(self.timings),
            "mean_ms": self.mean * 1000,
            "median_ms": self.median * 1000,
            "stdev_ms": self.stdev * 1000,
            "min_ms": self.min_val * 1000,
            "max_ms": self.max_val * 1000,
            "total_ms": self.total * 1000,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_current_mb": self.memory_current_mb,
            "throughput_rps": 1.0 / self.mean if self.mean > 0 else 0,
        }

    def __str__(self) -> str:
        return (
            f"{self.name}: "
            f"mean={self.mean*1000:.3f}ms "
            f"median={self.median*1000:.3f}ms "
            f"min={self.min_val*1000:.3f}ms "
            f"max={self.max_val*1000:.3f}ms "
            f"(n={len(self.timings)}, "
            f"peak_mem={self.memory_peak_mb:.2f}MB)"
        )


def timeit(func: Callable = None, *, runs: int = 10, warmup: int = 1, name: str = None):
    """Decorator to benchmark a function. Usage: @timeit or @timeit(runs=100)"""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Warmup
            for _ in range(warmup):
                fn(*args, **kwargs)

            timings = []
            tracemalloc.start()

            for _ in range(runs):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                end = time.perf_counter()
                timings.append(end - start)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            bench_name = name or fn.__name__
            bench = BenchmarkResult(
                bench_name, timings,
                peak / (1024 * 1024),
                current / (1024 * 1024)
            )
            print(bench)
            return result
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


class BenchmarkSuite:
    """Run and compare multiple function benchmarks."""

    def __init__(self, output_dir: str = "15_PERFORMANCE_REPORTS/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def run(self, func: Callable, *args, runs: int = 10, warmup: int = 1,
            name: str = None, memory: bool = True, **kwargs) -> BenchmarkResult:
        """Benchmark a function with given arguments."""
        bench_name = name or func.__name__

        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)

        timings = []
        if memory:
            tracemalloc.start()

        # Timed runs
        for _ in range(runs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            timings.append(end - start)

        memory_peak = 0
        memory_current = 0
        if memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_peak = peak / (1024 * 1024)
            memory_current = current / (1024 * 1024)

        bench = BenchmarkResult(bench_name, timings, memory_peak, memory_current)
        self.results.append(bench)
        print(bench)
        return result

    def compare(self, *funcs: Callable, args=(), runs: int = 10, **kwargs) -> dict:
        """Compare multiple functions with the same arguments."""
        for func in funcs:
            self.run(func, *args, runs=runs, **kwargs)

        return self.report()

    def report(self) -> dict:
        """Generate comparison report."""
        if not self.results:
            return {"error": "No benchmarks run"}

        report = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": [r.to_dict() for r in self.results],
        }

        if len(self.results) > 1:
            best = min(self.results, key=lambda r: r.mean)
            for r in self.results:
                speedup = r.mean / best.mean if best.mean > 0 else float('inf')
                r_dict = report["benchmarks"][self.results.index(r)]
                r_dict["vs_best"] = f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.2f}x faster"

            report["fastest"] = best.name

        return report

    def save(self, filename: str = None):
        """Save results to JSON."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{ts}.json"
        filepath = self.output_dir / filename
        report = self.report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved to: {filepath}")
        return filepath


@contextmanager
def profile_block(name: str = "block", print_stats: bool = True):
    """Profile a block of code using cProfile."""
    pr = cProfile.Profile()
    pr.enable()
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    if print_stats:
        print(f"\n--- Profile: {name} ({elapsed*1000:.1f}ms) ---")
        print(s.getvalue())


def profile_function(func: Callable, *args, **kwargs) -> str:
    """Profile a single function and return stats string."""
    pr = cProfile.Profile()
    pr.enable()
    result = pr.runcall(func, *args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    return s.getvalue()


def memory_snapshot():
    """Take a memory snapshot if psutil is available."""
    if not HAS_PSUTIL:
        return {}
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024),
    }


def track_memory(func: Callable, *args, **kwargs):
    """Track memory usage of a function."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {
        "peak_mb": peak / (1024 * 1024),
        "current_mb": current / (1024 * 1024),
    }


# CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AGI Lab Benchmark Suite")
    subparsers = parser.add_subparsers(dest="command")

    # Run memory snapshot
    subparsers.add_parser("memory", help="Take memory snapshot")

    # Profile a function from a module
    profile_parser = subparsers.add_parser("profile", help="Profile a function")
    profile_parser.add_argument("module", help="Module path (e.g., mymodule)")
    profile_parser.add_argument("function", help="Function name")

    # Quick benchmark
    quick = subparsers.add_parser("quick", help="Quick benchmark a function")
    quick.add_argument("module")
    quick.add_argument("function")
    quick.add_argument("--runs", type=int, default=100)

    args = parser.parse_args()

    if args.command == "memory":
        print(json.dumps(memory_snapshot(), indent=2))
    elif args.command == "profile":
        import importlib
        mod = importlib.import_module(args.module)
        func = getattr(mod, args.function)
        print(profile_function(func))
    elif args.command == "quick":
        import importlib
        mod = importlib.import_module(args.module)
        func = getattr(mod, args.function)
        suite = BenchmarkSuite()
        suite.run(func, runs=args.runs)
        suite.save()
    else:
        print("No command specified. Use --help for usage.")
