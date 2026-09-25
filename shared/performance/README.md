# Performance Tooling — AGI Research Lab

Continuous profiling, benchmarking, load testing, and budget enforcement.

## Tools

| Script | Purpose | Usage |
|--------|---------|-------|
| `continuous_profiler.py` | System monitoring (CPU, memory, I/O, GPU) | `python continuous_profiler.py --output-dir 15_PERFORMANCE_REPORTS/monitoring` |
| `benchmark_suite.py` | Function-level benchmarking | `@timeit` decorator or `BenchmarkSuite().run(func)` |
| `load_tester.py` | HTTP endpoint load testing | `python load_tester.py --url http://host:port --concurrency 50` |
| `performance_budget.py` | Budget enforcement and alerting | `python performance_budget.py --check benchmarks.json` |
| `performance_reporter.py` | Trend analysis and regression detection | `python performance_reporter.py summary` |

## Directories

- `15_PERFORMANCE_REPORTS/benchmarks/` — Benchmark result JSON files
- `15_PERFORMANCE_REPORTS/flamegraphs/` — Generated flame graphs
- `15_PERFORMANCE_REPORTS/monitoring/` — Continuous profiling snapshots
- `15_PERFORMANCE_REPORTS/traces/` — Distributed traces
- `shared/performance/` — Shared performance tooling

## Quick Start

```bash
# Quick system snapshot
python shared/performance/continuous_profiler.py snapshot

# Monitor a running process
python shared/performance/continuous_profiler.py pid <PID> --duration 60

# Run benchmarks
python -c "
from shared.performance.benchmark_suite import BenchmarkSuite
suite = BenchmarkSuite()
suite.run(my_function, runs=100)
suite.save()
"

# Load test an endpoint
python shared/performance/load_tester.py \
  --url http://localhost:8000 \
  --endpoint /infer \
  --payload '{"prompt": "Hello"}' \
  --concurrency 20 \
  --requests 1000
```
