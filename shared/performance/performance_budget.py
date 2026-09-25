#!/usr/bin/env python3
"""Performance budget checker and alerting for AGI Research Lab."""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Default performance budgets - modify per project needs
DEFAULT_BUDGETS = {
    "api": {
        "p50_ms": 50,
        "p95_ms": 200,
        "p99_ms": 500,
        "error_rate_percent": 1.0,
        "throughput_rps": 100,
    },
    "inference": {
        "p50_ms": 500,
        "p95_ms": 2000,
        "p99_ms": 5000,
        "time_to_first_token_ms": 200,
        "tokens_per_second": 20,
        "memory_gb": 8,
    },
    "memory": {
        "training_gb": 16,
        "inference_gb": 8,
        "embedding_gb": 4,
    },
    "storage": {
        "max_size_gb": 100,
        "read_iops": 3000,
        "write_iops": 1500,
    }
}


class PerformanceBudget:
    """Checks benchmark results against performance budgets."""

    def __init__(self, budget_file: Optional[str] = None):
        if budget_file and Path(budget_file).exists():
            with open(budget_file) as f:
                self.budgets = json.load(f)
        else:
            self.budgets = DEFAULT_BUDGETS

    def check_latency(self, category: str, metrics: dict) -> dict:
        """Check latency metrics against budget."""
        budget = self.budgets.get(category, {})
        violations = []

        for metric, limit in budget.items():
            if metric in metrics:
                actual = metrics[metric]
                if "ms" in metric and actual > limit:
                    violations.append({
                        "metric": metric,
                        "limit": limit,
                        "actual": actual,
                        "over_by_percent": ((actual - limit) / limit) * 100,
                    })
                elif "throughput" in metric and actual < limit:
                    violations.append({
                        "metric": metric,
                        "limit": limit,
                        "actual": actual,
                        "under_by_percent": ((limit - actual) / limit) * 100,
                    })

        return {
            "category": category,
            "violations": violations,
            "passed": len(violations) == 0,
        }

    def check_benchmark_file(self, filepath: str) -> dict:
        """Check a benchmark report file against all applicable budgets."""
        with open(filepath) as f:
            data = json.load(f)

        results = {"timestamp": datetime.now().isoformat(), "file": filepath, "checks": []}

        # Check API benchmarks
        if "latency" in data:
            check = self.check_latency("api", data["latency"])
            results["checks"].append(check)

        if "throughput_rps" in data:
            check = self.check_latency("api", {"throughput_rps": data["throughput_rps"]})
            results["checks"].append(check)

        # Check inference metrics
        if "tokens_per_second" in data:
            check = self.check_latency("inference", {
                "tokens_per_second": data["tokens_per_second"],
                "p50_ms": data.get("p50_ms", 0),
                "p95_ms": data.get("p95_ms", 0),
                "p99_ms": data.get("p99_ms", 0),
            })
            results["checks"].append(check)

        results["all_passed"] = all(c.get("passed", True) for c in results["checks"])
        return results

    def generate_budget_file(self, filepath: str = "shared/performance/budget.json"):
        """Generate a template budget file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.budgets, f, indent=2)
        print(f"Budget template saved to: {filepath}")
        return filepath


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Performance Budget Checker")
    parser.add_argument("--check", help="Benchmark JSON file to check")
    parser.add_argument("--budget", help="Custom budget JSON file")
    parser.add_argument("--generate-budget", action="store_true", help="Generate budget template")

    args = parser.parse_args()
    pb = PerformanceBudget(args.budget)

    if args.generate_budget:
        pb.generate_budget_file()
    elif args.check:
        results = pb.check_benchmark_file(args.check)
        print(json.dumps(results, indent=2))
        sys.exit(0 if results["all_passed"] else 1)
    else:
        print("Use --generate-budget to create a budget template or --check <file> to validate benchmarks")
