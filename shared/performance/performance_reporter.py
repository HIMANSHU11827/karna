#!/usr/bin/env python3
"""Performance reporting and trend analysis for AGI Research Lab."""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import statistics


class PerformanceReporter:
    """Analyzes benchmark trends and generates reports."""

    def __init__(self, reports_dir: str = "15_PERFORMANCE_REPORTS"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def analyze_trend(self, metric_name: str, json_files: list[str]) -> dict:
        """Track a metric across multiple benchmark files to detect regressions."""
        values = []
        timestamps = []

        for f in json_files:
            with open(f) as fp:
                data = json.load(fp)

            # Navigate nested dicts to find the metric
            val = data
            for key in metric_name.split('.'):
                if isinstance(val, dict):
                    val = val.get(key)
                else:
                    val = None
                    break

            if val is not None:
                values.append(val)
                timestamps.append(data.get("timestamp", Path(f).stem))

        if len(values) < 2:
            return {"error": f"Need at least 2 data points for {metric_name}", "found": len(values)}

        # Calculate trend
        first_half = values[:len(values) // 2]
        second_half = values[len(values) // 2:]
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)

        change = ((avg_second - avg_first) / avg_first * 100) if avg_first != 0 else 0
        is_regression = change > 10  # 10% worse = regression

        return {
            "metric": metric_name,
            "data_points": len(values),
            "first_half_avg": avg_first,
            "second_half_avg": avg_second,
            "change_percent": change,
            "is_regression": is_regression,
            "trend": "improving" if change < -5 else ("regressing" if is_regression else "stable"),
            "values": values,
            "timestamps": timestamps,
        }

    def generate_report(self, benchmark_dir: str = "15_PERFORMANCE_REPORTS/benchmarks") -> dict:
        """Generate a summary report from all benchmark files."""
        bench_dir = Path(benchmark_dir)
        if not bench_dir.exists():
            return {"error": f"Directory not found: {bench_dir}"}

        json_files = sorted(bench_dir.glob("benchmark_*.json"))
        all_benchmarks = []

        for f in json_files:
            with open(f) as fp:
                data = json.load(fp)
            all_benchmarks.append({
                "file": f.name,
                "timestamp": data.get("timestamp"),
                "benchmarks": data.get("benchmarks", []),
            })

        report = {
            "generated": datetime.now().isoformat(),
            "total_files": len(json_files),
            "benchmarks": all_benchmarks,
        }

        # Save report
        report_path = self.reports_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as fp:
            json.dump(report, fp, indent=2)
        print(f"Report saved: {report_path}")
        return report

    def compare_runs(self, file1: str, file2: str) -> dict:
        """Compare two benchmark runs for regressions."""
        with open(file1) as f:
            data1 = json.load(f)
        with open(file2) as f:
            data2 = json.load(f)

        comparisons = []
        for b1 in data1.get("benchmarks", []):
            for b2 in data2.get("benchmarks", []):
                if b1["name"] == b2["name"]:
                    mean1 = b1.get("mean_ms", 0)
                    mean2 = b2.get("mean_ms", 0)
                    if mean1 > 0:
                        change = ((mean2 - mean1) / mean1) * 100
                        comparisons.append({
                            "name": b1["name"],
                            "run1_mean_ms": mean1,
                            "run2_mean_ms": mean2,
                            "change_percent": change,
                            "status": "regression" if change > 10 else ("improvement" if change < -10 else "stable"),
                        })

        return {
            "file1": file1,
            "file2": file2,
            "comparisons": comparisons,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Performance Reporter")
    subparsers = parser.add_subparsers(dest="command")

    # Generate summary
    subparsers.add_parser("summary", help="Generate summary report")

    # Compare runs
    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark files")
    compare_parser.add_argument("file1")
    compare_parser.add_argument("file2")

    # Trend analysis
    trend_parser = subparsers.add_parser("trend", help="Analyze metric trend")
    trend_parser.add_argument("metric", help="Metric path (e.g., latency.p95_ms)")
    trend_parser.add_argument("files", nargs="+", help="JSON files to analyze")

    args = parser.parse_args()
    reporter = PerformanceReporter()

    if args.command == "summary":
        reporter.generate_report()
    elif args.command == "compare":
        result = reporter.compare_runs(args.file1, args.file2)
        print(json.dumps(result, indent=2))
    elif args.command == "trend":
        result = reporter.analyze_trend(args.metric, args.files)
        print(json.dumps(result, indent=2))
    else:
        reporter.generate_report()
