#!/usr/bin/env python3
"""Load testing and latency profiling for AGI Lab services."""

import time
import json
import statistics
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Optional


class LoadTester:
    """HTTP load tester for API endpoints."""

    def __init__(self, base_url: str, output_dir: str = "15_PERFORMANCE_REPORTS"):
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _single_request(self, session: aiohttp.ClientSession,
                               endpoint: str, payload: dict) -> dict:
        """Make a single request and measure latency."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        start = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                body = await resp.read()
                elapsed = time.perf_counter() - start
                return {
                    "status": resp.status,
                    "latency": elapsed,
                    "size": len(body),
                    "success": 200 <= resp.status < 300,
                }
        except Exception as e:
            elapsed = time.perf_counter() - start
            return {
                "status": 0,
                "latency": elapsed,
                "size": 0,
                "success": False,
                "error": str(e),
            }

    async def run_load_test(self, endpoint: str, payload: dict,
                            concurrency: int = 10, total_requests: int = 100) -> dict:
        """Run a load test with given concurrency and request count."""
        print(f"Load testing {endpoint} | concurrency={concurrency} | total={total_requests}")

        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrency)

            async def limited_request():
                async with semaphore:
                    return await self._single_request(session, endpoint, payload)

            tasks = [limited_request() for _ in range(total_requests)]
            results = await asyncio.gather(*tasks)

        return self._analyze_results(results)

    def _analyze_results(self, results: list[dict]) -> dict:
        """Analyze load test results."""
        total = len(results)
        successes = sum(1 for r in results if r["success"])
        failures = total - successes
        latencies = [r["latency"] for r in results if r["success"]]

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_requests": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total if total > 0 else 0,
        }

        if latencies:
            latencies.sort()
            report["latency"] = {
                "mean_ms": statistics.mean(latencies) * 1000,
                "median_ms": statistics.median(latencies) * 1000,
                "p50_ms": latencies[int(len(latencies) * 0.50)] * 1000,
                "p90_ms": latencies[int(len(latencies) * 0.90)] * 1000,
                "p95_ms": latencies[int(len(latencies) * 0.95)] * 1000,
                "p99_ms": latencies[int(len(latencies) * 0.99)] * 1000,
                "min_ms": min(latencies) * 1000,
                "max_ms": max(latencies) * 1000,
                "stdev_ms": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,
            }
            report["throughput_rps"] = total / sum(latencies) if sum(latencies) > 0 else 0

        return report

    async def latency_profile(self, endpoint: str, payload: dict,
                              samples: int = 100) -> dict:
        """Profile latency distribution over many sequential requests."""
        print(f"Profiling {endpoint} ({samples} sequential requests)")

        async with aiohttp.ClientSession() as session:
            results = []
            for i in range(samples):
                result = await self._single_request(session, endpoint, payload)
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{samples}")

        return self._analyze_results(results)

    def save_report(self, report: dict, filename: Optional[str] = None):
        """Save load test report to JSON."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loadtest_{ts}.json"
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved: {filepath}")
        return filepath


async def profile_endpoint(url: str, payload: dict, samples: int = 100):
    """Quick CLI endpoint profiling."""
    tester = LoadTester(url)
    report = await tester.latency_profile("/", payload, samples)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AGI Lab Load Tester")
    parser.add_argument("--url", required=True, help="Base URL to test")
    parser.add_argument("--endpoint", default="/", help="Endpoint to hit")
    parser.add_argument("--payload", default='{}', help="JSON payload string")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--profile", action="store_true", help="Profile latency distribution")

    args = parser.parse_args()
    payload = json.loads(args.payload)

    tester = LoadTester(args.url)
    if args.profile:
        result = asyncio.run(tester.latency_profile(args.endpoint, payload, args.requests))
    else:
        result = asyncio.run(tester.run_load_test(
            args.endpoint, payload, args.concurrency, args.requests
        ))
    print(json.dumps(result, indent=2))
    tester.save_report(result)
