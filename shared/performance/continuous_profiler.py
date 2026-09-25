#!/usr/bin/env python3
"""Continuous profiler for AGI Research Lab. Monitors CPU, memory, I/O, and GPU usage."""

import os
import sys
import time
import json
import threading
import argparse
from datetime import datetime
from pathlib import Path

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("psutil not installed. Limited monitoring available.", file=sys.stderr)

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class SystemProfiler:
    """Real-time system resource profiler."""

    def __init__(self, output_dir: str, interval: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.running = False
        self.thread = None
        self.samples = []

        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except Exception:
                self.gpu_count = 0
        else:
            self.gpu_count = 0

    def _sample(self) -> dict:
        """Take a single sample of system metrics."""
        sample = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_mb": 0.0,
            "io_read_mb": 0.0,
            "io_write_mb": 0.0,
            "gpu": []
        }

        if HAS_PSUTIL:
            sample["cpu_percent"] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            sample["memory_percent"] = mem.percent
            sample["memory_mb"] = mem.used / (1024 * 1024)

            io = psutil.disk_io_counters()
            if io:
                sample["io_read_mb"] = io.read_bytes / (1024 * 1024)
                sample["io_write_mb"] = io.write_bytes / (1024 * 1024)

        if HAS_NVML and self.gpu_count > 0:
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    sample["gpu"].append({
                        "name": name.decode() if isinstance(name, bytes) else name,
                        "memory_used_mb": mem.used / (1024 * 1024),
                        "memory_total_mb": mem.total / (1024 * 1024),
                        "utilization_percent": util.gpu,
                        "temperature_c": temp
                    })
                except Exception as e:
                    sample["gpu"].append({"error": str(e)})

        return sample

    def _monitor(self):
        """Background monitoring loop."""
        while self.running:
            sample = self._sample()
            self.samples.append(sample)
            time.sleep(self.interval)

    def start(self):
        """Start background profiling."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop profiling and return collected samples."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        return self.samples

    def save(self, filename: str = None):
        """Save profiling data to JSON."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profile_{ts}.json"
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.samples, f, indent=2)
        return filepath

    def summary(self) -> dict:
        """Generate summary statistics from collected samples."""
        if not self.samples:
            return {"error": "No samples collected"}

        cpu_values = [s["cpu_percent"] for s in self.samples]
        mem_values = [s["memory_percent"] for s in self.samples]

        summary = {
            "samples": len(self.samples),
            "duration_sec": len(self.samples) * self.interval,
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            },
            "memory": {
                "avg_percent": sum(mem_values) / len(mem_values),
                "max_percent": max(mem_values),
                "min_percent": min(mem_values),
            }
        }

        # GPU stats
        gpu_summaries = {}
        for sample in self.samples:
            for gpu in sample.get("gpu", []):
                if "name" in gpu:
                    name = gpu["name"]
                    if name not in gpu_summaries:
                        gpu_summaries[name] = {"mem": [], "util": [], "temp": []}
                    gpu_summaries[name]["mem"].append(gpu.get("memory_used_mb", 0))
                    gpu_summaries[name]["util"].append(gpu.get("utilization_percent", 0))
                    gpu_summaries[name]["temp"].append(gpu.get("temperature_c", 0))

        for name, data in gpu_summaries.items():
            if data["util"]:
                summary.setdefault("gpu", {})[name] = {
                    "avg_utilization": sum(data["util"]) / len(data["util"]),
                    "max_memory_mb": max(data["mem"]) if data["mem"] else 0,
                    "avg_temperature_c": sum(data["temp"]) / len(data["temp"]) if data["temp"] else 0,
                }

        return summary


def monitor_pid(pid: int, output_dir: str, interval: float = 1.0, duration: float = 60.0):
    """Monitor a specific PID for a given duration."""
    profiler = SystemProfiler(output_dir, interval)
    print(f"Monitoring PID {pid} for {duration}s (interval: {interval}s)")
    profiler.start()
    time.sleep(duration)
    profiler.stop()
    filepath = profiler.save(f"pid_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    summary = profiler.summary()
    print(f"Profile saved to: {filepath}")
    print(json.dumps(summary, indent=2))
    return summary


def watch_daemon(output_dir: str, interval: float = 5.0):
    """Run as a persistent monitoring daemon."""
    profiler = SystemProfiler(output_dir, interval)
    print(f"Monitoring daemon started. Sampling every {interval}s. Ctrl+C to stop.")
    profiler.start()
    try:
        while True:
            time.sleep(60)  # Save every minute
            filepath = profiler.save()
            print(f"Saved {len(profiler.samples)} samples to {filepath}")
            profiler.samples = []  # Reset
    except KeyboardInterrupt:
        profiler.stop()
        filepath = profiler.save("final_profile.json")
        print(f"\nFinal profile saved to: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGI Lab System Profiler")
    parser.add_argument("--output-dir", default="15_PERFORMANCE_REPORTS/monitoring")
    parser.add_argument("--interval", type=float, default=1.0, help="Sample interval in seconds")
    subparsers = parser.add_subparsers(dest="command")

    # Monitor a process
    pid_parser = subparsers.add_parser("pid", help="Monitor a specific PID")
    pid_parser.add_argument("pid", type=int)
    pid_parser.add_argument("--duration", type=float, default=60.0)

    # Daemon mode
    daemon_parser = subparsers.add_parser("daemon", help="Run as persistent monitor")
    daemon_parser.add_argument("--interval", type=float, default=5.0)

    # Quick snapshot
    snapshot_parser = subparsers.add_parser("snapshot", help="Take a single snapshot")

    args = parser.parse_args()

    if args.command == "pid":
        monitor_pid(args.pid, args.output_dir, args.interval, args.duration)
    elif args.command == "daemon":
        watch_daemon(args.output_dir, args.interval)
    elif args.command == "snapshot":
        profiler = SystemProfiler(args.output_dir, args.interval)
        print(json.dumps(profiler._sample(), indent=2))
    else:
        # Default: quick system summary
        profiler = SystemProfiler(args.output_dir, 0.5)
        profiler.start()
        time.sleep(2)
        profiler.stop()
        print(json.dumps(profiler.summary(), indent=2))
