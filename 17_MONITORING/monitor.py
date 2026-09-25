#!/usr/bin/env python3
"""
AGI Research Lab — Monitoring Dashboard
Tracks system health, resource usage, and project status.
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/himanshu/Desktop/AGI_RESEARCH_LAB")
LOG_DIR = ROOT / "17_MONITORING"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_disk_usage():
    """Get disk usage of the project."""
    result = subprocess.run(
        ["du", "-sh", str(ROOT)],
        capture_output=True, text=True
    )
    size = result.stdout.split()[0] if result.returncode == 0 else "N/A"
    return {"total_size": size}

def get_directory_stats():
    """Count files and directories."""
    result = subprocess.run(
        ["find", str(ROOT), "-type", "f"],
        capture_output=True, text=True
    )
    file_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    
    result = subprocess.run(
        ["find", str(ROOT), "-type", "d"],
        capture_output=True, text=True
    )
    dir_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    
    return {"files": file_count, "directories": dir_count}

def get_folder_sizes():
    """Get sizes of major folders."""
    folders = {}
    for folder in sorted(ROOT.iterdir()):
        if folder.is_dir():
            result = subprocess.run(
                ["du", "-sh", str(folder)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                folders[folder.name] = result.stdout.split()[0]
    return folders

def check_empty_directories():
    """Find empty directories that may need attention."""
    result = subprocess.run(
        ["find", str(ROOT), "-type", "d", "-empty"],
        capture_output=True, text=True
    )
    return result.stdout.strip().split('\n') if result.stdout.strip() else []

def get_code_metrics():
    """Count lines of code in Python files."""
    result = subprocess.run(
        ["find", str(ROOT), "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
    total = 0
    for line in lines:
        try:
            total += int(line.split()[0])
        except:
            pass
    return {"total_lines": total, "file_count": len(lines)}

def main():
    timestamp = datetime.now().isoformat()
    
    report = {
        "timestamp": timestamp,
        "project": "AGI Research Lab",
        "status": "operational",
        "disk": get_disk_usage(),
        "counts": get_directory_stats(),
        "code": get_code_metrics(),
        "folder_sizes": get_folder_sizes(),
        "empty_dirs": check_empty_directories(),
        "alerts": []
    }
    
    # Check for alerts
    if report["disk"]["total_size"].endswith("G"):
        report["alerts"].append(f"DISK WARNING: Project size is {report['disk']['total_size']}")
    
    empty = report["empty_dirs"]
    if len(empty) > 50:
        report["alerts"].append(f"STRUCTURE WARNING: {len(empty)} empty directories")
    
    # Save report
    report_path = LOG_DIR / f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2))
    
    # Print summary
    print(f"[MONITOR] AGI Research Lab Health Report — {timestamp}")
    print(f"  Disk: {report['disk']['total_size']}")
    print(f"  Files: {report['counts']['files']} | Dirs: {report['counts']['directories']}")
    print(f"  Code: {report['code']['total_lines']} lines in {report['code']['file_count']} files")
    print(f"  Empty Dirs: {len(empty)}")
    print(f"  Alerts: {len(report['alerts'])}")
    for alert in report['alerts']:
        print(f"  ⚠️  {alert}")

if __name__ == "__main__":
    main()
