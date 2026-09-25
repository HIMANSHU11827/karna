#!/usr/bin/env python3
"""
AGI Research Lab — Watcher Alerts
Monitors for issues and generates alerts.
"""

import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/himanshu/Desktop/AGI_RESEARCH_LAB")
ALERT_LOG = ROOT / "12_MONITORING" / "alerts.json"

def check_git_status():
    """Check for uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    if result.returncode == 0:
        changes = result.stdout.strip().split('\n') if result.stdout.strip() else []
        return {"git_initialized": True, "uncommitted": len(changes), "changes": changes[:10]}
    return {"git_initialized": False}

def check_large_files(threshold_mb=10):
    """Find files larger than threshold."""
    result = subprocess.run(
        ["find", str(ROOT), "-type", "f", "-size", f"+{threshold_mb}M"],
        capture_output=True, text=True
    )
    large = result.stdout.strip().split('\n') if result.stdout.strip() else []
    return large

def check_broken_symlinks():
    """Find broken symlinks."""
    result = subprocess.run(
        ["find", str(ROOT), "-type", "l", "-exec", "test", "!", "-e", "{}", ";", "-print"],
        capture_output=True, text=True
    )
    return result.stdout.strip().split('\n') if result.stdout.strip() else []

def check_for_secrets():
    """Basic check for potential secrets in code."""
    patterns = ["api_key", "password", "secret", "token"]
    alerts = []
    for pattern in patterns:
        result = subprocess.run(
            ["grep", "-r", "-i", pattern, "--include=*.py", str(ROOT)],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')[:5]
            alerts.append(f"Potential secrets matching '{pattern}': {len(lines)} occurrences")
    return alerts

def generate_alert_report():
    """Generate comprehensive alert report."""
    timestamp = datetime.now().isoformat()
    
    report = {
        "timestamp": timestamp,
        "git": check_git_status(),
        "large_files": check_large_files(5),
        "broken_symlinks": check_broken_symlinks(),
        "secret_warnings": check_for_secrets(),
        "overall_status": "healthy"
    }
    
    # Determine overall status
    if report["large_files"] or report["broken_symlinks"] or report["secret_warnings"]:
        report["overall_status"] = "warnings"
    
    # Save
    alerts = []
    if ALERT_LOG.exists():
        alerts = json.loads(ALERT_LOG.read_text())
    alerts.append(report)
    # Keep only last 50 alerts
    alerts = alerts[-50:]
    ALERT_LOG.write_text(json.dumps(alerts, indent=2))
    
    # Print summary
    print(f"[ALERTS] Report generated at {timestamp[:19]}")
    print(f"  Git: {'tracked' if report['git']['git_initialized'] else 'not tracked'}")
    print(f"  Uncommitted: {report['git'].get('uncommitted', 'N/A')}")
    print(f"  Large files: {len(report['large_files'])}")
    print(f"  Broken symlinks: {len(report['broken_symlinks'])}")
    print(f"  Secret warnings: {len(report['secret_warnings'])}")
    print(f"  Status: {report['overall_status'].upper()}")
    
    return report

if __name__ == "__main__":
    generate_alert_report()
