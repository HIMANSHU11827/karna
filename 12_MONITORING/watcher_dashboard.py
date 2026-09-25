#!/usr/bin/env python3
"""
AGI Research Lab — Watcher Dashboard
Tracks system health, project metrics, and team activity.
"""

import json
import subprocess
import platform
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/himanshu/Desktop/AGI_RESEARCH_LAB")
WATCHER_DIR = ROOT / "12_MONITORING"
WATCHER_DIR.mkdir(parents=True, exist_ok=True)

def system_health():
    """Get system CPU, memory, disk info."""
    info = {"hostname": platform.node(), "platform": platform.platform()}
    
    # Memory
    try:
        mem = shutil.disk_usage("/")
        info["disk_total"] = f"{mem.total / (1024**3):.1f}G"
        info["disk_used"] = f"{mem.used / (1024**3):.1f}G"
        info["disk_free"] = f"{mem.free / (1024**3):.1f}G"
        info["disk_pct"] = f"{(mem.used / mem.total) * 100:.1f}%"
    except:
        info["disk"] = "unknown"
    
    return info

def project_health():
    """Get project stats."""
    result = subprocess.run(["du", "-sh", str(ROOT)], capture_output=True, text=True)
    size = result.stdout.split()[0] if result.returncode == 0 else "N/A"
    
    result = subprocess.run(["find", str(ROOT), "-type", "f"], capture_output=True, text=True)
    files = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    
    result = subprocess.run(["find", str(ROOT), "-type", "d"], capture_output=True, text=True)
    dirs = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    
    result = subprocess.run(["find", str(ROOT), "-name", "*.py"], capture_output=True, text=True)
    py_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    
    return {"size": size, "files": files, "directories": dirs, "python_files": len(py_files)}

def team_activity():
    """Check team directories for recent work."""
    team_dirs = {
        "coder-1 (Forge)": "19_PROTOTYPES",
        "coder-2 (Anvil)": "19_PROTOTYPES/memory_system",
        "coder-3": "05_PERCEPTION",
        "coder-4": "06_EVALUATION",
        "definer": "04_ARCHITECTURE",
        "designer": "03_DESIGN",
        "documenter": "01_DOCUMENTATION",
        "orchestrator": "10_ORCHESTRATION",
        "performance-fixer": "10_PERFORMANCE",
        "planner": "06_PROJECT_PLANS",
        "researcher": "01_RESEARCH",
        "verifier": "11_VERIFICATION",
        "watcher": "12_MONITORING",
        "bug-finder": "13_SECURITY"
    }
    
    activity = {}
    for name, path in team_dirs.items():
        full_path = ROOT / path
        if full_path.exists():
            result = subprocess.run(["find", str(full_path), "-type", "f"], capture_output=True, text=True)
            count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            activity[name] = {"path": path, "files": count, "active": count > 0}
        else:
            activity[name] = {"path": path, "files": 0, "active": False}
    
    return activity

def generate_dashboard():
    """Generate full monitoring dashboard."""
    timestamp = datetime.now().isoformat()
    
    dashboard = {
        "timestamp": timestamp,
        "system": system_health(),
        "project": project_health(),
        "team": team_activity(),
        "alerts": []
    }
    
    # Alerts
    if dashboard["project"]["files"] < 10:
        dashboard["alerts"].append("LOW FILE COUNT: Project has very few files — team needs to start building")
    
    inactive = [name for name, info in dashboard["team"].items() if not info["active"] and name != "watcher"]
    if len(inactive) > 10:
        dashboard["alerts"].append(f"TEAM INACTIVITY: {len(inactive)} bots haven't produced output yet")
    
    # Save
    path = WATCHER_DIR / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(dashboard, indent=2))
    
    # Print
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║     AGI RESEARCH LAB — WATCHER DASHBOARD     ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║ Time: {timestamp[:19]}                 ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║ SYSTEM                                        ║")
    print(f"║   Host: {dashboard['system']['hostname'][:30]:30s}   ║")
    print(f"║   Disk: {dashboard['system'].get('disk_free', 'N/A'):>8s} free / {dashboard['system'].get('disk_total', 'N/A'):>8s} total   ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║ PROJECT                                       ║")
    print(f"║   Size: {dashboard['project']['size']:>10s}                     ║")
    print(f"║   Files: {dashboard['project']['files']:>4d}  Dirs: {dashboard['project']['directories']:>4d}  Py: {dashboard['project']['python_files']:>3d}    ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║ TEAM ACTIVITY                                 ║")
    for name, info in dashboard["team"].items():
        status = "●" if info["active"] else "○"
        print(f"║  {status} {name[:20]:20s} {info['files']:>4d} files        ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║ ALERTS: {len(dashboard['alerts'])}                                       ║")
    for alert in dashboard["alerts"]:
        print(f"║  ⚠️  {alert[:45]:45s} ║")
    print(f"╚══════════════════════════════════════════════╝")
    
    return dashboard

if __name__ == "__main__":
    generate_dashboard()
