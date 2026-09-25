#!/usr/bin/env python3
"""
AGI Research Lab — Design Phase Watcher
Monitors for FINAL_DESIGN.md and scans for violations when it appears.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/himanshu/Desktop/AGI_RESEARCH_LAB")
TARGET = ROOT / "04_ARCHITECTURE" / "FINAL_DESIGN.md"
WATCHER_DIR = ROOT / "12_MONITORING"
LOG = WATCHER_DIR / "design_watch.json"

# Known existing architectures to check against
BANNED_PATTERNS = [
    r"(?i)krotov",
    r"(?i)hopfield",
    r"(?i)oja",
    r"(?i)bienenstock[\s-]*cooper[\s-]*munro",
    r"(?i)bc\s?m\b",
    r"(?i)transformer",
    r"(?i)\bmamba\b",
    r"(?i)\bssm\b",
    r"(?i)spatiotemporal[\s-]*predictive[\s-]*coding",
]

# Suspicious novelty claims
NOVELTY_FLAGS = [
    r"(?i)from[\s-]*scratch",
    r"(?i)novel[\s-]*architecture",
    r"(?i)new[\s-]*paradigm",
    r"(?i)original[\s-]*design",
    r"(?i)deduced[\s-]*from[\s-]*first[\s-]*principles",
]

# Real-time keywords
RT_KEYWORDS = [
    r"(?i)real[\s-]*time",
    r"(?i)latency",
    r"(?i)streaming",
    r"(?i)online[\s-]*learning",
]

def scan_file(text):
    """Scan document text for violations."""
    results = {"violations": [], "novelty_claims": [], "rt_mentions": 0, "existing_refs": []}
    
    # Check for banned patterns
    for pattern in BANNED_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            results["violations"].append(f"{pattern}: {len(matches)} hits")
    
    # Check novelty claims
    for pattern in NOVELTY_FLAGS:
        matches = re.findall(pattern, text)
        if matches:
            results["novelty_claims"].append(f"{pattern}: {len(matches)} hits")
    
    # Count real-time mentions
    for pattern in RT_KEYWORDS:
        matches = re.findall(pattern, text)
        results["rt_mentions"] += len(matches)
    
    # Check for citations that reference known existing work
    cite_patterns = [
        r"(?i)rao[\s-]*&[\s-]*ballard[\s-]*1999",
        r"(?i)friston[\s-]*200[35]",
        r"(?i)whittington[\s-]*&[\s-]*bogacz[\s-]*2017",
        r"(?i)millidge[\s-]*et[\s-]*al[\s-]*2022",
        r"(?i)krotov[\s-]*&[\s-]*hopfield[\s-]*2019",
    ]
    for pattern in cite_patterns:
        if re.search(pattern, text):
            results["existing_refs"].append(pattern)
    
    return results

def main():
    log_data = {"checked_at": datetime.now().isoformat(), "target_exists": False}
    
    if not TARGET.exists():
        log_data["status"] = "WAITING"
        print(f"[WATCHER] {TARGET.name} not yet created. Watching...")
    else:
        log_data["target_exists"] = True
        text = TARGET.read_text()
        log_data["size"] = len(text)
        log_data["lines"] = len(text.split('\n'))
        
        results = scan_file(text)
        log_data.update(results)
        
        if results["violations"]:
            log_data["status"] = "VIOLATION_DETECTED"
            print(f"[WATCHER] 🚨 VIOLATIONS DETECTED in {TARGET.name}")
            for v in results["violations"]:
                print(f"  - {v}")
        
        if results["novelty_claims"]:
            print(f"[WATCHER] ⚠️ Novelty claims found: {len(results['novelty_claims'])}")
            for nc in results["novelty_claims"]:
                print(f"  - {nc}")
        
        if results["existing_refs"]:
            print(f"[WATCHER] 📚 Existing work citations: {len(results['existing_refs'])}")
        
        if results["rt_mentions"] == 0:
            print(f"[WATCHER] ⚠️ No real-time/latency mentions — potential feasibility gap")
        
        if not results["violations"] and len(results["novelty_claims"]) <= 3:
            log_data["status"] = "OK"
            print(f"[WATCHER] ✅ {TARGET.name} appears clean")
    
    # Append to log
    logs = []
    if LOG.exists():
        logs = json.loads(LOG.read_text())
    logs.append(log_data)
    logs = logs[-50:]
    LOG.write_text(json.dumps(logs, indent=2))

if __name__ == "__main__":
    main()
