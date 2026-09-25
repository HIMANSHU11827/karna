#!/usr/bin/env python3
"""Comprehensive benchmark — before/after comparison."""
import time, sys, json, tracemalloc, os
sys.path.insert(0, '/home/himanshu/Desktop/AGI_RESEARCH_LAB/19_PROTOTYPES')

REPORT_DIR = '/home/himanshu/Desktop/AGI_RESEARCH_LAB/15_PERFORMANCE_REPORTS'
os.makedirs(REPORT_DIR, exist_ok=True)

results = {"before_after": [], "system": {}, "recommendations": []}

# === System Info ===
import platform
results["system"] = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "cpu": "13th Gen Intel i3-1315U (6c/8t)",
    "memory": "15GB (4.5GB available)",
    "gpu": "none (CPU only)",
}

def compare(name, fn_old=None, fn_new=None, iterations=100):
    """Compare old vs new implementation."""
    if fn_old:
        tracemalloc.start()
        start = time.perf_counter()
        for _ in range(iterations):
            fn_old()
        old_time = time.perf_counter() - start
        old_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    else:
        old_time = 0
        old_peak = 0

    tracemalloc.start()
    start = time.perf_counter()
    for _ in range(iterations):
        fn_new()
    new_time = time.perf_counter() - start
    new_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    speedup = old_time / new_time if new_time > 0 else float("inf")
    mem_ratio = old_peak / new_peak if new_peak > 0 else float("inf")
    
    entry = {
        "name": name,
        "iterations": iterations,
        "old_ms": round(old_time * 1000, 2),
        "new_ms": round(new_time * 1000, 2),
        "speedup": f"{speedup:.0f}x",
        "old_peak_kb": round(old_peak / 1024, 1),
        "new_peak_kb": round(new_peak / 1024, 1),
        "mem_reduction": f"{mem_ratio:.0f}x",
    }
    results["before_after"].append(entry)
    print(f"  {name}: {entry['speedup']} faster ({entry['old_ms']}ms -> {entry['new_ms']}ms)")
    print(f"    Memory: {entry['mem_reduction']} reduction ({entry['old_peak_kb']}KB -> {entry['new_peak_kb']}KB)")

# === Comparison 1: World Model predict ===
print("=== COMPARISON 1: World Model predict ===")
from world_model import WorldModel, SimpleEnvironment, Transition, State, Action

def benchmark_world_model():
    """Full benchmark: collect experience + predict."""
    env = SimpleEnvironment(size=10, seed=42)
    model = WorldModel()
    rng = __import__('random').Random(42)
    
    # Collect experience (same as before)
    for episode in range(200):
        state = env.reset()
        done = False
        while not done:
            action = rng.choice(env.get_available_actions())
            next_state, reward, done = env.step(action)
            model.observe(Transition(state, action, next_state, reward, done))
            state = next_state
    
    # Predict
    state = env.reset()
    for action in env.get_available_actions():
        model.predict(state, action)

compare("world_model_predict", fn_new=benchmark_world_model, iterations=100)

# === Comparison 2: World Model Planning ===
print("=== COMPARISON 2: World Model Planning ===")
def benchmark_world_model_planning():
    env = SimpleEnvironment(size=10, seed=42)
    model = WorldModel()
    rng = __import__('random').Random(42)
    
    for episode in range(200):
        state = env.reset()
        done = False
        while not done:
            action = rng.choice(env.get_available_actions())
            next_state, reward, done = env.step(action)
            model.observe(Transition(state, action, next_state, reward, done))
            state = next_state
    
    state = env.reset()
    actions = env.get_available_actions()
    plan = model.plan(state, actions, depth=10)

compare("world_model_planning", fn_new=benchmark_world_model_planning, iterations=100)

# === Save report ===
report_path = os.path.join(REPORT_DIR, "benchmark_after_fix.json")
with open(report_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nReport saved to {report_path}")
