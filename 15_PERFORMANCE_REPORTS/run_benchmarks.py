#!/usr/bin/env python3
"""AGI Research Lab — Performance Benchmark Suite"""
import time, sys, json, tracemalloc, os
sys.path.insert(0, '/home/himanshu/Desktop/AGI_RESEARCH_LAB/19_PROTOTYPES')

REPORT_DIR = '/home/himanshu/Desktop/AGI_RESEARCH_LAB/15_PERFORMANCE_REPORTS'
os.makedirs(REPORT_DIR, exist_ok=True)

results = {"system": {}, "benchmarks": [], "recommendations": []}

# === System Info ===
import platform
results["system"] = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "cpu": os.cpu_count(),
    "memory_gb": "15 (4.5 available)",
    "gpu": "none",
}

def bench(name, fn, iterations=100, **kwargs):
    """Run a benchmark with memory tracking."""
    tracemalloc.start()
    start = time.perf_counter()
    result = fn(iterations, **kwargs)
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    entry = {
        "name": name,
        "iterations": iterations,
        "total_ms": round(elapsed * 1000, 2),
        "avg_ms": round(elapsed * 1000 / iterations, 4),
        "peak_kb": round(peak / 1002, 1),
    }
    results["benchmarks"].append(entry)
    print(f"  {name}: {entry['total_ms']:.2f}ms total, {entry['avg_ms']:.4f}ms avg, {entry['peak_kb']:.1f}KB peak")
    return result

# === Benchmark 1: Memory System ===
print("=== BENCHMARK 1: Memory System Recall ===")
from memory_system import MemorySystem

def bench_memory_recall(n):
    mem = MemorySystem()
    for i in range(500):
        mem.perceive(f"Observation {i}: agent encountered state {i} with context", importance=0.3 + (i%7)*0.1, tags=["observation"])
        mem.remember_fact(f"fact_{i}", f"Fact {i}: parameter {i} = {i*3.14}", importance=0.7)
    for i in range(50):
        mem.perceive(f"High importance {i}", importance=0.9, tags=["critical"])
    for _ in range(n):
        mem.recall(query="observation", n=10)

bench("memory_recall", bench_memory_recall, iterations=100)

# === Benchmark 2: World Model ===
print("=== BENCHMARK 2: World Model Prediction ===")
from world_model import WorldModel, SimpleEnvironment, Transition
import random

def bench_world_model_predict(n):
    env = SimpleEnvironment(size=10, seed=42)
    model = WorldModel()
    rng = random.Random(42)
    for episode in range(200):
        state = env.reset()
        done = False
        while not done:
            action = rng.choice(env.get_available_actions())
            next_state, reward, done = env.step(action)
            model.observe(Transition(state, action, next_state, reward, done))
            state = next_state
    for _ in range(n):
        state = env.reset()
        for action in env.get_available_actions():
            model.predict(state, action)

bench("world_model_predict", bench_world_model_predict, iterations=100)

# === Benchmark 3: NAS Mutation ===
print("=== BENCHMARK 3: NAS Mutation ===")
from neural_architecture_search import NeuralArchitectureSearch

def bench_nas_mutation(n):
    nas = NeuralArchitectureSearch(population_size=50, seed=42)
    nas.initialize_population()
    for _ in range(n):
        for arch in nas.population:
            nas.mutate(arch)

bench("nas_mutation", bench_nas_mutation, iterations=100)

# === Benchmark 4: Baseline Agent ===
print("=== BENCHMARK 4: Baseline Agent Step ===")
from baseline_agent import BaselineAgent, mock_llm

def bench_agent_step(n):
    agent = BaselineAgent(name="bench-agent", system_prompt="Test agent", llm_fn=mock_llm)
    agent.tools.register("search", "Search info", {"query": {"type": "string"}}, lambda query: f"Results for {query}")
    for _ in range(n):
        agent.step("What is AGI?")

bench("agent_step", bench_agent_step, iterations=100)

# === Benchmark 5: JSON Serialization ===
print("=== BENCHMARK 5: JSON Serialization ===")
def bench_json(n):
    data = {"messages": [{"role": "user", "content": f"Message {i}", "timestamp": time.time()} for i in range(100)]}
    for _ in range(n):
        json.dumps(data, indent=2)
        json.loads(json.dumps(data))

bench("json_serialize", bench_json, iterations=100)

# === Recommendations ===
print()
print("=== RECOMMENDATIONS ===")
recs = []
for b in results["benchmarks"]:
    if b["avg_ms"] > 10:
        recs.append(f"HIGH LATENCY: {b['name']} averages {b['avg_ms']}ms — optimize hot path")
    if b["peak_kb"] > 10000:
        recs.append(f"HIGH MEMORY: {b['name']} peaks at {b['peak_kb']}KB — reduce allocations")
if not recs:
    recs.append("All benchmarks within acceptable limits for prototype stage")

results["recommendations"] = recs
for r in recs:
    print(f"  • {r}")

# Save report
report_path = os.path.join(REPORT_DIR, "benchmark_report.json")
with open(report_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nReport saved to {report_path}")
