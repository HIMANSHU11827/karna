#!/usr/bin/env python3
"""Profile the predictive coding architecture at MNIST scale."""
import time, sys, json, tracemalloc, os, random, math
sys.path.insert(0, '/home/himanshu/Desktop/AGI_RESEARCH_LAB/19_PROTOTYPES')

REPORT_DIR = '/home/himanshu/Desktop/AGI_RESEARCH_LAB/15_PERFORMANCE_REPORTS'
os.makedirs(REPORT_DIR, exist_ok=True)

from learning_architecture.predictive_coding_hybrid import (
    LearningArchitecture, PredictiveCodingLayer, TemporalPredictiveCodingLayer,
    SparseCodingLayer, ValueNeuron, ErrorNeuron, HebbianSynapse
)

results = {"components": {}, "scaling": {}, "recommendations": []}

def profile_component(name, fn, iterations=100):
    """Profile a single component."""
    tracemalloc.start()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    entry = {
        "iterations": iterations,
        "total_ms": round(elapsed * 1000, 2),
        "avg_ms": round(elapsed * 1000 / iterations, 4),
        "peak_kb": round(peak / 1024, 1),
    }
    results["components"][name] = entry
    print(f"  {name}: {entry['avg_ms']:.4f}ms avg, {entry['peak_kb']:.1f}KB peak")
    return entry

# === Component Benchmarks ===
print("=== COMPONENT BENCHMARKS ===")

# 1. HebbianSynapse.update()
print("\n1. HebbianSynapse.update()")
syn = HebbianSynapse(rule="oja")
profile_component("hebbian_update_oja", lambda: syn.update(0.5, 0.3, 1.0), iterations=10000)

syn_ghl = HebbianSynapse(rule="ghl")
profile_component("hebbian_update_ghl", lambda: syn_ghl.update(0.5, 0.3, 1.0), iterations=10000)

# 2. ValueNeuron membrane update
print("\n2. ValueNeuron.update_membrane()")
neuron = ValueNeuron()
profile_component("neuron_update", lambda: neuron.update_membrane(0.5, 0.1), iterations=10000)

# 3. PredictiveCodingLayer inference step
print("\n3. PredictiveCodingLayer.inference_step()")
layer = PredictiveCodingLayer(n_value=64, n_inputs=64)
layer.set_input([random.gauss(0, 1) for _ in range(64)])
profile_component("pc_inference_step", layer.inference_step, iterations=1000)

# 4. PredictiveCodingLayer learn step
print("\n4. PredictiveCodingLayer.learn()")
layer2 = PredictiveCodingLayer(n_value=64, n_inputs=64)
layer2.set_input([random.gauss(0, 1) for _ in range(64)])
layer2.inference(n_steps=3)
profile_component("pc_learn", lambda: layer2.learn(0.01, 1.0), iterations=1000)

# 5. TemporalPredictiveCodingLayer predict_next
print("\n5. TemporalPredictiveCodingLayer.predict_next()")
tpc = TemporalPredictiveCodingLayer(state_dim=32, transition_rank=4)
tpc.state = [random.gauss(0, 1) for _ in range(32)]
profile_component("tpc_predict_next", tpc.predict_next, iterations=1000)

# 6. TemporalPredictiveCodingLayer inference
print("\n6. TemporalPredictiveCodingLayer.inference()")
tpc2 = TemporalPredictiveCodingLayer(state_dim=32, transition_rank=4)
obs = [random.gauss(0, 1) for _ in range(32)]
profile_component("tpc_inference", lambda: tpc2.inference(obs, n_steps=3), iterations=100)

# 7. SparseCodingLayer inference
print("\n7. SparseCodingLayer.inference()")
sc = SparseCodingLayer(input_dim=64, dict_size=128, sparsity=10)
sc_input = [random.gauss(0, 1) for _ in range(64)]
profile_component("sc_inference", lambda: sc.inference(sc_input, n_steps=10), iterations=100)

# 8. SparseCodingLayer learn
print("\n8. SparseCodingLayer.learn()")
sc2 = SparseCodingLayer(input_dim=64, dict_size=128, sparsity=10)
sc2.inference(sc_input, n_steps=10)
profile_component("sc_learn", lambda: sc2.learn(sc_input), iterations=100)

# === MNIST-Scale Benchmarks ===
print("\n=== MNIST-SCALE BENCHMARKS ===")

# MNIST: 784 inputs, hidden layers, 10 outputs
# Architecture: [784, 256, 64, 10]

def bench_mnist_forward():
    """Single forward pass at MNIST scale."""
    arch = LearningArchitecture(
        input_dim=784,
        layer_dims=[256, 64, 10]
    )
    sample = [random.gauss(0, 1) for _ in range(784)]
    arch.forward(sample)

print("\n9. MNIST forward pass (784 -> 256 -> 64 -> 10)")
tracemalloc.start()
start = time.perf_counter()
arch = LearningArchitecture(input_dim=784, layer_dims=[256, 64, 10])
sample = [random.gauss(0, 1) for _ in range(784)]
arch.forward(sample)
elapsed = time.perf_counter() - start
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
results["components"]["mnist_forward"] = {
    "iterations": 1,
    "total_ms": round(elapsed * 1000, 2),
    "avg_ms": round(elapsed * 1000, 2),
    "peak_kb": round(peak / 1024, 1),
}
print(f"  mnist_forward: {elapsed*1000:.2f}ms, {peak/1024:.1f}KB peak")

# MNIST learn step
def bench_mnist_learn():
    """Single learn step at MNIST scale."""
    arch = LearningArchitecture(input_dim=784, layer_dims=[256, 64, 10])
    sample = [random.gauss(0, 1) for _ in range(784)]
    arch.learn(sample, global_gradient=1.0)

print("\n10. MNIST learn step (784 -> 256 -> 64 -> 10)")
tracemalloc.start()
start = time.perf_counter()
arch = LearningArchitecture(input_dim=784, layer_dims=[256, 64, 10])
sample = [random.gauss(0, 1) for _ in range(784)]
arch.learn(sample, global_gradient=1.0)
elapsed = time.perf_counter() - start
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
results["components"]["mnist_learn"] = {
    "iterations": 1,
    "total_ms": round(elapsed * 1000, 2),
    "avg_ms": round(elapsed * 1000, 2),
    "peak_kb": round(peak / 1024, 1),
}
print(f"  mnist_learn: {elapsed*1000:.2f}ms, {peak/1024:.1f}KB peak")

# === Scaling Analysis ===
print("\n=== SCALING ANALYSIS ===")

# Estimate time for 1 epoch of MNIST (60,000 samples)
mnist_learn_ms = results["components"]["mnist_learn"]["avg_ms"]
epoch_time_s = (mnist_learn_ms * 60000) / 1000
results["scaling"]["mnist_epoch_estimated_s"] = round(epoch_time_s, 1)
results["scaling"]["mnist_epoch_estimated_min"] = round(epoch_time_s / 60, 1)
results["scaling"]["mnist_100_epochs_hours"] = round(epoch_time_s * 100 / 3600, 1)
print(f"  1 MNIST epoch (60K samples): ~{epoch_time_s:.1f}s ({epoch_time_s/60:.1f} min)")
print(f"  100 epochs: ~{epoch_time_s * 100 / 3600:.1f} hours")

# === Recommendations ===
print("\n=== RECOMMENDATIONS ===")
recs = []

# Check each component
for name, data in results["components"].items():
    if data["avg_ms"] > 100:
        recs.append(f"CRITICAL: {name} = {data['avg_ms']}ms — blocks real-time learning")
    elif data["avg_ms"] > 10:
        recs.append(f"HIGH: {name} = {data['avg_ms']}ms — significant overhead")
    elif data["avg_ms"] > 1:
        recs.append(f"MEDIUM: {name} = {data['avg_ms']}ms — could be optimized")

if epoch_time_s > 600:
    recs.append(f"CRITICAL: MNIST epoch = {epoch_time_s:.0f}s — need 10-100x speedup for practical training")

# Check memory
for name, data in results["components"].items():
    if data["peak_kb"] > 10000:
        recs.append(f"HIGH MEMORY: {name} = {data['peak_kb']:.0f}KB — reduce allocations")

results["recommendations"] = recs
for r in recs:
    print(f"  • {r}")

# Save report
report_path = os.path.join(REPORT_DIR, "learning_architecture_profile.json")
with open(report_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nReport saved to {report_path}")
