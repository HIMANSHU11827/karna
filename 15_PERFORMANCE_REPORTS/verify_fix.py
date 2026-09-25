#!/usr/bin/env python3
"""Verify the world_model.predict() optimization."""
import time, sys, tracemalloc, random
sys.path.insert(0, '/home/himanshu/Desktop/AGI_RESEARCH_LAB/19_PROTOTYPES')

from world_model import WorldModel, SimpleEnvironment, Transition, State, Action

# Reproduce the original benchmark exactly
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

print(f"Collected {sum(model.state_action_counts.values())} transitions ({len(model._index)} unique state-action pairs)")

# Benchmark predict
N = 100
tracemalloc.start()
start = time.perf_counter()
for _ in range(N):
    state = env.reset()
    for action in env.get_available_actions():
        next_state, reward = model.predict(state, action)
        # In the benchmark we don't actually step, so state stays at reset for fairness
elapsed = time.perf_counter() - start
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"\n{N * 4} predictions: {elapsed*1000:.2f}ms total, {elapsed*1000/(N*4):.4f}ms avg")
print(f"Peak memory: {peak/1024:.1f} KB")

# Also test planning (which calls predict repeatedly)
start = time.perf_counter()
for _ in range(N):
    state = env.reset()
    actions = env.get_available_actions()
    plan = model.plan(state, actions, depth=10)
elapsed = time.perf_counter() - start
print(f"\n{N} planning steps (depth=10): {elapsed*1000:.2f}ms total, {elapsed*1000/N:.2f}ms avg")
