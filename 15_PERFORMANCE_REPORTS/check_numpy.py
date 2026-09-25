#!/usr/bin/env python3
"""Check what's available and profile bottlenecks."""
import sys, time, json, os
sys.path.insert(0, '/home/himanshu/Desktop/AGI_RESEARCH_LAB/19_PROTOTYPES')

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

print(f"NumPy: {'YES' if HAS_NUMPY else 'NO'}")
print(f"SciPy: {'YES' if HAS_SCIPY else 'NO'}")
print(f"Numba: {'YES' if HAS_NUMBA else 'NO'}")

if HAS_NUMPY:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    # Profile key operations
    A = np.random.randn(784).astype(np.float32)
    W = np.random.randn(256, 784).astype(np.float32)
    
    # Matrix-vector product
    start = time.perf_counter()
    for _ in range(1000):
        out = W @ A
    elapsed = time.perf_counter() - start
    print(f"\nNumPy 784x256 matmul: {elapsed*1000/1000:.4f}ms avg")
    
    # Full learning architecture with NumPy
    from learning_architecture.predictive_coding_hybrid import (
        LearningArchitecture, PredictiveCodingLayer
    )
    
    # Profile numpy-accelerated version
    arch = LearningArchitecture(input_dim=784, layer_dims=[256, 64, 10])
    sample = [0.5]*784
    
    start = time.perf_counter()
    arch.forward(sample)
    elapsed = time.perf_counter() - start
    print(f"LearningArchitecture forward (pure Python lists): {elapsed*1000:.2f}ms")
    print(f"  - The bottleneck is list-of-list weights, not math")
