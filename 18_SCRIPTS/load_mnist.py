#!/usr/bin/env python3
"""Test MNIST loading and create train/test splits."""
import numpy as np
import gzip
from pathlib import Path

data_dir = Path("/home/himanshu/Desktop/AGI_RESEARCH_LAB/16_DATA")

def load_images(filename):
    with gzip.open(data_dir / filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float64) / 255.0

def load_labels(filename):
    with gzip.open(data_dir / filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data.astype(np.int64)

print("Loading MNIST...")
X_train = load_images("train-images-idx3-ubyte.gz")
y_train = load_labels("train-labels-idx1-ubyte.gz")
X_test = load_images("t10k-images-idx3-ubyte.gz")
y_test = load_labels("t10k-labels-idx1-ubyte.gz")

print(f"Train: {X_train.shape[0]} images, {y_train.shape[0]} labels")
print(f"Test: {X_test.shape[0]} images, {y_test.shape[0]} labels")
print(f"Image range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"Labels: {np.unique(y_train)}")

# Save as numpy arrays for easy loading
np.save(data_dir / "X_train.npy", X_train)
np.save(data_dir / "y_train.npy", y_train)
np.save(data_dir / "X_test.npy", X_test)
np.save(data_dir / "y_test.npy", y_test)

print("\nSaved as numpy arrays.")
