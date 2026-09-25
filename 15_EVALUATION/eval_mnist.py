"""
RAIE MNIST Evaluation — target: 95%+
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '19_PROTOTYPES'))
from raie_network import RAIENetwork
import numpy as np
import time

print("=== RAIE MNIST Evaluation ===")

# Load MNIST
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

# Subsample for quick test
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]

print(f"Training: {X_train.shape}, Testing: {X_test.shape}")

# Create RAIE network: 784 -> 128 -> 64 -> 10
net = RAIENetwork([784, 128, 64, 10], learning_rate=0.01, sparsity=0.9)

# Training
print("\nTraining...")
start = time.time()
for i, (x, label) in enumerate(zip(X_train, y_train)):
    net.train_step(x, label)
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(X_train)} samples trained")
train_time = time.time() - start
print(f"Training completed in {train_time:.1f}s")

# Evaluation
print("\nEvaluating...")
correct = 0
for x, label in zip(X_test, y_test):
    pred_label, _ = net.predict(x)
    if pred_label == label:
        correct += 1

accuracy = correct / len(X_test)
print(f"\n{'='*40}")
print(f"MNIST Accuracy: {accuracy*100:.1f}%")
print(f"Correct: {correct}/{len(X_test)}")
print(f"Target: 95%+")
print(f"{'='*40}")
