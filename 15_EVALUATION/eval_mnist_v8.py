"""
RAIE v8 — Raw pixel prototype classifier

Key insight: For MNIST, raw pixels are already discriminative.
Each digit has a distinct pixel pattern. Prototype classification on raw pixels should work.

Algorithm:
1. For each class, compute the average pixel pattern (prototype)
2. Classify by nearest prototype (cosine similarity)
3. No random projection needed

This is essentially a 1-NN classifier with class prototypes.
For MNIST, this should get ~90%+.
"""
import numpy as np
import time

print("=== RAIE v8 MNIST ===")

# Load MNIST
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

class RawPrototypeClassifier:
    """Prototype classification on raw pixels."""
    def __init__(self, num_classes):
        self.prototypes = np.zeros((num_classes, 784))
        self.counts = np.zeros(num_classes)
    
    def learn(self, x, label):
        self.prototypes[label] += x
        self.counts[label] += 1
    
    def predict(self, x):
        # Normalize prototypes
        prototypes = np.zeros_like(self.prototypes)
        for i in range(len(self.prototypes)):
            if self.counts[i] > 0:
                prototypes[i] = self.prototypes[i] / self.counts[i]
        
        # Cosine similarity
        similarities = np.zeros(len(prototypes))
        for i in range(len(prototypes)):
            norm = np.linalg.norm(prototypes[i]) * np.linalg.norm(x)
            if norm > 0:
                similarities[i] = np.dot(prototypes[i], x) / norm
        
        return np.argmax(similarities)
    
    def predict_batch(self, X):
        # Normalize prototypes
        prototypes = np.zeros_like(self.prototypes)
        for i in range(len(self.prototypes)):
            if self.counts[i] > 0:
                prototypes[i] = self.prototypes[i] / self.counts[i]
        
        # Vectorized cosine similarity
        norm_prototypes = np.linalg.norm(prototypes, axis=1, keepdims=True)
        norm_X = np.linalg.norm(X, axis=1, keepdims=True)
        
        dots = X @ prototypes.T
        similarities = dots / (norm_prototypes.T * norm_X + 1e-8)
        
        return np.argmax(similarities, axis=1)

# Build
classifier = RawPrototypeClassifier(10)

# Training
print("Training...")
start = time.time()
for i, (x, label) in enumerate(zip(X_train, y_train)):
    classifier.learn(x, label)
    if (i + 1) % 10000 == 0:
        print(f"  {i+1}/{len(X_train)} trained")

train_time = time.time() - start
print(f"Training: {train_time:.1f}s")

# Evaluation
print("\nEvaluating...")
correct = 0
for i in range(0, len(X_test), 100):
    X_batch = X_test[i:i+100]
    y_batch = y_test[i:i+100]
    
    preds = classifier.predict_batch(X_batch)
    correct += np.sum(preds == y_batch)

accuracy = correct / len(X_test)
print(f"\n{'='*40}")
print(f"MNIST Accuracy: {accuracy*100:.1f}%")
print(f"Correct: {correct}/{len(X_test)}")
print(f"Target: 95%+")
print(f"{'='*40}")
