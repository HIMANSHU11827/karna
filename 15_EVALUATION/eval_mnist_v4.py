"""
RAIE v4 — Sparse random features + prototype classification

Key insight: For from-scratch learning, fixed sparse random projections can work well.
Random projections preserve distances in high dimensions (Johnson-Lindenstrauss).
Sparse random = efficient, neuromorphic-friendly.

Algorithm:
1. Generate fixed sparse random projection matrix
2. For each input, compute sparse features via top-k
3. Store class prototypes (average features per class)
4. Classify by nearest prototype (cosine similarity)
"""
import numpy as np
import time

print("=== RAIE v4 MNIST ===")

# Load MNIST
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
X_train, y_train = X_train[:5000], y_train[:5000]
X_test, y_test = X_test[:1000], y_test[:1000]

class SparseRandomFeatureExtractor:
    """Fixed sparse random projection for feature extraction."""
    def __init__(self, input_size, feature_size, connection_rate=0.1):
        # Fixed sparse random projection matrix
        self.W = np.random.binomial(1, connection_rate, (feature_size, input_size)).astype(float)
        # Normalize
        self.W = self.W / np.sqrt(connection_rate * input_size)
        self.k = max(1, int(feature_size * 0.1))  # Top 10% activation
    
    def encode(self, x):
        # Project to feature space
        features = self.W @ x
        # Sparse activation: top-k
        k = self.k
        k = min(k, len(features))
        threshold = np.partition(features.flatten(), -k)[-k]
        sparse_features = np.where(features >= threshold, features, 0)
        # Normalize
        if sparse_features.sum() > 0:
            sparse_features = sparse_features / sparse_features.sum()
        return sparse_features

class PrototypeClassifier:
    """Prototype-based classification."""
    def __init__(self, feature_size, num_classes):
        self.prototypes = np.zeros((num_classes, feature_size))
        self.counts = np.zeros(num_classes)
    
    def learn(self, features, label):
        self.prototypes[label] += features
        self.counts[label] += 1
    
    def predict(self, features):
        # Normalize prototypes
        prototypes = np.zeros_like(self.prototypes)
        for i in range(len(self.prototypes)):
            if self.counts[i] > 0:
                prototypes[i] = self.prototypes[i] / self.counts[i]
        
        # Cosine similarity
        similarities = np.zeros(len(prototypes))
        for i in range(len(prototypes)):
            norm = np.linalg.norm(prototypes[i]) * np.linalg.norm(features)
            if norm > 0:
                similarities[i] = np.dot(prototypes[i], features) / norm
        
        return np.argmax(similarities)

# Build
extractor = SparseRandomFeatureExtractor(784, 512, connection_rate=0.1)
classifier = PrototypeClassifier(512, 10)

# Training
print("Training...")
start = time.time()
for i, (x, label) in enumerate(zip(X_train, y_train)):
    features = extractor.encode(x)
    classifier.learn(features, label)
    
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(X_train)} trained")

train_time = time.time() - start
print(f"Training: {train_time:.1f}s")

# Evaluation
print("\nEvaluating...")
correct = 0
for x, label in zip(X_test, y_test):
    features = extractor.encode(x)
    pred = classifier.predict(features)
    if pred == label:
        correct += 1

accuracy = correct / len(X_test)
print(f"\n{'='*40}")
print(f"MNIST Accuracy: {accuracy*100:.1f}%")
print(f"Correct: {correct}/{len(X_test)}")
print(f"Target: 95%+")
print(f"{'='*40}")
