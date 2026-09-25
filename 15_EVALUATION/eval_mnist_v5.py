"""
RAIE v5 — Sparse random prototype with supervised prototype refinement

Key insight: Random projections + nearest prototype works, but need more features.
Use MANY more random features (sparse) and let prototypes naturally cluster.

Algorithm:
1. Sparse random projection to high-dimensional space
2. Class prototypes = average projected features per class
3. Classification = nearest prototype via cosine similarity
"""
import numpy as np
import time

print("=== RAIE v5 MNIST ===")

# Load MNIST
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
X_train, y_train = X_train[:5000], y_train[:5000]
X_test, y_test = X_test[:1000], y_test[:1000]

class SparseRandomFeatureExtractor:
    """Fixed sparse random projection for feature extraction."""
    def __init__(self, input_size, feature_size, connection_rate=0.05):
        # Fixed sparse random projection matrix (each feature connects to 5% of inputs)
        self.W = np.random.randn(feature_size, input_size)
        mask = np.random.binomial(1, connection_rate, (feature_size, input_size))
        self.W = self.W * mask / np.sqrt(connection_rate * input_size)
        self.k = max(1, int(feature_size * 0.1))  # Top 10% activation
    
    def encode(self, x):
        features = self.W @ x
        # ReLU
        features = np.maximum(features, 0)
        # Sparse: top-k
        k = self.k
        k = min(k, len(features))
        threshold = np.partition(features.flatten(), -k)[-k]
        sparse_features = np.where(features >= threshold, features, 0)
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

# Build with more features
extractor = SparseRandomFeatureExtractor(784, 2048, connection_rate=0.05)
classifier = PrototypeClassifier(2048, 10)

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
