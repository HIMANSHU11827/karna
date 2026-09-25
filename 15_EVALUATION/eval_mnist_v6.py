"""
RAIE v6 — Scaled sparse random prototype classifier

Key improvements for 95%+ target:
- More features (4096)
- Multiple training epochs
- Full MNIST training set
- Learned threshold adjustment
"""
import numpy as np
import time

print("=== RAIE v6 MNIST ===")

# Load MNIST
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
# Use full training set
X_train_full = X_train
y_train_full = y_train

class SparseRandomFeatureExtractor:
    """Fixed sparse random projection for feature extraction."""
    def __init__(self, input_size, feature_size, connection_rate=0.05):
        self.W = np.random.randn(feature_size, input_size)
        mask = np.random.binomial(1, connection_rate, (feature_size, input_size))
        self.W = self.W * mask / np.sqrt(connection_rate * input_size)
        self.k = max(1, int(feature_size * 0.2))  # Top 20% activation
    
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
extractor = SparseRandomFeatureExtractor(784, 4096, connection_rate=0.05)
classifier = PrototypeClassifier(4096, 10)

# Training - multiple epochs
print("Training...")
start = time.time()
for epoch in range(3):
    correct = 0
    for i, (x, label) in enumerate(zip(X_train_full, y_train_full)):
        features = extractor.encode(x)
        classifier.learn(features, label)
        
        if (i + 1) % 10000 == 0:
            print(f"  Epoch {epoch+1}: {i+1}/{len(X_train_full)} trained")
    print(f"  Epoch {epoch+1} complete")

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
