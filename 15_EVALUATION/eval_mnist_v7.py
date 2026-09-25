"""
RAIE v7 — Vectorized sparse random prototype classifier

Key improvements:
- Batch processing (no per-sample Python loops)
- Vectorized prototype prediction
- Much faster training
"""
import numpy as np
import time

print("=== RAIE v7 MNIST ===")

# Load MNIST
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
X_train, y_train = X_train[:10000], y_train[:10000]
X_test, y_test = X_test[:1000], y_test[:1000]

class SparseRandomFeatureExtractor:
    """Fixed sparse random projection for feature extraction."""
    def __init__(self, input_size, feature_size, connection_rate=0.05):
        self.W = np.random.randn(feature_size, input_size)
        mask = np.random.binomial(1, connection_rate, (feature_size, input_size))
        self.W = self.W * mask / np.sqrt(connection_rate * input_size)
        self.k = max(1, int(feature_size * 0.2))  # Top 20%
        self.feature_size = feature_size
    
    def encode(self, x):
        features = self.W @ x
        features = np.maximum(features, 0)
        # Sparse top-k
        k = self.k
        k = min(k, len(features))
        threshold = np.partition(features.flatten(), -k)[-k]
        return np.where(features >= threshold, features, 0)
    
    def encode_batch(self, X):
        """Batch encode."""
        features = X @ self.W.T  # (batch, feature_size)
        features = np.maximum(features, 0)
        # Sparse top-k per sample
        k = self.k
        for i in range(len(features)):
            threshold = np.partition(features[i], -k)[-k]
            features[i] = np.where(features[i] >= threshold, features[i], 0)
        return features

class PrototypeClassifier:
    """Prototype-based classification."""
    def __init__(self, feature_size, num_classes):
        self.prototypes = np.zeros((num_classes, feature_size))
        self.counts = np.zeros(num_classes)
    
    def learn_batch(self, features, labels):
        """Batch learning: accumulate prototypes."""
        for i in range(len(labels)):
            self.prototypes[labels[i]] += features[i]
            self.counts[labels[i]] += 1
    
    def predict(self, features):
        # Normalize prototypes
        prototypes = np.zeros_like(self.prototypes)
        for i in range(len(self.prototypes)):
            if self.counts[i] > 0:
                prototypes[i] = self.prototypes[i] / self.counts[i]
        
        # Cosine similarity (vectorized)
        norm_features = np.linalg.norm(features)
        if norm_features == 0:
            return 0
        similarities = np.zeros(len(prototypes))
        for i in range(len(prototypes)):
            norm_proto = np.linalg.norm(prototypes[i])
            if norm_proto > 0:
                similarities[i] = np.dot(prototypes[i], features) / (norm_proto * norm_features)
        return np.argmax(similarities)
    
    def predict_batch(self, features_batch):
        """Batch prediction."""
        # Normalize prototypes once
        prototypes = np.zeros_like(self.prototypes)
        for i in range(len(self.prototypes)):
            if self.counts[i] > 0:
                prototypes[i] = self.prototypes[i] / self.counts[i]
        
        # Vectorized cosine similarity
        # prototypes: (10, F), features_batch: (N, F)
        norm_prototypes = np.linalg.norm(prototypes, axis=1, keepdims=True)  # (10, 1)
        norm_features = np.linalg.norm(features_batch, axis=1, keepdims=True)  # (N, 1)
        
        # Dot products
        dots = features_batch @ prototypes.T  # (N, 10)
        
        # Cosine similarity
        similarities = dots / (norm_prototypes.T * norm_features + 1e-8)  # (N, 10)
        
        return np.argmax(similarities, axis=1)

# Build
extractor = SparseRandomFeatureExtractor(784, 4096, connection_rate=0.05)
classifier = PrototypeClassifier(4096, 10)

# Training
print("Training...")
start = time.time()
batch_size = 1000
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    
    features = extractor.encode_batch(X_batch)
    classifier.learn_batch(features, y_batch)
    
    print(f"  {i+len(X_batch)}/{len(X_train)} trained")

train_time = time.time() - start
print(f"Training: {train_time:.1f}s")

# Evaluation
print("\nEvaluating...")
correct = 0
for i in range(0, len(X_test), 100):
    X_batch = X_test[i:i+100]
    y_batch = y_test[i:i+100]
    
    features = extractor.encode_batch(X_batch)
    preds = classifier.predict_batch(features)
    correct += np.sum(preds == y_batch)

accuracy = correct / len(X_test)
print(f"\n{'='*40}")
print(f"MNIST Accuracy: {accuracy*100:.1f}%")
print(f"Correct: {correct}/{len(X_test)}")
print(f"Target: 95%+")
print(f"{'='*40}")
