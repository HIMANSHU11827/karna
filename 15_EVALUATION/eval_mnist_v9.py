"""
RAIE v9 — Multi-prototype sparse classifier

Key improvement: Multiple prototypes per class to capture intra-class variance.
Each digit class has multiple "subtypes" (different writing styles).

Algorithm:
1. Sparse random projection
2. Cluster each class into k sub-clusters (multiple prototypes)
3. Classify by nearest prototype among all sub-clusters
"""
import numpy as np
import time

print("=== RAIE v9 MNIST ===")

# Load MNIST
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

class SparseRandomFeatureExtractor:
    def __init__(self, input_size, feature_size, connection_rate=0.05):
        self.W = np.random.randn(feature_size, input_size)
        mask = np.random.binomial(1, connection_rate, (feature_size, input_size))
        self.W = self.W * mask / np.sqrt(connection_rate * input_size)
        self.k = max(1, int(feature_size * 0.2))
    
    def encode(self, x):
        features = self.W @ x
        features = np.maximum(features, 0)
        k = self.k
        k = min(k, len(features))
        threshold = np.partition(features.flatten(), -k)[-k]
        return np.where(features >= threshold, features, 0)
    
    def encode_batch(self, X):
        features = X @ self.W.T
        features = np.maximum(features, 0)
        k = self.k
        for i in range(len(features)):
            threshold = np.partition(features[i], -k)[-k]
            features[i] = np.where(features[i] >= threshold, features[i], 0)
        return features

class MultiPrototypeClassifier:
    """Multiple prototypes per class via online k-means."""
    def __init__(self, feature_size, num_classes, prototypes_per_class=10):
        self.num_classes = num_classes
        self.k = prototypes_per_class
        self.prototypes = np.random.randn(num_classes * prototypes_per_class, feature_size) * 0.01
        self.counts = np.zeros(num_classes * prototypes_per_class)
        self.labels = np.repeat(np.arange(num_classes), prototypes_per_class)
    
    def find_nearest_prototype(self, features):
        """Find nearest prototype index."""
        similarities = np.zeros(len(self.prototypes))
        for i in range(len(self.prototypes)):
            norm = np.linalg.norm(self.prototypes[i]) * np.linalg.norm(features)
            if norm > 0:
                similarities[i] = np.dot(self.prototypes[i], features) / norm
        return np.argmax(similarities)
    
    def learn(self, features, label):
        # Find nearest prototype
        proto_idx = self.find_nearest_prototype(features)
        
        # If it belongs to the right class, move prototype toward features
        if self.labels[proto_idx] == label:
            lr = 0.1
            self.prototypes[proto_idx] = (1 - lr) * self.prototypes[proto_idx] + lr * features
        else:
            # Find a prototype of the correct class and move it
            correct_prototypes = np.where(self.labels == label)[0]
            # Find nearest correct prototype
            best_idx = correct_prototypes[0]
            best_sim = -1
            for idx in correct_prototypes:
                norm = np.linalg.norm(self.prototypes[idx]) * np.linalg.norm(features)
                if norm > 0:
                    sim = np.dot(self.prototypes[idx], features) / norm
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = idx
            lr = 0.1
            self.prototypes[best_idx] = (1 - lr) * self.prototypes[best_idx] + lr * features
        
        self.counts[proto_idx] += 1
    
    def learn_batch(self, features_batch, labels_batch):
        for i in range(len(labels_batch)):
            self.learn(features_batch[i], labels_batch[i])
    
    def predict(self, features):
        # Classify by majority vote among top prototypes
        similarities = np.zeros(len(self.prototypes))
        for i in range(len(self.prototypes)):
            norm = np.linalg.norm(self.prototypes[i]) * np.linalg.norm(features)
            if norm > 0:
                similarities[i] = np.dot(self.prototypes[i], features) / norm
        
        # Get top-k prototypes
        k = min(5, len(similarities))
        top_indices = np.argsort(similarities)[-k:]
        top_labels = self.labels[top_indices]
        
        # Majority vote
        votes = np.zeros(self.num_classes)
        for label in top_labels:
            votes[label] += 1
        
        return np.argmax(votes)
    
    def predict_batch(self, features_batch):
        # Vectorized
        similarities = np.zeros((len(features_batch), len(self.prototypes)))
        for i in range(len(self.prototypes)):
            norm_proto = np.linalg.norm(self.prototypes[i])
            if norm_proto > 0:
                dots = features_batch @ self.prototypes[i]
                norms = np.linalg.norm(features_batch, axis=1)
                similarities[:, i] = dots / (norm_proto * norms + 1e-8)
        
        # For each sample, get majority vote among top-k prototypes
        predictions = np.zeros(len(features_batch), dtype=int)
        for i in range(len(features_batch)):
            k = min(5, len(similarities[i]))
            top_indices = np.argsort(similarities[i])[-k:]
            top_labels = self.labels[top_indices]
            votes = np.zeros(self.num_classes)
            for label in top_labels:
                votes[label] += 1
            predictions[i] = np.argmax(votes)
        
        return predictions

# Build
extractor = SparseRandomFeatureExtractor(784, 2048, connection_rate=0.05)
classifier = MultiPrototypeClassifier(2048, 10, prototypes_per_class=10)

# Training
print("Training...")
start = time.time()
for i, (x, label) in enumerate(zip(X_train, y_train)):
    features = extractor.encode(x)
    classifier.learn(features, label)
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
    
    features = extractor.encode_batch(X_batch)
    preds = classifier.predict_batch(features)
    correct += np.sum(preds == y_batch)

accuracy = correct / len(X_test)
print(f"\n{'='*40}")
print(f"MNIST Accuracy: {accuracy*100:.1f}%")
print(f"Correct: {correct}/{len(X_test)}")
print(f"Target: 95%+")
print(f"{'='*40}")
