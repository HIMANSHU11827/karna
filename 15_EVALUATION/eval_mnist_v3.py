"""
RAIE v3 — Prototype-based classification with sparse features

Key insight: For from-scratch Hebbian learning, prototype classification is more effective.
1. Learn sparse features via Hebbian learning
2. Store class prototypes (average feature vector per class)
3. Classify by nearest prototype (cosine similarity)
"""
import numpy as np
import time

print("=== RAIE v3 MNIST ===")

# Load MNIST
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
X_train, y_train = X_train[:5000], y_train[:5000]
X_test, y_test = X_test[:1000], y_test[:1000]

class KWTA:
    @staticmethod
    def activate(x, k):
        k = min(k, len(x))
        threshold = np.partition(x.flatten(), -k)[-k]
        result = np.where(x >= threshold, x, 0)
        if result.sum() > 0:
            result = result / result.sum()
        return result

def srbi_init(out_size, in_size, rate=0.1):
    W = np.random.binomial(1, rate, (out_size, in_size)).astype(float)
    return W / max(1, in_size * rate)

class SparseAutoencoder:
    """Sparse feature extractor via Hebbian learning."""
    def __init__(self, in_size, hidden_size, lr=0.01, sparsity=0.8):
        self.W = srbi_init(hidden_size, in_size, 0.1)
        self.M = np.zeros((hidden_size, in_size))
        self.lr = lr
        self.k = max(1, int(hidden_size * (1 - sparsity)))
    
    def encode(self, x):
        drive = self.W @ x
        memory = self.M @ x
        activation = drive + 0.1 * memory
        return KWTA.activate(activation, self.k)
    
    def learn(self, x):
        y = self.encode(x)
        # Hebbian update
        self.W += self.lr * np.outer(y, x)
        self.M += self.lr * 0.1 * np.outer(y, x)
        self.W *= 0.999
        self.M *= 0.999
        return y

class PrototypeClassifier:
    """Store class prototypes and classify by nearest neighbor."""
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

# Build network
encoder = SparseAutoencoder(784, 256, lr=0.01, sparsity=0.8)
classifier = PrototypeClassifier(256, 10)

# Training
print("Training...")
start = time.time()
for i, (x, label) in enumerate(zip(X_train, y_train)):
    features = encoder.learn(x)
    classifier.learn(features, label)
    
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(X_train)} trained")

train_time = time.time() - start
print(f"Training: {train_time:.1f}s")

# Evaluation
print("\nEvaluating...")
correct = 0
for x, label in zip(X_test, y_test):
    features = encoder.encode(x)
    pred = classifier.predict(features)
    if pred == label:
        correct += 1

accuracy = correct / len(X_test)
print(f"\n{'='*40}")
print(f"MNIST Accuracy: {accuracy*100:.1f}%")
print(f"Correct: {correct}/{len(X_test)}")
print(f"Target: 95%+")
print(f"{'='*40}")
