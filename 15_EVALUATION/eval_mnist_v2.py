"""
RAIE v2 — Proper classification layer for MNIST

Key fix: Last layer produces class probabilities via supervised Hebbian learning.
First layers: feature extraction (KWTA + Hebbian)
Last layer: prototype-based classification
"""
import numpy as np
import time

print("=== RAIE v2 MNIST ===")

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

# SRBI init
def srbi_init(out_size, in_size, rate=0.1):
    W = np.random.binomial(1, rate, (out_size, in_size)).astype(float)
    return W / max(1, in_size * rate)

# Feature extraction layer
class FeatureLayer:
    def __init__(self, in_size, out_size, lr=0.01, sparsity=0.9):
        self.W = srbi_init(out_size, in_size, 0.1)
        self.M = np.zeros((out_size, in_size))
        self.lr = lr
        self.k = max(1, int(out_size * (1 - sparsity)))
    
    def encode(self, x):
        drive = self.W @ x
        memory = self.M @ x
        activation = drive + 0.1 * memory
        return KWTA.activate(activation, self.k)
    
    def learn(self, x, y):
        self.W += self.lr * np.outer(y, x)
        self.M += self.lr * 0.1 * np.outer(y, x)
        self.W *= 0.999
        self.M *= 0.999

# Supervised classifier layer
class PrototypeClassifier:
    """Each neuron corresponds to a class. Learns class prototypes."""
    def __init__(self, feature_size, num_classes, lr=0.1):
        self.W = srbi_init(num_classes, feature_size, 0.5)  # 10 classes
        self.lr = lr
        self.num_classes = num_classes
    
    def predict(self, features):
        return np.argmax(self.W @ features)
    
    def learn(self, features, label):
        # Supervised Hebbian: strengthen connection to correct class
        target = np.zeros(self.num_classes)
        target[label] = 1.0
        # Also weaken connections to incorrect classes (competitive)
        pred = self.W @ features
        pred_label = np.argmax(pred)
        if pred_label != label:
            # Move weights toward correct class
            self.W[label] += self.lr * features
            # Move weights away from incorrect class
            self.W[pred_label] -= self.lr * 0.5 * features

# Build network
feature1 = FeatureLayer(784, 256, lr=0.01, sparsity=0.9)
feature2 = FeatureLayer(256, 64, lr=0.01, sparsity=0.9)
classifier = PrototypeClassifier(64, 10, lr=0.1)

# Training
print("Training...")
start = time.time()
for i, (x, label) in enumerate(zip(X_train, y_train)):
    # Forward pass
    h1 = feature1.encode(x)
    h2 = feature2.encode(h1)
    
    # Learn features
    feature1.learn(x, h1)
    feature2.learn(h1, h2)
    
    # Learn classifier
    classifier.learn(h2, label)
    
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(X_train)} trained")

train_time = time.time() - start
print(f"Training: {train_time:.1f}s")

# Evaluation
print("\nEvaluating...")
correct = 0
for x, label in zip(X_test, y_test):
    h1 = feature1.encode(x)
    h2 = feature2.encode(h1)
    pred = classifier.predict(h2)
    if pred == label:
        correct += 1

accuracy = correct / len(X_test)
print(f"\n{'='*40}")
print(f"MNIST Accuracy: {accuracy*100:.1f}%")
print(f"Correct: {correct}/{len(X_test)}")
print(f"Target: 95%+")
print(f"{'='*40}")
