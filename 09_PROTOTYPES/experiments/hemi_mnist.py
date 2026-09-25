import numpy as np
import os
import time
import json
import urllib.request
import gzip
from pathlib import Path


# =============================================================================
# MNIST Loader
# =============================================================================

def download_mnist(data_dir="/tmp/mnist_data"):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ]
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            continue
        for mirror in mirrors:
            try:
                urllib.request.urlretrieve(mirror + filename, filepath)
                print(f"  Downloaded {filename}")
                break
            except Exception as e:
                pass
    return data_dir

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int32)

def load_mnist(data_dir="/tmp/mnist_data"):
    try:
        train_images = load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
        train_labels = load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
        test_images = load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
        test_labels = load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    except FileNotFoundError:
        raise FileNotFoundError("MNIST data not available")
    return train_images, train_labels, test_images, test_labels


# =============================================================================
# Linear Classifier
# =============================================================================

class LinearClassifier:
    def __init__(self, input_dim, output_dim, lr=0.1):
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01
        self.b = np.zeros(output_dim, dtype=np.float32)
        self.lr = lr
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)
        self.momentum = 0.9

    def forward(self, x):
        logits = x @ self.W + self.b
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def train_step(self, features, target):
        probs = self.forward(features)
        loss = -np.log(max(probs[target], 1e-8))
        delta = probs.copy()
        delta[target] -= 1.0
        grad_W = np.outer(features, delta)
        grad_b = delta
        self.v_W = self.momentum * self.v_W - self.lr * grad_W
        self.v_b = self.momentum * self.v_b - self.lr * grad_b
        self.W += self.v_W
        self.b += self.v_b
        return loss

    def predict(self, features):
        return np.argmax(self.forward(features))


# =============================================================================
# Stacked Hebbian Extractor
# =============================================================================

class StackedHebbianExtractor:
    def __init__(self, input_dim, layer_sizes, lr=0.01):
        self.layers = []
        prev_dim = input_dim
        for size in layer_sizes:
            self.layers.append({
                'W': np.random.randn(prev_dim, size).astype(np.float32) * 0.1,
                'b': np.zeros(size, dtype=np.float32),
                'lr': lr
            })
            prev_dim = size

    def _train_layer(self, layer_idx, data, n_epochs=5):
        layer = self.layers[layer_idx]
        W = layer['W']
        lr = layer['lr']
        norms = np.linalg.norm(W, axis=0, keepdims=True)
        W = W / np.maximum(norms, 1e-8)
        layer['W'] = W
        n_samples = min(5000, len(data))
        data = data[:n_samples]
        for epoch in range(n_epochs):
            np.random.shuffle(data)
            for x in data:
                h = W.T @ x
                delta = lr * (np.outer(x, h) - W * (h ** 2)[None, :])
                W += delta
                W = np.clip(W, -2, 2)
            norms = np.linalg.norm(W, axis=0, keepdims=True)
            W = W / np.maximum(norms, 1e-8)
            layer['W'] = W

    def train_unsupervised(self, train_images, n_epochs=5):
        current_data = train_images
        for i, layer in enumerate(self.layers):
            print(f"  Training layer {i+1} ({self.layers[i]['W'].shape})...")
            self._train_layer(i, current_data, n_epochs)
            new_data = []
            for x in current_data[:3000]:
                h = self.layers[i]['W'].T @ x
                h = np.maximum(h, 0)
                new_data.append(h)
            current_data = np.array(new_data, dtype=np.float32)

    def extract_features(self, x):
        current = x
        for layer in self.layers:
            h = layer['W'].T @ current
            current = np.maximum(h, 0)
        return current


# =============================================================================
# Two-Layer Hebbian + Linear
# =============================================================================

class TwoLayerHebbianLinear:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.classifier = LinearClassifier(hidden_dim, output_dim, lr=0.1)
        self.lr = lr

    def _normalize_W1(self):
        norms = np.linalg.norm(self.W1, axis=0, keepdims=True)
        self.W1 = self.W1 / np.maximum(norms, 1e-8)

    def train_hidden(self, train_images, n_epochs=5):
        self._normalize_W1()
        n_samples = min(6000, len(train_images))
        for epoch in range(n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for idx in indices:
                x = train_images[idx]
                h = self.W1.T @ x + self.b1
                delta = self.lr * (np.outer(x, h) - self.W1 * (h ** 2)[None, :])
                self.W1 += delta
                self.b1 += self.lr * h * 0.1
                self.W1 = np.clip(self.W1, -2, 2)
            self._normalize_W1()

    def extract_hidden(self, x):
        return np.maximum(self.W1.T @ x + self.b1, 0)

    def train_classifier(self, train_images, train_labels, n_epochs=20):
        for epoch in range(n_epochs):
            indices = np.arange(len(train_images))
            np.random.shuffle(indices)
            total_loss = 0
            correct = 0
            total = 0
            for idx in indices:
                features = self.extract_hidden(train_images[idx])
                loss = self.classifier.train_step(features, train_labels[idx])
                total_loss += loss
                pred = self.classifier.predict(features)
                if pred == train_labels[idx]:
                    correct += 1
                total += 1
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: loss={total_loss/total:.4f}, "
                      f"train_acc={correct/total:.4f}")

    def evaluate(self, test_images, test_labels):
        correct = 0
        for i in range(len(test_images)):
            features = self.extract_hidden(test_images[i])
            if self.classifier.predict(features) == test_labels[i]:
                correct += 1
        return correct / len(test_labels)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("MNIST — Hebbian Features + Linear Classifier")
    print("=" * 70)

    data_dir = download_mnist()
    try:
        train_images, train_labels, test_images, test_labels = load_mnist()
        print(f"Loaded MNIST: {len(train_images)} train, {len(test_images)} test")
    except Exception as e:
        print(f"Failed: {e}")
        return

    n_train = 6000
    n_test = 1000
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]

    input_dim = 784
    hidden_dim = 128
    output_dim = 10

    results = {}

    # Experiment 1: Random Projection + kNN
    print("\n--- Experiment 1: Random Projection + kNN ---")
    rng = np.random.RandomState(42)
    proj = rng.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    train_features = np.array([np.maximum(proj.T @ x, 0) for x in train_images])
    correct = 0
    for i in range(n_test):
        test_feat = np.maximum(proj.T @ test_images[i], 0)
        dists = np.linalg.norm(train_features - test_feat, axis=1)
        nearest = np.argsort(dists)[:5]
        votes = train_labels[nearest]
        pred = np.bincount(votes).argmax()
        if pred == test_labels[i]:
            correct += 1
    rand_acc = correct / n_test
    print(f"  Random Projection + kNN(5): {rand_acc:.4f}")
    results["random_proj_knn"] = rand_acc

    # Experiment 2: Single Hebbian Layer + Linear
    print("\n--- Experiment 2: Single Hebbian + Linear ---")
    single = TwoLayerHebbianLinear(input_dim, hidden_dim, output_dim, lr=0.01)
    single.train_hidden(train_images, n_epochs=5)
    single.train_classifier(train_images, train_labels, n_epochs=20)
    acc2 = single.evaluate(test_images, test_labels)
    print(f"  Single Hebbian + Linear: {acc2:.4f}")
    results["single_hebbian_linear"] = acc2

    # Experiment 3: Stacked Hebbian (2 layers) + Linear
    print("\n--- Experiment 3: Stacked Hebbian (2 layers) + Linear ---")
    stacked = StackedHebbianExtractor(input_dim, [64, 32], lr=0.01)
    stacked.train_unsupervised(train_images, n_epochs=3)
    train_feat = np.array([stacked.extract_features(x) for x in train_images])
    test_feat = np.array([stacked.extract_features(x) for x in test_images])
    classifier = LinearClassifier(32, output_dim, lr=0.1)
    for epoch in range(20):
        indices = np.arange(n_train)
        np.random.shuffle(indices)
        for idx in indices:
            classifier.train_step(train_feat[idx], train_labels[idx])
    correct = 0
    for i in range(n_test):
        if classifier.predict(test_feat[i]) == test_labels[i]:
            correct += 1
    stacked_acc = correct / n_test
    print(f"  Stacked Hebbian + Linear: {stacked_acc:.4f}")
    results["stacked_hebbian_linear"] = stacked_acc

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for name, acc in results.items():
        print(f"  {name:30s}: {acc:.4f}")

    output = {"dataset": "mnist", "config": {"input_dim": input_dim,
              "hidden_dim": hidden_dim, "n_train": n_train, "n_test": n_test},
              "results": {k: float(v) for k, v in results.items()}}
    path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/11_LOGS/hembi_fixed.json"
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {path}")

    return output


if __name__ == "__main__":
    main()
