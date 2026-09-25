"""
Krotov & Hopfield (2019) Style — Single Layer
==============================================
Simplified to single hidden layer for MNIST.

Key design:
- Input (784) → Output (10) directly
- No hidden layers (avoids credit assignment problem)
- WTA activation with power p=2
- Anti-Hebbian on inactive neurons
- Linear classifier on top is the output itself

This SHOULD work because:
1. MNIST digits are ~90% linearly separable
2. Hebbian updates find good input→class mappings
3. No backprop needed (single layer)
"""

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
                break
            except:
                pass
    return data_dir

def load_mnist(data_dir="/tmp/mnist_data"):
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int32)
    return (load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz")),
            load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz")),
            load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")),
            load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")))


# =============================================================================
# Krotov & Hopfield Single Layer
# =============================================================================

class KrotovLayer:
    """
    Krotov & Hopfield (2019) style single-layer network.
    
    Forward:
      h = |W^T @ x|^p    (powered similarity)
      WTA: only top-k active
    
    Learning:
      Winners: Δw = η * h * (x - h * w)   (Hebbian + Oja)
      Losers:  Δw = -η * h * w            (Anti-Hebbian decay)
    """
    
    def __init__(self, input_dim, output_dim, p=2, lr=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p = p
        self.lr = lr
        
        # Weights (input_dim x output_dim)
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
        self._normalize()
        
    def _normalize(self):
        """Normalize columns."""
        norms = np.linalg.norm(self.W, axis=0, keepdims=True)
        self.W /= np.maximum(norms, 1e-8)
    
    def forward(self, x):
        """Forward pass."""
        # Similarity
        h = np.abs(self.W.T @ x) ** self.p
        self.last_input = x
        self.last_h = h
        return h
    
    def learn(self, target_class=None):
        """
        Krotov & Hopfield learning.
        
        If target_class is provided: that neuron is forced to be winner.
        Otherwise: use WTA.
        """
        x = self.last_input
        h = self.last_h
        
        # Find winner
        if target_class is not None:
            winner = target_class
        else:
            winner = np.argmax(h)
        
        # Hebbian update for winner
        # Δw = η * h * (x - h * w)
        w_winner = self.W[:, winner].copy()
        delta = self.lr * h[winner] * (x - h[winner] * w_winner)
        self.W[:, winner] += delta
        
        # Anti-Hebbian for all others (decay)
        for j in range(self.output_dim):
            if j != winner:
                self.W[:, j] *= (1 - self.lr * 0.01)
        
        # Renormalize
        self._normalize()
        
        return winner
    
    def predict(self, x):
        """Predict class."""
        h = self.forward(x)
        return np.argmax(h)


# =============================================================================
# Dense Hebbian (No WTA) — Alternative
# =============================================================================

class DenseHebbianLayer:
    """
    Dense Hebbian learning (no WTA).
    
    Forward: h = ReLU(W^T @ x)
    Learn:   ΔW = η * (target - h) * x  (perceptron/Hebbian hybrid)
    
    This is equivalent to training a perceptron with SGD,
    but framed as Hebbian learning.
    """
    
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01
        self.b = np.zeros(output_dim, dtype=np.float32)
    
    def forward(self, x):
        """Forward pass."""
        self.last_input = x
        h = self.W.T @ x + self.b
        self.last_h = h
        return h
    
    def learn(self, target):
        """
        Perceptron/Hebbian hybrid learning.
        
        For target class: Δw = η * (1 - h) * x  (push up)
        For others:       Δw = η * (0 - h) * x  (push down)
        """
        x = self.last_input
        h = self.last_h
        
        # One-hot target
        target_vec = np.zeros(self.output_dim, dtype=np.float32)
        target_vec[target] = 1.0
        
        # Error
        error = target_vec - h
        
        # Update (outer product)
        self.W += self.lr * np.outer(x, error)
        self.b += self.lr * error
        
        return np.mean(error ** 2)
    
    def predict(self, x):
        """Predict class."""
        h = self.forward(x)
        return np.argmax(h)


# =============================================================================
# Two-Layer Hybrid (Dense Hebbian hidden + Linear readout)
# =============================================================================

class TwoLayerHybrid:
    """
    Two-layer network with:
    - Hidden layer: Dense Hebbian (unsupervised, finds features)
    - Output layer: Linear classifier (supervised, trained with labels)
    
    The hidden layer learns with Oja's rule (unsupervised).
    The output layer learns with perceptron rule (supervised).
    No backprop between them.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.hidden = DenseHebbianLayer(input_dim, hidden_dim, lr)
        self.output = DenseHebbianLayer(hidden_dim, output_dim, lr)
        
        # Override hidden learning to be unsupervised Oja
        self.hidden.W = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        self.hidden.b = np.zeros(hidden_dim, dtype=np.float32)
        self.hidden.lr = lr  # fix: add learning_rate attribute
        
    def forward(self, x):
        """Forward pass."""
        h = self.hidden.forward(x)
        h = np.maximum(h, 0)  # ReLU
        self.last_hidden = h
        o = self.output.forward(h)
        return o
    
    def learn_unsupervised(self, x, n_steps=3):
        """Unsupervised Oja learning on hidden layer."""
        for step in range(n_steps):
            h = self.hidden.forward(x)
            h_act = np.maximum(h, 0)
            
            # Oja's rule
            lr = self.hidden.lr
            delta = lr * (
                np.outer(x, h_act) - self.hidden.W * (h_act ** 2)[None, :]
            )
            self.hidden.W += delta
            self.hidden.W = np.clip(self.hidden.W, -2, 2)
    
    def learn_supervised(self, x, target):
        """Supervised perceptron learning on output layer."""
        o = self.forward(x)
        loss = self.output.learn(target)
        return loss
    
    def predict(self, x):
        """Predict class."""
        o = self.forward(x)
        return np.argmax(o)


# =============================================================================
# Training Loop
# =============================================================================

def train_krotov(model, train_images, train_labels, test_images, test_labels,
                 epochs=20):
    """Train Krotov model."""
    n_samples = len(train_images)
    
    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        correct = 0
        total = 0
        
        for idx in indices:
            x = train_images[idx]
            label = train_labels[idx]
            
            # Forward
            h = model.forward(x)
            pred = np.argmax(h)
            
            if pred == label:
                correct += 1
            total += 1
            
            # Learn (with target class as winner)
            model.learn(target_class=label)
        
        train_acc = correct / total
        
        # Test
        test_correct = 0
        for i in range(len(test_images)):
            pred = model.predict(test_images[i])
            if pred == test_labels[i]:
                test_correct += 1
        test_acc = test_correct / len(test_images)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    
    return test_acc


def train_dense_hebbian(model, train_images, train_labels, test_images, test_labels,
                        epochs=20):
    """Train Dense Hebbian model."""
    n_samples = len(train_images)
    
    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for idx in indices:
            x = train_images[idx]
            label = train_labels[idx]
            
            # Forward
            pred = model.predict(x)
            if pred == label:
                correct += 1
            total += 1
            
            # Learn
            loss = model.learn(target=label)
            total_loss += loss
        
        train_acc = correct / total
        
        # Test
        test_correct = 0
        for i in range(len(test_images)):
            pred = model.predict(test_images[i])
            if pred == test_labels[i]:
                test_correct += 1
        test_acc = test_correct / len(test_images)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
                  f"loss={total_loss/n_samples:.4f}")
    
    return test_acc


def train_two_layer(model, train_images, train_labels, test_images, test_labels,
                    epochs=20):
    """Train two-layer hybrid."""
    n_samples = len(train_images)
    
    # Phase 1: Unsupervised feature learning
    print("  Unsupervised phase...")
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    for idx in indices[:1000]:
        model.learn_unsupervised(train_images[idx], n_steps=1)
    
    # Phase 2: Supervised classifier training
    print("  Supervised phase...")
    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        correct = 0
        total = 0
        
        for idx in indices:
            x = train_images[idx]
            label = train_labels[idx]
            
            pred = model.predict(x)
            if pred == label:
                correct += 1
            total += 1
            
            model.learn_supervised(x, label)
        
        train_acc = correct / total
        
        # Test
        test_correct = 0
        for i in range(len(test_images)):
            pred = model.predict(test_images[i])
            if pred == test_labels[i]:
                test_correct += 1
        test_acc = test_correct / len(test_images)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    
    return test_acc


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("MNIST — Krotov & Hopfield (2019) Style")
    print("=" * 70)
    
    # Load data
    data_dir = download_mnist()
    try:
        train_images, train_labels, test_images, test_labels = load_mnist()
        print(f"Loaded MNIST: {len(train_images)} train, {len(test_images)} test")
    except Exception as e:
        print(f"Failed to load MNIST: {e}")
        return
    
    # Use subset
    n_train = min(6000, len(train_images))
    n_test = min(1000, len(test_images))
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]
    
    print(f"Using: {n_train} train, {n_test} test")
    
    results = {}
    
    # Model 1: Krotov WTA (single layer, 784 -> 10)
    print("\n" + "-" * 50)
    print("Model 1: Krotov WTA (784 -> 10)")
    print("-" * 50)
    krotov = KrotovLayer(784, 10, p=2, lr=0.1)
    krotov_acc = train_krotov(krotov, train_images, train_labels,
                               test_images, test_labels, epochs=20)
    print(f"  Final test_acc: {krotov_acc:.4f}")
    results["krotov_wta"] = krotov_acc
    
    # Model 2: Dense Hebbian (single layer, 784 -> 10)
    print("\n" + "-" * 50)
    print("Model 2: Dense Hebbian (784 -> 10)")
    print("-" * 50)
    dense = DenseHebbianLayer(784, 10, lr=0.01)
    dense_acc = train_dense_hebbian(dense, train_images, train_labels,
                                     test_images, test_labels, epochs=20)
    print(f"  Final test_acc: {dense_acc:.4f}")
    results["dense_hebbian"] = dense_acc
    
    # Model 3: Two-layer hybrid (784 -> 128 -> 10)
    print("\n" + "-" * 50)
    print("Model 3: Two-layer Hybrid (784 -> 128 -> 10)")
    print("-" * 50)
    hybrid = TwoLayerHybrid(784, 128, 10, lr=0.01)
    hybrid_acc = train_two_layer(hybrid, train_images, train_labels,
                                  test_images, test_labels, epochs=20)
    print(f"  Final test_acc: {hybrid_acc:.4f}")
    results["two_layer_hybrid"] = hybrid_acc
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for name, acc in results.items():
        print(f"  {name:25s}: {acc:.4f}")
    
    # Save
    output = {"dataset": "mnist", "n_train": n_train, "n_test": n_test,
              "results": {k: float(v) for k, v in results.items()}}
    path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/11_LOGS/krotov_results.json"
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {path}")
    
    return output


if __name__ == "__main__":
    main()
