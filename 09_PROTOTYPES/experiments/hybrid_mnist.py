"""
Hybrid Approach: Hebbian Feature Extraction + Linear Classifier
================================================================

Based on Krotov & Hopfield (2019) and modern Hebbian-Linear hybrids.

Architecture:
  Input (784) → Hebbian/BCM Feature Extraction (untrained) → Sparse Features
  Features → Linear Classifier (logistic regression, trained separately)

Key insight: Hebbian learning builds good representations.
             Linear classifier on those features = strong baseline.

NO backprop through the feature layers.
Only the top classifier is trained (and even that is just gradient descent
on a single linear layer — not "deep learning").
"""

import numpy as np
import os
import sys
import time
import json
import urllib.request
import gzip
from pathlib import Path


# =============================================================================
# MNIST Loader (with fallback)
# =============================================================================

def download_mnist(data_dir="/tmp/mnist_data"):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ]
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            continue
        for mirror in mirrors:
            try:
                urllib.request.urlretrieve(mirror + filename, filepath)
                break
            except:
                if os.path.exists(filepath):
                    os.remove(filepath)
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

def generate_synthetic_mnist(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    images = rng.rand(n, 784).astype(np.float32) * 0.5 + 0.2
    labels = rng.randint(0, 10, size=n)
    for i in range(n):
        cls = labels[i]
        r_s, c_s = (cls // 3) * 8, (cls % 3) * 8
        for r in range(r_s, min(r_s + 12, 28)):
            for c in range(c_s, min(c_s + 12, 28)):
                images[i, r * 28 + c] += 0.3
    return np.clip(images, 0, 1), labels


# =============================================================================
# Hebbian Feature Extractors (Krotov & Hopfield style)
# =============================================================================

class HebbianFeatureLayer:
    """
    Unsupervised Hebbian feature extraction layer.
    
    Implements Krotov & Hopfield (2019) style:
    - Compute similarity: h = W^p * x (element-wise power)
    - Winner-Take-All (WTA): only top-k neurons active
    - Anti-Hebbian update on inactive neurons (prevents redundancy)
    
    This extracts sparse, discriminative features.
    """
    
    def __init__(self, input_dim, output_dim, sparsity=0.1, lr=0.01, p=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.lr = lr
        self.p = p  # power for similarity (Krotov uses p=2)
        
        # Initialize weights (normalized)
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
        self._normalize_weights()
        
        # Running mean for BCM threshold
        self.running_mean = np.zeros(output_dim, dtype=np.float32)
        self.momentum = 0.9
        
    def _normalize_weights(self):
        """L2 normalize each column (feature vector)."""
        norms = np.linalg.norm(self.W, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.W /= norms
    
    def extract_features(self, x):
        """
        Extract sparse features from input.
        Implements Krotov & Hopfield WTA mechanism.
        """
        # Compute similarity (powered dot product)
        h = np.abs(self.W.T @ x) ** self.p  # (output_dim,)
        
        # Winner-Take-All: keep top-k active
        k = max(1, int(self.output_dim * self.sparsity))
        top_k_idx = np.argsort(h)[-k:]
        
        features = np.zeros(self.output_dim, dtype=np.float32)
        features[top_k_idx] = h[top_k_idx]
        
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm
        
        self.last_input = x
        self.last_features = features
        self.last_top_k = top_k_idx
        
        return features
    
    def learn(self):
        """
        Hebbian + Anti-Hebbian learning (Krotov & Hopfield style).
        
        Active neurons (winners): Hebbian (strengthen connection to input)
        Inactive neurons: Anti-Hebbian (weaken connection, prevent redundancy)
        """
        x = self.last_input
        features = self.last_features
        top_k = self.last_top_k
        
        # Hebbian update for winners
        for idx in top_k:
            # Strengthen: Δw = η * feature * (input - feature * weight)
            # This is Oja-like stabilization
            self.W[:, idx] += self.lr * features[idx] * (
                x - features[idx] * self.W[:, idx]
            )
        
        # Anti-Hebbian for non-winners (reduce correlation with current input)
        inactive = np.ones(self.output_dim, dtype=bool)
        inactive[top_k] = False
        inactive_idx = np.where(inactive)[0]
        
        for idx in inactive_idx:
            # Decay weights toward zero for non-active neurons
            self.W[:, idx] *= 0.999
        
        # Update running mean (for BCM-style threshold)
        self.running_mean = (self.momentum * self.running_mean + 
                           (1 - self.momentum) * np.abs(features))
        
        # Normalize
        self._normalize_weights()
        
        return np.mean(features[top_k])


class BCMFeatureLayer:
    """
    Bienenstock-Cooper-Munro (BCM) feature extraction.
    
    BCM solves a key problem with pure Hebbian: unbounded growth.
    It uses a sliding threshold that adapts to average activity.
    """
    
    def __init__(self, input_dim, output_dim, sparsity=0.1, lr=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.lr = lr
        
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
        self._normalize_weights()
        
        # BCM sliding threshold (per neuron)
        self.theta = np.zeros(output_dim, dtype=np.float32)
        self.momentum = 0.9
        
    def _normalize_weights(self):
        norms = np.linalg.norm(self.W, axis=0, keepdims=True)
        self.W /= np.maximum(norms, 1e-8)
    
    def extract_features(self, x):
        """Extract features with BCM activation."""
        # Linear projection
        h = self.W.T @ x
        
        # BCM activation: h * (h - θ)  (only positive part)
        activation = h * np.maximum(h - self.theta, 0)
        
        # Sparsify (top-k)
        k = max(1, int(self.output_dim * self.sparsity))
        top_k = np.argsort(activation)[-k]
        features = np.zeros(self.output_dim, dtype=np.float32)
        features[top_k] = activation[top_k]
        
        self.last_input = x
        self.last_features = features
        self.last_raw = h
        
        return features
    
    def learn(self):
        """BCM learning rule."""
        x = self.last_input
        features = self.last_features
        h = self.last_raw
        
        # BCM update
        delta = features * (h - self.theta)
        self.W += self.lr * np.outer(x, delta)
        
        # Update sliding threshold
        self.theta = (self.momentum * self.theta + 
                     (1 - self.momentum) * (features ** 2))
        
        self._normalize_weights()
        
        return np.mean(features[features > 0]) if np.any(features > 0) else 0


class PredictiveFeatureLayer:
    """
    Predictive Coding feature extraction.
    
    Learns to reconstruct input from sparse features.
    Minimizes: ||x - W @ features||² + λ * ||features||₁
    """
    
    def __init__(self, input_dim, dict_size, sparsity=5, lr=0.01):
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity = sparsity  # hard k-sparsity
        self.lr = lr
        
        # Dictionary
        self.D = np.random.randn(input_dim, dict_size).astype(np.float32) * 0.1
        self._normalize_dict()
        
    def _normalize_dict(self):
        norms = np.linalg.norm(self.D, axis=0, keepdims=True)
        self.D /= np.maximum(norms, 1e-8)
    
    def extract_features(self, x, n_steps=10):
        """
        ISTA (Iterative Shrinkage-Thresholding) for sparse coding.
        Finds sparse c that minimizes ||x - Dc||² + λ||c||₁
        """
        # Initialize
        c = np.zeros(self.dict_size, dtype=np.float32)
        
        # D^T @ D (precompute)
        DtD = self.D.T @ self.D
        
        for step in range(n_steps):
            # Gradient
            residual = x - self.D @ c
            gradient = self.D.T @ residual - DtD @ c
            
            # Update
            c += 0.1 * gradient
            
            # Soft threshold (proximal operator for L1)
            threshold = 0.1
            c = np.where(c > threshold, c - threshold,
                        np.where(c < -threshold, c + threshold, 0))
            
            # Hard k-sparsity
            if np.count_nonzero(c) > self.sparsity:
                top_k = np.argsort(np.abs(c))[-self.sparsity:]
                c_sparse = np.zeros_like(c)
                c_sparse[top_k] = c[top_k]
                c = c_sparse
        
        self.last_input = x
        self.last_features = c
        
        return c
    
    def learn(self):
        """Hebbian dictionary learning."""
        x = self.last_input
        c = self.last_features
        
        # Reconstruction error
        error = x - self.D @ c
        
        # Hebbian update on dictionary
        self.D += self.lr * np.outer(error, c)
        
        self._normalize_dict()
        
        return np.mean(error ** 2)


# =============================================================================
# Linear Classifier (trained on extracted features)
# =============================================================================

class LinearClassifier:
    """
    Simple logistic regression classifier.
    
    Trained with gradient descent (NOT backprop through feature layers).
    This is just a single linear layer — the "supervised" part is minimal.
    """
    
    def __init__(self, input_dim, output_dim, lr=0.1):
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01
        self.b = np.zeros(output_dim, dtype=np.float32)
        self.lr = lr
    
    def forward(self, x):
        """Forward pass."""
        logits = x @ self.W + self.b
        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
    def train_step(self, features, target):
        """Single training step (gradient descent on cross-entropy)."""
        probs = self.forward(features)
        
        # Cross-entropy loss
        loss = -np.log(max(probs[target], 1e-8))
        
        # Gradient (softmax + cross-entropy simplifies to: probs - target)
        delta = probs.copy()
        delta[target] -= 1.0
        
        # Update
        self.W -= self.lr * np.outer(features, delta)
        self.b -= self.lr * delta
        
        return loss
    
    def predict(self, features):
        """Predict class."""
        probs = self.forward(features)
        return np.argmax(probs)


# =============================================================================
# Hybrid Model
# =============================================================================

class HybridModel:
    """
    Hybrid: Unsupervised Hebbian Features + Supervised Linear Classifier
    
    Feature layers: trained with Hebbian/BCM/Predictive (NO labels)
    Classifier: trained with labels (gradient descent on cross-entropy)
    
    This is NOT backprop through the whole network.
    Feature layers never see labels or gradients from the classifier.
    """
    
    def __init__(self, feature_layers, classifier):
        self.feature_layers = feature_layers
        self.classifier = classifier
    
    def extract_features(self, x):
        """Extract features through all layers."""
        current = x
        for layer in self.feature_layers:
            current = layer.extract_features(current)
        return current
    
    def train_features(self, train_images, n_epochs=5):
        """Train feature layers (unsupervised, no labels)."""
        print("Training feature layers (unsupervised)...")
        for epoch in range(n_epochs):
            indices = np.arange(len(train_images))
            np.random.shuffle(indices)
            
            total_loss = 0
            for idx in indices:
                x = train_images[idx]
                features = self.extract_features(x)
                
                # Learn each layer
                for layer in self.feature_layers:
                    layer.learn()
                
                total_loss += np.sum(features ** 2)
            
            avg_loss = total_loss / len(train_images)
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}: feature_energy={avg_loss:.4f}")
    
    def train_classifier(self, train_images, train_labels, n_epochs=10):
        """Train linear classifier (supervised, with labels)."""
        print("Training linear classifier (supervised)...")
        for epoch in range(n_epochs):
            indices = np.arange(len(train_images))
            np.random.shuffle(indices)
            
            total_loss = 0
            correct = 0
            
            for idx in indices:
                x = train_images[idx]
                label = train_labels[idx]
                
                # Extract features
                features = self.extract_features(x)
                
                # Train classifier
                loss = self.classifier.train_step(features, label)
                total_loss += loss
                
                # Check accuracy
                pred = self.classifier.predict(features)
                if pred == label:
                    correct += 1
            
            avg_loss = total_loss / len(train_images)
            acc = correct / len(train_images)
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.4f}")
    
    def evaluate(self, test_images, test_labels):
        """Evaluate on test set."""
        correct = 0
        for i in range(len(test_images)):
            features = self.extract_features(test_images[i])
            pred = self.classifier.predict(features)
            if pred == test_labels[i]:
                correct += 1
        return correct / len(test_images)


# =============================================================================
# Experiment Runner
# =============================================================================

def run_hybrid_experiment():
    """Run hybrid experiment."""
    print("=" * 70)
    print("Hybrid Experiment: Hebbian Features + Linear Classifier")
    print("=" * 70)
    
    # Load data
    data_dir = download_mnist()
    try:
        train_images, train_labels, test_images, test_labels = load_mnist()
        dataset = "mnist"
        print(f"Loaded MNIST: {len(train_images)} train, {len(test_images)} test")
    except:
        train_images, train_labels = generate_synthetic_mnist(6000, 42)
        test_images, test_labels = generate_synthetic_mnist(1000, 43)
        dataset = "synthetic"
        print(f"Using synthetic data")
    
    # Use subset
    n_train = min(6000, len(train_images))
    n_test = min(1000, len(test_images))
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]
    
    print(f"Using: {n_train} train, {n_test} test")
    
    # Config
    input_dim = 784
    feat_dim1 = 256
    feat_dim2 = 128
    output_dim = 10
    
    # Create models
    models = {}
    
    # Model 1: Krotov-style WTA + Linear
    print("\n" + "-" * 50)
    print("Model 1: Krotov WTA + Linear")
    feat1 = HebbianFeatureLayer(input_dim, feat_dim1, sparsity=0.1, lr=0.01, p=2)
    classifier1 = LinearClassifier(feat_dim1, output_dim, lr=0.1)
    models["krotov_wta"] = HybridModel([feat1], classifier1)
    
    # Model 2: BCM + Linear
    print("-" * 50)
    print("Model 2: BCM + Linear")
    feat2 = BCMFeatureLayer(input_dim, feat_dim1, sparsity=0.1, lr=0.01)
    classifier2 = LinearClassifier(feat_dim1, output_dim, lr=0.1)
    models["bcm"] = HybridModel([feat2], classifier2)
    
    # Model 3: Predictive Sparse Coding + Linear
    print("-" * 50)
    print("Model 3: Predictive Sparse Coding + Linear")
    feat3 = PredictiveFeatureLayer(input_dim, feat_dim1 * 2, sparsity=10, lr=0.01)
    classifier3 = LinearClassifier(feat_dim1 * 2, output_dim, lr=0.1)
    models["predictive_sc"] = HybridModel([feat3], classifier3)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'=' * 70}")
        print(f"Training: {name}")
        print("=" * 70)
        
        start = time.time()
        
        # Phase 1: Unsupervised feature learning
        model.train_features(train_images, n_epochs=3)
        
        # Phase 2: Supervised classifier training
        model.train_classifier(train_images, train_labels, n_epochs=10)
        
        # Evaluate
        test_acc = model.evaluate(test_images, test_labels)
        
        elapsed = time.time() - start
        
        results[name] = {
            "test_accuracy": float(test_acc),
            "time": float(elapsed)
        }
        
        print(f"\n{name}: test_acc={test_acc:.4f}, time={elapsed:.2f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for name, res in results.items():
        print(f"  {name:20s}: acc={res['test_accuracy']:.4f}, time={res['time']:.2f}s")
    
    # Save
    output = {
        "dataset": dataset,
        "config": {"input_dim": input_dim, "feat_dim1": feat_dim1,
                   "output_dim": output_dim, "n_train": n_train, "n_test": n_test},
        "results": results
    }
    path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/11_LOGS/hybrid_results.json"
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {path}")
    
    return output


if __name__ == "__main__":
    run_hybrid_experiment()
