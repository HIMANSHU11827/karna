"""
AGI Research Lab - Phase 2 PIVOT: Hybrid Hebbian + Supervised
==============================================================

Pure Hebbian gets ~12% MNIST — confirmed.

NEW STRATEGY:
1. Unsupervised Hebbian feature extraction (train layers with local rules)
2. Supervised linear classifier on top (logistic regression)
3. Krotov & Hopfield (2019) expanded approach for 95%+ target

Based on: Krotov & Hopfield "Unsupervised learning by competing hidden units"
Neural Computation 2019 — reports ~96% MNIST without backprop.
"""

import numpy as np
import os
import sys
import time
import json
import urllib.request
import gzip
from pathlib import Path
from numpy.linalg import norm


# =============================================================================
# MNIST Loader
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
                pass
    return data_dir


def load_mnist(data_dir="/tmp/mnist_data"):
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data.astype(np.int32)
    return (load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz")),
            load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz")),
            load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")),
            load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")))


def generate_synthetic(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    images = rng.rand(n, 784).astype(np.float32) * 0.5 + 0.2
    labels = rng.randint(0, 10, n)
    for i in range(n):
        cls = labels[i]
        r_s, c_s = (cls // 3) * 8, (cls % 3) * 8
        for r in range(r_s, min(r_s+12, 28)):
            for c in range(c_s, min(c_s+12, 28)):
                images[i, r*28+c] += 0.3
    images = np.clip(images, 0, 1)
    return images, labels


# =============================================================================
# 1. Hybrid Hebbian + Logistic Regression
# =============================================================================

class HebbianFeatureExtractor:
    """
    Unsupervised Hebbian feature extraction.
    Train each layer with local Hebbian rule (no labels needed).
    Then extract features for a supervised classifier.
    """

    def __init__(self, input_dim, hidden_dims, lr=0.01):
        self.layers = []
        prev = input_dim
        for h in hidden_dims:
            W = np.random.randn(prev, h).astype(np.float32) * 0.1
            b = np.zeros(h, dtype=np.float32)
            self.layers.append({'W': W, 'b': b, 'lr': lr})
            prev = h

    def forward(self, x):
        """Forward pass through all layers."""
        h = x.copy()
        activations = [h]
        for layer in self.layers:
            h = h @ layer['W'] + layer['b']
            h = np.maximum(h, 0)  # ReLU
            # Top-k sparsity
            k = max(1, int(h.shape[-1] * 0.15))
            top_k = np.argsort(h, axis=-1)[..., -k:]
            mask = np.zeros_like(h)
            for i in range(h.shape[0]):
                mask[i, top_k[i]] = 1.0
            h = h * mask
            activations.append(h)
        return activations

    def train_unsupervised(self, images, epochs=10, batch_size=32):
        """
        Unsupervised training: each layer learns to represent its input
        using Hebbian outer product updates.
        """
        n = len(images)
        indices = np.arange(n)
        losses = []

        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = images[indices[start:end]]

                activations = self.forward(batch)

                # Train each layer with Hebbian rule
                for i, layer in enumerate(self.layers):
                    input_act = activations[i]  # (batch, prev_dim)
                    output_act = activations[i+1]  # (batch, hidden_dim)

                    # Hebbian: ΔW = mean over batch of input^T @ output
                    delta_W = input_act.T @ output_act / batch_size

                    # Oja stabilization
                    delta_W -= 0.0001 * layer['W'] * (output_act ** 2).sum(axis=0) / batch_size

                    layer['W'] += layer['lr'] * delta_W
                    layer['W'] *= 0.999  # weight decay
                    layer['W'] = np.clip(layer['W'], -2, 2)

                    # Reconstruction error
                    if i == 0:
                        recon = output_act @ layer['W'].T
                        loss = np.mean((input_act - recon) ** 2)
                        total_loss += loss

                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            losses.append(float(avg_loss))
            if (epoch + 1) % 2 == 0:
                print(f"    Hebbian epoch {epoch+1}: recon_loss={avg_loss:.4f}")

        return losses

    def extract_features(self, images):
        """Extract features from final layer."""
        activations = self.forward(images)
        return activations[-1]


class LogisticRegression:
    """Multinomial logistic regression (softmax classifier)."""

    def __init__(self, input_dim, n_classes=10, lr=0.1):
        self.W = np.random.randn(input_dim, n_classes).astype(np.float32) * 0.01
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.lr = lr

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, x):
        return self.softmax(x @ self.W + self.b)

    def train(self, images, labels, epochs=20, batch_size=32):
        n = len(images)
        indices = np.arange(n)
        losses = []

        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_x = images[indices[start:end]]
                batch_y = labels[indices[start:end]]

                # One-hot
                targets = np.zeros((len(batch_y), 10), dtype=np.float32)
                targets[np.arange(len(batch_y)), batch_y] = 1.0

                # Forward
                probs = self.forward(batch_x)
                loss = -np.mean(np.sum(targets * np.log(probs + 1e-8), axis=1))

                # Gradient
                delta = probs - targets
                dW = batch_x.T @ delta / len(batch_x)
                db = np.mean(delta, axis=0)

                self.W -= self.lr * dW
                self.b -= self.lr * db

                total_loss += loss
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            losses.append(float(avg_loss))
            if (epoch + 1) % 5 == 0:
                preds = np.argmax(self.forward(images), axis=1)
                acc = np.mean(preds == labels)
                print(f"    LR epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.4f}")

        return losses

    def predict(self, images):
        return np.argmax(self.forward(images), axis=1)


# =============================================================================
# 2. Krotov & Hopfield Style Network (2019)
# =============================================================================

class KrotovHopfieldNetwork:
    """
    Expanded Hopfield-like network with Winner-Take-All (WTA) competition.

    Based on Krotov & Hopfield (2019):
    "Unsupervised learning by competing hidden units"

    Key ideas:
    - WTA competition selects sparse active neurons
    - Hebbian learning with anti-Hebbian expansion term
    - Power iteration (p>2) for stronger separation
    - Reports ~96% MNIST with single layer + SVM
    """

    def __init__(self, input_dim, hidden_dim, p=4, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p = p  # power parameter for WTA (p=4 in paper)
        self.lr = lr

        # Initialize weights
        self.W = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        self.b = np.zeros(hidden_dim, dtype=np.float32)

    def forward(self, x):
        """
        Forward pass with WTA competition.
        Returns sparse hidden activation.
        """
        # Total input to hidden neurons
        h = x @ self.W + self.b  # (batch, hidden)

        # Power nonlinearity (p=4 gives stronger separation)
        h_p = np.sign(h) * (np.abs(h) ** self.p)

        # WTA: only top-k neurons fire
        k = max(1, int(self.hidden_dim * 0.1))
        top_k = np.argsort(h_p, axis=-1)[:, -k:]
        sparse_h = np.zeros_like(h_p)
        for i in range(h_p.shape[0]):
            sparse_h[i, top_k[i]] = h_p[i, top_k[i]]

        return sparse_h

    def train_unsupervised(self, images, epochs=10, batch_size=32):
        """
        Unsupervised training with expanded Hebbian rule.

        Krotov's rule:
        Δw_ij = η * sum_over_samples [ h_j * (x_i - sum_k w_ik * h_k) ]

        This includes:
        - Hebbian term: h_j * x_i
        - Anti-Hebbian (expansion) term: -h_j * sum_k w_ik * h_k

        The expansion term prevents duplicate features.
        """
        n = len(images)
        indices = np.arange(n)
        losses = []

        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = images[indices[start:end]]

                # Forward
                h = self.forward(batch)

                # Krotov's expanded Hebbian rule
                # Reconstruction
                recon = h @ self.W.T  # (batch, input_dim)

                # Error
                error = batch - recon  # (batch, input_dim)

                # Weight update: outer product of error and h
                delta_W = error.T @ h / batch_size  # (input_dim, hidden)

                # Anti-Hebbian expansion: subtract duplicate correlations
                # This is the key to preventing feature collapse
                expansion = np.zeros_like(self.W)
                for j in range(self.hidden_dim):
                    # Subtract influence of other active neurons
                    other_h = h.copy()
                    other_h[:, j] = 0
                    expansion[:, j] = (other_h.T @ h[:, j]) / batch_size

                delta_W -= 0.01 * expansion

                self.W += self.lr * delta_W
                self.W = np.clip(self.W, -2, 2)

                # Normalize weights per hidden unit
                for j in range(self.hidden_dim):
                    w_norm = norm(self.W[:, j])
                    if w_norm > 1.0:
                        self.W[:, j] /= w_norm

                loss = np.mean(error ** 2)
                total_loss += loss

                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            losses.append(float(avg_loss))
            if (epoch + 1) % 2 == 0:
                print(f"    Krotov epoch {epoch+1}: recon_loss={avg_loss:.4f}")

        return losses

    def extract_features(self, images):
        return self.forward(images)


# =============================================================================
# 3. Expanded Hopfield Layer (Dense Associative Memory)
# =============================================================================

class ExpandedHopfieldLayer:
    """
    Modern Hopfield layer with expanded capacity.

    Based on Krotov & Hopfield (2019) + Ramsauer et al. (2021)
    "Hopfield Networks is All You Need"

    Key difference from classical Hopfield:
    - Uses softmax-based energy (ΔE = -β * log(sum(exp(β * W^T x))))
    - Separation: patterns can be stored with higher capacity
    - Retrieval via energy minimization (associative memory)
    """

    def __init__(self, input_dim, hidden_dim, beta=1.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.beta = beta

        # Stored patterns (keys and values)
        self.keys = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.1
        self.values = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.1

        # Hebbian storage
        self.stored = np.zeros((hidden_dim, input_dim), dtype=np.float32)

    def store_patterns(self, patterns):
        """Store patterns using Hebbian outer product."""
        self.stored = patterns.T @ patterns / len(patterns)

    def retrieve(self, query, steps=3):
        """
        Retrieve stored pattern via energy minimization.
        Uses softmax attention mechanism (but NOT transformer attention).
        """
        current = query.copy()
        for _ in range(steps):
            # Compute similarity
            sim = current @ self.keys.T  # (batch, hidden)
            # Softmax weighting (modern Hopfield)
            weights = np.exp(self.beta * sim - np.max(self.beta * sim, axis=-1, keepdims=True))
            weights /= np.sum(weights, axis=-1, keepdims=True)
            # Retrieve values
            current = weights @ self.values
        return current

    def train_unsupervised(self, images, epochs=5):
        """Store training patterns."""
        # Simple: use stored patterns as prototypes
        n_store = min(1000, len(images))
        self.store_patterns(images[:n_store])
        # Initialize keys and values
        self.keys = images[:self.hidden_dim].T @ images[:n_store]
        self.keys = self.keys.T.astype(np.float32)


# =============================================================================
# 4. Full Experiment Pipeline
# =============================================================================

def run_hybrid_experiment(train_images, train_labels, test_images, test_labels,
                          feature_dim=256, lr=0.01, dataset="mnist"):
    """
    Run hybrid Hebbian + Logistic Regression experiment.
    """
    print(f"\n--- Hybrid Hebbian + Logistic Regression ({dataset}) ---")
    input_dim = train_images.shape[1]

    # 1. Unsupervised feature extraction
    print("Step 1: Unsupervised Hebbian feature extraction...")
    extractor = HebbianFeatureExtractor(
        input_dim=input_dim,
        hidden_dims=[feature_dim],
        lr=lr
    )
    extractor.train_unsupervised(train_images, epochs=10, batch_size=64)

    # 2. Extract features
    print("Step 2: Extracting features...")
    train_features = extractor.extract_features(train_images)
    test_features = extractor.extract_features(test_images)

    print(f"  Feature shape: {train_features.shape}")
    print(f"  Feature sparsity: {np.mean(train_features > 0):.2%}")
    print(f"  Feature norm mean: {np.mean(np.linalg.norm(train_features, axis=1)):.3f}")

    # 3. Supervised classification
    print("Step 3: Training logistic regression...")
    classifier = LogisticRegression(
        input_dim=feature_dim,
        n_classes=10,
        lr=0.1
    )
    classifier.train(train_features, train_labels, epochs=30, batch_size=64)

    # 4. Evaluate
    preds = classifier.predict(test_features)
    acc = np.mean(preds == test_labels)
    print(f"  Final test accuracy: {acc:.4f}")

    return acc


def run_krotov_experiment(train_images, train_labels, test_images, test_labels,
                          hidden_dim=256, lr=0.01, dataset="mnist"):
    """
    Run Krotov & Hopfield style experiment.
    """
    print(f"\n--- Krotov & Hopfield Network ({dataset}) ---")
    input_dim = train_images.shape[1]

    # 1. Unsupervised Krotov training
    print("Step 1: Krotov WTA training...")
    krotov = KrotovHopfieldNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        p=4,  # p=4 gives better separation than p=2
        lr=lr
    )
    krotov.train_unsupervised(train_images, epochs=10, batch_size=64)

    # 2. Extract features
    print("Step 2: Extracting features...")
    train_features = krotov.extract_features(train_images)
    test_features = krotov.extract_features(test_images)

    print(f"  Feature shape: {train_features.shape}")
    print(f"  Feature sparsity: {np.mean(train_features > 0):.2%}")

    # 3. Classify
    print("Step 3: Training logistic regression...")
    classifier = LogisticRegression(
        input_dim=hidden_dim,
        n_classes=10,
        lr=0.1
    )
    classifier.train(train_features, train_labels, epochs=30, batch_size=64)

    # 4. Evaluate
    preds = classifier.predict(test_features)
    acc = np.mean(preds == test_labels)
    print(f"  Final test accuracy: {acc:.4f}")

    return acc


def run_backprop_baseline(train_images, train_labels, test_images, test_labels,
                          hidden_dim=64, lr=0.001, dataset="mnist"):
    """
    Backprop baseline: same architecture, trained with backprop.
    """
    print(f"\n--- Backprop Baseline ({dataset}) ---")
    input_dim = train_images.shape[1]

    # Initialize
    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(hidden_dim, 10).astype(np.float32) * 0.1
    b2 = np.zeros(10, dtype=np.float32)

    n = len(train_images)
    epochs = 20
    batch_size = 32

    for epoch in range(epochs):
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_x = train_images[indices[start:end]]
            batch_y = train_labels[indices[start:end]]

            # One-hot
            targets = np.zeros((len(batch_y), 10), dtype=np.float32)
            targets[np.arange(len(batch_y)), batch_y] = 1.0

            # Forward
            h = np.maximum(batch_x @ W1 + b1, 0)
            out = np.exp(h @ W2 + b2 - np.max(h @ W2 + b2, axis=-1, keepdims=True))
            out /= np.sum(out, axis=-1, keepdims=True)

            # Backward
            delta2 = out - targets
            dW2 = h.T @ delta2 / len(batch_x)
            db2 = np.mean(delta2, axis=0)

            delta1 = (delta2 @ W2.T) * (h > 0).astype(np.float32)
            dW1 = batch_x.T @ delta1 / len(batch_x)
            db1 = np.mean(delta1, axis=0)

            # Update
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

    # Evaluate
    h = np.maximum(test_images @ W1 + b1, 0)
    out = np.exp(h @ W2 + b2 - np.max(h @ W2 + b2, axis=-1, keepdims=True))
    out /= np.sum(out, axis=-1, keepdims=True)
    preds = np.argmax(out, axis=1)
    acc = np.mean(preds == test_labels)
    print(f"  Final test accuracy: {acc:.4f}")
    return acc


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("AGI Research Lab - Phase 2 PIVOT")
    print("Hybrid Hebbian + Supervised Classification")
    print("Krotov & Hopfield (2019) expanded approach")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data_dir = download_mnist()
    if data_dir and all(os.path.exists(os.path.join(data_dir, f))
                       for f in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]):
        train_images, train_labels, test_images, test_labels = load_mnist()
        dataset = "mnist"
        print(f"  Real MNIST: {len(train_images)} train, {len(test_images)} test")
    else:
        print("  Download failed, using synthetic data...")
        train_images, train_labels = generate_synthetic(6000, 42)
        test_images, test_labels = generate_synthetic(1000, 43)
        dataset = "synthetic"
        print(f"  Synthetic: {len(train_images)} train, {len(test_images)} test")

    # Subset for speed
    n_train = min(6000, len(train_images))
    n_test = min(1000, len(test_images))
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]

    results = {"dataset": dataset}

    # Experiment 1: Hybrid Hebbian + Logistic Regression
    hybrid_acc = run_hybrid_experiment(
        train_images, train_labels, test_images, test_labels,
        feature_dim=256, lr=0.01, dataset=dataset
    )
    results["hybrid_hebbian_lr"] = hybrid_acc

    # Experiment 2: Krotov & Hopfield
    krotov_acc = run_krotov_experiment(
        train_images, train_labels, test_images, test_labels,
        hidden_dim=256, lr=0.01, dataset=dataset
    )
    results["krotov_hopfield"] = krotov_acc

    # Experiment 3: Backprop baseline
    bp_acc = run_backprop_baseline(
        train_images, train_labels, test_images, test_labels,
        hidden_dim=256, lr=0.01, dataset=dataset
    )
    results["backprop_baseline"] = bp_acc

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Dataset: {dataset}")
    print(f"  Hybrid Hebbian + LR:  {results['hybrid_hebbian_lr']:.4f}")
    print(f"  Krotov & Hopfield:    {results['krotov_hopfield']:.4f}")
    print(f"  Backprop baseline:    {results['backprop_baseline']:.4f}")

    # Save
    path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/11_LOGS/pivot_results.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")

    return results


if __name__ == "__main__":
    main()
