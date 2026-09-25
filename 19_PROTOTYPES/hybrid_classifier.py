"""
AGI Research Lab — Hybrid Hebbian + Linear Classifier
Phase 3 Pivot: Unsupervised Hebbian feature extraction → Supervised linear readout.

Key insight (Krotov & Hopfield 2019): Hebbian networks can learn meaningful features
if the architecture enforces competition and homeostasis. A simple linear readout
on top of these features can achieve high accuracy — no backprop needed.
"""

import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent))
from hebbian_layers import HebbianLayer, PredictiveCodingLayer


class HybridHebbianClassifier:
    """
    Phase 3 Hybrid Model:
    1. Unsupervised Hebbian layer learns features from raw data
    2. Linear classifier (logistic regression) trained on frozen features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 200,
        n_classes: int = 10,
        hebbian_lr: float = 0.01,
        classifier_lr: float = 0.1,
        hebbian_epochs: int = 30,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.hebbian_lr = hebbian_lr
        self.classifier_lr = classifier_lr
        self.hebbian_epochs = hebbian_epochs
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Layer 1: Hebbian feature extractor
        self.feature_extractor = HebbianLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            learning_rate=hebbian_lr,
            inhibition_strength=0.2,
            homeostasis_target=0.15,
            weight_bound=1.0,
            decay=0.0001,
            seed=seed,
        )

        # Layer 2: Linear classifier (trained separately)
        # We'll use one-hot encoded labels to set prototypes
        self.classifier_weights = np.zeros((n_classes, hidden_dim))
        self.classifier_bias = np.zeros(n_classes)
        self.prototypes = np.zeros((n_classes, hidden_dim))
        self.counts = np.zeros(n_classes)

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract features using Hebbian layer."""
        x = x / (np.linalg.norm(x) + 1e-8)
        return self.feature_extractor.forward(x)

    def pretrain_hebbian(self, X: np.ndarray, epochs: int = 30, verbose: bool = True):
        """Unsupervised pretraining — no labels needed."""
        if verbose:
            print(f"Pretraining Hebbian layer for {epochs} epochs...")

        n_samples = len(X)
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            self.rng.shuffle(indices)
            total_error = 0.0

            for i in indices:
                x = X[i].astype(np.float64)
                x = x / (np.linalg.norm(x) + 1e-8)

                # Forward + Hebbian update
                h = self.feature_extractor.forward(x)
                self.feature_extractor.update(x, h)

                # Track error (prediction error for monitoring)
                total_error += np.linalg.norm(x - self.feature_extractor.weights.T @ h) / self.input_dim

            if verbose and (epoch + 1) % 5 == 0:
                mean_error = total_error / n_samples
                print(f"  Epoch {epoch + 1}/{epochs}: mean_error={mean_error:.4f}, "
                      f"sparsity={self.feature_extractor.sparsity:.2%}")

    def train_classifier_prototypes(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Train linear classifier using prototype-based approach.
        For each class, compute mean feature representation → use as prototype.
        """
        if verbose:
            print("Training linear classifier (prototypes)...")

        # Extract features for all training samples
        features = []
        for i in range(len(X)):
            h = self.extract_features(X[i])
            features.append(h)
        features = np.array(features)

        # Compute class prototypes
        for c in range(self.n_classes):
            mask = y == c
            if mask.sum() > 0:
                self.prototypes[c] = features[mask].mean(axis=0)
                self.counts[c] = mask.sum()

        if verbose:
            for c in range(self.n_classes):
                print(f"  Class {c}: {int(self.counts[c])} samples")

    def train_classifier_sgd(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        verbose: bool = True,
    ):
        """
        Train linear classifier using SGD (not backprop through Hebbian layer).
        The Hebbian features are FROZEN — we only train the linear readout.
        """
        if verbose:
            print(f"Training classifier (SGD, {epochs} epochs)...")

        n_samples = len(X)
        lr = self.classifier_lr

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            total_loss = 0.0
            correct = 0

            for i in indices:
                # Extract frozen features
                h = self.extract_features(X[i])

                # Linear classifier
                logits = self.classifier_weights @ h + self.classifier_bias

                # Softmax
                exp = np.exp(logits - logits.max())
                probs = exp / exp.sum()

                label = int(y[i])

                # Cross-entropy loss
                loss = -np.log(probs[label] + 1e-8)
                total_loss += loss

                if np.argmax(probs) == label:
                    correct += 1

                # Gradient (softmax cross-entropy)
                dlogits = probs.copy()
                dlogits[label] -= 1

                # Update weights (only classifier, not Hebbian layer)
                self.classifier_weights -= lr * np.outer(dlogits, h)
                self.classifier_bias -= lr * dlogits

            if verbose and (epoch + 1) % 10 == 0:
                acc = correct / n_samples
                print(f"  Epoch {epoch + 1}: loss={total_loss / n_samples:.4f}, acc={acc:.2%}")

    def predict(self, x: np.ndarray) -> int:
        """Predict class label."""
        h = self.extract_features(x)
        logits = self.classifier_weights @ h + self.classifier_bias
        return int(np.argmax(logits))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        h = self.extract_features(x)
        logits = self.classifier_weights @ h + self.classifier_bias
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate accuracy."""
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                correct += 1
        accuracy = correct / len(X)
        return {"accuracy": accuracy, "correct": correct, "total": len(X)}

    def save(self, path: str):
        state = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "n_classes": self.n_classes,
            "feature_extractor": self.feature_extractor.save(),
            "classifier_weights": self.classifier_weights.tolist(),
            "classifier_bias": self.classifier_bias.tolist(),
        }
        Path(path).write_text(json.dumps(state, indent=2))


def generate_synthetic_mnist(n_samples: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate richer synthetic digit-like data."""
    rng = np.random.default_rng(seed)

    def make_digit(digit: int, size: int = 784) -> np.ndarray:
        """Create varied patterns for each digit with noise."""
        img = np.zeros(size)

        # Base patterns (more complex than Phase 2)
        if digit == 0:
            for i in range(28):
                img[i * 28] = 1
                img[i * 28 + 27] = 1
            img[:28] = 1
            img[-28:] = 1
        elif digit == 1:
            for i in range(28):
                img[i * 28 + 14] = 1
        elif digit == 2:
            for i in range(28):
                img[i * 28 + i] = 1
                img[i * 28 + 27 - i] = 0.5
        elif digit == 3:
            for i in range(28):
                img[i * 28 + 27 - i] = 1
        elif digit == 4:
            for i in range(28):
                img[i * 28] = 1
                img[i * 28 + 27] = 1
        elif digit == 5:
            for i in range(28):
                img[i * 28 + 14] = 1
            img[:784:7] = 0.5
        elif digit == 6:
            img[::3] = 0.8
            img[::5] = 0.6
        elif digit == 7:
            img[::4] = 0.9
            img[::6] = 0.7
        elif digit == 8:
            img[::5] = 0.7
            img[::3] = 0.5
        elif digit == 9:
            img[::7] = 0.6
            img[::4] = 0.8

        # Add multiple noise sources for richer data
        img += rng.normal(0, 0.08, size)
        img += rng.choice([0, 0.3], size=size, p=[0.95, 0.05])  # salt noise
        img = np.clip(img, 0, 1)
        return img

    X, y = [], []
    samples_per_class = n_samples // 10
    for digit in range(10):
        for _ in range(samples_per_class):
            X.append(make_digit(digit))
            y.append(digit)

    X = np.array(X)
    y = np.array(y)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:]


if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid Hebbian + Linear Classifier — Phase 3")
    print("=" * 60)

    # Create model
    model = HybridHebbianClassifier(
        input_dim=784,
        hidden_dim=300,
        n_classes=10,
        hebbian_lr=0.01,
        classifier_lr=0.05,
        hebbian_epochs=30,
        seed=42,
    )
    print(f"Architecture: 784 -> 300 (Hebbian) -> 10 (linear)")

    # Generate data
    print("\nGenerating synthetic data...")
    X_train, y_train, X_test, y_test = generate_synthetic_mnist(2000)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Phase 1: Unsupervised Hebbian pretraining (no labels!)
    model.pretrain_hebbian(X_train, epochs=30)

    # Phase 2: Supervised linear classifier (Hebbian frozen)
    model.train_classifier_sgd(X_train, y_train, epochs=50)

    # Evaluate
    print("\nEvaluating...")
    results = model.evaluate(X_test, y_test)
    print(f"  Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

    # Save
    output_path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/20_RESULTS/hybrid_phase3.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nResults saved to {output_path}")
