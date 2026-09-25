"""
AGI Research Lab — Krotov & Hopfield Approach (Vectorized)
Based on "Unsupervised learning by hidden layer prediction" (Krotov & Hopfield 2019).

Key idea: Use competitive learning where neurons compete to respond to inputs.
Classification = nearest prototype in feature space.

This version is fully vectorized for speed.
"""

import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any


class VectorizedHopfieldLayer:
    """
    Vectorized Hopfield-style competitive layer.
    All operations are matrix operations — no Python loops.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.01,
        temperature: float = 0.1,
        sparsity: float = 0.1,  # fraction of active neurons
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.temperature = temperature
        self.sparsity = sparsity
        self.rng = np.random.default_rng(seed)

        # Each row is a prototype pattern
        self.weights = self.rng.normal(0, 0.1, (output_dim, input_dim)) / np.sqrt(input_dim)
        self.bias = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sparse activations."""
        x = x / (np.linalg.norm(x) + 1e-8)
        logits = self.weights @ x + self.bias

        # Temperature-scaled softmax
        logits = (logits - logits.max()) / self.temperature
        exp = np.exp(logits)
        activations = exp / (exp.sum() + 1e-8)

        # Top-k sparsification
        k = max(1, int(self.sparsity * self.output_dim))
        top_k = np.argpartition(activations, -k)[-k:]
        sparse = np.zeros(self.output_dim)
        sparse[top_k] = activations[top_k]
        if sparse.sum() > 0:
            sparse = sparse / sparse.sum()

        return sparse

    def update(self, x: np.ndarray, h: np.ndarray):
        """Vectorized Krotov-Hopfield update."""
        x = x / (np.linalg.norm(x) + 1e-8)

        # Active neurons (h > threshold)
        active_mask = h > 0.01

        # Update active neurons toward input
        self.weights[active_mask] += self.lr * np.outer(h[active_mask], x) - self.lr * 0.01 * self.weights[active_mask]

        # Update inactive neurons away from input (anti-Hebbian)
        inactive = ~active_mask
        if inactive.any():
            self.weights[inactive] -= self.lr * 0.005 * np.outer(np.ones(inactive.sum()), x)

        # Normalize all weights
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        self.weights = self.weights / (norms + 1e-8)


class KrotovHopfieldNetwork:
    """Krotov-Hopfield network with vectorized operations."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 300,
        n_classes: int = 10,
        hebbian_epochs: int = 20,
        classifier_epochs: int = 50,
        learning_rate: float = 0.01,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.hebbian_epochs = hebbian_epochs
        self.classifier_epochs = classifier_epochs
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)

        self.hopfield = VectorizedHopfieldLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed,
        )

        self.classifier_weights = np.zeros((n_classes, hidden_dim))
        self.classifier_bias = np.zeros(n_classes)

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        x = x / (np.linalg.norm(x) + 1e-8)
        return self.hopfield.forward(x)

    def pretrain(self, X: np.ndarray, verbose: bool = True):
        if verbose:
            print(f"Pretraining for {self.hebbian_epochs} epochs...")

        n_samples = len(X)
        indices = np.arange(n_samples)

        for epoch in range(self.hebbian_epochs):
            self.rng.shuffle(indices)
            for i in indices:
                x = X[i].astype(np.float64)
                h = self.extract_features(x)
                self.hopfield.update(x, h)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{self.hebbian_epochs}")

    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ):
        if verbose:
            print(f"Training classifier for {self.classifier_epochs} epochs...")

        n_samples = len(X)

        for epoch in range(self.classifier_epochs):
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            total_loss = 0.0
            correct = 0

            for i in indices:
                h = self.extract_features(X[i])
                logits = self.classifier_weights @ h + self.classifier_bias

                exp = np.exp(logits - logits.max())
                probs = exp / exp.sum()

                label = int(y[i])
                loss = -np.log(probs[label] + 1e-8)
                total_loss += loss

                if np.argmax(probs) == label:
                    correct += 1

                dlogits = probs.copy()
                dlogits[label] -= 1

                self.classifier_weights -= self.lr * np.outer(dlogits, h)
                self.classifier_bias -= self.lr * dlogits

            if verbose and (epoch + 1) % 10 == 0:
                acc = correct / n_samples
                print(f"  Epoch {epoch + 1}: loss={total_loss / n_samples:.4f}, acc={acc:.2%}")

    def predict(self, x: np.ndarray) -> int:
        h = self.extract_features(x)
        logits = self.classifier_weights @ h + self.classifier_bias
        return int(np.argmax(logits))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == y[i])
        accuracy = correct / len(X)
        return {"accuracy": accuracy, "correct": correct, "total": len(X)}


def generate_synthetic_mnist(n_samples: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate richer synthetic digit-like data."""
    rng = np.random.default_rng(seed)

    def make_digit(digit: int, size: int = 784) -> np.ndarray:
        img = np.zeros(size)
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
        elif digit == 6:
            img[::3] = 0.8
        elif digit == 7:
            img[::4] = 0.9
        elif digit == 8:
            img[::5] = 0.7
        elif digit == 9:
            img[::7] = 0.6
        img += rng.normal(0, 0.08, size)
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
    print("Krotov-Hopfield Network (Vectorized) — Phase 3")
    print("=" * 60)

    net = KrotovHopfieldNetwork(
        input_dim=784,
        hidden_dim=300,
        n_classes=10,
        hebbian_epochs=20,
        classifier_epochs=50,
        learning_rate=0.01,
        seed=42,
    )
    print(f"Architecture: 784 -> 300 (Hopfield) -> 10 (linear)")

    print("\nGenerating data...")
    X_train, y_train, X_test, y_test = generate_synthetic_mnist(2000)

    net.pretrain(X_train, verbose=True)
    net.train_classifier(X_train, y_train, verbose=True)

    print("\nEvaluating...")
    results = net.evaluate(X_test, y_test)
    print(f"  Accuracy: {results['accuracy']:.2%}")

    output_path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/20_RESULTS/krotov_hopfield.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nResults saved to {output_path}")
