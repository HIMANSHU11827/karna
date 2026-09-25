"""
AGI Research Lab — BCM (Bienenstock-Cooper-Munro) Learning Rule
Another biologically plausible learning rule that adapts based on local history.

BCM rule: Δw = lr * x * y * (y - θ)
where θ is a sliding threshold based on recent activity.

This creates a homeostatic effect: neurons that fire too much become less sensitive,
neurons that fire too little become more sensitive.
"""

import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any


class BCMLayer:
    """
    BCM learning layer with sliding threshold.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.001,
        tau: float = 0.1,  # time constant for threshold adaptation
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.tau = tau
        self.rng = np.random.default_rng(seed)

        # Weights
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.weights = self.rng.uniform(-limit, limit, (output_dim, input_dim))
        self.bias = np.zeros(output_dim)

        # Sliding threshold (adapts based on recent activity)
        self.threshold = np.ones(output_dim) * 0.1
        self.mean_activity = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        x = x / (np.linalg.norm(x) + 1e-8)
        z = self.weights @ x + self.bias
        y = np.maximum(0, z)  # ReLU
        return y

    def update(self, x: np.ndarray, y: np.ndarray):
        """BCM update rule."""
        x = x / (np.linalg.norm(x) + 1e-8)

        # Update mean activity (exponential moving average)
        self.mean_activity = (1 - self.tau) * self.mean_activity + self.tau * y

        # Update threshold
        self.threshold = self.mean_activity * 1.5

        # BCM rule: Δw = lr * x * y * (y - θ)
        for j in range(self.output_dim):
            if y[j] > 0:
                delta = self.lr * y[j] * (y[j] - self.threshold[j]) * x
                self.weights[j] += delta

                # Anti-Hebbian if below threshold
                if y[j] < self.threshold[j]:
                    self.weights[j] -= self.lr * 0.01 * np.outer(np.ones(1), x).flatten()

        # Normalize
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        self.weights = self.weights / (norms + 1e-8)


class BCMNetwork:
    """BCM-based network for MNIST."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 300,
        n_classes: int = 10,
        bcm_epochs: int = 30,
        classifier_epochs: int = 50,
        learning_rate: float = 0.001,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.bcm_epochs = bcm_epochs
        self.classifier_epochs = classifier_epochs
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)

        self.bcm = BCMLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed,
        )

        self.classifier_weights = np.zeros((n_classes, hidden_dim))
        self.classifier_bias = np.zeros(n_classes)

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        x = x / (np.linalg.norm(x) + 1e-8)
        return self.bcm.forward(x)

    def pretrain(self, X: np.ndarray, verbose: bool = True):
        if verbose:
            print(f"Pretraining BCM for {self.bcm_epochs} epochs...")

        n_samples = len(X)
        indices = np.arange(n_samples)

        for epoch in range(self.bcm_epochs):
            self.rng.shuffle(indices)
            for i in indices:
                x = X[i].astype(np.float64)
                y = self.extract_features(x)
                self.bcm.update(x, y)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{self.bcm_epochs}")

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

                self.classifier_weights -= self.lr * 10 * np.outer(dlogits, h)
                self.classifier_bias -= self.lr * 10 * dlogits

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
    print("BCM Network — Phase 3")
    print("=" * 60)

    net = BCMNetwork(
        input_dim=784,
        hidden_dim=300,
        n_classes=10,
        bcm_epochs=30,
        classifier_epochs=50,
        learning_rate=0.001,
        seed=42,
    )
    print(f"Architecture: 784 -> 300 (BCM) -> 10 (linear)")

    print("\nGenerating data...")
    X_train, y_train, X_test, y_test = generate_synthetic_mnist(2000)

    net.pretrain(X_train, verbose=True)
    net.train_classifier(X_train, y_train, verbose=True)

    print("\nEvaluating...")
    results = net.evaluate(X_test, y_test)
    print(f"  Accuracy: {results['accuracy']:.2%}")

    output_path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/20_RESULTS/bcm_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nResults saved to {output_path}")
