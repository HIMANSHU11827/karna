"""
AGI Research Lab — Hebbian Learning Module
From-scratch neural learning without backpropagation.
"""

import numpy as np
from typing import Optional, Tuple
import time


class HebbianLayer:
    """
    Single layer with Hebbian learning.
    Δw_ij = η * (x_i * y_j - decay * w_ij)
    """

    def __init__(self, input_size: int, output_size: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(input_size, output_size) * 0.01
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.01
        self.decay = 0.001
        self.last_input = None
        self.last_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.last_input = x.copy()
        # Linear + ReLU
        output = x @ self.weights
        output = np.maximum(output, 0)  # ReLU
        self.last_output = output.copy()
        return output

    def update(self):
        """Hebbian weight update."""
        if self.last_input is None or self.last_output is None:
            return

        # Outer product
        delta = np.outer(self.last_input, self.last_output)

        # Update weights with decay
        self.weights += self.learning_rate * delta
        self.weights -= self.learning_rate * self.decay * self.weights

        # Clip weights
        self.weights = np.clip(self.weights, -1.0, 1.0)


class PredictiveCodingLayer:
    """
    Hierarchical predictive coding layer.
    Each layer tries to predict the activity of the layer below.
    """

    def __init__(self, size: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.size = size
        self.activation = np.zeros(size)
        self.prediction = np.zeros(size)
        self.error = np.zeros(size)
        self.weights = rng.randn(size, size) * 0.01
        self.learning_rate = 0.01

    def predict(self) -> np.ndarray:
        """Generate prediction."""
        self.prediction = self.activation @ self.weights
        self.prediction = np.maximum(self.prediction, 0)  # ReLU
        return self.prediction

    def compute_error(self, target: np.ndarray):
        """Compute prediction error."""
        self.error = target - self.prediction
        return self.error

    def update_weights(self):
        """Update weights based on prediction error."""
        delta = np.outer(self.activation, self.error)
        self.weights += self.learning_rate * delta

    def update_activation(self, target: np.ndarray, steps: int = 5):
        """Update activation to minimize prediction error (gradient descent)."""
        self.activation = target.copy()
        for _ in range(steps):
            self.predict()
            self.compute_error(target)
            self.activation += self.learning_rate * self.error @ self.weights.T
            self.activation = np.maximum(self.activation, 0)


class Network:
    """Simple feedforward network with Hebbian learning."""

    def __init__(self, sizes: list, seed: int = 42):
        self.layers = []
        for i in range(len(sizes) - 1):
            layer = HebbianLayer(sizes[i], sizes[i + 1], seed=seed + i)
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train_step(self, x: np.ndarray, target: np.ndarray):
        """Single Hebbian training step."""
        # Forward pass
        output = self.forward(x)

        # Update each layer
        for layer in self.layers:
            layer.update()

        # Compute loss (MSE)
        loss = np.mean((output - target) ** 2)
        return loss

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train on dataset."""
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(X)):
                loss = self.train_step(X[i], y[i])
                epoch_loss += loss
            epoch_loss /= len(X)
            losses.append(epoch_loss)
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: loss={epoch_loss:.4f}")
        return losses


def generate_synthetic_data(n_samples: int = 100, input_size: int = 10, n_classes: int = 3, seed: int = 42):
    """Generate simple synthetic classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_size)
    y = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        label = rng.randint(n_classes)
        y[i, label] = 1.0
        # Make class-dependent pattern
        X[i, label * 3:(label + 1) * 3] += 2.0
    return X, y


def demo_hebbian():
    """Demo Hebbian learning on synthetic data."""
    print("=" * 60)
    print("Hebbian Learning — Demo")
    print("=" * 60)

    X, y = generate_synthetic_data(n_samples=200, input_size=12, n_classes=3)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} classes")

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Train
    net = Network([12, 32, 16, 3], seed=42)
    print("\nTraining with Hebbian learning...")
    losses = net.train(X, y, epochs=100)

    # Evaluate
    correct = 0
    for i in range(len(X)):
        output = net.forward(X[i])
        pred = np.argmax(output)
        actual = np.argmax(y[i])
        if pred == actual:
            correct += 1

    accuracy = correct / len(X) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    print(f"Final loss: {losses[-1]:.4f}")


def demo_predictive_coding():
    """Demo predictive coding."""
    print("\n" + "=" * 60)
    print("Predictive Coding — Demo")
    print("=" * 60)

    size = 20
    layer = PredictiveCodingLayer(size)

    # Random target pattern
    rng = np.random.RandomState(42)
    target = rng.randn(size)
    target = np.maximum(target, 0)

    print("\nMinimizing prediction error...")
    for step in range(50):
        layer.update_activation(target, steps=3)
        layer.predict()
        error = np.mean(layer.error ** 2)
        if step % 10 == 0:
            print(f"  Step {step}: prediction error = {error:.4f}")


if __name__ == "__main__":
    demo_hebbian()
    demo_predictive_coding()
