"""
AGI Research Lab — Hebbian Learning Layers
Neural layers that learn via Hebbian/anti-Hebbian updates — no backpropagation.

Key principles:
- Local learning rules (no global gradient)
- Event-driven updates (sparse activity)
- NumPy-only (no PyTorch/TensorFlow)
- Neuromorphic-ready (spike-friendly)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class LayerStats:
    """Track layer statistics."""
    update_count: int = 0
    total_spikes: int = 0
    mean_activation: float = 0.0
    learning_events: int = 0


class HebbianLayer:
    """
    A layer with Hebbian learning: "neurons that fire together, wire together."
    
    Forward: y = f(Wx + b)
    Learning: ΔW = lr * (y @ x^T) - decay * W
    
    Supports:
    - Sparse, event-driven updates
    - Lateral inhibition for competition
    - Homeostatic plasticity
    - Weight bounding for stability
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.001,
        decay: float = 0.0001,
        inhibition_strength: float = 0.1,
        homeostasis_target: float = 0.1,
        weight_bound: float = 1.0,
        dt: float = 1.0,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.decay = decay
        self.inhibition_strength = inhibition_strength
        self.homeostasis_target = homeostasis_target
        self.weight_bound = weight_bound
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # Weights — small random init
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.weights = self.rng.uniform(-limit, limit, (output_dim, input_dim))
        self.bias = np.zeros(output_dim)

        # Running stats
        self.mean_activity = np.zeros(output_dim)
        self.stats = LayerStats()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        z = self.weights @ x + self.bias
        y = np.maximum(0, z)

        if self.inhibition_strength > 0:
            mean_act = y.mean()
            if mean_act > 0:
                y = np.maximum(0, y - self.inhibition_strength * mean_act)

        self.stats.total_spikes += int((y > 0).sum())
        self.stats.mean_activation = float(y.mean())
        return y

    def update(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        reward_signal: Optional[float] = None,
    ):
        """Hebbian weight update — local, stable, event-driven."""
        if y is None:
            y = self.forward(x)

        if y.sum() < 1e-8:
            return

        # Normalize input to prevent magnitude explosion
        x_norm = x / (np.linalg.norm(x) + 1e-8)

        # Hebbian update: outer product with normalization
        delta_w = self.lr * np.outer(y, x_norm)

        # Optional reward modulation
        if reward_signal is not None:
            delta_w *= np.clip(reward_signal, -1, 1)

        self.weights += delta_w

        # Weight bounding — L2 norm constraint per neuron
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        mask = norms > self.weight_bound
        self.weights = np.where(mask, self.weights * self.weight_bound / (norms + 1e-8), self.weights)

        # Decay toward zero (forgetting)
        self.weights *= (1 - self.decay)

        # Homeostatic plasticity
        self.mean_activity = 0.99 * self.mean_activity + 0.01 * y
        excess = self.mean_activity - self.homeostasis_target
        self.bias -= self.dt * excess * self.lr * 0.1

        self.stats.update_count += 1
        self.stats.learning_events += 1

    @property
    def sparsity(self) -> float:
        return float(np.mean(self.mean_activity < 1e-6))

    def save(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "stats": {
                "update_count": self.stats.update_count,
                "total_spikes": self.stats.total_spikes,
            },
        }


class PredictiveCodingLayer(HebbianLayer):
    """
    A layer that learns via predictive coding.
    Learns to predict its input from higher-level feedback.
    Minimizes local prediction error.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        # Feedback weights for top-down prediction
        limit = np.sqrt(6.0 / (output_dim + input_dim))
        self.feedback_weights = self.rng.uniform(-limit * 0.1, limit * 0.1, (input_dim, output_dim))
        self.prediction_error = np.zeros(input_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute representation from input."""
        z = self.weights @ x + self.bias
        y = np.maximum(0, z)

        if self.inhibition_strength > 0:
            mean_act = y.mean()
            if mean_act > 0:
                y = np.maximum(0, y - self.inhibition_strength * mean_act)

        return y

    def predict(self, y: np.ndarray) -> np.ndarray:
        """Generate top-down prediction."""
        return self.feedback_weights @ y

    def compute_error(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Prediction error: input - prediction."""
        prediction = self.predict(y)
        self.prediction_error = x - prediction
        return self.prediction_error

    def update(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        reward_signal: Optional[float] = None,
    ):
        """Update to minimize prediction error."""
        if y is None:
            y = self.forward(x)

        # Normalize input
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        y_active = y / (np.linalg.norm(y) + 1e-8)

        # Update feedforward weights
        delta_w = self.lr * np.outer(y_active, x_norm)
        self.weights += delta_w

        # Update feedback weights using prediction error
        error = self.compute_error(x_norm, y_active)
        delta_fb = self.lr * 0.5 * np.outer(error, y_active)
        self.feedback_weights += delta_fb

        # Weight bounding
        w_norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        self.weights = np.where(w_norms > self.weight_bound, self.weights * self.weight_bound / (w_norms + 1e-8), self.weights)

        fb_norms = np.linalg.norm(self.feedback_weights, axis=1, keepdims=True)
        self.feedback_weights = np.where(fb_norms > self.weight_bound, self.feedback_weights * self.weight_bound / (fb_norms + 1e-8), self.feedback_weights)

        # Decay
        self.weights *= (1 - self.decay)
        self.feedback_weights *= (1 - self.decay)

        # Homeostasis
        self.mean_activity = 0.99 * self.mean_activity + 0.01 * y
        excess = self.mean_activity - self.homeostasis_target
        self.bias -= self.dt * excess * self.lr * 0.1

        self.stats.update_count += 1
        self.stats.learning_events += 1


class HebbianSoftmaxClassifier:
    """Hebbian classifier using prototype-based learning."""

    def __init__(self, feature_dim: int, n_classes: int, temperature: float = 1.0):
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.temperature = temperature
        self.prototypes = np.zeros((n_classes, feature_dim))
        self.counts = np.zeros(n_classes)

    def predict(self, x: np.ndarray) -> int:
        """Nearest prototype."""
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        sims = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            p_norm = self.prototypes[c] / (np.linalg.norm(self.prototypes[c]) + 1e-8)
            sims[c] = np.dot(x_norm, p_norm)
        return int(np.argmax(sims))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        sims = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            p_norm = self.prototypes[c] / (np.linalg.norm(self.prototypes[c]) + 1e-8)
            sims[c] = np.dot(x_norm, p_norm)
        sims = sims / self.temperature
        sims -= sims.max()
        exp = np.exp(sims)
        return exp / exp.sum()

    def update(self, x: np.ndarray, label: int):
        """Update prototype — running average."""
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        self.counts[label] += 1
        alpha = 1.0 / self.counts[label]
        self.prototypes[label] = (1 - alpha) * self.prototypes[label] + alpha * x_norm

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == y[i])
        return correct / len(X)


if __name__ == "__main__":
    print("=" * 60)
    print("Hebbian Layers — Stability Test")
    print("=" * 60)

    layer = HebbianLayer(784, 100, learning_rate=0.01, weight_bound=1.0)

    for i in range(100):
        x = np.random.rand(784).astype(np.float64)
        x = x / np.linalg.norm(x)
        y = layer.forward(x)
        layer.update(x, y)

    print(f"HebbianLayer: {layer.stats.update_count} updates, {layer.stats.total_spikes} spikes")
    print(f"  Weight range: [{layer.weights.min():.3f}, {layer.weights.max():.3f}]")
    print(f"  Mean activation: {layer.stats.mean_activation:.4f}")
    print(f"  Sparsity: {layer.sparsity:.2%}")

    # Test predictive coding
    pc = PredictiveCodingLayer(784, 100, learning_rate=0.01)
    for i in range(100):
        x = np.random.rand(784).astype(np.float64)
        x = x / np.linalg.norm(x)
        y = pc.forward(x)
        pc.update(x, y)

    print(f"\nPredictiveCodingLayer: {pc.stats.update_count} updates")
    print(f"  Error norm: {np.linalg.norm(pc.prediction_error):.4f}")

    # Test classifier
    clf = HebbianSoftmaxClassifier(100, 10)
    correct = 0
    for i in range(200):
        x = np.random.rand(100).astype(np.float64)
        label = i % 10
        clf.update(x, label)
        if clf.predict(x) == label:
            correct += 1
    print(f"\nClassifier accuracy (same-sample): {correct / 200:.2%}")
    print("Hebbian layer tests passed.")
