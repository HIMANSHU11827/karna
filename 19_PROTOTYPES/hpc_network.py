"""
AGI Research Lab — Hierarchical Predictive Coding Network
Multi-layer predictive coding architecture for unsupervised feature learning.
Trains layer-by-layer with local Hebbian/anti-Hebbian updates.
"""

import numpy as np
from pathlib import Path
import json
import time
from typing import List, Tuple, Optional, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent))
from hebbian_layers import HebbianLayer, PredictiveCodingLayer, HebbianSoftmaxClassifier


class HierarchicalPredictiveCodingNetwork:
    """
    Multi-layer network that learns via hierarchical predictive coding.
    Each layer predicts the layer below's activity.
    Learning is purely local — no backpropagation.
    
    Training protocol:
    1. Layer-wise pretraining: train each layer independently
    2. Fine-tuning: adjust all layers together with slow learning rate
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01, seed: int = 42):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.layers: List[PredictiveCodingLayer] = []
        for i in range(len(layer_sizes) - 1):
            layer = PredictiveCodingLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i + 1],
                learning_rate=learning_rate,
                inhibition_strength=0.1,
                homeostasis_target=0.1,
                weight_bound=1.0,
                seed=seed + i,
            )
            self.layers.append(layer)

        self.classifier = HebbianSoftmaxClassifier(
            feature_dim=layer_sizes[-1],
            n_classes=10,
        )

        self.training_history: List[Dict[str, Any]] = []

    def encode(self, x: np.ndarray) -> List[np.ndarray]:
        """Encode input through all layers."""
        representations = []
        current = x.astype(np.float64)
        for layer in self.layers:
            current = layer.forward(current)
            representations.append(current)
        return representations

    def train_step(self, x: np.ndarray, label: Optional[int] = None, update_layers: bool = True) -> Dict[str, float]:
        """Single training step."""
        x = x.astype(np.float64)
        x_norm = x / (np.linalg.norm(x) + 1e-8)

        # Forward pass
        representations = self.encode(x_norm)

        # Update each layer locally
        total_error = 0.0
        prev = x_norm
        for i, layer in enumerate(self.layers):
            if update_layers:
                layer.update(prev, representations[i])
            error = layer.compute_error(prev / (np.linalg.norm(prev) + 1e-8), representations[i])
            total_error += float(np.linalg.norm(error))
            prev = representations[i]

        if label is not None:
            self.classifier.update(representations[-1], label)

        return {"total_error": total_error}

    def predict(self, x: np.ndarray) -> int:
        x = x.astype(np.float64)
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        representations = self.encode(x_norm)
        return self.classifier.predict(representations[-1])

    def train_epoch(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        update_layers: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)

        total_error = 0.0
        for i in indices:
            x = X[i].astype(np.float64)
            x = x / (np.linalg.norm(x) + 1e-8)
            label = int(y[i]) if y is not None else None
            metrics = self.train_step(x, label, update_layers)
            total_error += metrics["total_error"]

        mean_error = total_error / max(n_samples, 1)
        result = {"mean_error": mean_error, "n_steps": n_samples}

        if verbose:
            print(f"  Mean prediction error: {mean_error:.4f}")

        self.training_history.append(result)
        return result

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                correct += 1
        accuracy = correct / len(X)
        return {"accuracy": accuracy, "correct": correct, "total": len(X)}

    def save(self, path: str):
        state = {
            "layer_sizes": self.layer_sizes,
            "layers": [layer.save() for layer in self.layers],
            "classifier_prototypes": self.classifier.prototypes.tolist(),
            "training_history": self.training_history,
        }
        Path(path).write_text(json.dumps(state, indent=2))

    def summary(self) -> str:
        total_params = sum(
            layer.weights.size + layer.bias.size + layer.feedback_weights.size
            for layer in self.layers
        )
        return (
            f"HierarchicalPredictiveCodingNetwork\n"
            f"  Architecture: {' -> '.join(str(s) for s in self.layer_sizes)}\n"
            f"  Layers: {len(self.layers)}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Learning rate: {self.lr}"
        )


def generate_synthetic_mnist(n_samples: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic digit-like data."""
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
        img += rng.normal(0, 0.05, size)
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
    print("Hierarchical Predictive Coding Network — Demo")
    print("=" * 60)

    net = HierarchicalPredictiveCodingNetwork(
        layer_sizes=[784, 256, 100],
        learning_rate=0.005,
    )
    print(net.summary())

    print("\nGenerating synthetic data...")
    X_train, y_train, X_test, y_test = generate_synthetic_mnist(1000)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    print("\nPretraining layers (unsupervised)...")
    # Train first layer on raw data
    for epoch in range(3):
        print(f"  Layer 1 pretrain epoch {epoch + 1}:")
        for i in range(min(200, len(X_train))):
            x = X_train[i].astype(np.float64)
            x = x / (np.linalg.norm(x) + 1e-8)
            net.layers[0].update(x, net.layers[0].forward(x))

    # Train second layer on first layer representations
    print("  Pretraining layer 2...")
    for epoch in range(3):
        for i in range(min(200, len(X_train))):
            x = X_train[i].astype(np.float64)
            x = x / (np.linalg.norm(x) + 1e-8)
            h1 = net.layers[0].forward(x)
            net.layers[1].update(h1, net.layers[1].forward(h1))

    # Fine-tune with labels
    print("\nFine-tuning with labels...")
    for epoch in range(5):
        print(f"Epoch {epoch + 1}:")
        net.train_epoch(X_train, y_train)

    print("\nEvaluating...")
    results = net.evaluate(X_test, y_test)
    print(f"  Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

    output_path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/20_OUTPUT/hpc_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    net.save(output_path)
    print(f"\nResults saved to {output_path}")
