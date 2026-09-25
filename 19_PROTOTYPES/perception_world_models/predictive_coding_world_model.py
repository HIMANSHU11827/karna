"""
Predictive Coding World Model — Hebbian Learning Only
======================================================

A world model based on Predictive Coding (PC) theory:
- Hierarchical layers predict each other
- Prediction errors drive learning (no backprop)
- Hebbian update rules at each layer
- Event-driven, sparse representations

Architecture:
- Layer 0: Sensory input
- Layer 1-3: Hidden layers with predictive connections
- Each layer predicts the layer below
- Prediction errors flow upward, predictions flow downward
"""

import numpy as np
from typing import Optional


class PredictiveCodingLayer:
    """A single layer in the hierarchical predictive coding network.

    Each layer:
    - Receives predictions from above (top-down)
    - Receives input from below (bottom-up)
    - Computes prediction error
    - Updates weights via Hebbian learning
    """

    def __init__(self, n_neurons: int, n_below: int, n_above: int,
                 learning_rate: float = 0.01, sparsity: float = 0.1):
        self.n_neurons = n_neurons
        self.n_below = n_below
        self.n_above = n_above
        self.lr = learning_rate
        self.k = max(1, int(n_neurons * sparsity))

        # Feedforward weights (bottom-up): encode input from below
        self.W_ff = np.random.randn(n_neurons, n_below) * 0.1
        # Feedback weights (top-down): predict from above
        self.W_fb = np.random.randn(n_neurons, n_above) * 0.1 if n_above > 0 else None
        # Lateral weights (within layer)
        self.W_lat = np.random.randn(n_neurons, n_neurons) * 0.05

        # Biases
        self.bias = np.zeros(n_neurons)

        # State
        self.activation = np.zeros(n_neurons)
        self.prediction = np.zeros(n_neurons)
        self.error = np.zeros(n_neurons)

    def predict(self, input_below: np.ndarray, prediction_above: np.ndarray = None) -> np.ndarray:
        """Generate prediction for this layer."""
        # Bottom-up drive
        ff_drive = self.W_ff @ input_below if input_below is not None else np.zeros(self.n_neurons)

        # Top-down prediction
        if prediction_above is not None and self.W_fb is not None:
            fb_drive = self.W_fb @ prediction_above
        else:
            fb_drive = np.zeros(self.n_neurons)

        # Combined prediction
        self.prediction = np.tanh(ff_drive + fb_drive + self.bias)
        return self.prediction

    def compute_error(self, actual: np.ndarray) -> np.ndarray:
        """Compute prediction error."""
        self.error = actual - self.prediction
        return self.error

    def activate(self, input_below: np.ndarray, prediction_above: np.ndarray = None) -> np.ndarray:
        """Activate layer with sparse k-winner-take-all."""
        # Compute prediction
        self.predict(input_below, prediction_above)

        # Bottom-up drive
        drive = self.W_ff @ input_below if input_below is not None else np.zeros(self.n_neurons)

        # Sparse activation
        self.activation = np.zeros(self.n_neurons)
        if self.k >= self.n_neurons:
            self.activation = np.maximum(drive, 0)
        else:
            top_k = np.argsort(drive)[-self.k:]
            self.activation[top_k] = np.maximum(drive[top_k], 0)

        return self.activation

    def update(self, input_below: np.ndarray, prediction_above: np.ndarray = None) -> None:
        """Update weights using Hebbian learning (Oja's rule).

        ΔW_ff[i,j] = lr * error[i] * input[j]
        This is a Hebbian update: correlated neurons strengthen connections.
        """
        # Feedforward update (Hebbian: error-driven)
        if input_below is not None:
            for i in range(self.n_neurons):
                error_i = self.activation[i] - self.prediction[i]
                self.W_ff[i] += self.lr * error_i * input_below

        # Feedback update (Hebbian)
        if prediction_above is not None and self.W_fb is not None:
            for i in range(self.n_neurons):
                self.W_fb[i] += self.lr * self.error[i] * prediction_above

        # Lateral update (decorrelate)
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j:
                    self.W_lat[i, j] += self.lr * 0.1 * (
                        self.activation[i] * self.activation[j] -
                        self.activation[i] ** 2 * self.W_lat[i, j]
                    )

        # Renormalize
        self.W_ff /= (np.linalg.norm(self.W_ff, axis=1, keepdims=True) + 1e-8)
        if self.W_fb is not None:
            self.W_fb /= (np.linalg.norm(self.W_fb, axis=1, keepdims=True) + 1e-8)


class PredictiveCodingNetwork:
    """Hierarchical Predictive Coding Network.

    Layers stacked vertically. Each layer predicts the layer below.
    Learning is purely local (Hebbian), no backpropagation.
    """

    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.01,
                 sparsity: float = 0.1):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.layers: list[PredictiveCodingLayer] = []

        for i, size in enumerate(layer_sizes):
            n_below = layer_sizes[i - 1] if i > 0 else size
            n_above = layer_sizes[i + 1] if i < self.n_layers - 1 else 0

            layer = PredictiveCodingLayer(
                n_neurons=size,
                n_below=n_below,
                n_above=n_above,
                learning_rate=learning_rate,
                sparsity=sparsity,
            )
            self.layers.append(layer)

        self._update_count = 0

    def forward_pass(self, sensory_input: np.ndarray) -> list[np.ndarray]:
        """Bottom-up pass: encode input through hierarchy."""
        activations = [sensory_input]

        for i, layer in enumerate(self.layers):
            pred_above = activations[-1] if i < self.n_layers - 1 else None
            input_below = activations[0] if i == 0 else activations[-1]

            layer.activate(input_below, pred_above)
            activations.append(layer.activation.copy())

        return activations

    def update_all(self, sensory_input: np.ndarray) -> None:
        """Update all layers via Hebbian learning.

        1. Forward pass to get predictions
        2. Compute errors at each layer
        3. Update weights (Hebbian, local)
        """
        activations = self.forward_pass(sensory_input)

        # Update each layer
        for i, layer in enumerate(self.layers):
            input_below = activations[i] if i > 0 else sensory_input
            pred_above = activations[i + 2] if i < self.n_layers - 1 else None

            layer.update(input_below, pred_above)

        self._update_count += 1

    def encode(self, sensory_input: np.ndarray, update: bool = True) -> list[np.ndarray]:
        """Encode sensory input into hierarchical representation.

        Args:
            sensory_input: Raw input vector
            update: Whether to update weights

        Returns:
            List of activations at each layer
        """
        if update:
            self.update_all(sensory_input)
        else:
            self.forward_pass(sensory_input)

        return [layer.activation.copy() for layer in self.layers]

    def get_reconstruction(self, layer_idx: int = 0) -> np.ndarray:
        """Reconstruct input from a layer's representation."""
        if layer_idx >= self.n_layers:
            return np.zeros(self.layer_sizes[0])

        layer = self.layers[layer_idx]
        # Simple reconstruction: activation @ W_ff.T
        return layer.activation @ layer.W_ff

    def get_stats(self) -> dict:
        """Get network statistics."""
        return {
            "n_layers": self.n_layers,
            "layer_sizes": self.layer_sizes,
            "total_updates": self._update_count,
            "total_parameters": sum(
                layer.W_ff.size + (layer.W_fb.size if layer.W_fb is not None else 0) + layer.W_lat.size
                for layer in self.layers
            ),
        }


def train_pc_network(n_epochs: int = 5, n_samples: int = 500) -> dict:
    """Train the predictive coding network on synthetic data."""
    print("=" * 60)
    print("  Predictive Coding Network — Training")
    print("=" * 60)
    print()

    # Generate synthetic data
    np.random.seed(42)
    dim = 784
    data = np.random.randn(n_samples, dim).astype(np.float32) * 0.1

    # Add structure: clusters
    for i in range(n_samples):
        cluster = i % 10
        data[i, cluster * 78:(cluster + 1) * 78] += 1.0

    # Normalize
    data /= (np.linalg.norm(data, axis=1, keepdims=True) + 1e-8)

    # Create network: 784 -> 256 -> 128 -> 64
    pc_net = PredictiveCodingNetwork(
        layer_sizes=[256, 128, 64],
        learning_rate=0.01,
        sparsity=0.1,
    )

    print(f"Network: {' -> '.join(map(str, pc_net.layer_sizes))}")
    print(f"Total parameters: {pc_net.get_stats()['total_parameters']}")
    print(f"Training samples: {n_samples}")
    print()

    # Training
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        total_error = 0.0

        for idx in indices:
            x = data[idx]
            pc_net.encode(x, update=True)

            # Track reconstruction error
            recon = pc_net.get_reconstruction(layer_idx=0)
            error = np.linalg.norm(x - recon)
            total_error += error

        avg_error = total_error / n_samples
        print(f"Epoch {epoch + 1}/{n_samples}: Avg reconstruction error = {avg_error:.4f}")

    print()
    print("=" * 60)
    print("  Final Statistics")
    print("=" * 60)
    stats = pc_net.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Test encoding
    print("  Test encoding (first 3 samples):")
    for i in range(3):
        activations = pc_net.encode(data[i], update=False)
        print(f"    Sample {i}: "
              f"layer0={np.sum(activations[0] > 0)} active, "
              f"layer1={np.sum(activations[1] > 0)} active, "
              f"layer2={np.sum(activations[2] > 0)} active")

    print()
    print("=" * 60)
    print("  Training Complete")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    train_pc_network(n_epochs=5, n_samples=500)
