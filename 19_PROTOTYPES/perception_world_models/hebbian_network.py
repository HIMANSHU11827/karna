"""
Hebbian Neural Network — Phase 2 AGI Research Lab
==================================================

NumPy-only implementation of a Hebbian learning network.
No backpropagation, no autograd, no PyTorch/TensorFlow.

Architecture:
- Oja's Rule: normalized Hebbian learning (prevents unbounded growth)
- HebbianLayer: single layer with Hebbian weight updates
- HebbianNetwork: multi-layer network with layer-wise learning
- Supports continual learning (Split-MNIST)

"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional


# ============================================================
# Oja's Rule
# ============================================================

class OjaRule:
    """Oja's learning rule for Hebbian learning.

    Oja's rule: Δw_ij = η * y_j * (x_i - y_j * w_ij)
    This normalizes Hebbian updates to prevent unbounded weight growth.

    Args:
        learning_rate: Step size for weight updates (η)
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Update weights using Oja's rule.

        Args:
            weights: Current weight matrix (input_dim, output_dim)
            x: Input vector (input_dim,)
            y: Output vector (output_dim,)

        Returns:
            Updated weight matrix (input_dim, output_dim)
        """
        # Oja's rule: Δw_ij = η * y_j * (x_i - y_j * w_ij)
        # Vectorized: ΔW = η * (outer(x, y) - y² * W)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        xy_outer = np.outer(x, y)  # (input_dim, output_dim) where [i,j] = x[i] * y[j]
        decay = (y ** 2)[np.newaxis, :] * weights  # y[j]² * W[i,j]
        dW = self.learning_rate * (xy_outer - decay)

        new_weights = weights + dW

        # Clip weights to prevent overflow (stability constraint)
        max_weight_norm = 10.0
        weight_norm = np.linalg.norm(new_weights)
        if weight_norm > max_weight_norm:
            new_weights = new_weights * (max_weight_norm / weight_norm)

        return new_weights


# ============================================================
# HebbianLayer
# ============================================================

class HebbianLayer:
    """A single layer with Hebbian learning.

    Supports both unsupervised (Oja's rule) and supervised Hebbian learning.
    In supervised mode, uses target outputs to guide weight updates for
    classification tasks.

    Args:
        input_dim: Number of input features
        output_dim: Number of output features
        learning_rate: Learning rate for Hebbian updates
        supervised: If True, use supervised Hebbian learning with targets
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.01,
        supervised: bool = False,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.supervised = supervised

        # Initialize weights with small random values (Xavier/Glorot initialization)
        scale = math.sqrt(1.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim) * scale
        self.bias = np.zeros(output_dim)

        # Learning rule
        self.rule = OjaRule(learning_rate)

        # For tracking
        self._forward_count = 0
        self._learn_count = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer.

        Args:
            x: Input array of shape (input_dim,) or (batch_size, input_dim)

        Returns:
            Output array of shape (output_dim,) or (batch_size, output_dim)
        """
        self._forward_count += 1

        # Center input to zero-mean (critical for Oja's rule stability)
        # Oja's rule assumes zero-mean inputs; MNIST pixels are [0, 1]
        x = np.asarray(x, dtype=np.float64)
        x = x - 0.5  # Center around zero (MNIST range is [0, 1])

        # Linear transformation
        output = x @ self.weights + self.bias

        # Activation: tanh for non-linearity
        output = np.tanh(output)

        return output

    def learn(self, x: np.ndarray, target: np.ndarray | None = None) -> None:
        """Update weights using Hebbian learning.

        Args:
            x: Input array of shape (input_dim,) or (batch_size, input_dim)
            target: Target output for supervised learning (output_dim,)
        """
        self._learn_count += 1

        # Forward pass to get output
        output = self.forward(x)

        if self.supervised and target is not None:
            # Supervised Hebbian: use target as the output signal
            # This is a simplified version of the "Hebbian learning with target" rule
            if x.ndim == 1:
                self.weights = self.rule.update(self.weights, x, target)
            else:
                batch_size = x.shape[0]
                weight_updates = np.zeros_like(self.weights)
                for i in range(batch_size):
                    weight_updates += self.rule.update(self.weights, x[i], target[i])
                self.weights = self.weights + weight_updates / batch_size
        else:
            # Unsupervised: use Oja's rule with the actual output
            if x.ndim == 1:
                self.weights = self.rule.update(self.weights, x, output)
            else:
                batch_size = x.shape[0]
                weight_updates = np.zeros_like(self.weights)
                for i in range(batch_size):
                    weight_updates += self.rule.update(self.weights, x[i], output[i])
                self.weights = self.weights + weight_updates / batch_size

    def get_stats(self) -> dict:
        """Get layer statistics."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "supervised": self.supervised,
            "forward_count": self._forward_count,
            "learn_count": self._learn_count,
            "weight_norm": float(np.linalg.norm(self.weights)),
        }


# ============================================================
# HebbianNetwork
# ============================================================

class LogisticRegression:
    """Simple logistic regression classifier for supervised classification.

    Trained via gradient descent (not backpropagation through the network).
    This is a standalone classifier that operates on features extracted
    by the Hebbian network.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        learning_rate: Learning rate for gradient descent
    """

    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 0.1):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Initialize weights (Xavier initialization)
        scale = math.sqrt(1.0 / input_dim)
        self.weights = np.random.randn(input_dim, num_classes) * scale
        self.bias = np.zeros(num_classes)

        self._trained = False
        self._loss_history = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: compute class probabilities via softmax.

        Args:
            X: Input features of shape (batch_size, input_dim)

        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = X @ self.weights + self.bias
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input features of shape (batch_size, input_dim)

        Returns:
            Predicted labels of shape (batch_size,)
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        verbose: bool = False,
    ) -> list[float]:
        """Train the classifier via gradient descent.

        Args:
            X: Training features of shape (n_samples, input_dim)
            y: Training labels of shape (n_samples,) — integer class indices
            epochs: Number of training epochs
            verbose: Print progress

        Returns:
            List of loss values per epoch
        """
        n_samples = X.shape[0]
        losses = []

        # One-hot encode labels
        y_onehot = np.zeros((n_samples, self.num_classes))
        y_onehot[np.arange(n_samples), y] = 1.0

        for epoch in range(epochs):
            # Forward pass
            probabilities = self.forward(X)

            # Compute loss (cross-entropy)
            loss = -np.mean(np.sum(y_onehot * np.log(probabilities + 1e-10), axis=1))
            losses.append(loss)

            # Backpropagation (gradient descent) — only through the classifier
            # This is NOT backpropagation through the Hebbian network
            error = probabilities - y_onehot  # (n_samples, num_classes)
            grad_weights = (X.T @ error) / n_samples  # (input_dim, num_classes)
            grad_bias = np.mean(error, axis=0)  # (num_classes,)

            # Update weights
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Classifier Epoch {epoch + 1}/{epochs} — Loss: {loss:.4f}")

        self._trained = True
        self._loss_history = losses
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the classifier.

        Args:
            X: Test features
            y: Test labels

        Returns:
            Dictionary with accuracy and loss
        """
        probabilities = self.forward(X)
        predictions = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predictions == y)

        # Compute loss
        n_samples = X.shape[0]
        y_onehot = np.zeros((n_samples, self.num_classes))
        y_onehot[np.arange(n_samples), y] = 1.0
        loss = -np.mean(np.sum(y_onehot * np.log(probabilities + 1e-10), axis=1))

        return {"accuracy": accuracy, "loss": loss}


class HybridHebbianClassifier:
    """Hybrid classifier: Hebbian feature extractor + logistic regression.

    Architecture:
    1. Hebbian network (unsupervised Oja's rule) extracts features
    2. Logistic regression (supervised) classifies the features

    This is NOT backpropagation through the network. The Hebbian layers
    learn unsupervised features, and only the final classifier layer is
    trained via gradient descent.
    """

    def __init__(self, hebbian_network: HebbianNetwork, learning_rate: float = 0.1):
        self.hebbian_network = hebbian_network
        output_dim = hebbian_network.layer_dims[-1]
        self.classifier = LogisticRegression(
            input_dim=output_dim,
            num_classes=output_dim,
            learning_rate=learning_rate,
        )

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features using the Hebbian network.

        Args:
            X: Input data of shape (batch_size, input_dim)

        Returns:
            Features of shape (batch_size, output_dim)
        """
        # Forward pass through Hebbian layers (without the last supervised layer)
        current = X
        for i, layer in enumerate(self.hebbian_network.layers):
            if i < len(self.hebbian_network.layers) - 1:
                # Unsupervised layers: forward pass only
                current = layer.forward(current)
            else:
                # Last layer: use pre-activation output as features
                current = np.asarray(current, dtype=np.float64)
                current = current - 0.5  # Center
                current = current @ layer.weights + layer.bias
                # Apply tanh
                current = np.tanh(current)
        return current

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        verbose: bool = True,
    ) -> list[float]:
        """Train the hybrid classifier.

        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of training epochs for the classifier
            verbose: Print progress

        Returns:
            List of loss values
        """
        # Step 1: Train Hebbian layers (unsupervised)
        if verbose:
            print("  Step 1: Training Hebbian layers (unsupervised)...")
        self.hebbian_network.train(X_train, y=None, epochs=5, verbose=False)

        # Step 2: Extract features
        if verbose:
            print("  Step 2: Extracting features...")
        features = self.extract_features(X_train)

        # Step 3: Train classifier (supervised)
        if verbose:
            print("  Step 3: Training classifier (supervised)...")
        losses = self.classifier.train(features, y_train, epochs=epochs, verbose=verbose)

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input data

        Returns:
            Predicted labels
        """
        features = self.extract_features(X)
        return self.classifier.predict(features)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the hybrid classifier.

        Args:
            X: Test data
            y: Test labels

        Returns:
            Dictionary with accuracy and loss
        """
        features = self.extract_features(X)
        return self.classifier.evaluate(features, y)


class HebbianNetwork:
    """Multi-layer Hebbian neural network.

    Each layer learns independently via Hebbian updates (no backpropagation).
    Supports continual learning by freezing older layers.

    Args:
        layer_dims: List of layer dimensions, e.g. [784, 64, 10]
        learning_rate: Learning rate for all layers
        supervised_last: If True, last layer uses supervised Hebbian with targets
    """

    def __init__(
        self,
        layer_dims: list[int],
        learning_rate: float = 0.01,
        supervised_last: bool = True,
    ):
        if len(layer_dims) < 2:
            raise ValueError("Need at least input and output dimensions")

        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.layers: list[HebbianLayer] = []

        # Create layers
        for i in range(len(layer_dims) - 1):
            is_last = (i == len(layer_dims) - 2)
            layer = HebbianLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                learning_rate=learning_rate,
                supervised=(is_last and supervised_last),
            )
            self.layers.append(layer)

        # For tracking
        self._forward_count = 0
        self._learn_count = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the entire network.

        Args:
            x: Input array of shape (input_dim,) or (batch_size, input_dim)

        Returns:
            Output array of shape (output_dim,) or (batch_size, output_dim)
        """
        self._forward_count += 1

        # Pass through each layer
        current = x
        for layer in self.layers:
            current = layer.forward(current)

        return current

    def learn(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Update all layers using Hebbian learning.

        Args:
            x: Input array
            y: Target output (used for supervised Hebbian learning)
        """
        self._learn_count += 1

        # Forward pass through all layers
        outputs = []
        current = x
        for layer in self.layers:
            current = layer.forward(current)
            outputs.append(current)

        # Update each layer's weights
        prev_output = x
        for i, layer in enumerate(self.layers):
            if layer.supervised and y is not None:
                # Supervised Hebbian: use target as output signal
                layer.learn(prev_output, target=y)
            else:
                # Unsupervised: use Oja's rule
                layer.learn(prev_output)
            prev_output = outputs[i]

    def train(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 10,
        verbose: bool = False,
    ) -> list[float]:
        """Train the network for multiple epochs.

        Args:
            X: Training data of shape (n_samples, input_dim)
            y: Target data of shape (n_samples, output_dim)
            epochs: Number of training epochs
            verbose: Print progress

        Returns:
            List of loss values per epoch
        """
        losses = []
        n_samples = X.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Shuffle data
            indices = np.random.permutation(n_samples)

            for idx in indices:
                x = X[idx]
                target = y[idx] if y is not None else None

                # Forward pass
                output = self.forward(x)

                # Compute loss (MSE)
                if target is not None:
                    loss = np.mean((output - target) ** 2)
                    epoch_loss += loss

                # Learn
                self.learn(x, target)

            avg_loss = epoch_loss / n_samples
            losses.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for given inputs.

        Args:
            X: Input array of shape (n_samples, input_dim)

        Returns:
            Predictions of shape (n_samples, output_dim)
        """
        return self.forward(X)

    def freeze_layer(self, layer_idx: int) -> None:
        """Freeze a layer to prevent further learning (for continual learning).

        Args:
            layer_idx: Index of layer to freeze
        """
        if 0 <= layer_idx < len(self.layers):
            self.layers[layer_idx].learning_rate = 0.0

    def unfreeze_layer(self, layer_idx: int) -> None:
        """Unfreeze a layer to allow learning.

        Args:
            layer_idx: Index of layer to unfreeze
        """
        if 0 <= layer_idx < len(self.layers):
            self.layers[layer_idx].learning_rate = self.learning_rate

    def get_stats(self) -> dict:
        """Get network statistics."""
        return {
            "num_layers": len(self.layers),
            "layer_dims": self.layer_dims,
            "learning_rate": self.learning_rate,
            "forward_count": self._forward_count,
            "learn_count": self._learn_count,
            "layer_stats": [layer.get_stats() for layer in self.layers],
        }


# ============================================================
# Demo
# ============================================================

def main():
    """Run Hebbian network demo."""
    print("=" * 60)
    print("  Hebbian Neural Network Demo")
    print("=" * 60)
    print()

    # Demo 1: Simple pattern learning
    print("[1] Learning simple patterns...")
    np.random.seed(42)

    # Create network
    network = HebbianNetwork(layer_dims=[10, 20, 5], learning_rate=0.01)

    # Generate simple data
    X = np.random.randn(100, 10) * 0.5
    y = np.zeros((100, 5))
    for i in range(100):
        y[i, i % 5] = 1.0

    # Train
    losses = network.train(X, y, epochs=20, verbose=False)

    print(f"    Initial loss: {losses[0]:.4f}")
    print(f"    Final loss: {losses[-1]:.4f}")
    print(f"    Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    print()

    # Demo 2: Forward pass
    print("[2] Testing forward pass...")
    test_input = np.random.randn(10)
    output = network.forward(test_input)
    print(f"    Input shape: {test_input.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")
    print()

    # Demo 3: Batch prediction
    print("[3] Batch prediction...")
    batch = np.random.randn(32, 10)
    predictions = network.predict(batch)
    print(f"    Batch shape: {batch.shape}")
    print(f"    Predictions shape: {predictions.shape}")
    print()

    # Demo 4: Continual learning
    print("[4] Testing continual learning...")
    network2 = HebbianNetwork(layer_dims=[20, 10, 2], learning_rate=0.01)

    # Task A
    np.random.seed(42)
    X_A = np.random.randn(50, 20)
    y_A = (X_A[:, :10].sum(axis=1) > 0).astype(float).reshape(-1, 1)
    network2.train(X_A, y_A, epochs=10, verbose=False)

    # Evaluate A
    pred_A_before = network2.predict(X_A)
    error_A_before = np.mean((pred_A_before - y_A) ** 2)

    # Task B
    np.random.seed(123)
    X_B = np.random.randn(50, 20)
    y_B = (X_B[:, 10:].sum(axis=1) > 0).astype(float).reshape(-1, 1)
    network2.train(X_B, y_B, epochs=10, verbose=False)

    # Evaluate A again
    pred_A_after = network2.predict(X_A)
    error_A_after = np.mean((pred_A_after - y_A) ** 2)

    print(f"    Task A error before: {error_A_before:.4f}")
    print(f"    Task A error after:  {error_A_after:.4f}")
    print(f"    Forgetting ratio:    {error_A_after / error_A_before:.2f}x")
    print()

    # Demo 5: Network stats
    print("[5] Network statistics:")
    stats = network.get_stats()
    print(f"    Num layers: {stats['num_layers']}")
    print(f"    Layer dims: {stats['layer_dims']}")
    print(f"    Forward passes: {stats['forward_count']}")
    print(f"    Learn passes: {stats['learn_count']}")
    for i, layer_stat in enumerate(stats['layer_stats']):
        print(f"    Layer {i}: {layer_stat['input_dim']} -> {layer_stat['output_dim']}, "
              f"weight norm: {layer_stat['weight_norm']:.4f}")
    print()

    print("=" * 60)
    print("  Hebbian Neural Network Demo — Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
