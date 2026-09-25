"""
Event-Driven Sparse Perception with Hebbian Learning
=====================================================

A biologically-inspired perception system using:
- Event-driven processing (only process when something changes)
- Sparse representations (k-winner-take-all)
- Hebbian learning (no backpropagation)
- NumPy only, CPU-compatible

Architecture:
- Input layer: sensory encoding
- Sparse coding layer: k-winners-take-all activation
- Hebbian updates: Oja's rule / BCM theory
"""

import numpy as np
from typing import Optional


# ============================================================
# Sparse Coding Layer
# ============================================================

class SparseCoder:
    """Sparse coding layer with k-winner-take-all activation.

    Learns a dictionary of features via competitive learning.
    Activates only k neurons per input (sparse activation).
    """

    def __init__(self, input_dim: int, n_neurons: int, k: int = 10,
                 learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.k = k  # Number of active neurons (sparsity)
        self.lr = learning_rate

        # Dictionary of learned features (random init)
        self.dictionary = np.random.randn(n_neurons, input_dim) * 0.1
        # Normalize
        self.dictionary /= (np.linalg.norm(self.dictionary, axis=1, keepdims=True) + 1e-8)

        # Neuron biases
        self.bias = np.zeros(n_neurons)

        # Activation history for statistics
        self.activation_counts = np.zeros(n_neurons)
        self.total_updates = 0

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input into sparse representation.

        Args:
            x: Input vector of shape (input_dim,)

        Returns:
            Sparse activation vector of shape (n_neurons,)
        """
        # Compute similarity (dot product with each neuron)
        activations = self.dictionary @ x + self.bias

        # k-winner-take-all: only top-k neurons activate
        sparse = np.zeros(self.n_neurons)
        if self.k >= self.n_neurons:
            # All neurons active
            sparse = np.maximum(activations, 0)
        else:
            # Only top-k neurons
            top_k_idx = np.argsort(activations)[-self.k:]
            sparse[top_k_idx] = np.maximum(activations[top_k_idx], 0)

        return sparse

    def update(self, x: np.ndarray, sparse: np.ndarray) -> None:
        """Update dictionary using Oja's rule (Hebbian learning).

        Oja's rule: Δw_i = lr * (x * y_i - y_i^2 * w_i)
        This is a normalized Hebbian rule that prevents unbounded growth.
        """
        active_idx = np.where(sparse > 0)[0]
        for i in active_idx:
            y_i = sparse[i]
            # Oja's rule
            dw = self.lr * (y_i * x - y_i ** 2 * self.dictionary[i])
            self.dictionary[i] += dw

        # Renormalize
        norms = np.linalg.norm(self.dictionary, axis=1, keepdims=True)
        self.dictionary /= (norms + 1e-8)

        # Update statistics
        self.activation_counts[active_idx] += 1
        self.total_updates += 1

    def get_sparsity(self) -> float:
        """Get average sparsity (fraction of neurons active)."""
        return self.k / self.n_neurons

    def get_stats(self) -> dict:
        """Get coder statistics."""
        return {
            "n_neurons": self.n_neurons,
            "k_active": self.k,
            "sparsity": self.get_sparsity(),
            "total_updates": self.total_updates,
            "activation_mean": float(np.mean(self.activation_counts)),
            "activation_std": float(np.std(self.activation_counts)),
        }


# ============================================================
# Event-Driven Perception
# ============================================================

class EventDrivenPerception:
    """Event-driven perception system.

    Only processes inputs when they differ significantly from
    the last processed input (change detection).
    """

    def __init__(self, input_dim: int, n_neurons: int = 256, k: int = 10,
                 change_threshold: float = 0.1):
        self.input_dim = input_dim
        self.change_threshold = change_threshold

        # Sparse coding layer
        self.coder = SparseCoder(input_dim, n_neurons, k)

        # Previous input for change detection
        self.prev_input: Optional[np.ndarray] = None
        self.prev_sparse: Optional[np.ndarray] = None

        # Statistics
        self.n_events = 0
        self.n_skipped = 0
        self.total_inputs = 0

    def process(self, x: np.ndarray, update: bool = True) -> Optional[np.ndarray]:
        """Process input, but only if it's different enough.

        Args:
            x: Input vector
            update: Whether to update the dictionary

        Returns:
            Sparse representation, or None if input was skipped
        """
        self.total_inputs += 1

        # Change detection
        if self.prev_input is not None:
            diff = np.linalg.norm(x - self.prev_input)
            if diff < self.change_threshold:
                self.n_skipped += 1
                return None  # Skip — not different enough

        # Process the event
        self.n_events += 1
        sparse = self.coder.encode(x)

        # Update dictionary
        if update:
            self.coder.update(x, sparse)

        self.prev_input = x.copy()
        self.prev_sparse = sparse.copy()

        return sparse

    def get_event_rate(self) -> float:
        """Get fraction of inputs that triggered events."""
        if self.total_inputs == 0:
            return 0.0
        return self.n_events / self.total_inputs

    def get_stats(self) -> dict:
        """Get perception system statistics."""
        return {
            "total_inputs": self.total_inputs,
            "events_processed": self.n_events,
            "inputs_skipped": self.n_skipped,
            "event_rate": self.get_event_rate(),
            "change_threshold": self.change_threshold,
            "coder_stats": self.coder.get_stats(),
        }


# ============================================================
# MNIST Loader (NumPy only)
# ============================================================

def load_mnist_from_keras() -> tuple:
    """Load MNIST dataset (uses keras for data, then converts to NumPy)."""
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize and flatten
        x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
        x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

        return x_train, y_train, x_test, y_test
    except ImportError:
        raise ImportError("TensorFlow/Keras not available. Install with: pip install tensorflow")


def generate_synthetic_data(n_samples: int = 1000, dim: int = 784) -> tuple:
    """Generate synthetic data for testing without external deps."""
    np.random.seed(42)
    # Create clustered data
    n_clusters = 10
    data = np.zeros((n_samples, dim))
    labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        cluster = i % n_clusters
        labels[i] = cluster
        # Random prototype
        prototype = np.random.randn(dim) * 0.1
        prototype[cluster * 78:(cluster + 1) * 78] += 1.0
        data[i] = prototype + np.random.randn(dim) * 0.05

    # Normalize
    data = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-8)
    return data, labels


# ============================================================
# Demo: Train and Test Event-Driven Perception
# ============================================================

def train_perception(n_epochs: int = 3, n_samples: int = 1000,
                     use_mnist: bool = False) -> dict:
    """Train the event-driven perception system.

    Args:
        n_epochs: Number of training epochs
        n_samples: Number of training samples
        use_mnist: Whether to use MNIST (requires TF) or synthetic data

    Returns:
        Training statistics
    """
    print("=" * 60)
    print("  Event-Driven Sparse Perception — Training")
    print("=" * 60)
    print()

    # Load data
    if use_mnist:
        try:
            x_train, y_train, x_test, y_test = load_mnist_from_keras()
            x_train = x_train[:n_samples]
            print(f"Using MNIST: {len(x_train)} samples")
        except ImportError:
            print("MNIST not available, using synthetic data")
            x_train, y_train = generate_synthetic_data(n_samples)
            x_test, y_test = generate_synthetic_data(200)
    else:
        x_train, y_train = generate_synthetic_data(n_samples)
        x_test, y_test = generate_synthetic_data(200)
        print(f"Using synthetic data: {len(x_train)} samples")

    # Initialize perception system
    input_dim = x_train.shape[1]
    perception = EventDrivenPerception(
        input_dim=input_dim,
        n_neurons=256,
        k=20,  # Top-20 active neurons (7.8% sparsity)
        change_threshold=0.05,
    )

    print(f"Input dim: {input_dim}")
    print(f"Neurons: {perception.coder.n_neurons}")
    print(f"k (active): {perception.coder.k}")
    print(f"Sparsity: {perception.coder.get_sparsity():.2%}")
    print()

    # Training loop
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")

        # Shuffle
        indices = np.random.permutation(len(x_train))

        for idx in indices:
            x = x_train[idx]
            perception.process(x, update=True)

        stats = perception.get_stats()
        print(f"  Events: {stats['events_processed']}/{stats['total_inputs']}")
        print(f"  Event rate: {stats['event_rate']:.2%}")
        print(f"  Skipped: {stats['inputs_skipped']}")
        print()

    # Final statistics
    print("=" * 60)
    print("  Final Statistics")
    print("=" * 60)
    stats = perception.get_stats()
    for key, value in stats.items():
        if key != "coder_stats":
            print(f"  {key}: {value}")
    print()
    print("  Coder stats:")
    for key, value in stats["coder_stats"].items():
        print(f"    {key}: {value}")
    print()

    # Test on a few samples
    print("  Test encoding (first 5 samples):")
    for i in range(min(5, len(x_test))):
        sparse = perception.process(x_test[i], update=False)
        if sparse is not None:
            active = np.sum(sparse > 0)
            print(f"    Sample {i}: {active} active neurons, "
                  f"sparsity={active/perception.coder.n_neurons:.2%}")

    print()
    print("=" * 60)
    print("  Training Complete")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    train_perception(n_epochs=3, n_samples=500, use_mnist=False)
