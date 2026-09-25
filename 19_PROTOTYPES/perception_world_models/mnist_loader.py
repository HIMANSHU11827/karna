"""
MNIST Data Loader and Training Pipeline for Hebbian Network
============================================================

Loads MNIST dataset and trains Hebbian network with evaluation.
NumPy-only implementation — no external ML libraries.

Usage:
    from mnist_loader import load_mnist
    from hebbian_network import HebbianNetwork

    X_train, y_train, X_test, y_test = load_mnist()
    network = HebbianNetwork(layer_dims=[784, 256, 10], learning_rate=0.01)
    results = train_and_evaluate(network, X_train, y_train, X_test, y_test)
"""

import gzip
import json
import os
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np


# ============================================================
# MNIST Data Loader
# ============================================================

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_file(url: str, filepath: Path) -> None:
    """Download a file if it doesn't exist."""
    if filepath.exists():
        return
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"  Saved to {filepath}")


def read_idx_file(filepath: Path) -> np.ndarray:
    """Read an IDX file (MNIST format)."""
    with gzip.open(filepath, "rb") as f:
        # Read magic number and dimensions
        magic = struct.unpack(">I", f.read(4))[0]
        if magic == 2051:  # Images
            num_images = struct.unpack(">I", f.read(4))[0]
            num_rows = struct.unpack(">I", f.read(4))[0]
            num_cols = struct.unpack(">I", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, num_rows * num_cols)
        elif magic == 2049:  # Labels
            num_labels = struct.unpack(">I", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown magic number: {magic}")
    return data


def load_mnist(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset.

    Returns:
        X_train: (60000, 784) float32, normalized to [0, 1]
        y_train: (60000,) int64
        X_test: (10000, 784) float32, normalized to [0, 1]
        y_test: (10000,) int64
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    # Download if needed
    files = {}
    for name, url in MNIST_URLS.items():
        filepath = data_path / Path(url).name
        download_file(url, filepath)
        files[name] = filepath

    # Read data
    print("Loading MNIST dataset...")
    X_train = read_idx_file(files["train_images"]).astype(np.float32) / 255.0
    y_train = read_idx_file(files["train_labels"]).astype(np.int64)
    X_test = read_idx_file(files["test_images"]).astype(np.float32) / 255.0
    y_test = read_idx_file(files["test_labels"]).astype(np.int64)

    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"  Image size: {X_train.shape[1]} pixels")

    return X_train, y_train, X_test, y_test


# ============================================================
# One-Hot Encoding
# ============================================================

def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot encoded vectors."""
    encoded = np.zeros((len(labels), num_classes), dtype=np.float32)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


# ============================================================
# Training and Evaluation
# ============================================================

def train_and_evaluate(
    network,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    verbose: bool = True,
) -> dict:
    """Train Hebbian network and evaluate on test set.

    Uses a hybrid approach:
    - Hidden layers: unsupervised Oja's rule (feature extraction)
    - Output layer: supervised Hebbian with targets

    Args:
        network: HebbianNetwork instance
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Print progress

    Returns:
        Dictionary with training results and metrics
    """
    n_samples = X_train.shape[0]
    y_train_onehot = one_hot_encode(y_train, network.layer_dims[-1])

    # Track metrics
    losses = []
    accuracies = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0

        # Shuffle
        indices = np.random.permutation(n_samples)

        for batch_start in range(0, n_samples, batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train_onehot[batch_idx]

            # Train on batch
            network.learn(X_batch, y_batch)

            # Compute batch loss
            predictions = network.forward(X_batch)
            batch_loss = np.mean((predictions - y_batch) ** 2)
            epoch_loss += batch_loss * len(batch_idx)

            # Compute batch accuracy
            pred_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            correct += np.sum(pred_labels == true_labels)

        # Epoch metrics
        avg_loss = epoch_loss / n_samples
        train_accuracy = correct / n_samples
        losses.append(avg_loss)
        accuracies.append(train_accuracy)

        if verbose:
            print(f"  Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}")

    # Test evaluation
    test_predictions = network.forward(X_test)
    test_pred_labels = np.argmax(test_predictions, axis=1)
    test_accuracy = np.mean(test_pred_labels == y_test)
    test_loss = np.mean((test_predictions - one_hot_encode(y_test, network.layer_dims[-1])) ** 2)

    elapsed = time.time() - start_time

    results = {
        "epochs": epochs,
        "batch_size": batch_size,
        "final_train_loss": losses[-1],
        "final_train_accuracy": accuracies[-1],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_time_seconds": elapsed,
        "loss_history": losses,
        "accuracy_history": accuracies,
        "network_stats": network.get_stats(),
    }

    if verbose:
        print(f"\n  Final Results:")
        print(f"    Train Accuracy: {results['final_train_accuracy']:.4f}")
        print(f"    Test Accuracy:  {results['test_accuracy']:.4f}")
        print(f"    Test Loss:      {test_loss:.4f}")
        print(f"    Training Time:  {elapsed:.1f}s")

    return results


# ============================================================
# Demo
# ============================================================

def main():
    """Run MNIST training demo with hybrid classifier."""
    print("=" * 60)
    print("  MNIST Hybrid Hebbian + Logistic Regression")
    print("=" * 60)
    print()

    from hebbian_network import HebbianNetwork, HybridHebbianClassifier

    # Load data
    X_train, y_train, X_test, y_test = load_mnist()

    # Use subset for quick demo
    n_train = 10000
    n_test = 1000
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_test]
    y_test = y_test[:n_test]

    print(f"\nUsing {n_train} train, {n_test} test samples")

    # Create hybrid classifier with larger hidden layer
    network = HebbianNetwork(
        layer_dims=[784, 256, 10],
        learning_rate=0.001,
        supervised_last=False,  # Hebbian layers are unsupervised
    )
    hybrid = HybridHebbianClassifier(network, learning_rate=0.1)

    print(f"Network: {network.layer_dims}")
    print()

    # Train with more epochs
    print("Training hybrid classifier...")
    losses = hybrid.train(
        X_train, y_train,
        epochs=200,
        verbose=True,
    )

    # Evaluate
    print("\nEvaluating...")
    results = hybrid.evaluate(X_test, y_test)

    print(f"\n  Final Results:")
    print(f"    Test Accuracy:  {results['accuracy']:.4f}")
    print(f"    Test Loss:      {results['loss']:.4f}")
    print(f"    Classifier Loss History: {losses[0]:.4f} -> {losses[-1]:.4f}")

    print()
    print("=" * 60)
    print("  MNIST Hybrid Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
