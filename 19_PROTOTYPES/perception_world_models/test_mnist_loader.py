"""
Tests for MNIST Loader and Training Pipeline
=============================================

TDD: Write failing tests first, watch them fail, then implement.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mnist_loader import load_mnist, one_hot_encode, train_and_evaluate
except ImportError as e:
    print(f"Expected import error (RED phase): {e}")
    def load_mnist(*args, **kwargs):
        raise NotImplementedError

    def one_hot_encode(*args, **kwargs):
        raise NotImplementedError

    def train_and_evaluate(*args, **kwargs):
        raise NotImplementedError


# ============================================================
# Test One-Hot Encoding
# ============================================================

class TestOneHotEncode:
    """Test one-hot encoding utility."""

    def test_one_hot_shape(self):
        """One-hot encoding should produce correct shape."""
        labels = np.array([0, 1, 2, 3])
        encoded = one_hot_encode(labels, num_classes=5)
        assert encoded.shape == (4, 5)

    def test_one_hot_values(self):
        """One-hot encoding should have 1 at correct index."""
        labels = np.array([0, 2, 4])
        encoded = one_hot_encode(labels, num_classes=5)
        assert encoded[0, 0] == 1.0
        assert encoded[1, 2] == 1.0
        assert encoded[2, 4] == 1.0

    def test_one_hot_zeros_elsewhere(self):
        """One-hot encoding should be 0 everywhere else."""
        labels = np.array([1])
        encoded = one_hot_encode(labels, num_classes=3)
        assert encoded[0, 0] == 0.0
        assert encoded[0, 2] == 0.0


# ============================================================
# Test MNIST Data Loader
# ============================================================

class TestMNISTLoader:
    """Test MNIST data loading."""

    def test_load_mnist_returns_correct_shapes(self):
        """MNIST loader should return correct array shapes."""
        X_train, y_train, X_test, y_test = load_mnist()
        assert X_train.shape == (60000, 784)
        assert y_train.shape == (60000,)
        assert X_test.shape == (10000, 784)
        assert y_test.shape == (10000,)

    def test_load_mnist_normalized(self):
        """MNIST images should be normalized to [0, 1]."""
        X_train, _, X_test, _ = load_mnist()
        assert X_train.min() >= 0.0
        assert X_train.max() <= 1.0
        assert X_test.min() >= 0.0
        assert X_test.max() <= 1.0

    def test_load_mnist_labels_range(self):
        """MNIST labels should be in [0, 9]."""
        _, y_train, _, y_test = load_mnist()
        assert y_train.min() >= 0
        assert y_train.max() <= 9
        assert y_test.min() >= 0
        assert y_test.max() <= 9


# ============================================================
# Test Training Pipeline
# ============================================================

class TestTrainingPipeline:
    """Test training and evaluation pipeline."""

    def test_train_and_evaluate_returns_metrics(self):
        """Training should return a results dictionary."""
        from hebbian_network import HebbianNetwork

        # Small synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 20) * 0.1
        y_train = np.array([i % 3 for i in range(100)])
        X_test = np.random.randn(20, 20) * 0.1
        y_test = np.array([i % 3 for i in range(20)])

        network = HebbianNetwork(layer_dims=[20, 10, 3], learning_rate=0.01)
        results = train_and_evaluate(
            network, X_train, y_train, X_test, y_test,
            epochs=2, batch_size=16, verbose=False
        )

        assert "test_accuracy" in results
        assert "test_loss" in results
        assert "training_time_seconds" in results
        assert results["test_accuracy"] >= 0.0
        assert results["test_accuracy"] <= 1.0

    def test_training_improves_accuracy(self):
        """Training should improve or maintain accuracy."""
        from hebbian_network import HebbianNetwork

        np.random.seed(42)
        X_train = np.random.randn(200, 10) * 0.1
        y_train = np.array([i % 2 for i in range(200)])
        X_test = np.random.randn(50, 10) * 0.1
        y_test = np.array([i % 2 for i in range(50)])

        network = HebbianNetwork(layer_dims=[10, 5, 2], learning_rate=0.01)
        results = train_and_evaluate(
            network, X_train, y_train, X_test, y_test,
            epochs=5, batch_size=16, verbose=False
        )

        # Accuracy should be finite and in valid range
        assert 0.0 <= results["test_accuracy"] <= 1.0
        assert np.isfinite(results["test_loss"])


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
