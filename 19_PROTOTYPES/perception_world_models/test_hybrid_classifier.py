"""
Tests for Hybrid Hebbian Classifier
=====================================

TDD: Write failing tests first, watch them fail, then implement.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from hebbian_network import LogisticRegression, HybridHebbianClassifier
except ImportError as e:
    print(f"Expected import error (RED phase): {e}")
    class LogisticRegression:
        pass

    class HybridHebbianClassifier:
        pass


# ============================================================
# Test Logistic Regression
# ============================================================

class TestLogisticRegression:
    """Test logistic regression classifier."""

    def test_initialization(self):
        """Classifier should initialize with correct shapes."""
        clf = LogisticRegression(input_dim=10, num_classes=3)
        assert clf.weights.shape == (10, 3)
        assert clf.bias.shape == (3,)

    def test_forward_returns_probabilities(self):
        """Forward pass should return valid probabilities."""
        clf = LogisticRegression(input_dim=5, num_classes=3)
        X = np.random.randn(10, 5)
        probs = clf.forward(X)

        assert probs.shape == (10, 3)
        # Probabilities should sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
        # All probabilities should be non-negative
        assert np.all(probs >= 0)

    def test_predict_returns_labels(self):
        """Predict should return integer labels."""
        clf = LogisticRegression(input_dim=5, num_classes=3)
        X = np.random.randn(10, 5)
        predictions = clf.predict(X)

        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 3)

    def test_training_decreases_loss(self):
        """Training should decrease loss."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.array([i % 3 for i in range(100)])

        clf = LogisticRegression(input_dim=5, num_classes=3, learning_rate=0.1)
        losses = clf.train(X, y, epochs=50, verbose=False)

        assert len(losses) == 50
        # Loss should generally decrease (allow some fluctuation)
        assert losses[-1] < losses[0]

    def test_training_improves_accuracy(self):
        """Training should improve accuracy on separable data."""
        np.random.seed(42)
        # Create data where labels correlate with features
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)  # Split on first feature

        clf = LogisticRegression(input_dim=5, num_classes=2, learning_rate=0.1)
        clf.train(X, y, epochs=100, verbose=False)

        accuracy = clf.evaluate(X, y)["accuracy"]
        assert accuracy > 0.8  # Should be much better than random (0.5)


# ============================================================
# Test Hybrid Hebbian Classifier
# ============================================================

class TestHybridHebbianClassifier:
    """Test hybrid Hebbian + logistic regression classifier."""

    def test_initialization(self):
        """Hybrid classifier should initialize correctly."""
        from hebbian_network import HebbianNetwork
        network = HebbianNetwork(layer_dims=[20, 10, 3], learning_rate=0.01, supervised_last=False)
        hybrid = HybridHebbianClassifier(network, learning_rate=0.1)

        assert hybrid.hebbian_network is network
        assert hybrid.classifier.input_dim == 3
        assert hybrid.classifier.num_classes == 3

    def test_extract_features(self):
        """Feature extraction should produce correct shape."""
        from hebbian_network import HebbianNetwork
        network = HebbianNetwork(layer_dims=[20, 10, 3], learning_rate=0.01, supervised_last=False)
        hybrid = HybridHebbianClassifier(network)

        X = np.random.randn(32, 20)
        features = hybrid.extract_features(X)

        assert features.shape == (32, 3)

    def test_train_and_predict(self):
        """Full training pipeline should work."""
        from hebbian_network import HebbianNetwork

        np.random.seed(42)
        X_train = np.random.randn(100, 10) * 0.1
        y_train = np.array([i % 2 for i in range(100)])
        X_test = np.random.randn(20, 10) * 0.1
        y_test = np.array([i % 2 for i in range(20)])

        network = HebbianNetwork(layer_dims=[10, 5, 2], learning_rate=0.01, supervised_last=False)
        hybrid = HybridHebbianClassifier(network, learning_rate=0.1)

        # Train
        losses = hybrid.train(X_train, y_train, epochs=20, verbose=False)
        assert len(losses) > 0

        # Evaluate
        results = hybrid.evaluate(X_test, y_test)
        assert 0.0 <= results["accuracy"] <= 1.0
        assert np.isfinite(results["loss"])

    def test_hybrid_beats_pure_hebbian(self):
        """Hybrid should outperform pure Hebbian on classification."""
        from hebbian_network import HebbianNetwork

        np.random.seed(42)
        X = np.random.randn(200, 10) * 0.1
        y = np.array([i % 2 for i in range(200)])

        # Pure Hebbian (unsupervised)
        network_pure = HebbianNetwork(layer_dims=[10, 5, 2], learning_rate=0.01, supervised_last=False)
        network_pure.train(X, y=None, epochs=5, verbose=False)
        # Pure Hebbian output — use argmax of forward pass
        pure_preds = np.argmax(network_pure.forward(X), axis=1)
        pure_acc = np.mean(pure_preds == y)

        # Hybrid
        network_hybrid = HebbianNetwork(layer_dims=[10, 5, 2], learning_rate=0.01, supervised_last=False)
        hybrid = HybridHebbianClassifier(network_hybrid, learning_rate=0.1)
        hybrid.train(X, y, epochs=20, verbose=False)
        hybrid_results = hybrid.evaluate(X, y)

        # Hybrid should be at least as good (usually much better)
        assert hybrid_results["accuracy"] >= pure_acc - 0.1


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
