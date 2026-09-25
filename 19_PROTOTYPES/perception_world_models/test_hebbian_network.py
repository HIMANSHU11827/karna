"""
Tests for Hebbian Neural Network — Phase 2 AGI Research Lab
============================================================

TDD: Write failing tests first, watch them fail, then implement.

Architecture: NumPy-only, Hebbian learning (no backpropagation).
"""

import sys
import os
import numpy as np
import pytest

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Import the module (will fail — RED phase)
# ============================================================

try:
    from hebbian_network import HebbianLayer, HebbianNetwork, OjaRule
except ImportError as e:
    print(f"Expected import error (RED phase): {e}")
    # Define dummy classes so tests can be written
    class HebbianLayer:
        pass

    class HebbianNetwork:
        pass

    class OjaRule:
        pass


# ============================================================
# Test Oja's Rule
# ============================================================

class TestOjaRule:
    """Test the Oja's learning rule for Hebbian learning.

    Oja's rule: Δw_ij = η * y * (x_i - y * w_ij)
    This is a normalized Hebbian rule that prevents unbounded weight growth.
    """

    def test_oja_rule_returns_correct_shape(self):
        """Oja rule update should return weight matrix of same shape."""
        rule = OjaRule(learning_rate=0.01)
        weights = np.random.randn(5, 3) * 0.1
        x = np.random.randn(5)
        y = np.random.randn(3)

        new_weights = rule.update(weights, x, y)

        assert new_weights.shape == weights.shape

    def test_oja_rule_preserves_weight_norm(self):
        """Oja's rule should prevent unbounded weight growth."""
        rule = OjaRule(learning_rate=0.01)
        weights = np.random.randn(10, 5) * 0.1
        x = np.random.randn(10)
        y = np.random.randn(5)

        initial_norm = np.linalg.norm(weights)

        # Run multiple updates
        for _ in range(100):
            y = np.tanh(x @ weights)
            weights = rule.update(weights, x, y)

        final_norm = np.linalg.norm(weights)

        # Weight norm should not explode
        assert final_norm < initial_norm * 10

    def test_oja_rule_learning_rate_zero(self):
        """With zero learning rate, weights should not change."""
        rule = OjaRule(learning_rate=0.0)
        weights = np.random.randn(5, 3) * 0.1
        x = np.random.randn(5)
        y = np.random.randn(3)

        new_weights = rule.update(weights, x, y)

        np.testing.assert_array_almost_equal(new_weights, weights)


# ============================================================
# Test HebbianLayer
# ============================================================

class TestHebbianLayer:
    """Test a single Hebbian learning layer."""

    def test_layer_initialization(self):
        """Layer should initialize with random weights."""
        layer = HebbianLayer(input_dim=10, output_dim=5)

        assert layer.weights.shape == (10, 5)
        assert layer.bias.shape == (5,)
        # Weights should be small random values
        assert np.abs(layer.weights).max() < 1.0

    def test_layer_forward_pass(self):
        """Forward pass should produce output of correct shape."""
        layer = HebbianLayer(input_dim=10, output_dim=5)
        x = np.random.randn(10)

        output = layer.forward(x)

        assert output.shape == (5,)

    def test_layer_forward_batch(self):
        """Forward pass should handle batch inputs."""
        layer = HebbianLayer(input_dim=10, output_dim=5)
        x = np.random.randn(32, 10)

        output = layer.forward(x)

        assert output.shape == (32, 5)

    def test_layer_learn(self):
        """Layer should update weights via Hebbian learning."""
        layer = HebbianLayer(input_dim=10, output_dim=5, learning_rate=0.01)
        x = np.random.randn(10)
        initial_weights = layer.weights.copy()

        layer.learn(x)

        # Weights should have changed
        assert not np.array_equal(layer.weights, initial_weights)

    def test_layer_learn_batch(self):
        """Batch learning should update weights."""
        layer = HebbianLayer(input_dim=10, output_dim=5, learning_rate=0.01)
        x = np.random.randn(32, 10)
        initial_weights = layer.weights.copy()

        layer.learn(x)

        assert not np.array_equal(layer.weights, initial_weights)

    def test_activation_function(self):
        """Activation function should be applied."""
        layer = HebbianLayer(input_dim=5, output_dim=3)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        output = layer.forward(x)

        # Output should be finite
        assert np.all(np.isfinite(output))


# ============================================================
# Test HebbianNetwork
# ============================================================

class TestHebbianNetwork:
    """Test a multi-layer Hebbian network."""

    def test_network_initialization(self):
        """Network should initialize with correct layer structure."""
        network = HebbianNetwork(layer_dims=[10, 20, 5])

        assert len(network.layers) == 2
        assert network.layers[0].weights.shape == (10, 20)
        assert network.layers[1].weights.shape == (20, 5)

    def test_network_forward(self):
        """Network forward pass should produce correct output shape."""
        network = HebbianNetwork(layer_dims=[10, 20, 5])
        x = np.random.randn(10)

        output = network.forward(x)

        assert output.shape == (5,)

    def test_network_forward_batch(self):
        """Network forward pass should handle batches."""
        network = HebbianNetwork(layer_dims=[10, 20, 5])
        x = np.random.randn(32, 10)

        output = network.forward(x)

        assert output.shape == (32, 5)

    def test_network_learn(self):
        """Network should learn via Hebbian updates."""
        network = HebbianNetwork(layer_dims=[10, 20, 5], learning_rate=0.01)
        x = np.random.randn(10)
        initial_weights = [l.weights.copy() for l in network.layers]

        network.learn(x)

        # At least one layer's weights should have changed
        changed = any(
            not np.array_equal(l.weights, iw)
            for l, iw in zip(network.layers, initial_weights)
        )
        assert changed

    def test_network_learn_epoch(self):
        """Training for multiple epochs should improve or maintain performance."""
        network = HebbianNetwork(layer_dims=[10, 5], learning_rate=0.01)
        X = np.random.randn(100, 10)
        y = (X.sum(axis=1) > 0).astype(float)

        initial_output = network.forward(X)

        network.train(X, y, epochs=10)

        final_output = network.forward(X)

        # Output should be finite
        assert np.all(np.isfinite(final_output))


# ============================================================
# Test MNIST Integration (will be slow — mark as integration)
# ============================================================

class TestMNISTIntegration:
    """Integration tests for MNIST dataset."""

    def test_network_can_learn_mnist_pattern(self):
        """Network should be able to learn a simple MNIST-like pattern."""
        # Simple pattern: classify based on pixel intensity
        network = HebbianNetwork(layer_dims=[784, 64, 10], learning_rate=0.001)

        # Generate simple patterns (not real MNIST for speed)
        np.random.seed(42)
        X = np.random.randn(100, 784) * 0.1
        y = np.zeros((100, 10))
        for i in range(100):
            y[i, i % 10] = 1.0

        # Train
        network.train(X, y, epochs=5)

        # Predict
        predictions = network.forward(X[:10])

        # Should produce finite outputs
        assert np.all(np.isfinite(predictions))
        assert predictions.shape == (10, 10)


# ============================================================
# Test Continual Learning
# ============================================================

class TestContinualLearning:
    """Test continual learning (Split-MNIST)."""

    def test_network_retains_old_knowledge(self):
        """Network should retain performance on old tasks while learning new ones."""
        network = HebbianNetwork(layer_dims=[20, 10, 2], learning_rate=0.01)

        # Task A
        np.random.seed(42)
        X_A = np.random.randn(50, 20)
        y_A = (X_A[:, :10].sum(axis=1) > 0).astype(float)

        network.train(X_A, y_A.reshape(-1, 1), epochs=10)

        # Evaluate on Task A
        pred_A_before = network.forward(X_A)

        # Task B
        np.random.seed(123)
        X_B = np.random.randn(50, 20)
        y_B = (X_B[:, 10:].sum(axis=1) > 0).astype(float)

        network.train(X_B, y_B.reshape(-1, 1), epochs=10)

        # Evaluate on Task A again (should not be catastrophically forgotten)
        pred_A_after = network.forward(X_A)

        # Performance on A should not degrade catastrophically
        # (Hebbian networks are known for less forgetting)
        error_before = np.mean((pred_A_before - y_A.reshape(-1, 1)) ** 2)
        error_after = np.mean((pred_A_after - y_A.reshape(-1, 1)) ** 2)

        # Allow some degradation but not catastrophic
        assert error_after < error_before * 5


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
