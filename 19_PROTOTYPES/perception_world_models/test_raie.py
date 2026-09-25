"""
Tests for RAIE (Real-time AI Everything) — Unified Architecture
===============================================================

Merges PEN + HAPN into one system with:
- KWTA activation
- Iterative Attractor Convergence (IAC)
- XOR binding for semantic memory
- Complementary learning (fast episodic + slow semantic)
- SRBI weight initialization
- SHD learning rule

TDD: RED phase — all tests fail until implementation.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raie_network import (
    srbi_init,
    shd_update,
    kwta,
    iterative_attractor,
    xor_bind,
    xor_unbind,
    ComplementaryMemory,
    RAIENetwork,
)


# ============================================================
# Test SRBI (Sparse Random Binary Init)
# ============================================================

class TestSRBI:
    """Test Sparse Random Binary Initialization."""

    def test_srbi_shape(self):
        """SRBI should produce correct shape."""
        weights = srbi_init(100, 50, sparsity=0.1)
        assert weights.shape == (100, 50)

    def test_srbi_sparsity(self):
        """SRBI should have approximately the requested sparsity."""
        weights = srbi_init(1000, 500, sparsity=0.1)
        actual_sparsity = np.sum(weights != 0) / weights.size
        assert abs(actual_sparsity - 0.1) < 0.05

    def test_srbi_binary_values(self):
        """SRBI should produce binary values (+1/-1)."""
        weights = srbi_init(100, 50, sparsity=0.1)
        non_zero = weights[weights != 0]
        assert np.all(np.abs(non_zero) == 1.0)

    def test_srbi_seed_reproducible(self):
        """SRBI with same seed should produce same weights."""
        w1 = srbi_init(50, 50, sparsity=0.1, seed=42)
        w2 = srbi_init(50, 50, sparsity=0.1, seed=42)
        np.testing.assert_array_equal(w1, w2)


# ============================================================
# Test SHD (Sparse Hebbian with Decay)
# ============================================================

class TestSHD:
    """Test Sparse Hebbian with Decay learning rule."""

    def test_shd_update_shape(self):
        """SHD update should preserve weight matrix shape."""
        weights = np.random.randn(10, 5) * 0.1
        x = np.random.randn(10)
        y = np.random.randn(5)
        new_weights = shd_update(weights, x, y, learning_rate=0.01, decay=0.001)
        assert new_weights.shape == weights.shape

    def test_shd_decay_reduces_weights(self):
        """SHD should decay weights toward zero."""
        weights = np.ones((10, 5)) * 0.5
        x = np.zeros(10)
        y = np.zeros(5)
        new_weights = shd_update(weights, x, y, learning_rate=0.0, decay=0.1)
        assert np.all(np.abs(new_weights) < np.abs(weights))

    def test_shd_hebbian_strengthens(self):
        """SHD should strengthen correlated weights."""
        weights = np.zeros((10, 5))
        x = np.ones(10)
        y = np.ones(5)
        new_weights = shd_update(weights, x, y, learning_rate=0.1, decay=0.0)
        assert np.all(new_weights > 0)


# ============================================================
# Test KWTA (k-Winners-Take-All)
# ============================================================

class TestKWTA:
    """Test k-Winners-Take-All activation."""

    def test_kwta_sparsity(self):
        """KWTA should activate exactly k neurons."""
        from raie_network import kwta
        x = np.random.randn(100)
        k = 10
        result = kwta(x, k)
        assert np.sum(result > 0) == k

    def test_kwta_preserves_top_k(self):
        """KWTA should keep the top k values."""
        from raie_network import kwta
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        k = 2
        result = kwta(x, k)
        # Top 2 values (4.0, 5.0) should be preserved
        assert result[3] == 4.0
        assert result[4] == 5.0
        # Others should be zero
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_kwta_batch(self):
        """KWTA should handle batch inputs."""
        from raie_network import kwta
        X = np.random.randn(32, 100)
        k = 10
        result = kwta(X, k)
        assert result.shape == (32, 100)
        # Each row should have exactly k non-zero
        for i in range(32):
            assert np.sum(result[i] > 0) == k


# ============================================================
# Test Iterative Attractor Convergence (IAC)
# ============================================================

class TestIAC:
    """Test Iterative Attractor Convergence for memory retrieval."""

    def test_iac_converges(self):
        """IAC should converge to a stable state."""
        from raie_network import iterative_attractor
        # Simple attractor: memory matrix with one stored pattern
        stored = np.array([1.0, -1.0, 1.0, -1.0])
        memory = np.outer(stored, stored)  # Hopfield-like
        # Start from partial cue
        cue = np.array([0.5, 0.0, 0.5, 0.0])
        result = iterative_attractor(cue, memory, n_steps=10)
        # Should converge toward stored pattern
        assert np.allclose(np.sign(result), stored, atol=0.1)

    def test_iac_output_shape(self):
        """IAC should preserve input shape."""
        from raie_network import iterative_attractor
        memory = np.eye(10)
        cue = np.random.randn(10)
        result = iterative_attractor(cue, memory, n_steps=5)
        assert result.shape == cue.shape

    def test_iac_stable_fixed_point(self):
        """IAC should reach a stable fixed point."""
        from raie_network import iterative_attractor
        stored = np.array([1.0, -1.0, 1.0, -1.0])
        memory = np.outer(stored, stored)
        cue = stored.copy()  # Start at stored pattern
        result1 = iterative_attractor(cue, memory, n_steps=5)
        result2 = iterative_attractor(result1, memory, n_steps=5)
        # Should be stable (no change between iterations)
        np.testing.assert_allclose(result1, result2, atol=0.01)


# ============================================================
# Test XOR Binding for Semantic Memory
# ============================================================

class TestXORBinding:
    """Test XOR binding for semantic memory operations."""

    def test_xor_binding_shape(self):
        """XOR binding should produce output of correct shape."""
        from raie_network import xor_bind
        a = np.array([1.0, -1.0, 1.0])
        b = np.array([-1.0, 1.0, -1.0])
        result = xor_bind(a, b)
        assert result.shape == a.shape

    def test_xor_binding_values(self):
        """XOR binding should compute element-wise XOR."""
        from raie_network import xor_bind
        a = np.array([1.0, 1.0, -1.0, -1.0])
        b = np.array([1.0, -1.0, 1.0, -1.0])
        result = xor_bind(a, b)
        # XOR: same → -1, different → +1
        expected = np.array([-1.0, 1.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_xor_binding_inverse(self):
        """XOR binding should be invertible."""
        from raie_network import xor_bind, xor_unbind
        a = np.array([1.0, -1.0, 1.0, -1.0])
        b = np.array([-1.0, 1.0, -1.0, 1.0])
        bound = xor_bind(a, b)
        recovered = xor_unbind(bound, b)
        np.testing.assert_array_almost_equal(recovered, a)

    def test_xor_binding_batch(self):
        """XOR binding should handle batches."""
        from raie_network import xor_bind
        A = np.random.randn(32, 10)
        B = np.random.randn(32, 10)
        result = xor_bind(A, B)
        assert result.shape == (32, 10)


# ============================================================
# Test Complementary Learning System
# ============================================================

class TestComplementaryLearning:
    """Test complementary learning (fast episodic + slow semantic)."""

    def test_dual_memory_init(self):
        """System should have two memory stores."""
        from raie_network import ComplementaryMemory
        mem = ComplementaryMemory(input_dim=100, episodic_dim=200, semantic_dim=50)
        assert mem.episodic_weights.shape == (100, 200)
        assert mem.semantic_weights.shape == (100, 50)

    def test_fast_learning(self):
        """Fast learning should quickly store a pattern."""
        from raie_network import ComplementaryMemory
        mem = ComplementaryMemory(input_dim=100, episodic_dim=200, semantic_dim=50)
        x = np.random.randn(100)
        # Fast learning on episodic system
        mem.fast_learn(x)
        # Should be able to recall
        retrieved = mem.fast_recall(x)
        assert retrieved.shape == x.shape

    def test_slow_learning(self):
        """Slow learning should consolidate to semantic memory."""
        from raie_network import ComplementaryMemory
        mem = ComplementaryMemory(input_dim=100, episodic_dim=200, semantic_dim=50)
        x = np.random.randn(100)
        # Slow learning (multiple exposures)
        for _ in range(10):
            mem.slow_learn(x)
        # Should be able to recall from semantic system
        retrieved = mem.slow_recall(x)
        assert retrieved.shape == x.shape


# ============================================================
# Test RAIE Network
# ============================================================

class TestRAIE:
    """Test the unified RAIE network."""

    def test_raie_init(self):
        """RAIE should initialize with correct dimensions."""
        from raie_network import RAIENetwork
        net = RAIENetwork(
            input_dim=784,
            encoder_dims=[256, 128],
            memory_dim=64,
            output_dim=10,
        )
        assert net.input_dim == 784
        assert net.output_dim == 10

    def test_raie_forward(self):
        """Forward pass should produce correct output shape."""
        from raie_network import RAIENetwork
        net = RAIENetwork(
            input_dim=20,
            encoder_dims=[10],
            memory_dim=5,
            output_dim=3,
        )
        x = np.random.randn(20)
        output = net.forward(x)
        assert output.shape == (3,)

    def test_raie_forward_batch(self):
        """Batch forward pass should work."""
        from raie_network import RAIENetwork
        net = RAIENetwork(
            input_dim=20,
            encoder_dims=[10],
            memory_dim=5,
            output_dim=3,
        )
        X = np.random.randn(32, 20)
        output = net.forward(X)
        assert output.shape == (32, 3)

    def test_raie_learn(self):
        """Learning should update weights."""
        from raie_network import RAIENetwork
        net = RAIENetwork(
            input_dim=20,
            encoder_dims=[10],
            memory_dim=5,
            output_dim=3,
        )
        x = np.random.randn(20)
        initial_weights = [w.copy() for w in net.encoder_weights]
        net.learn(x)
        # At least some weights should have changed
        changed = any(
            not np.array_equal(w, iw)
            for w, iw in zip(net.encoder_weights, initial_weights)
        )
        assert changed

    def test_raie_mnist_accuracy(self):
        """RAIE should achieve >95% on MNIST (integration test)."""
        from raie_network import RAIENetwork
        from mnist_loader import load_mnist

        X_train, y_train, X_test, y_test = load_mnist()
        X_train = X_train[:5000]
        y_train = y_train[:5000]
        X_test = X_test[:1000]
        y_test = y_test[:1000]

        net = RAIENetwork(
            input_dim=784,
            encoder_dims=[512, 256],
            memory_dim=128,
            output_dim=10,
        )

        # Train
        net.train(X_train, y_train, epochs=20, verbose=False)

        # Evaluate
        predictions = net.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        assert accuracy > 0.95


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
