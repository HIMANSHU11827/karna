"""
Tests for RT-ALM — Real-Time Adaptive Language Model
======================================================

Components:
1. SDR encoder (character n-gram hashing)
2. k-WTA activation
3. Attractor network (Hopfield-style)
4. Episodic memory (key-value with content hashing)
5. Online learning rule (eligibility traces + error-driven)
6. Response retriever (brute-force similarity)

All NumPy. No backprop.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Test SDR Encoder
# ============================================================

class TestSDREncoder:
    """Test character n-gram hashing to sparse distributed representations."""

    def test_sdr_shape(self):
        """SDR encoder should produce correct shape."""
        from rtalm import SDREncoder
        encoder = SDREncoder(dim=1000, ngram_size=3)
        sdr = encoder.encode("hello")
        assert sdr.shape == (1000,)

    def test_sdr_binary(self):
        """SDR should be binary (0/1 values)."""
        from rtalm import SDREncoder
        encoder = SDREncoder(dim=1000, ngram_size=3)
        sdr = encoder.encode("hello")
        assert np.all(np.isin(sdr, [0, 1]))

    def test_sdr_similarity(self):
        """Similar strings should have higher SDR overlap."""
        from rtalm import SDREncoder
        encoder = SDREncoder(dim=1000, ngram_size=3)
        sdr1 = encoder.encode("hello")
        sdr2 = encoder.encode("hello")
        sdr3 = encoder.encode("world")
        overlap_same = np.sum(sdr1 & sdr2)
        overlap_diff = np.sum(sdr1 & sdr3)
        assert overlap_same > overlap_diff

    def test_sdr_batch(self):
        """SDR encoder should handle batch encoding."""
        from rtalm import SDREncoder
        encoder = SDREncoder(dim=1000, ngram_size=3)
        sdrs = encoder.encode_batch(["hello", "world", "test"])
        assert len(sdrs) == 3
        for sdr in sdrs:
            assert sdr.shape == (1000,)


# ============================================================
# Test k-WTA Activation
# ============================================================

class TestKWTA:
    """Test k-Winners-Take-All activation."""

    def test_kwta_sparsity(self):
        """k-WTA should activate exactly k neurons."""
        from rtalm import kwta
        x = np.random.randn(100)
        result = kwta(x, k=10)
        assert np.sum(result > 0) == 10

    def test_kwta_preserves_top_k(self):
        """k-WTA should preserve top-k values."""
        from rtalm import kwta
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = kwta(x, k=2)
        assert result[3] == 4.0
        assert result[4] == 5.0
        assert result[0] == 0.0

    def test_kwta_batch(self):
        """k-WTA should handle batch inputs."""
        from rtalm import kwta
        X = np.random.randn(32, 100)
        result = kwta(X, k=10)
        assert result.shape == (32, 100)
        for i in range(32):
            assert np.sum(result[i] > 0) == 10


# ============================================================
# Test Attractor Network
# ============================================================

class TestAttractorNetwork:
    """Test Hopfield-style attractor convergence."""

    def test_attractor_converges(self):
        """Attractor should converge to stored pattern."""
        from rtalm import AttractorMemory
        mem = AttractorMemory(dim=100)
        pattern = np.random.choice([0, 1], size=100).astype(np.int8)
        mem.store(pattern)
        # Query with noisy version
        noisy = pattern.copy()
        noisy[:10] = 1 - noisy[:10]
        result = mem.retrieve(noisy, steps=5)
        # Should converge close to original
        overlap = np.sum(result & pattern)
        assert overlap > 80

    def test_attractor_stores_multiple(self):
        """Attractor should store and retrieve multiple patterns."""
        from rtalm import AttractorMemory
        mem = AttractorMemory(dim=100)
        patterns = [np.random.choice([0, 1], size=100).astype(np.int8) for _ in range(5)]
        for p in patterns:
            mem.store(p)
        # Retrieve each
        for p in patterns:
            result = mem.retrieve(p, steps=5)
            overlap = np.sum(result & p)
            assert overlap > 80

    def test_attractor_capacity(self):
        """Attractor should handle capacity limit."""
        from rtalm import AttractorMemory
        mem = AttractorMemory(dim=100, capacity=10)
        for i in range(15):
            pattern = np.random.choice([0, 1], size=100).astype(np.int8)
            mem.store(pattern)
        assert len(mem.patterns) == 10


# ============================================================
# Test Episodic Memory
# ============================================================

class TestEpisodicMemory:
    """Test key-value episodic memory with content hashing."""

    def test_store_and_retrieve(self):
        """Should store and retrieve episodes."""
        from rtalm import EpisodicMemory
        mem = EpisodicMemory(dim=100)
        key = np.random.choice([0, 1], size=100).astype(np.int8)
        value = np.random.choice([0, 1], size=100).astype(np.int8)
        mem.store(key, value)
        retrieved = mem.retrieve(key)
        assert retrieved is not None
        assert np.array_equal(retrieved, value)

    def test_content_hashing(self):
        """Content hashing should deduplicate identical content."""
        from rtalm import EpisodicMemory
        mem = EpisodicMemory(dim=100)
        key = np.random.choice([0, 1], size=100).astype(np.int8)
        value = np.random.choice([0, 1], size=100).astype(np.int8)
        mem.store(key, value)
        # Store same value with different key
        key2 = np.random.choice([0, 1], size=100).astype(np.int8)
        mem.store(key2, value)
        # Should deduplicate
        assert len(mem.values) == 1

    def test_retrieve_similar(self):
        """Should retrieve similar content with noisy query."""
        from rtalm import EpisodicMemory
        mem = EpisodicMemory(dim=100)
        key = np.random.choice([0, 1], size=100).astype(np.int8)
        value = np.random.choice([0, 1], size=100).astype(np.int8)
        mem.store(key, value)
        # Query with noisy key
        noisy = key.copy()
        noisy[:5] = 1 - noisy[:5]
        retrieved = mem.retrieve(noisy)
        assert retrieved is not None
        # Should be similar to stored value
        overlap = np.sum(retrieved & value)
        assert overlap > 50


# ============================================================
# Test Online Learning Rule
# ============================================================

class TestOnlineLearningRule:
    """Test eligibility traces + error-driven updates."""

    def test_update_shape(self):
        """Weight update should preserve shape."""
        from rtalm import OnlineLearningRule
        rule = OnlineLearningRule(dim=100, lr=0.01)
        pre = np.random.randn(100)
        post = np.random.randn(100)
        target = np.random.randn(100)
        neuromodulator = 1.0
        update = rule.compute_update(pre, post, target, neuromodulator)
        assert update.shape == (100, 100)

    def test_error_driven(self):
        """Update should reduce error over time."""
        from rtalm import OnlineLearningRule
        rule = OnlineLearningRule(dim=10, lr=0.1)
        pre = np.random.randn(10)
        target = np.random.randn(10)
        post = np.zeros(10)  # Zero initial activation
        initial_error = np.sum((target - post) ** 2)
        for _ in range(20):
            update = rule.compute_update(pre, post, target, 1.0)
            post += update @ pre
        final_error = np.sum((target - post) ** 2)
        assert final_error < initial_error

    def test_neuromodulator_gates(self):
        """Zero neuromodulator should produce zero update."""
        from rtalm import OnlineLearningRule
        rule = OnlineLearningRule(dim=10, lr=0.1)
        pre = np.random.randn(10)
        post = np.random.randn(10)
        target = np.random.randn(10)
        update = rule.compute_update(pre, post, target, 0.0)
        assert np.allclose(update, 0)


# ============================================================
# Test Response Retriever
# ============================================================

class TestResponseRetriever:
    """Test brute-force similarity response retrieval."""

    def test_retrieve_exact(self):
        """Should retrieve exact match."""
        from rtalm import ResponseRetriever
        retriever = ResponseRetriever(dim=100)
        query = np.random.choice([0, 1], size=100).astype(np.int8)
        response = np.random.choice([0, 1], size=100).astype(np.int8)
        retriever.store(query, response)
        retrieved = retriever.retrieve(query)
        assert retrieved is not None
        assert np.array_equal(retrieved, response)

    def test_retrieve_similar(self):
        """Should retrieve similar response with noisy query."""
        from rtalm import ResponseRetriever
        retriever = ResponseRetriever(dim=100)
        query = np.random.choice([0, 1], size=100).astype(np.int8)
        response = np.random.choice([0, 1], size=100).astype(np.int8)
        retriever.store(query, response)
        # Noisy query
        noisy = query.copy()
        noisy[:5] = 1 - noisy[:5]
        retrieved = retriever.retrieve(noisy)
        assert retrieved is not None

    def test_top_k(self):
        """Should return top-k responses."""
        from rtalm import ResponseRetriever
        retriever = ResponseRetriever(dim=100)
        for i in range(10):
            q = np.random.choice([0, 1], size=100).astype(np.int8)
            r = np.random.choice([0, 1], size=100).astype(np.int8)
            retriever.store(q, r)
        query = np.random.choice([0, 1], size=100).astype(np.int8)
        results = retriever.retrieve_top_k(query, k=3)
        assert len(results) == 3


# ============================================================
# Integration: Full RT-ALM
# ============================================================

class TestRTALMIntegration:
    """Integration tests for full RT-ALM system."""

    def test_full_pipeline(self):
        """Full pipeline: encode → attractor → retrieve → respond."""
        from rtalm import RTALM
        model = RTALM(sdr_dim=1000, ngram_size=3)
        # Store a response
        model.store("hello", "hi there!")
        # Retrieve
        response = model.respond("hello")
        assert response == "hi there!"

    def test_noisy_retrieval(self):
        """Should retrieve similar response with noisy input."""
        from rtalm import RTALM
        model = RTALM(sdr_dim=1000, ngram_size=3)
        model.store("hello", "hi there!")
        # Noisy input (typo)
        response = model.respond("helo")
        assert response is not None

    def test_learning_improves(self):
        """Learning should improve response quality."""
        from rtalm import RTALM
        model = RTALM(sdr_dim=1000, ngram_size=3)
        # Store multiple examples
        for _ in range(10):
            model.store("hello", "hi there!")
        response = model.respond("hello")
        assert response == "hi there!"
