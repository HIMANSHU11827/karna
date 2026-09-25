"""
RT-ALM — Real-Time Adaptive Language Model
=============================================

A unified architecture combining:
1. Multimodal encoder (text/image/audio/video → SDR fusion)
2. SDR encoder (character n-gram hashing)
3. k-WTA activation
4. Attractor network (Hopfield-style)
5. Episodic memory (key-value with content hashing)
6. Online learning rule (eligibility traces + error-driven)
7. Response retriever (brute-force similarity)

All NumPy. No backprop.
"""

import hashlib
from typing import Optional, Tuple, List, Dict, Any

import numpy as np


# ============================================================
# SDR Encoder
# ============================================================

class SDREncoder:
    """Character n-gram hashing to sparse distributed representations.

    Encodes text into binary SDRs by hashing character n-grams.
    """

    def __init__(self, dim: int = 10000, ngram_size: int = 3):
        """Initialize SDR encoder.

        Args:
            dim: SDR dimensionality
            ngram_size: Character n-gram size
        """
        self.dim = dim
        self.ngram_size = ngram_size
        # Use MurmurHash3 for fast hashing
        self._seed = 42

    def _hash_ngram(self, ngram: str) -> int:
        """Hash n-gram to bit position using MD5 (platform-independent)."""
        h = hashlib.md5(ngram.encode('utf-8')).digest()
        return int.from_bytes(h, 'little') % self.dim

    def encode(self, text: str) -> np.ndarray:
        """Encode text to SDR.

        Args:
            text: Input string

        Returns:
            Binary SDR vector (dim,)
        """
        sdr = np.zeros(self.dim, dtype=np.int8)

        # Pad text for n-grams at boundaries
        padded = '\0' * (self.ngram_size - 1) + text + '\0' * (self.ngram_size - 1)

        # Hash each n-gram
        for i in range(len(padded) - self.ngram_size + 1):
            ngram = padded[i:i + self.ngram_size]
            bit_pos = self._hash_ngram(ngram)
            sdr[bit_pos] = 1

        return sdr

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode batch of texts to SDRs.

        Args:
            texts: List of input strings

        Returns:
            List of binary SDR vectors
        """
        return [self.encode(text) for text in texts]


# ============================================================
# k-WTA Activation
# ============================================================

def kwta(x: np.ndarray, k: int) -> np.ndarray:
    """k-Winners-Take-All activation.

    Keeps top-k activations, zeros the rest.

    Args:
        x: Input array (n,) or (batch, n)
        k: Number of winners

    Returns:
        Sparse activation array
    """
    if x.ndim == 1:
        result = np.zeros_like(x)
        top_k_indices = np.argsort(x)[-k:]
        result[top_k_indices] = x[top_k_indices]
        return result
    else:
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            top_k_indices = np.argsort(x[i])[-k:]
            result[i, top_k_indices] = x[i, top_k_indices]
        return result


# ============================================================
# Attractor Memory (Hopfield-style)
# ============================================================

class AttractorMemory:
    """Hopfield-like attractor network for pattern completion.

    Stores binary patterns and retrieves them via energy minimization.
    """

    def __init__(self, dim: int, capacity: int = 10000):
        """Initialize attractor memory.

        Args:
            dim: Pattern dimensionality
            capacity: Maximum number of stored patterns
        """
        self.dim = dim
        self.capacity = capacity
        self.patterns: List[np.ndarray] = []
        self.weights = np.zeros((dim, dim), dtype=np.float32)

    def store(self, pattern: np.ndarray) -> None:
        """Store pattern via incremental Hebbian outer product.

        Args:
            pattern: Binary pattern (dim,)
        """
        if len(self.patterns) >= self.capacity:
            # FIFO eviction
            self.patterns.pop(0)

        self.patterns.append(pattern.copy())

        # Rebuild weights from all stored patterns (normalized)
        self.weights.fill(0)
        for p in self.patterns:
            x = p.astype(np.float32)
            self.weights += np.outer(x, x)
        np.fill_diagonal(self.weights, 0)
        if len(self.patterns) > 0:
            self.weights /= len(self.patterns)

    def retrieve(self, query: np.ndarray, steps: int = 5) -> np.ndarray:
        """Retrieve nearest stored pattern via attractor dynamics.

        Args:
            query: Query pattern (dim,)
            steps: Number of iterations

        Returns:
            Retrieved pattern (dim,)
        """
        x = query.astype(np.float32).copy()
        for _ in range(steps):
            h = self.weights @ x
            new_x = (h > 0).astype(np.float32)
            if np.array_equal(new_x, x):
                break
            x = new_x
        return (x > 0).astype(np.int8)

    def retrieve_with_overlap(self, query: np.ndarray, steps: int = 5) -> Tuple[np.ndarray, float]:
        """Retrieve pattern and return overlap with query.

        Args:
            query: Query pattern (dim,)
            steps: Number of iterations

        Returns:
            Tuple of (retrieved pattern, overlap ratio)
        """
        result = self.retrieve(query, steps)
        overlap = np.sum(result & query)
        total = np.sum(query)
        ratio = float(overlap) / float(total) if total > 0 else 0.0
        return result, ratio


# ============================================================
# Episodic Memory
# ============================================================

class EpisodicMemory:
    """Key-value episodic memory with content hashing.

    Stores episodes with deduplication via content hashing.
    """

    def __init__(self, dim: int, capacity: int = 10000):
        """Initialize episodic memory.

        Args:
            dim: SDR dimensionality
            capacity: Maximum number of episodes
        """
        self.dim = dim
        self.capacity = capacity
        self.keys: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.content_hashes: set = set()

    def _content_hash(self, value: np.ndarray) -> str:
        """Compute hash of value for deduplication."""
        return hashlib.md5(value.tobytes()).hexdigest()

    def store(self, key: np.ndarray, value: np.ndarray) -> None:
        """Store episode (key-value pair).

        Args:
            key: Key SDR
            value: Value SDR
        """
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)

        ch = self._content_hash(value)
        if ch in self.content_hashes:
            return

        self.content_hashes.add(ch)
        self.keys.append(key.copy())
        self.values.append(value.copy())

    def retrieve(self, query: np.ndarray) -> Optional[np.ndarray]:
        """Retrieve value by similarity to query.

        Args:
            query: Query SDR

        Returns:
            Retrieved value or None if memory empty
        """
        if not self.keys:
            return None

        best_idx = 0
        best_overlap = -1
        for i, key in enumerate(self.keys):
            overlap = np.sum(query & key)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        return self.values[best_idx].copy()


# ============================================================
# Online Learning Rule
# ============================================================

class OnlineLearningRule:
    """Eligibility traces + error-driven weight updates.

    ΔW = η * eligibility_trace * error * neuromodulator
    """

    def __init__(self, dim: int, lr: float = 0.01, trace_decay: float = 0.95):
        """Initialize online learning rule.

        Args:
            dim: Weight matrix dimensionality
            lr: Learning rate
            trace_decay: Eligibility trace decay rate
        """
        self.dim = dim
        self.lr = lr
        self.trace_decay = trace_decay
        self.trace = np.zeros((dim, dim), dtype=np.float32)

    def compute_update(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        target: np.ndarray,
        neuromodulator: float,
    ) -> np.ndarray:
        """Compute weight update.

        Args:
            pre: Pre-synaptic activity
            post: Post-synaptic activity
            target: Target post-synaptic activity
            neuromodulator: Gating signal

        Returns:
            Weight update matrix
        """
        # Prediction error
        error = target - post

        # Update eligibility trace: e(t+1) = λ * e(t) + outer(post, pre)
        outer = np.outer(post, pre)
        self.trace = self.trace_decay * self.trace + outer

        # Weight update: ΔW = η * (trace + direct) * error * neuromodulator
        # Include direct Hebbian term for faster learning from zero init
        direct = outer
        update = self.lr * (self.trace + direct) * error[:, np.newaxis] * neuromodulator
        return update

    def reset(self) -> None:
        """Reset eligibility trace."""
        self.trace.fill(0)


# ============================================================
# Response Retriever
# ============================================================

class ResponseRetriever:
    """Brute-force similarity response retrieval.

    Stores query-response pairs and retrieves by SDR overlap.
    """

    def __init__(self, dim: int, capacity: int = 10000):
        """Initialize response retriever.

        Args:
            dim: SDR dimensionality
            capacity: Maximum number of stored responses
        """
        self.dim = dim
        self.capacity = capacity
        self.queries: List[np.ndarray] = []
        self.responses: List[np.ndarray] = []
        self.texts: List[str] = []

    def store(self, query: np.ndarray, response: np.ndarray, text: str = "") -> None:
        """Store query-response pair.

        Args:
            query: Query SDR
            response: Response SDR
            text: Optional text for decoding
        """
        if len(self.queries) >= self.capacity:
            self.queries.pop(0)
            self.responses.pop(0)
            self.texts.pop(0)

        self.queries.append(query.copy())
        self.responses.append(response.copy())
        self.texts.append(text)

    def retrieve(self, query: np.ndarray) -> Optional[np.ndarray]:
        """Retrieve best matching response.

        Args:
            query: Query SDR

        Returns:
            Best matching response or None
        """
        if not self.queries:
            return None

        best_idx = 0
        best_overlap = -1
        for i, q in enumerate(self.queries):
            overlap = np.sum(query & q)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        return self.responses[best_idx].copy()

    def retrieve_with_text(self, query: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Retrieve best matching response with text.

        Args:
            query: Query SDR

        Returns:
            Tuple of (response SDR, response text)
        """
        if not self.queries:
            return None, None

        best_idx = 0
        best_overlap = -1
        for i, q in enumerate(self.queries):
            overlap = np.sum(query & q)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        return self.responses[best_idx].copy(), self.texts[best_idx]

    def retrieve_top_k(self, query: np.ndarray, k: int = 3) -> List[np.ndarray]:
        """Retrieve top-k matching responses.

        Args:
            query: Query SDR
            k: Number of results

        Returns:
            List of top-k responses
        """
        if not self.queries:
            return []

        overlaps = np.array([np.sum(query & q) for q in self.queries])
        top_k_indices = np.argsort(overlaps)[-k:][::-1]

        return [self.responses[i].copy() for i in top_k_indices]


# ============================================================
# Multimodal Encoder
# ============================================================

class MultimodalEncoder:
    """Encodes inputs from any modality to unified SDR space.

    Supports text, image, audio, and video modalities.
    """

    def __init__(self, sdr_dim: int = 10000, input_dims: Optional[Dict[str, int]] = None):
        """Initialize multimodal encoder.

        Args:
            sdr_dim: Unified SDR dimensionality
            input_dims: Dict mapping modality to input dimension
        """
        self.sdr_dim = sdr_dim
        self.input_dims = input_dims or {
            'text': 256,      # Character-level max
            'image': 784,    # 28x28 flattened
            'audio': 128,    # MFCC-like features
            'video': 1024,   # Frame features
        }

        # Random projection matrices per modality
        np.random.seed(42)
        self.projections = {
            modality: np.random.randn(sdr_dim, dim) * 0.01
            for modality, dim in self.input_dims.items()
        }

    def encode(self, modality: str, input_vec: np.ndarray) -> np.ndarray:
        """Encode input from specific modality to SDR.

        Args:
            modality: Input modality ('text', 'image', 'audio', 'video')
            input_vec: Input vector

        Returns:
            Binary SDR
        """
        if modality not in self.projections:
            raise ValueError(f"Unknown modality: {modality}")

        projection = self.projections[modality] @ input_vec
        sdr = np.zeros(self.sdr_dim, dtype=np.int8)

        # Top-k activation for sparsity
        k = max(1, int(self.sdr_dim * 0.02))
        top_k = np.argsort(projection)[-k:]
        sdr[top_k] = 1

        return sdr

    def fuse(self, sdr_list: List[np.ndarray]) -> np.ndarray:
        """Fuse multiple SDRs via union.

        Args:
            sdr_list: List of SDRs to fuse

        Returns:
            Fused SDR
        """
        if not sdr_list:
            return np.zeros(self.sdr_dim, dtype=np.int8)

        result = sdr_list[0].copy()
        for sdr in sdr_list[1:]:
            result = result | sdr
        return result


# ============================================================
# Main RT-ALM Class
# ============================================================

class RTALM:
    """Real-Time Adaptive Language Model.

    Combines all components into a unified system.
    """

    def __init__(
        self,
        sdr_dim: int = 10000,
        ngram_size: int = 3,
        max_episodes: int = 10000,
    ):
        """Initialize RT-ALM.

        Args:
            sdr_dim: SDR dimensionality
            ngram_size: Character n-gram size
            max_episodes: Maximum number of stored episodes
        """
        self.sdr_dim = sdr_dim
        self.ngram_size = ngram_size

        # Components
        self.encoder = SDREncoder(dim=sdr_dim, ngram_size=ngram_size)
        self.multimodal_encoder = MultimodalEncoder(sdr_dim=sdr_dim)
        self.attractor = AttractorMemory(dim=sdr_dim, capacity=max_episodes)
        self.episodic_memory = EpisodicMemory(dim=sdr_dim, capacity=max_episodes)
        self.response_retriever = ResponseRetriever(dim=sdr_dim, capacity=max_episodes)
        self.learning_rule = OnlineLearningRule(dim=sdr_dim)

        # Text storage for decoding responses
        self.text_store: Dict[str, str] = {}

    def store(self, input_text: str, response_text: str) -> None:
        """Store input-response pair.

        Args:
            input_text: Input text
            response_text: Response text
        """
        input_sdr = self.encoder.encode(input_text)
        response_sdr = self.encoder.encode(response_text)

        # Store in all memory systems
        self.attractor.store(input_sdr)
        self.episodic_memory.store(input_sdr, response_sdr)
        self.response_retriever.store(input_sdr, response_sdr, response_text)

    def respond(self, input_text: str) -> Optional[str]:
        """Generate response to input.

        Args:
            input_text: Input text

        Returns:
            Response text or None if no match
        """
        input_sdr = self.encoder.encode(input_text)

        # Retrieve from response retriever (returns SDR + text)
        _, response_text = self.response_retriever.retrieve_with_text(input_sdr)
        return response_text

    def learn(
        self,
        input_text: str,
        target_text: str,
        neuromodulator: float = 1.0,
    ) -> None:
        """Online learning from single example.

        Args:
            input_text: Input text
            target_text: Target response text
            neuromodulator: Gating signal
        """
        input_sdr = self.encoder.encode(input_text)
        target_sdr = self.encoder.encode(target_text)

        # Compute prediction error
        retrieved = self.episodic_memory.retrieve(input_sdr)
        if retrieved is not None:
            error = target_sdr.astype(np.float32) - retrieved.astype(np.float32)
        else:
            error = target_sdr.astype(np.float32)

        # Update learning rule
        pre = input_sdr.astype(np.float32)
        post = retrieved.astype(np.float32) if retrieved is not None else np.zeros(self.sdr_dim)
        update = self.learning_rule.compute_update(pre, post, target_sdr.astype(np.float32), neuromodulator)

        # Store new knowledge
        self.store(input_text, target_text)


# ============================================================
# Utility Functions
# ============================================================

def compute_overlap(sdr1: np.ndarray, sdr2: np.ndarray) -> int:
    """Compute overlap between two SDRs.

    Args:
        sdr1: First SDR
        sdr2: Second SDR

    Returns:
        Number of shared active bits
    """
    return int(np.sum(sdr1 & sdr2))


def compute_similarity(sdr1: np.ndarray, sdr2: np.ndarray) -> float:
    """Compute similarity between two SDRs.

    Args:
        sdr1: First SDR
        sdr2: Second SDR

    Returns:
        Similarity score (0.0 to 1.0)
    """
    k = np.sum(sdr1)
    if k == 0:
        return 0.0
    return float(np.sum(sdr1 & sdr2)) / float(k)


def decode_sdr(sdr: np.ndarray, encoder: SDREncoder, texts: List[str]) -> Optional[str]:
    """Decode SDR to most similar text.

    Args:
        sdr: SDR to decode
        encoder: SDR encoder
        texts: Candidate texts

    Returns:
        Most similar text or None
    """
    if not texts:
        return None

    best_text = texts[0]
    best_overlap = -1

    for text in texts:
        text_sdr = encoder.encode(text)
        overlap = np.sum(sdr & text_sdr)
        if overlap > best_overlap:
            best_overlap = overlap
            best_text = text

    return best_text
