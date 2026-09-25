"""
Episodic memory module for persistent state across time.

This is not a standard attention mechanism. It is a content-addressable
memory matrix that stores patterns via local Hebbian writes and reads
via similarity-based retrieval. Designed for sparse, event-driven operation.

Math:
  M[i,j] = stored association between address i and content j
  read: out = M @ query (sparse dot product)
  write: ΔM = η · (query ⊗ target) - γ · M (forgetting)

Key properties:
  - Content-addressable (no separate addressing)
  - Local updates (no global gradient)
  - Forgetting prevents saturation
  - Sparse queries reduce interference
"""

from typing import Optional, Tuple
import numpy as np


class EpisodicMemory:
    """
    Content-addressable episodic memory.
    
    Memory stores associations between address (key) patterns and
    content (value) patterns. Reading retrieves similar stored patterns;
    writing stores new associations with local Hebbian updates.
    
    Args:
        address_size: dimensionality of key vectors
        content_size: dimensionality of value vectors  
        capacity: number of memory slots (rows)
        learning_rate: write strength
        forgetting_rate: decay per write (prevents saturation)
        retrieval_threshold: minimum similarity for retrieval
    """
    
    def __init__(
        self,
        address_size: int,
        content_size: int,
        capacity: int = 1000,
        learning_rate: float = 0.05,
        forgetting_rate: float = 0.001,
        retrieval_threshold: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ):
        self.address_size = address_size
        self.content_size = content_size
        self.capacity = capacity
        self.lr = learning_rate
        self.gamma = forgetting_rate
        self.threshold = retrieval_threshold
        self.rng = rng or np.random.default_rng(42)
        
        # Memory matrix: each row is a stored association
        # W[address, content] — Hebbian outer product accumulated
        self.W = np.zeros((address_size, content_size), dtype=np.float64)
        
        # Usage tracking (for capacity management)
        self.usage = np.zeros(capacity, dtype=np.float64)
        self.write_head = 0  # circular buffer position
        self.stored = 0      # total patterns seen
        
        # History
        self.retrieval_history: list = []
    
    def read(self, query: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Retrieve stored content from a query (address/key) pattern.
        
        Uses sparse dot-product similarity. Returns weighted sum of
        stored content for addresses matching the query.
        
        Args:
            query: sparse query vector of shape (address_size,)
        
        Returns:
            (content, confidence): retrieved content and retrieval confidence
        """
        if np.sum(np.abs(query)) < 1e-8:
            return np.zeros(self.content_size), 0.0
        
        # Read: content = W^T @ query
        content = self.W.T @ query

        # Confidence: scale-invariant retrieval strength
        # ||W^T q|| / (||W||_F * ||q||) = 1.0 for exact single-pattern match,
        # degrades gracefully with superposition. Old mean|content|/sum|query|
        # never exceeded threshold after a single write.
        w_norm = float(np.linalg.norm(self.W))
        q_norm = float(np.linalg.norm(query))
        c_norm = float(np.linalg.norm(content))
        if w_norm < 1e-8 or q_norm < 1e-8:
            return np.zeros(self.content_size), 0.0
        confidence = c_norm / (w_norm * q_norm + 1e-8)
        
        # Threshold: return zeros if retrieval is too weak
        if confidence < self.threshold:
            return np.zeros(self.content_size), 0.0
        
        self.retrieval_history.append(confidence)
        return content, confidence
    
    def write(self, address: np.ndarray, content: np.ndarray) -> None:
        """
        Store an address→content association.
        
        Uses local Hebbian update with forgetting:
          ΔW = η · (address ⊗ content) - γ · W
        
        Args:
            address: key pattern to store at
            content: value pattern to store
        """
        if np.sum(np.abs(address)) < 1e-8 or np.sum(np.abs(content)) < 1e-8:
            return
        
        # Hebbian outer product write
        outer = np.outer(address, content)

        # Update memory with forgetting
        self.W = (1 - self.gamma) * self.W + self.lr * outer

        # Capacity: usage-tracked slot accounting (fixes dead usage/write_head)
        self.stored += 1
        self.usage[self.write_head] += 1.0
        self.write_head = (self.write_head + 1) % self.capacity
        
        # Normalize to prevent unbounded growth
        max_val = np.max(np.abs(self.W))
        if max_val > 10.0:
            self.W *= 10.0 / max_val

    def get_stats(self) -> dict:
        """Memory statistics for monitoring."""
        return {
            "stored": self.stored,
            "write_head": self.write_head,
            "mean_usage": float(np.mean(self.usage)),
            "sparsity": float(np.mean(np.abs(self.W) < 1e-6)),
            "mean_weight": float(np.mean(np.abs(self.W))),
            "max_weight": float(np.max(np.abs(self.W))),
            "mean_confidence": float(np.mean(self.retrieval_history)) if self.retrieval_history else 0.0,
        }


class WorkingMemoryBuffer:
    """
    Short-term working memory with temporal decay.
    
    Holds a small number of recent patterns. Items decay over time
    unless refreshed. Models the transient, capacity-limited nature
    of biological working memory.
    
    Args:
        size: dimensionality of each item
        capacity: number of items held (typically 4-7)
        decay_rate: how quickly items fade per timestep
    """
    
    def __init__(self, size: int, capacity: int = 5, decay_rate: float = 0.3):
        self.size = size
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.buffer: list = []
        self.decays: list = []
    
    def store(self, pattern: np.ndarray) -> None:
        """Store a new pattern in working memory."""
        self.buffer.append(pattern.copy())
        self.decays.append(1.0)
        
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.decays.pop(0)
    
    def decay_step(self) -> None:
        """Apply temporal decay to all held items."""
        self.decays = [d * (1 - self.decay_rate) for d in self.decays]
        # Remove items that decayed below threshold
        alive = [(p, d) for p, d in zip(self.buffer, self.decays) if d > 0.1]
        if alive:
            self.buffer, self.decays = zip(*alive)
            self.buffer, self.decays = list(self.buffer), list(self.decays)
        else:
            self.buffer, self.decays = [], []
    
    def get_blend(self) -> np.ndarray:
        """Get blended (weighted average) of all held patterns."""
        if not self.buffer:
            return np.zeros(self.size)
        weights = np.array(self.decays)
        weights /= weights.sum() + 1e-8
        blend = np.zeros(self.size)
        for w, p in zip(weights, self.buffer):
            blend += w * p
        return blend
    
    def is_empty(self) -> bool:
        return len(self.buffer) == 0
