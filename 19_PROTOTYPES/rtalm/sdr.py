"""
RT-ALM: Real-Time Adaptive Language Model
Core SDR (Sparse Distributed Representation) class.
"""
import numpy as np


class SDR:
    """Sparse Distributed Representation with binary vectors."""
    
    def __init__(self, N=1024, k=20):
        self.N = N
        self.k = k
        self.bits = np.zeros(N, dtype=bool)
    
    def randomize(self):
        """Set k random bits."""
        self.bits = np.zeros(self.N, dtype=bool)
        active = np.random.choice(self.N, self.k, replace=False)
        self.bits[active] = True
        return self
    
    def overlap(self, other):
        """Compute overlap (number of shared bits)."""
        return int(np.sum(self.bits & other.bits))
    
    def similarity(self, other):
        """Compute cosine-like similarity."""
        return self.overlap(other) / max(self.k * other.k, 1) ** 0.5
    
    def union(self, other):
        """Union (OR) of two SDRs."""
        result = SDR(self.N, self.k)
        result.bits = self.bits | other.bits
        return result
    
    def xor(self, other):
        """XOR binding of two SDRs."""
        result = SDR(self.N, self.k)
        result.bits = self.bits ^ other.bits
        return result
    
    def copy(self):
        """Deep copy."""
        result = SDR(self.N, self.k)
        result.bits = self.bits.copy()
        return result
    
    def active_indices(self):
        """Return indices of active bits."""
        return np.where(self.bits)[0]
    
    def to_dense(self):
        """Convert to dense float array."""
        return self.bits.astype(np.float32)
    
    @classmethod
    def from_dense(cls, dense, k=20):
        """Create SDR from dense vector (top-k)."""
        N = len(dense)
        sdr = cls(N, k)
        top_k = np.argsort(dense)[-k:]
        sdr.bits[top_k] = True
        return sdr
    
    def __repr__(self):
        return f"SDR(N={self.N}, k={self.k}, active={np.sum(self.bits)})"
