"""
RT-ALM: Attractor Memory
Hopfield-style attractor network with SHD (Sparse Hebbian with Decay) learning.
"""
import numpy as np
from sdr import SDR


class AttractorMemory:
    """Attractor memory with online Hebbian learning."""
    
    def __init__(self, N=1024, k=20, eta=0.01, lam=0.001, max_iter=10):
        self.N = N
        self.k = k
        self.eta = eta
        self.lam = lam
        self.max_iter = max_iter
        self.W = np.zeros((N, N), dtype=np.float32)
        self.stored_count = 0
    
    def converge(self, query_sdr):
        """Iterate to nearest attractor state."""
        state = query_sdr.bits.astype(np.float32)
        
        for _ in range(self.max_iter):
            # Compute input to each neuron
            input_ = self.W @ state
            
            # k-WTA: top k become active
            top_k = np.argsort(input_)[-self.k:]
            new_state = np.zeros(self.N, dtype=np.float32)
            new_state[top_k] = 1.0
            
            # Check convergence
            if np.allclose(state, new_state):
                break
            
            state = new_state
        
        result = SDR(self.N, self.k)
        result.bits = state > 0
        return result
    
    def store(self, query_sdr, target_sdr):
        """Store pattern using SHD rule."""
        # Hebbian term: strengthen co-active connections
        hebbian = np.outer(target_sdr.bits.astype(np.float32), 
                          query_sdr.bits.astype(np.float32))
        
        # Decay term: weaken inactive connections
        decay = self.lam * self.W
        
        # Update
        self.W += self.eta * (hebbian - decay)
        self.W = np.clip(self.W, 0, 1)
        self.stored_count += 1
    
    def recall(self, query_sdr):
        """Store and recall in one step."""
        converged = self.converge(query_sdr)
        self.store(query_sdr, converged)
        return converged
    
    def query(self, query_sdr):
        """Query without storing (inference only)."""
        return self.converge(query_sdr)
    
    def get_stats(self):
        """Return memory statistics."""
        return {
            'stored': self.stored_count,
            'W_mean': float(np.mean(self.W)),
            'W_max': float(np.max(self.W)),
            'W_nonzero': int(np.count_nonzero(self.W))
        }
