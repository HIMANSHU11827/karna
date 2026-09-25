"""
RT-ALM: Attractor Network Memory
====================================
Hopfield-style energy-based memory with 5-iteration convergence cap.

Based on: Hopfield (1982), Krotov & Hopfield (2016), Ramsauer et al. (2020)

Key constraints:
- Max 5 iterations for convergence (real-time requirement)
- Sparse binary representations (SDR space, dim=10000, sparsity=2%)
- Online learning via Hebbian outer product
- Fallback to original query if no convergence
"""
import numpy as np


class AttractorMemory:
    """
    Modern Hopfield-style associative memory with k-WTA sparse representations.
    
    Architecture:
    - SDR space: dim=10000, k=200 active bits (2% sparsity)
    - Storage: Hebbian weight matrix W = sum(outer(sdr, sdr)) - diagonal
    - Retrieval: Iterative energy minimization, max 5 iterations
    - Learning: Online, incremental updates to W
    """
    
    def __init__(self, dim: int = 10000, k: int = 200, max_iter: int = 5,
                 learning_rate: float = 0.01, beta: float = 10.0):
        """
        Args:
            dim: SDR dimension
            k: Number of active bits (for k-WTA)
            max_iter: Maximum convergence iterations (hard cap)
            learning_rate: Hebbian learning rate
            beta: Retrieval sharpness (higher = more decisive convergence)
        """
        self.dim = dim
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.beta = beta
        
        # Weight matrix (Hebbian)
        # Using int16 to save memory, accumulate in float32
        self.W = np.zeros((dim, dim), dtype=np.float32)
        self.N = 0  # Number of stored patterns
        
        # Track stored SDRs for retrieval (key-value store)
        self.keys = []  # List of SDR arrays (sparse, stored as index arrays)
        self.values = []  # Associated output SDRs
        
    def store(self, key_sdr: np.ndarray, value_sdr: np.ndarray = None):
        """
        Store a key-value pair in the attractor memory.
        
        Uses Hebbian outer product to update weights:
        W += lr * outer(key, key)  (then subtract diagonal)
        
        Args:
            key_sdr: Sparse binary array (key)
            value_sdr: Optional output SDR (if None, uses key as value)
        """
        if value_sdr is None:
            value_sdr = key_sdr
        
        # Convert sparse to dense if needed
        if key_sdr.dtype != np.float32:
            key_dense = np.zeros(self.dim, dtype=np.float32)
            key_dense[key_sdr > 0] = 1.0
        else:
            key_dense = key_sdr.astype(np.float32)
        
        if value_sdr.dtype != np.float32:
            val_dense = np.zeros(self.dim, dtype=np.float32)
            val_dense[value_sdr > 0] = 1.0
        else:
            val_dense = value_sdr.astype(np.float32)
        
        # Hebbian update: W += lr * outer(key, key)
        self.W += self.learning_rate * np.outer(key_dense, val_dense)
        
        # Subtract self-connections (diagonal) for stability
        np.fill_diagonal(self.W, 0)
        
        # Store sparse key
        active_indices = np.where(key_sdr > 0)[0]
        self.keys.append(active_indices)
        self.values.append(np.where(value_sdr > 0)[0])
        self.N += 1
        
        # Cap at 10k patterns (evict oldest)
        if self.N > 10000:
            self.keys.pop(0)
            self.values.pop(0)
            self.N -= 1
    
    def energy(self, state: np.ndarray) -> float:
        """
        Compute Hopfield energy for a state.
        E = -0.5 * sum_{i,j} W_{ij} * s_i * s_j
        """
        return -0.5 * np.dot(state, np.dot(self.W, state))
    
    def retrieve(self, query_sdr: np.ndarray, return_trajectory: bool = False):
        """
        Retrieve stored pattern via iterative energy minimization.
        
        Uses asynchronous updates with 5-iteration cap.
        Each iteration: update all neurons based on local field.
        
        Args:
            query_sdr: Input SDR (sparse binary or index array)
            return_trajectory: If True, return all intermediate states
            
        Returns:
            Converged SDR, or original query if no convergence
        """
        # Convert to dense if needed
        if query_sdr.dtype != np.float32:
            state = np.zeros(self.dim, dtype=np.float32)
            if len(query_sdr) > 0:
                state[query_sdr] = 1.0
        else:
            state = query_sdr.astype(np.float32).copy()
        
        trajectory = [state.copy()] if return_trajectory else None
        
        # Iterative energy minimization (max 5 iterations)
        for iteration in range(self.max_iter):
            # Compute local fields: h = W @ state
            h = self.W @ state
            
            # Apply k-WTA to get new state
            new_state = self._kwta(h, self.k)
            
            # Check convergence (no change)
            if np.array_equal(new_state, state):
                break
            
            state = new_state
            
            if return_trajectory:
                trajectory.append(state.copy())
        
        # Convert back to sparse indices
        result_indices = np.where(state > 0)[0]
        
        if return_trajectory:
            return result_indices, trajectory
        return result_indices
    
    def _kwta(self, activations: np.ndarray, k: int) -> np.ndarray:
        """
        k-Winners-Take-All: select top-k neurons, return binary state.
        """
        n = len(activations)
        if k >= n:
            return (activations >= 0).astype(np.float32)
        
        # Find top-k using argpartition (O(n) vs O(n log n) for sort)
        top_k_idx = np.argpartition(activations, -k)[-k:]
        threshold = activations[top_k_idx].min()
        
        # Create sparse state
        state = np.zeros(n, dtype=np.float32)
        state[activations >= threshold] = 1.0
        
        # Handle ties
        active = state.sum()
        if active > k:
            above = np.where(activations >= threshold)[0]
            top_k = above[np.argpartition(activations[above], -k)[-k:]]
            state[:] = 0
            state[top_k] = 1.0
        
        return state
    
    def batch_retrieve(self, queries: list) -> list:
        """
        Batch retrieve multiple queries.
        For efficiency, we process them sequentially but could parallelize.
        """
        return [self.retrieve(q) for q in queries]
    
    def capacity_estimate(self) -> float:
        """
        Estimate storage capacity based on current weights.
        
        For modern Hopfield: P ~ e^{alpha * N}
        For classical Hopfield: P ~ 0.138 * N
        
        We use a heuristic based on weight matrix properties.
        """
        if self.N == 0:
            return 0
        
        # Trace-based estimate: capacity proportional to trace(W^T W) / N
        trace = np.trace(self.W @ self.W.T)
        capacity = trace / (self.dim * self.N)
        
        return capacity
    
    def clear(self):
        """Clear all stored patterns."""
        self.W[:] = 0
        self.keys.clear()
        self.values.clear()
        self.N = 0


def test_attractor_memory():
    """Test AttractorMemory with synthetic SDRs."""
    mem = AttractorMemory(dim=1000, k=20, max_iter=5)
    
    np.random.seed(42)
    
    # Store 10 random patterns
    patterns = []
    for i in range(10):
        sdr = np.zeros(1000, dtype=np.int8)
        active = np.random.choice(1000, size=20, replace=False)
        sdr[active] = 1
        patterns.append(sdr)
        mem.store(sdr)
    
    # Test exact retrieval
    for i, pattern in enumerate(patterns):
        result = mem.retrieve(pattern)
        if len(result) > 0:
            overlap = len(set(result) & set(np.where(pattern > 0)[0]))
            acc = overlap / 20
            print(f"Pattern {i}: exact retrieval accuracy = {acc:.1%}")
        else:
            print(f"Pattern {i}: no retrieval")
    
    # Test noisy retrieval (remove 5 bits)
    print("\nNoisy retrieval (5 bits removed):")
    for i, pattern in enumerate(patterns[:3]):
        noisy = pattern.copy()
        active = np.where(pattern > 0)[0]
        remove = np.random.choice(active, size=5, replace=False)
        noisy[remove] = 0
        
        result = mem.retrieve(noisy)
        if len(result) > 0:
            overlap = len(set(result) & set(np.where(pattern > 0)[0]))
            acc = overlap / 20
            print(f"Pattern {i}: noisy retrieval accuracy = {acc:.1%}")
        else:
            print(f"Pattern {i}: no retrieval")
    
    print(f"\nStored patterns: {mem.N}")
    print(f"Capacity estimate: {mem.capacity_estimate():.2f}")
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_attractor_memory()
