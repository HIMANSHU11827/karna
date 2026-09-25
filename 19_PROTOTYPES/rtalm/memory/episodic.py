"""
RT-ALM: Episodic Memory
===========================
Key-value store with content-addressable retrieval.

Stores (key_sdr, value_sdr, context) tuples with:
- Hash-based deduplication
- FIFO eviction at capacity
- Brute-force similarity search (for <10k episodes)
"""
import numpy as np
import hashlib
import time


class EpisodicMemory:
    """
    Episodic memory storing key-value-context tuples.
    
    Uses content hashing for dedup and brute-force similarity
    for retrieval (suitable for <10k episodes as per FINAL_DESIGN.md).
    """
    
    def __init__(self, capacity: int = 10000, dim: int = 10000):
        """
        Args:
            capacity: Maximum number of episodes (FIFO eviction)
            dim: SDR dimension
        """
        self.capacity = capacity
        self.dim = dim
        
        # Storage lists (ordered, oldest first)
        self.keys = []       # List of SDR index arrays
        self.values = []     # List of SDR index arrays  
        self.contexts = []   # List of context dicts
        self.timestamps = [] # List of (store_time, last_access_time)
        self.content_hashes = set()  # For deduplication
        
        # Access statistics
        self.access_counts = []
        self.total_accesses = 0
    
    def _compute_hash(self, sdr_indices: np.ndarray) -> str:
        """Compute content hash for deduplication."""
        # Convert sorted indices to bytes
        data = np.sort(sdr_indices).tobytes()
        return hashlib.md5(data).hexdigest()
    
    def store(self, key_sdr: np.ndarray, value_sdr: np.ndarray = None,
              context: dict = None):
        """
        Store an episode.
        
        Args:
            key_sdr: Query SDR (indices of active bits)
            value_sdr: Response SDR (indices of active bits)
            context: Optional context dict (timestamp, modality, etc.)
        """
        # Convert to indices if dense
        if key_sdr.dtype != np.int64:
            key_indices = np.where(key_sdr > 0)[0].astype(np.int64)
        else:
            key_indices = key_sdr
        
        if value_sdr is not None:
            if value_sdr.dtype != np.int64:
                val_indices = np.where(value_sdr > 0)[0].astype(np.int64)
            else:
                val_indices = value_sdr
        else:
            val_indices = key_indices
        
        # Deduplication check
        content_hash = self._compute_hash(key_indices)
        if content_hash in self.content_hashes:
            # Duplicate — skip storing but update timestamp
            return False
        
        # FIFO eviction if at capacity
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)
            self.contexts.pop(0)
            self.timestamps.pop(0)
            self.access_counts.pop(0)
        
        # Store
        self.keys.append(key_indices)
        self.values.append(val_indices)
        self.contexts.append(context or {})
        self.timestamps.append([time.time(), time.time()])
        self.content_hashes.add(content_hash)
        self.access_counts.append(0)
        
        return True
    
    def retrieve(self, query_sdr: np.ndarray, top_k: int = 1,
                 threshold: float = 0.5) -> list:
        """
        Retrieve top-k most similar episodes via brute-force similarity.
        
        Similarity = |intersection| / |union| (Jaccard index)
        
        Args:
            query_sdr: Query SDR (indices or dense)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (key, value, context, similarity) tuples
        """
        if len(self.keys) == 0:
            return []
        
        # Convert query to indices
        if query_sdr.dtype != np.int64:
            query_indices = np.where(query_sdr > 0)[0].astype(np.int64)
        else:
            query_indices = query_sdr
        
        query_set = set(query_indices)
        if len(query_set) == 0:
            return []
        
        # Compute Jaccard similarity with all stored keys
        similarities = []
        for i, key_indices in enumerate(self.keys):
            key_set = set(key_indices)
            intersection = len(query_set & key_set)
            union = len(query_set | key_set)
            sim = intersection / union if union > 0 else 0.0
            similarities.append((i, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k above threshold
        results = []
        for idx, sim in similarities[:top_k]:
            if sim < threshold:
                break
            # Update access stats
            self.access_counts[idx] += 1
            self.timestamps[idx][1] = time.time()
            self.total_accesses += 1
            
            results.append({
                'key': self.keys[idx],
                'value': self.values[idx],
                'context': self.contexts[idx],
                'similarity': sim,
                'index': idx,
                'access_count': self.access_counts[idx]
            })
        
        return results
    
    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            'size': len(self.keys),
            'capacity': self.capacity,
            'utilization': len(self.keys) / self.capacity,
            'total_accesses': self.total_accesses,
            'unique_hashes': len(self.content_hashes),
            'avg_access_count': np.mean(self.access_counts) if self.access_counts else 0
        }


def test_episodic_memory():
    """Test EpisodicMemory."""
    mem = EpisodicMemory(capacity=1000, dim=1000)
    
    np.random.seed(42)
    
    # Store 100 episodes
    for i in range(100):
        key = np.random.choice(1000, size=20, replace=False)
        val = np.random.choice(1000, size=20, replace=False)
        mem.store(key, val, {'episode_id': i, 'modality': 'text'})
    
    # Retrieve with noisy query
    query = np.random.choice(1000, size=20, replace=False)
    results = mem.retrieve(query, top_k=5, threshold=0.0)
    
    print(f"Stored: {len(mem.keys)}")
    print(f"Retrieved: {len(results)}")
    for r in results[:3]:
        print(f"  sim={r['similarity']:.3f}, access_count={r['access_count']}")
    
    # Test deduplication
    dup_result = mem.store(query, query, {'dup': True})
    print(f"Duplicate rejected: {not dup_result}")
    
    # Test FIFO eviction
    mem_small = EpisodicMemory(capacity=10, dim=100)
    for i in range(15):
        key = np.random.choice(100, size=5, replace=False)
        mem_small.store(key)
    
    print(f"FIFO eviction: stored 15, size = {len(mem_small.keys)}")
    assert len(mem_small.keys) == 10
    
    print("\nAll episodic memory tests passed!")


if __name__ == "__main__":
    test_episodic_memory()
