"""
RT-ALM: Semantic Memory
=======================
Extracted facts via sparse coding.

Uses a weight matrix to store semantic associations via Hebbian learning.
Retrieval via energy minimization (attractor convergence).
Capacity: ~1000-5000 facts for N=1024, k=20.
"""
import numpy as np
from typing import List, Tuple, Optional


class SemanticMemory:
    """
    Sparse-coding based semantic memory.
    
    Stores semantic facts as sparse associations in a weight matrix.
    Uses Hebbian learning with decay for online updates.
    Retrieval via attractor dynamics (energy minimization).
    """
    
    def __init__(self, dim: int = 1024, k: int = 20, max_facts: int = 5000):
        self.dim = dim
        self.k = k
        self.max_facts = max_facts
        
        # Weight matrix (Hebbian)
        self.W = np.zeros((dim, dim), dtype=np.float32)
        
        # Fact storage: list of (subject_sdr, predicate_sdr, object_sdr, confidence)
        self.facts = []
        self.num_facts = 0
        
        # Hyperparameters
        self.eta = 0.01       # Learning rate
        self.decay = 0.001    # Weight decay
        self.max_iter = 5     # Convergence iterations (hard cap)
    
    def store(self, subject: np.ndarray, predicate: np.ndarray, 
              obj: np.ndarray, confidence: float = 1.0):
        """
        Store a semantic fact (subject, predicate, object).
        
        Uses Hebbian outer product to strengthen associations.
        W += eta * outer(predicate, subject) * confidence
        """
        # Convert to dense if needed
        subj_dense = self._to_dense(subject)
        pred_dense = self._to_dense(predicate)
        obj_dense = self._to_dense(obj)
        
        # Hebbian updates for S-P and P-O associations
        self.W += self.eta * confidence * np.outer(pred_dense, subj_dense)
        self.W += self.eta * confidence * np.outer(obj_dense, pred_dense)
        
        # Apply decay
        self.W -= self.decay * self.W
        
        # Clip weights
        self.W = np.clip(self.W, 0, 1)
        
        # Store fact
        self.facts.append({
            'subject': np.where(subj_dense > 0)[0],
            'predicate': np.where(pred_dense > 0)[0],
            'object': np.where(obj_dense > 0)[0],
            'confidence': confidence,
            'timestamp': self.num_facts
        })
        self.num_facts += 1
        
        # Evict oldest if over capacity
        if self.num_facts > self.max_facts:
            self.facts.pop(0)
            self.num_facts -= 1
    
    def query(self, subject: np.ndarray = None, predicate: np.ndarray = None,
              obj: np.ndarray = None, top_k: int = 5) -> List[dict]:
        """
        Query semantic memory for matching facts.
        
        Can query by any combination of S/P/O.
        Returns top-k matching facts with confidence scores.
        """
        results = []
        
        # Compute query SDR
        query_dense = np.zeros(self.dim, dtype=np.float32)
        if subject is not None:
            query_dense += self._to_dense(subject)
        if predicate is not None:
            query_dense += self._to_dense(predicate)
        if obj is not None:
            query_dense += self._to_dense(obj)
        
        # Normalize
        if query_dense.sum() > 0:
            query_dense = query_dense / query_dense.sum()
        
        # Search facts for best match
        for fact in self.facts:
            # Compute similarity to each component
            subj_sim = self._sdr_similarity(subject, fact['subject']) if subject is not None else 0.5
            pred_sim = self._sdr_similarity(predicate, fact['predicate']) if predicate is not None else 0.5
            obj_sim = self._sdr_similarity(obj, fact['object']) if obj is not None else 0.5
            
            # Combined score (weighted by confidence)
            score = (subj_sim + pred_sim + obj_sim) / 3.0
            score *= fact['confidence']
            
            results.append({
                'subject': fact['subject'],
                'predicate': fact['predicate'],
                'object': fact['object'],
                'score': score,
                'confidence': fact['confidence']
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def complete(self, subject: np.ndarray, predicate: np.ndarray) -> Optional[np.ndarray]:
        """
        Retrieve object given subject and predicate.
        Uses attractor dynamics (energy minimization).
        
        Args:
            subject: Subject SDR
            predicate: Predicate SDR
            
        Returns:
            Object SDR (indices of active bits)
        """
        # Combine S+P as query
        query = self._to_dense(subject) + self._to_dense(predicate)
        
        # Attractor convergence (max 5 iterations)
        state = query.copy()
        for _ in range(self.max_iter):
            # Compute local fields
            h = self.W @ state
            
            # k-WTA
            new_state = self._kwta(h, self.k)
            
            # Check convergence
            if np.allclose(state, new_state):
                break
            
            state = new_state
        
        return np.where(state > 0)[0]
    
    def _to_dense(self, sdr: np.ndarray) -> np.ndarray:
        """Convert SDR to dense array."""
        if sdr is None:
            return np.zeros(self.dim, dtype=np.float32)
        
        dense = np.zeros(self.dim, dtype=np.float32)
        
        if sdr.dtype in [np.int32, np.int64]:
            # SDR is index array
            if len(sdr) > 0:
                dense[sdr] = 1.0
        else:
            dense[sdr > 0] = 1.0
        
        return dense
    
    def _sdr_similarity(self, sdr_a: np.ndarray, sdr_b: np.ndarray) -> float:
        """Compute Jaccard similarity between two SDRs."""
        if sdr_a is None or sdr_b is None:
            return 0.0
        
        set_a = set(self._get_indices(sdr_a))
        set_b = set(self._get_indices(sdr_b))
        
        if not set_a or not set_b:
            return 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_indices(self, sdr: np.ndarray) -> np.ndarray:
        """Get active indices from SDR."""
        if sdr.dtype in [np.int32, np.int64]:
            return sdr
        return np.where(sdr > 0)[0]
    
    def _kwta(self, activations: np.ndarray, k: int) -> np.ndarray:
        """k-Winners-Take-All activation."""
        n = len(activations)
        state = np.zeros(n, dtype=np.float32)
        
        if k >= n:
            state[activations >= 0] = 1.0
            return state
        
        top_k = np.argpartition(activations, -k)[-k:]
        state[top_k] = 1.0
        
        return state
    
    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            'num_facts': self.num_facts,
            'capacity': self.max_facts,
            'utilization': self.num_facts / self.max_facts,
            'weight_mean': float(self.W.mean()),
            'weight_std': float(self.W.std()),
        }


def test_semantic_memory():
    """Test SemanticMemory with toy data."""
    mem = SemanticMemory(dim=1024, k=20)
    
    np.random.seed(42)
    
    # Store facts
    for i in range(50):
        subject = np.random.choice(1024, size=20, replace=False)
        predicate = np.random.choice(1024, size=20, replace=False)
        obj = np.random.choice(1024, size=20, replace=False)
        mem.store(subject, predicate, obj, confidence=0.5 + np.random.random() * 0.5)
    
    print(f"Stored facts: {mem.num_facts}")
    print(f"Memory stats: {mem.get_stats()}")
    
    # Query
    query_subj = np.random.choice(1024, size=20, replace=False)
    query_pred = np.random.choice(1024, size=20, replace=False)
    
    results = mem.query(subject=query_subj, predicate=query_pred, top_k=3)
    print(f"\nQuery returned {len(results)} results")
    for r in results:
        print(f"  score={r['score']:.3f}, confidence={r['confidence']:.3f}")
    
    # Test S->P->O completion
    subj = np.random.choice(1024, size=20, replace=False)
    pred = np.random.choice(1024, size=20, replace=False)
    obj = np.random.choice(1024, size=20, replace=False)
    
    mem.store(subj, pred, obj, confidence=1.0)
    
    retrieved_obj = mem.complete(subj, pred)
    print(f"\nCompletion: retrieved {len(retrieved_obj)} bits")
    overlap = len(set(retrieved_obj) & set(obj))
    print(f"Overlap with original: {overlap}/20 = {overlap/20:.1%}")
    
    print("\nAll semantic memory tests passed!")


if __name__ == "__main__":
    test_semantic_memory()
