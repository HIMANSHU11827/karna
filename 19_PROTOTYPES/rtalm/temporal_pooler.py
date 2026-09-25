"""
RT-ALM: Temporal Pooler
XOR binding-based temporal sequence learning and prediction.
"""
import numpy as np
from sdr import SDR


class TemporalPooler:
    """Learn and predict temporal sequences via XOR binding."""
    
    def __init__(self, N=1024, k=20, eta=0.01):
        self.N = N
        self.k = k
        self.eta = eta
        self.W = np.zeros((N, N), dtype=np.float32)
        self.transitions_learned = 0
    
    def bind(self, sdr_a, sdr_b):
        """Bind two SDRs via XOR."""
        return sdr_a.xor(sdr_b)
    
    def unbind(self, bound_sdr, sdr_b):
        """Unbind: recover sdr_a from bound(sdr_a, sdr_b) and sdr_b."""
        return bound_sdr.xor(sdr_b)
    
    def learn_transition(self, current_sdr, next_sdr):
        """Learn transition from current to next state."""
        # Transition = current XOR next
        transition = current_sdr.xor(next_sdr)
        
        # Store: current -> transition -> next
        # W[next, transition] += 1
        self.W += self.eta * np.outer(
            next_sdr.bits.astype(np.float32),
            transition.bits.astype(np.float32)
        )
        self.transitions_learned += 1
    
    def predict_next(self, current_sdr):
        """Predict next state from current."""
        # Predicted = current XOR (W @ current)
        input_ = self.W @ current_sdr.bits.astype(np.float32)
        
        # Keep top k
        top_k = np.argsort(input_)[-self.k:]
        result = SDR(self.N, self.k)
        result.bits[top_k] = True
        return result
    
    def learn_sequence(self, sdr_sequence):
        """Learn all transitions in a sequence."""
        for i in range(len(sdr_sequence) - 1):
            self.learn_transition(sdr_sequence[i], sdr_sequence[i + 1])
    
    def get_stats(self):
        """Return temporal pooler statistics."""
        return {
            'transitions_learned': self.transitions_learned,
            'W_mean': float(np.mean(self.W)),
            'W_max': float(np.max(self.W)),
            'W_nonzero': int(np.count_nonzero(self.W))
        }
