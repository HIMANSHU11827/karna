"""
RT-ALM: Spatial Pooler (k-WTA)
Competitive learning layer that maintains sparse representations.
"""
import numpy as np
from sdr import SDR


class SpatialPooler:
    """k-WTA spatial pooling with adaptive thresholds."""
    
    def __init__(self, N=1024, k=20, target_sparsity=0.02):
        self.N = N
        self.k = k
        self.target_sparsity = target_sparsity
        self.thresholds = np.ones(N) * 0.5
        self.activity_history = np.zeros(N)
        self.history_count = 0
    
    def compete(self, input_sdr):
        """Select top k bits via k-WTA competition."""
        activity = input_sdr.bits.astype(np.float32)
        
        # Apply thresholds
        activity[activity < self.thresholds] = 0.0
        
        # k-WTA: keep only top k
        result = SDR(self.N, self.k)
        if np.sum(activity > 0) >= self.k:
            top_k = np.argsort(activity)[-self.k:]
            result.bits[top_k] = True
        else:
            # Not enough active bits — take top k from input
            top_k = np.argsort(input_sdr.bits)[-self.k:]
            result.bits[top_k] = True
        
        # Update activity history for threshold adaptation
        self.activity_history += result.bits.astype(np.float32)
        self.history_count += 1
        
        return result
    
    def adapt_thresholds(self):
        """Adapt thresholds to maintain target sparsity (ATM)."""
        if self.history_count == 0:
            return
        
        avg_activity = self.activity_history / self.history_count
        actual_sparsity = np.mean(avg_activity > 0)
        
        # If too active, raise thresholds; if too inactive, lower them
        self.thresholds += 0.001 * (actual_sparsity - self.target_sparsity)
        self.thresholds = np.clip(self.thresholds, 0.01, 0.99)
        
        # Reset history
        self.activity_history = np.zeros(self.N)
        self.history_count = 0
    
    def process(self, input_sdr):
        """Full processing: compete + adapt."""
        result = self.compete(input_sdr)
        self.adapt_thresholds()
        return result
