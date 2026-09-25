"""
RT-ALM: Online Learning Rule — Eligibility traces + error-driven updates.
"""
import numpy as np

class OnlineLearner:
    """
    Online discriminative learning with eligibility traces.
    No backprop. Local updates only.
    """
    
    def __init__(self, dim: int, learning_rate: float = 0.01, trace_decay: float = 0.95):
        self.dim = dim
        self.lr = learning_rate
        self.trace_decay = trace_decay
        self.eligibility = np.zeros(dim, dtype=np.float32)
        self.weights = np.zeros(dim, dtype=np.float32)
    
    def predict(self, x: np.ndarray) -> float:
        """Predict output for input."""
        return np.dot(self.weights, x)
    
    def learn(self, x: np.ndarray, target: float, prediction: float = None):
        """
        Online learning step with eligibility traces.
        error = target - prediction
        update = lr * error * eligibility
        """
        if prediction is None:
            prediction = self.predict(x)
        
        error = target - prediction
        
        # Update eligibility trace
        self.eligibility = self.trace_decay * self.eligibility + x
        
        # Error-driven weight update
        self.weights += self.lr * error * self.eligibility
        
        return error
    
    def reset_trace(self):
        """Reset eligibility trace (e.g., between episodes)."""
        self.eligibility[:] = 0
