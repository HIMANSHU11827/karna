"""
Core layer module — predictive coding with sparse activation.

Each layer maintains a state vector, receives predictions from above
and prediction errors from below, and performs local Hebbian learning.

Math:
  x_hat = σ(W↓ · a↑)          # top-down prediction
  e = x - x_hat                 # prediction error
  ΔW↑ = η · e · a↑ · 𝟙[e>θ]   # Hebbian plasticity on positive errors
"""

from typing import Tuple, Optional
import numpy as np


def sparse_activation(x: np.ndarray, k: float = 0.02) -> np.ndarray:
    """
    k-Winners-Take-All sparse activation.
    
    Only top-k neurons fire; rest are zero. This gives us:
    - Sparse, energy-efficient representations (~2% active)
    - Better separation between patterns
    - Natural interference reduction for memory
    
    Args:
        x: pre-activation values, shape (n,)
        k: fraction of neurons to activate (default 2%)
    
    Returns:
        Sparse activation with same shape as x.
    """
    n = x.shape[0]
    n_active = max(1, int(np.ceil(k * n)))
    result = np.zeros_like(x)
    # Find top-k indices
    top_k_indices = np.argpartition(x, -n_active)[-n_active:]
    # Scale active neurons by their activation strength (soft scaling)
    result[top_k_indices] = x[top_k_indices]
    return result


def heaviside_threshold(x: np.ndarray, theta: float) -> np.ndarray:
    """Indicator function for error thresholding in Hebbian update."""
    return (x > theta).astype(np.float64)


class PredictiveLayer:
    """
    A single layer in the predictive hierarchy.
    
    State:
      - x: current activity state (n,)
      - e: prediction error (n,)
      - W_up: bottom-up weights to layer above (n_above, n)
      - W_down: top-down weights from layer above (n, n_above)
      - bias: intrinsic bias term (n,)
    """
    
    def __init__(
        self,
        size: int,
        below_size: int,
        above_size: int,
        learning_rate: float = 0.01,
        error_threshold: float = 0.1,
        sparsity: float = 0.02,
        rng: Optional[np.random.Generator] = None,
    ):
        self.size = size
        self.lr = learning_rate
        self.theta = error_threshold
        self.k = sparsity
        self.rng = rng or np.random.default_rng(42)
        
        # Initialize weights with Xavier-like scaling
        scale_up = np.sqrt(2.0 / (below_size + size))
        self.W_up = self.rng.normal(0, scale_up, (size, below_size))
        self.b_up = np.zeros(size)
        
        scale_down = np.sqrt(2.0 / (size + above_size))
        self.W_down = self.rng.normal(0, scale_down, (size, above_size))
        self.b_down = np.zeros(size)
        
        # State
        self.x = np.zeros(size)
        self.e = np.zeros(size)
        self.a = np.zeros(size)  # sparse activation
        
        # History for analysis
        self.error_history: list = []
    
    def predict(self, above_activation: np.ndarray) -> np.ndarray:
        """Top-down prediction from layer above."""
        return np.tanh(self.W_down @ above_activation + self.b_down)
    
    def encode(self, below_activation: np.ndarray) -> np.ndarray:
        """Bottom-up encoding to layer above."""
        return np.maximum(0, self.W_up @ below_activation + self.b_up)
    
    def update(
        self,
        input_from_below: np.ndarray,
        prediction_from_above: Optional[np.ndarray] = None,
        error_from_above: Optional[np.ndarray] = None,
        below_activation: Optional[np.ndarray] = None,
        above_activation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One inference+learning step.

        Args:
            input_from_below: drive signal from layer below (dim = size)
            prediction_from_above: x_hat from layer above (None for top layer)
            error_from_above: prediction error from layer above (for propagation)
            below_activation: raw activation of layer below (dim = below_size),
                presynaptic term for W_up plasticity
            above_activation: raw activation of layer above (dim = above_size),
                presynaptic term for W_down plasticity

        Returns:
            (activation, error): this layer's sparse activation and prediction error
        """
        # Compute prediction from above (if not top layer)
        if prediction_from_above is not None:
            x_hat = prediction_from_above
        else:
            x_hat = np.zeros(self.size)

        # Update state: integrate bottom-down prediction error with top-down error signal
        # x ← x + α·(bottom-up error - top-down error)
        top_down_correction = self.W_down @ error_from_above if error_from_above is not None else 0
        self.x = np.clip(
            self.x + self.lr * (input_from_below - top_down_correction),
            -5, 5,
        )

        # Sparse activation
        self.a = sparse_activation(self.x, self.k)

        # Prediction error
        self.e = self.x - x_hat

        # Hebbian learning on bottom-up weights
        # ΔW_up = η · (e ⊙ 𝟙[e > θ]) ⊗ a_below  (true pre × post)
        pre_below = below_activation if below_activation is not None else input_from_below
        if pre_below is not None and pre_below.shape[0] == self.W_up.shape[1]:
            mask = heaviside_threshold(self.e, self.theta)
            self.W_up += self.lr * np.outer(self.e * mask, pre_below)
            # Normalize weight rows to prevent runaway
            row_norms = np.linalg.norm(self.W_up, axis=1, keepdims=True)
            self.W_up = np.where(row_norms > 3.0, self.W_up * 3.0 / row_norms, self.W_up)

        # Hebbian learning on top-down weights
        # ΔW_down = η · e ⊗ a_above  (true pre × post)
        if prediction_from_above is not None and above_activation is not None:
            if above_activation.shape[0] == self.W_down.shape[1]:
                self.W_down += 0.5 * self.lr * np.outer(self.e, above_activation)
                row_norms = np.linalg.norm(self.W_down, axis=1, keepdims=True)
                self.W_down = np.where(row_norms > 3.0, self.W_down * 3.0 / row_norms, self.W_down)
        
        self.error_history.append(float(np.mean(np.abs(self.e))))
        return self.a, self.e
    
    def get_error_stats(self) -> dict:
        """Return error statistics for monitoring."""
        if not self.error_history:
            return {"mean": 0, "std": 0}
        return {
            "mean": float(np.mean(self.error_history)),
            "std": float(np.std(self.error_history)),
            "last": float(self.error_history[-1]),
            "sparsity": float(np.mean(self.a > 0)),
        }
