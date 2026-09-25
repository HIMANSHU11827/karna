"""
HPMN — Hierarchical Predictive Memory Network
Deduced from first principles for AGI.

Core architecture:
  - Layers form a hierarchy where each level predicts the one below
  - Bottom-up prediction errors propagate upward for inference
  - Top-down predictions suppress predicted input (predictive coding)
  - k-Winners-Take-All sparse activation (neuromorphic, ~2% active)
  - Local Hebbian/plastic updates — no backpropagation
  - Explicit episodic memory matrix for persistence across time

Mathematical foundation:
  - Prediction: x_hat_i^l = σ(Σ_j W^{l+1→l}_{ij} · a_j^{l+1})
  - Error:       e_i^l = x_i^l - x_hat_i^l
  - Inference:   x_i^l ← x_i^l + α·(Σ_j W^{l→l+1}_{ij}·e_j^{l+1} - Σ_j W^{l+1→l}_{ji}·e_i^l)
  - Sparse act:  a_i^l = x_i^l if x_i^l ∈ top-k(x^l), else 0
  - Learning:    ΔW^{l→l+1}_{ij} = η·e_i^l · a_j^{l+1} · 𝟙[e_i^l > θ]  (Oja-like)
"""

__version__ = "0.1.0"
__all__ = ["HPMN", "PredictiveLayer", "EpisodicMemory", "sparse_activation", "predictive_step"]

from .layer import PredictiveLayer, sparse_activation
from .memory import EpisodicMemory
from .network import HPMN, predictive_step
