"""
Sparse Predictive Memory Network (SPMN) — From-Scratch Neural Architecture

Novel architecture deduced from first principles for AGI:
1. Local predictive coding (each layer minimizes its own prediction error)
2. Sparse event-driven activation (neuromorphic, energy-efficient)
3. Hebbian learning (no backprop, biologically plausible)
4. Hierarchical composition (simple units → complex representations)
5. Memory slots that don't interfere with learning (continual learning)

Mathematical Formulation:
    Prediction error at layer l: e_l = x_l - p_l
    Activation (sparse): y_l = threshold(e_l, θ)  [θ = sparsity threshold]
    Top-down prediction: p_l = W_l^T y_{l+1}
    Hebbian update: ΔW_l = η * y_l * e_l^T  [η = learning rate]
    
    Memory (novel): M_l stores stable patterns via Hebbian consolidation
    Retrieval: y_l = threshold(e_l + α * M_l y_l, θ)  [α = memory gain]
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json
import time
from pathlib import Path


class SPMNLayer:
    """A single Sparse Predictive Memory Network layer."""
    
    def __init__(self, input_size: int, output_size: int, 
                 learning_rate: float = 0.01, 
                 sparsity_threshold: float = 0.1,
                 memory_gain: float = 0.1,
                 decay_rate: float = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.theta = sparsity_threshold
        self.alpha = memory_gain
        self.decay_rate = decay_rate
        
        # SRBI: Sparse Random Binary Init (neuromorphic-friendly)
        # Instead of Gaussian, use sparse binary: each neuron connects to ~10% of inputs
        self.W = np.random.binomial(1, 0.1, (output_size, input_size)).astype(float) * 0.1
        
        # Memory matrix (novel: input-output familiarity storage)
        self.M = np.zeros((output_size, input_size))
        
        # Activation statistics (for adaptive threshold)
        self.mean_activation = 0.0
        self.activation_count = 0
    
    def predict(self, y_above: Optional[np.ndarray]) -> np.ndarray:
        """Generate top-down prediction."""
        if y_above is None:
            return np.zeros(self.input_size)
        return self.W.T @ y_above
    
    def activate(self, x: np.ndarray, y_above: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        KWTA (k-Winners-Take-All) activation with predictive coding.
        
        Bottom-up drive: W @ x
        Top-down prediction: W.T @ y_above (if available, from higher layer)
        Error = drive - prediction
        """
        # Bottom-up: drive activation from input
        drive = self.W @ x
        
        # Top-down: predict what activation should be (from higher layer)
        # For feedforward, this is None during first pass
        if y_above is not None and y_above.shape[0] == self.output_size:
            prediction = self.W.T @ y_above
        else:
            prediction = np.zeros(self.output_size)
        
        # Prediction error (surprise)
        e = drive - prediction
        
        # Memory enhancement
        memory_contribution = self.alpha * (self.M @ x)
        
        # KWTA: k-Winners-Take-All (sparse activation)
        activation_raw = np.abs(e) + memory_contribution
        k = max(1, int(self.output_size * (1 - self.theta)))
        threshold = np.partition(activation_raw.flatten(), -k)[-k]
        
        y = np.where(activation_raw >= threshold, activation_raw, 0)
        
        # Normalize
        if np.sum(y) > 0:
            y = y / np.sum(y)
        
        self.mean_activation += np.sum(y)
        self.activation_count += 1
        
        return y, e
    
    def learn(self, y: np.ndarray, x: np.ndarray) -> None:
        """SHD learning: Sparse Hebbian with Decay.
        
        Strengthen connections from active inputs to active outputs.
        Sparse: only update for non-zero activations.
        Decay: prevent runaway growth.
        """
        # Hebbian update: outer product of output activation and input
        delta_W = self.learning_rate * np.outer(y, x)
        self.W += delta_W
        
        # Weight decay (SHD: Sparse Hebbian with Decay)
        self.W *= (1 - self.decay_rate)
        
        # Memory consolidation: store input-output familiarity
        self.M += self.learning_rate * 0.1 * np.outer(y, x)
        self.M *= (1 - self.decay_rate)
    
    def forward(self, x: np.ndarray, y_above: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass: activate and learn."""
        y, e = self.activate(x, y_above)
        self.learn(y, x)
        return y
    
    def get_stats(self) -> Dict[str, float]:
        """Get layer statistics for monitoring."""
        return {
            "mean_activation": self.mean_activation / max(1, self.activation_count),
            "weight_norm": np.linalg.norm(self.W),
            "memory_norm": np.linalg.norm(self.M),
            "sparsity": 1.0 - (np.count_nonzero(self.W) / self.W.size),
        }


class SPMNetwork:
    """Sparse Predictive Memory Network — multi-layer architecture."""
    
    def __init__(self, layer_sizes: List[int], 
                 learning_rate: float = 0.01,
                 sparsity_threshold: float = 0.9,
                 memory_gain: float = 0.1):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = SPMNLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                learning_rate=learning_rate,
                sparsity_threshold=sparsity_threshold,
                memory_gain=memory_gain
            )
            self.layers.append(layer)
        self.learning_rate = learning_rate
    
    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        """Bottom-up feedforward pass (no top-down during initial pass)."""
        activations = [x]
        
        for layer in self.layers:
            y = layer.forward(activations[-1], y_above=None)
            activations.append(y)
        
        return activations
    
    def forward_with_iac(self, x: np.ndarray, num_iterations: int = 3) -> List[np.ndarray]:
        """
        Forward pass with Iterative Attractor Convergence (IAC).
        
        1. Bottom-up pass (feedforward)
        2. Iterative refinement: top-down predictions + error correction
        """
        # Initial feedforward
        activations = self.forward(x)
        
        # Iterative refinement
        for iteration in range(num_iterations):
            # Top-down pass: propagate predictions from top to bottom
            for i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[i]
                if i < len(self.layers) - 1:
                    # Get top-down prediction from layer above
                    y_above = activations[i + 2]  # +2 because activations[0] is input
                    prediction = layer.W.T @ y_above
                else:
                    prediction = np.zeros(layer.output_size)
                
                # Re-activate with top-down context
                y_refined, _ = layer.activate(activations[i + 1], y_above)
                activations[i + 1] = y_refined
        
        return activations
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Inference without learning (for evaluation)."""
        activations = [x]
        y_prev = None
        
        for layer in self.layers:
            y, _ = layer.activate(activations[-1], y_prev)
            activations.append(y)
            y_prev = y
        
        return activations[-1]
    
    def get_all_stats(self) -> List[Dict[str, float]]:
        """Get statistics for all layers."""
        return [layer.get_stats() for layer in self.layers]
    
    def total_sparsity(self) -> float:
        """Get network-wide sparsity."""
        total_nonzero = sum(np.count_nonzero(l.W) for l in self.layers)
        total_params = sum(l.W.size for l in self.layers)
        return 1.0 - (total_nonzero / total_params)
    
    def save(self, path: str) -> None:
        """Save network weights."""
        data = {
            "layer_sizes": [self.layers[0].input_size] + [l.output_size for l in self.layers],
            "weights": [l.W.tolist() for l in self.layers],
            "memories": [l.M.tolist() for l in self.layers],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "SPMNetwork":
        """Load network weights."""
        with open(path) as f:
            data = json.load(f)
        net = cls(data["layer_sizes"])
        for i, layer in enumerate(net.layers):
            layer.W = np.array(data["weights"][i])
            layer.M = np.array(data["memories"][i])
        return net
