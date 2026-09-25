"""
RAIE — Real-time Adaptive Intelligence Engine

Unified from-scratch neural architecture:
- KWTA (k-Winners-Take-All) activation
- SRBI (Sparse Random Binary Init)
- SHD (Sparse Hebbian with Decay) learning
- IAC (Iterative Attractor Convergence)
- XOR binding for semantic memory
- Complementary learning (fast episodic + slow semantic)
- Real-time multimodal + continual learning + sparse representations

Mathematical Formulation:
    Bottom-up: d_l = W_l @ x_{l-1}
    Memory retrieval: m_l = M_l @ x_{l-1}
    Activation: y_l = KWTA(d_l + alpha * m_l)
    Prediction error: e_l = y_l - W_l @ x_{l+1}
    Hebbian update: W_l += eta * y_l @ x_{l-1}^T
    Memory update: M_l += beta * y_l @ x_{l-1}^T
    
    KWTA: keep top-k activations, zero the rest
    XOR binding: y_bind = y_A XOR y_B (sparse binding)
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import json
import time
from pathlib import Path


class KWTA:
    """k-Winners-Take-All activation — sparse, event-driven."""
    
    @staticmethod
    def activate(x: np.ndarray, k: int) -> np.ndarray:
        """Keep top-k activations, zero the rest."""
        k = min(k, len(x))
        threshold = np.partition(x.flatten(), -k)[-k]
        result = np.where(x >= threshold, x, 0)
        if result.sum() > 0:
            result = result / result.sum()
        return result
    
    @staticmethod
    def activate_rate(x: np.ndarray, target_rate: float) -> np.ndarray:
        """Keep top target_rate fraction of activations."""
        k = max(1, int(len(x) * (1 - target_rate)))
        return KWTA.activate(x, k)


class SRBI:
    """Sparse Random Binary Initialization — neuromorphic-friendly."""
    
    @staticmethod
    def init_weights(output_size: int, input_size: int, connection_rate: float = 0.1) -> np.ndarray:
        """Initialize sparse binary weights."""
        W = np.random.binomial(1, connection_rate, (output_size, input_size)).astype(float)
        W = W / max(1, input_size * connection_rate)  # Normalize
        return W


class ComplementaryStore:
    """Complementary Learning — fast episodic + slow semantic (CLS theory)."""
    
    def __init__(self, input_size: int, output_size: int, 
                 fast_lr: float = 0.1, slow_lr: float = 0.001):
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        
        # Hippocampus-like: fast learning, pattern separation
        self.episodic_W = SRBI.init_weights(output_size, input_size, 0.2)
        self.episodic_M = np.zeros((output_size, input_size))
        
        # Neocortex-like: slow learning, pattern completion
        self.semantic_W = SRBI.init_weights(output_size, input_size, 0.1)
        self.semantic_M = np.zeros((output_size, input_size))
    
    def store_fast(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fast episodic learning (hippocampus-like)."""
        delta = self.fast_lr * np.outer(y, x)
        self.episodic_W += delta
        self.episodic_M += self.fast_lr * 0.1 * np.outer(y, x)
    
    def store_slow(self, x: np.ndarray, y: np.ndarray) -> None:
        """Slow semantic learning (neocortex-like)."""
        delta = self.slow_lr * np.outer(y, x)
        self.semantic_W += delta
        self.semantic_M += self.slow_lr * 0.1 * np.outer(y, x)
    
    def consolidate(self) -> None:
        """Consolidate episodic -> semantic (hippocampal replay)."""
        self.semantic_W += 0.1 * (self.episodic_W - self.semantic_W)
        self.semantic_M += 0.1 * (self.episodic_M - self.semantic_M)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from semantic memory (pattern completion)."""
        return self.semantic_W @ x
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input (fast path)."""
        return KWTA.activate_rate(self.episodic_W @ x, 0.9)


class XORMemory:
    """XOR-based sparse binding for semantic memory."""
    
    def __init__(self, size: int, binding_rate: float = 0.1):
        self.size = size
        # Binding matrix: sparse random binary
        self.B = np.random.binomial(1, binding_rate, (size, size)).astype(float)
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """XOR binding: combine two sparse representations."""
        # Element-wise XOR-like operation on sparse vectors
        b_expanded = self.B @ b  # Transform b into binding space
        bound = np.abs(a - b_expanded)  # XOR approximation
        return bound / max(1, bound.sum())
    
    def unbind(self, bound: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Unbind: retrieve b from bound given a."""
        a_transformed = self.B.T @ a
        unbound = np.abs(bound - a_transformed)
        return unbound / max(1, unbound.sum())


class IACMemory:
    """Iterative Attractor Convergence — memory retrieval via convergence."""
    
    def __init__(self, size: int, num_attractors: int = 10):
        self.size = size
        # Attractor states (learned prototypes)
        self.attractors = np.random.randn(num_attractors, size) * 0.01
        self.attractor_labels = np.zeros(num_attractors, dtype=int)
        self.attractor_counts = np.zeros(num_attractors)
    
    def store(self, pattern: np.ndarray, label: int) -> int:
        """Store pattern as attractor. Returns attractor index."""
        # Find closest attractor
        distances = [np.linalg.norm(pattern - a) for a in self.attractors]
        idx = np.argmin(distances)
        
        if distances[idx] < 0.5 or self.attractor_counts[idx] < 5:
            # Update existing attractor
            self.attractors[idx] = 0.9 * self.attractors[idx] + 0.1 * pattern
            self.attractor_labels[idx] = label
            self.attractor_counts[idx] += 1
        else:
            # Create new attractor at first empty slot
            empty = np.where(self.attractor_counts == 0)[0]
            if len(empty) > 0:
                self.attractors[empty[0]] = pattern
                self.attractor_labels[empty[0]] = label
                self.attractor_counts[empty[0]] = 1
                idx = empty[0]
        
        return idx
    
    def retrieve(self, pattern: np.ndarray, num_iterations: int = 5) -> Tuple[np.ndarray, int]:
        """
        Retrieve via iterative attractor convergence.
        
        Iteratively refine pattern toward nearest attractor.
        """
        current = pattern.copy()
        
        for _ in range(num_iterations):
            # Find nearest attractor
            distances = [np.linalg.norm(current - a) for a in self.attractors]
            idx = np.argmin(distances)
            
            # Move toward attractor
            current = 0.7 * current + 0.3 * self.attractors[idx]
            
            # Normalize
            if current.sum() > 0:
                current = current / current.sum()
        
        label = self.attractor_labels[idx]
        return current, label
    
    def predict_label(self, pattern: np.ndarray) -> int:
        """Predict label via attractor convergence."""
        _, label = self.retrieve(pattern)
        return label


class RAIELayer:
    """RAIE layer: unified processing unit."""
    
    def __init__(self, input_size: int, output_size: int,
                 learning_rate: float = 0.01,
                 sparsity: float = 0.9,
                 memory_gain: float = 0.1,
                 decay: float = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.sparsity = sparsity
        self.memory_gain = memory_gain
        self.decay = decay
        
        # SRBI initialization
        self.W = SRBI.init_weights(output_size, input_size, 0.1)
        self.M = np.zeros((output_size, input_size))
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input: bottom-up + memory."""
        # Bottom-up drive
        drive = self.W @ x
        
        # Memory enhancement
        memory = self.M @ x
        
        # Combined activation
        activation = drive + self.memory_gain * memory
        
        # KWTA
        k = max(1, int(self.output_size * (1 - self.sparsity)))
        y = KWTA.activate(activation, k)
        
        return y
    
    def learn(self, x: np.ndarray, y: np.ndarray) -> None:
        """SHD: Sparse Hebbian with Decay learning."""
        # Hebbian update
        delta = self.learning_rate * np.outer(y, x)
        self.W += delta
        
        # Memory update
        self.M += self.learning_rate * 0.1 * np.outer(y, x)
        
        # Decay
        self.W *= (1 - self.decay)
        self.M *= (1 - self.decay)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: encode and learn."""
        y = self.encode(x)
        self.learn(x, y)
        return y
    
    def get_stats(self) -> Dict[str, float]:
        """Get layer statistics."""
        return {
            "weight_norm": np.linalg.norm(self.W),
            "memory_norm": np.linalg.norm(self.M),
            "weight_sparsity": 1.0 - (np.count_nonzero(self.W) / self.W.size),
        }


class RAIENetwork:
    """RAIE Network — full system."""
    
    def __init__(self, layer_sizes: List[int],
                 learning_rate: float = 0.01,
                 sparsity: float = 0.9):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = RAIELayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                learning_rate=learning_rate,
                sparsity=sparsity
            )
            self.layers.append(layer)
        
        # Complementary learning stores for each layer
        self.complementary = []
        for i in range(len(layer_sizes) - 1):
            store = ComplementaryStore(layer_sizes[i], layer_sizes[i + 1])
            self.complementary.append(store)
        
        # XOR binding memory
        self.xor_memory = XORMemory(layer_sizes[-1])
        
        # IAC memory for retrieval
        self.iac_memory = IACMemory(layer_sizes[-1], num_attractors=20)
    
    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        """Feedforward pass."""
        activations = [x]
        
        for layer in self.layers:
            y = layer.encode(activations[-1])
            activations.append(y)
        
        return activations
    
    def train_step(self, x: np.ndarray, label: int) -> None:
        """Single training step: encode + learn + store."""
        # Feedforward
        activations = [x]
        
        for i, layer in enumerate(self.layers):
            layer_input = activations[-1]  # Current input (before forward)
            y = layer.encode(layer_input)
            layer.learn(layer_input, y)
            activations.append(y)
            
            # Store in complementary system for this layer
            self.complementary[i].store_fast(layer_input, y)
            self.complementary[i].store_slow(layer_input, y)
        
        # Store final output in IAC memory
        output = activations[-1]
        self.iac_memory.store(output, label)
    
    def predict(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict label using complementary semantic memory."""
        activations = self.forward(x)
        pre_output = activations[-2]  # Second-to-last (correct dims for complementary)
        output = activations[-1]
        
        # Use semantic memory for classification
        semantic_output = self.complementary[-1].predict(pre_output)
        label = np.argmax(semantic_output)
        return label, output
    
    def save(self, path: str) -> None:
        """Save model."""
        data = {
            "layer_sizes": [self.layers[0].input_size] + [l.output_size for l in self.layers],
            "weights": [l.W.tolist() for l in self.layers],
            "memories": [l.M.tolist() for l in self.layers],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "RAIENetwork":
        """Load model."""
        with open(path) as f:
            data = json.load(f)
        net = cls(data["layer_sizes"])
        for i, layer in enumerate(net.layers):
            layer.W = np.array(data["weights"][i])
            layer.M = np.array(data["memories"][i])
        return net
