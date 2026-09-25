"""
HPMN — Hierarchical Predictive Memory Network

The full network architecture. Stacks PredictiveLayers into a hierarchy,
adds episodic memory for persistence, and provides training/inference
loops for classification tasks.

Architecture:
  Input → Layer1 → Layer2 → ... → LayerN → Output
              ↕           ↕
         EpisodicMemory (for persistence across time)

Key design decisions:
  - All learning is local (no backprop through time)
  - Sparse activation throughout (~2% neurons active)
  - Top-down predictions suppress predicted input
  - Bottom-up errors drive inference and learning
  - Episodic memory stores associations for later retrieval
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

from .layer import PredictiveLayer, sparse_activation
from .memory import EpisodicMemory, WorkingMemoryBuffer


def predictive_step(
    network: "HPMN",
    input_pattern: np.ndarray,
    target_pattern: Optional[np.ndarray] = None,
    n_iterations: int = 5,
) -> Dict[str, any]:
    """
    Run one full predictive coding step through the hierarchy.
    
    For each iteration:
      1. Bottom-up pass: encode input through layers
      2. Top-down pass: generate predictions
      3. Compute errors at each level
      4. Update states and learn (local Hebbian)
    
    Args:
        network: the HPMN network
        input_pattern: sensory input (flattened image or feature vector)
        target_pattern: optional supervised signal for output layer
        n_iterations: inference iterations per input
    
    Returns:
        dict with activations, errors, and diagnostics
    """
    # Set input layer state
    network.layers[0].x = input_pattern.copy()
    network.layers[0].a = sparse_activation(input_pattern, network.layers[0].k)
    
    # If target provided, set output layer state
    if target_pattern is not None:
        network.layers[-1].x = target_pattern.copy()
        network.layers[-1].a = target_pattern  # output is not sparse
    
    # Iterative inference
    for iteration in range(n_iterations):
        # Bottom-up pass: compute encodings (layer i activation -> dim size[i+1])
        encodings = []
        for i in range(len(network.layers) - 1):
            nxt = network.layers[i + 1]
            enc = nxt.encode(network.layers[i].a)
            encodings.append(enc)

        # Top-down pass: compute predictions and errors
        for i in range(1, len(network.layers)):
            # Prediction from layer above
            if i < len(network.layers) - 1:
                pred = network.layers[i].predict(network.layers[i + 1].a)
            else:
                pred = np.zeros(network.layers[i].size)

            # Drive from layer below (already projected to this layer size)
            error_below = encodings[i - 1] if i - 1 < len(encodings) else np.zeros(network.layers[i].size)

            # Error from layer above
            error_above = network.layers[i + 1].e if i + 1 < len(network.layers) else None

            # Update layer with true pre/post activations for Hebbian terms
            network.layers[i].update(
                input_from_below=error_below,
                prediction_from_above=pred,
                error_from_above=error_above,
                below_activation=network.layers[i - 1].a,
                above_activation=network.layers[i + 1].a if i + 1 < len(network.layers) else None,
            )
        
        # Update input layer (no layer below)
        network.layers[0].e = network.layers[0].x - network.layers[0].a
    
    # Episodic memory: store current state
    if target_pattern is not None:
        # Store association between input and target
        network.episodic_memory.write(
            address=network.layers[0].a,
            content=target_pattern,
        )
    
    # Collect diagnostics
    diagnostics = {
        "layer_errors": [layer.get_error_stats() for layer in network.layers],
        "mean_error": float(np.mean([float(np.mean(np.abs(layer.e))) for layer in network.layers])),
        "mean_sparsity": float(np.mean([np.mean(layer.a > 0) for layer in network.layers])),
    }
    
    return {
        "activations": [layer.a.copy() for layer in network.layers],
        "errors": [layer.e.copy() for layer in network.layers],
        "diagnostics": diagnostics,
    }


class HPMN:
    """
    Hierarchical Predictive Memory Network.
    
    Args:
        layer_sizes: list of layer dimensions [input, hidden1, ..., output]
        learning_rate: global learning rate
        error_threshold: threshold for Hebbian plasticity
        sparsity: fraction of active neurons (k in kWTA)
        memory_capacity: episodic memory capacity
        rng: random number generator
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        error_threshold: float = 0.1,
        sparsity: float = 0.02,
        memory_capacity: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.theta = error_threshold
        self.k = sparsity
        self.rng = rng or np.random.default_rng(42)
        
        # Build layers
        self.layers: List[PredictiveLayer] = []
        for i, size in enumerate(layer_sizes):
            below_size = layer_sizes[i - 1] if i > 0 else size
            above_size = layer_sizes[i + 1] if i < len(layer_sizes) - 1 else size
            layer = PredictiveLayer(
                size=size,
                below_size=below_size,
                above_size=above_size,
                learning_rate=learning_rate,
                error_threshold=error_threshold,
                sparsity=sparsity,
                rng=self.rng,
            )
            self.layers.append(layer)
        
        # Episodic memory: connects input to output
        self.episodic_memory = EpisodicMemory(
            address_size=layer_sizes[0],
            content_size=layer_sizes[-1],
            capacity=memory_capacity,
            learning_rate=learning_rate * 5,
            rng=self.rng,
        )
        
        # Working memory for temporal context
        self.working_memory = WorkingMemoryBuffer(
            size=layer_sizes[-1],
            capacity=5,
        )
        
        # Training history
        self.training_history: List[Dict] = []
    
    def forward(
        self,
        input_pattern: np.ndarray,
        n_iterations: int = 5,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass (inference only, no learning).

        Learning is disabled during inference by restoring weights
        after the internal predictive_step dynamics run.
        """
        # Set input
        self.layers[0].x = input_pattern.copy()
        self.layers[0].a = sparse_activation(input_pattern, self.layers[0].k)

        # Snapshot weights so inference does not learn
        saved = [(l.W_up.copy(), l.W_down.copy()) for l in self.layers]

        # Iterative inference
        for _ in range(n_iterations):
            # Bottom-up (layer i activation -> dim size[i+1])
            encodings = []
            for i in range(len(self.layers) - 1):
                enc = self.layers[i + 1].encode(self.layers[i].a)
                encodings.append(enc)

            # Top-down
            for i in range(1, len(self.layers)):
                if i < len(self.layers) - 1:
                    pred = self.layers[i].predict(self.layers[i + 1].a)
                else:
                    pred = np.zeros(self.layers[i].size)

                error_below = encodings[i - 1] if i - 1 < len(encodings) else np.zeros(self.layers[i].size)
                error_above = self.layers[i + 1].e if i + 1 < len(self.layers) else None

                self.layers[i].update(
                    input_from_below=error_below,
                    prediction_from_above=pred,
                    error_from_above=error_above,
                    below_activation=self.layers[i - 1].a,
                    above_activation=self.layers[i + 1].a if i + 1 < len(self.layers) else None,
                )

        # Restore weights: forward() is inference-only
        for l, (wu, wd) in zip(self.layers, saved):
            l.W_up = wu
            l.W_down = wd
        
        # Try episodic memory retrieval
        retrieved, confidence = self.episodic_memory.read(self.layers[0].a)
        
        # Blend hierarchy output with episodic retrieval
        hierarchy_output = self.layers[-1].a
        if confidence > 0.1:
            output = 0.7 * hierarchy_output + 0.3 * retrieved
        else:
            output = hierarchy_output
        
        diagnostics = {
            "episodic_confidence": confidence,
            "hierarchy_sparsity": float(np.mean(self.layers[-1].a > 0)),
            "output_sparsity": float(np.mean(output > 0)),
        }
        
        return output, diagnostics
    
    def train_step(
        self,
        input_pattern: np.ndarray,
        target_pattern: np.ndarray,
        n_iterations: int = 5,
    ) -> Dict:
        """
        One training step with learning.
        
        Args:
            input_pattern: input vector
            target_pattern: target output vector
            n_iterations: inference iterations
        
        Returns:
            dict with loss and diagnostics
        """
        result = predictive_step(
            network=self,
            input_pattern=input_pattern,
            target_pattern=target_pattern,
            n_iterations=n_iterations,
        )
        
        # Compute loss
        output = result["activations"][-1]
        loss = float(np.mean((output - target_pattern) ** 2))
        
        # Store in working memory
        self.working_memory.store(target_pattern)
        self.working_memory.decay_step()
        
        # Record history
        record = {
            "loss": loss,
            "mean_error": result["diagnostics"]["mean_error"],
            "mean_sparsity": result["diagnostics"]["mean_sparsity"],
        }
        self.training_history.append(record)
        
        return record
    
    def get_diagnostics(self) -> Dict:
        """Full network diagnostics."""
        return {
            "layer_errors": [layer.get_error_stats() for layer in self.layers],
            "episodic_memory": self.episodic_memory.get_stats(),
            "training_history_length": len(self.training_history),
            "recent_loss": np.mean([h["loss"] for h in self.training_history[-100:]]) if self.training_history else 0,
        }
