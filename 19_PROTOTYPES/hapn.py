"""
AGI Research Lab — HAPN Core Implementation
Hierarchical Attractor Predictive Network
Built from the formal spec in 02_ARCHITECTURE/HAPN_formal_spec.md
"""

import numpy as np
from pathlib import Path
import json
import time
from typing import List, Tuple, Optional, Dict, Any


# =============================================================================
# SDR Operations
# =============================================================================

class SDR:
    """Sparse Distributed Representation — binary vector with sparse active bits."""
    
    def __init__(self, dim: int, sparsity: float = 0.05, seed: int = 42):
        self.dim = dim
        self.sparsity = sparsity
        self.k = max(1, int(dim * sparsity))
        self.rng = np.random.default_rng(seed)
        self.vector = np.zeros(dim, dtype=np.int8)
    
    def randomize(self) -> np.ndarray:
        """Generate a random SDR."""
        self.vector = np.zeros(self.dim, dtype=np.int8)
        indices = self.rng.choice(self.dim, self.k, replace=False)
        self.vector[indices] = 1
        return self.vector
    
    def set_from_dense(self, dense: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Convert dense vector to SDR via top-k threshold."""
        if threshold is None:
            # Top-k
            indices = np.argpartition(dense, -self.k)[-self.k:]
            self.vector = np.zeros(self.dim, dtype=np.int8)
            self.vector[indices] = 1
        else:
            self.vector = (dense > threshold).astype(np.int8)
        return self.vector
    
    def overlap(self, other: np.ndarray) -> int:
        """Count of overlapping active bits."""
        return int(np.sum(self.vector * other))
    
    def similarity(self, other: np.ndarray) -> float:
        """Cosine similarity."""
        dot = self.overlap(other)
        norms = np.sum(self.vector) * np.sum(other)
        if norms == 0:
            return 0.0
        return dot / norms
    
    def xor_bind(self, other: np.ndarray) -> np.ndarray:
        """XOR binding (invertible)."""
        return np.bitwise_xor(self.vector, other).astype(np.int8)
    
    def xor_unbind(self, bound: np.ndarray) -> np.ndarray:
        """XOR unbinding (same as binding)."""
        return np.bitwise_xor(self.vector, bound).astype(np.int8)
    
    def circular_convolve(self, other: np.ndarray) -> np.ndarray:
        """Circular convolution (for continuous SDRs)."""
        return np.real(np.fft.ifft(np.fft.fft(self.vector) * np.fft.fft(other))).astype(np.int8)


def sdr_union(sdrs: List[np.ndarray]) -> np.ndarray:
    """Union of multiple SDRs."""
    if not sdrs:
        return np.zeros(1, dtype=np.int8)
    result = np.zeros(sdrs[0].shape, dtype=np.int8)
    for sdr in sdrs:
        result = np.logical_or(result, sdr).astype(np.int8)
    return result


def sdr_intersection(sdrs: List[np.ndarray]) -> np.ndarray:
    """Intersection of multiple SDRs."""
    if not sdrs:
        return np.zeros(1, dtype=np.int8)
    result = sdrs[0].copy()
    for sdr in sdrs[1:]:
        result = np.logical_and(result, sdr).astype(np.int8)
    return result


# =============================================================================
# K-WTA (k-Winner-Take-All)
# =============================================================================

def kwta(activations: np.ndarray, k: int) -> np.ndarray:
    """k-Winner-Take-All: keep only top-k active neurons."""
    result = np.zeros_like(activations)
    if k >= len(activations):
        result[activations > 0] = activations[activations > 0]
        return result
    top_k = np.argpartition(activations, -k)[-k:]
    result[top_k] = activations[top_k]
    return result


def kwta_binary(activations: np.ndarray, k: int) -> np.ndarray:
    """Binary k-WTA: top-k become 1, rest 0."""
    result = np.zeros_like(activations, dtype=np.int8)
    if k >= len(activations):
        result[activations > 0] = 1
        return result
    top_k = np.argpartition(activations, -k)[-k:]
    result[top_k] = 1
    return result


# =============================================================================
# Attractor Network (Hopfield-style)
# =============================================================================

class AttractorNetwork:
    """
    Attractor network for storing and recalling SDR patterns.
    Uses Hebbian storage rule and async convergence.
    """

    def __init__(self, dim: int, capacity: int = 1000, seed: int = 42):
        self.dim = dim
        self.capacity = capacity
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Weight matrix
        self.weights = np.zeros((dim, dim))
        # Adaptive thresholds
        self.threshold = np.ones(dim) * 0.1
        # Stored patterns
        self.stored_patterns: List[np.ndarray] = []
        self.n_stored = 0

    def store(self, pattern: np.ndarray):
        """Store a pattern using Hebbian learning rule."""
        if self.n_stored >= self.capacity:
            return

        # Normalize to -1, 1
        p = 2 * pattern.astype(np.float64) - 1

        # Hebbian update: outer product
        self.weights += np.outer(p, p)

        # Adaptive threshold (BCM-like)
        self.threshold = 0.99 * self.threshold + 0.01 * (pattern ** 2)

        # Normalize weights
        norm = np.linalg.norm(self.weights)
        if norm > 0:
            self.weights /= norm

        self.stored_patterns.append(pattern.copy())
        self.n_stored += 1

    def recall(self, cue: np.ndarray, max_steps: int = 50) -> np.ndarray:
        """Recall pattern from cue via attractor convergence."""
        state = cue.astype(np.float64).copy()

        for _ in range(max_steps):
            # Update one random neuron at a time (async)
            i = self.rng.integers(0, self.dim)
            activation = np.dot(self.weights[i], state) - self.threshold[i]
            state[i] = 1.0 if activation > 0 else 0.0

            # Check convergence (periodic)
            if _ % 10 == 0:
                new_state = (state > 0.5).astype(np.int8)
                if np.array_equal(new_state, (state > 0.5).astype(np.int8)):
                    return new_state

        return (state > 0.5).astype(np.int8)

    def get_capacity(self) -> int:
        """Get current stored pattern count."""
        return self.n_stored


# =============================================================================
# Level 1: Perception Encoder
# =============================================================================

class VisionEncoder:
    """
    Encodes dense input (MNIST pixels) to sparse SDR.
    Uses random projection + k-WTA.
    """

    def __init__(self, input_dim: int = 784, output_dim: int = 2000, sparsity: float = 0.05, seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.k = max(1, int(output_dim * sparsity))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Random projection matrix
        self.projection = self.rng.normal(0, 0.1, (output_dim, input_dim)) / np.sqrt(input_dim)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode dense input to SDR."""
        x = x / (np.linalg.norm(x) + 1e-8)
        activations = self.projection @ x
        return kwta_binary(activations, self.k)


class LanguageEncoder:
    """
    Encodes text to SDR.
    Simple bag-of-words hashing approach.
    """

    def __init__(self, vocab_size: int = 10000, output_dim: int = 2000, sparsity: float = 0.03, seed: int = 42):
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.k = max(1, int(output_dim * sparsity))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Hash-based projection
        self.hash_seeds = self.rng.integers(0, 2**31, size=vocab_size)

    def encode(self, tokens: List[str]) -> np.ndarray:
        """Encode token list to SDR."""
        activations = np.zeros(self.output_dim, dtype=np.float64)

        for token in tokens:
            # Hash token to output indices
            token_hash = hash(token) & 0x7FFFFFFF
            # Select k indices via hash
            indices = np.array([
                (token_hash * (i + 1)) % self.output_dim
                for i in range(self.k * 3)
            ])
            activations[indices] += 1.0

        # Top-k
        return kwta_binary(activations, self.k)


class ActionDecoder:
    """
    Decodes SDR to discrete action.
    Simple linear readout.
    """

    def __init__(self, input_dim: int = 2000, n_actions: int = 10, seed: int = 42):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Linear readout weights
        self.weights = self.rng.normal(0, 0.1, (n_actions, input_dim))

    def decode(self, sdr: np.ndarray) -> int:
        """Decode SDR to action index."""
        logits = self.weights @ sdr.astype(np.float64)
        return int(np.argmax(logits))

    def train_step(self, sdr: np.ndarray, target: int, lr: float = 0.1):
        """Train readout on single example."""
        logits = self.weights @ sdr.astype(np.float64)
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        # Gradient
        dlogits = probs.copy()
        dlogits[target] -= 1

        self.weights -= lr * np.outer(dlogits, sdr.astype(np.float64))


# =============================================================================
# Level 2: Memory Systems
# =============================================================================

class EpisodicMemory:
    """
    Stores specific experiences as attractor states.
    Retrieval: partial cue → convergence to nearest attractor.
    """

    def __init__(self, dim: int = 2000, sparsity: float = 0.05, capacity: int = 1000, seed: int = 42):
        self.dim = dim
        self.sparsity = sparsity
        self.attractor = AttractorNetwork(dim, capacity, seed)
        self.seed = seed

    def store_experience(self, state: np.ndarray, action: np.ndarray, outcome: np.ndarray):
        """Store an episode as binding of state, action, outcome."""
        # Episode = state ⊗ action ⊗ outcome
        episode = state.copy()
        episode = np.logical_xor(episode, action).astype(np.int8)
        episode = np.logical_xor(episode, outcome).astype(np.int8)
        self.attractor.store(episode)

    def retrieve(self, cue: np.ndarray, max_steps: int = 50) -> np.ndarray:
        """Retrieve episode from cue."""
        return self.attractor.recall(cue, max_steps)


class SemanticMemory:
    """
    Stores facts as binding: subject ⊗ predicate → object.
    Retrieval: (subject ⊗ predicate) ⊘ subject = object.
    """

    def __init__(self, dim: int = 2000, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.facts: Dict[int, np.ndarray] = {}  # hash → bound fact

    def store_fact(self, subject: np.ndarray, predicate: np.ndarray, obj: np.ndarray):
        """Store fact: subject ⊗ predicate → object."""
        fact_key = np.logical_xor(subject, predicate).astype(np.int8)
        self.facts[hash(fact_key.tobytes())] = obj.copy()

    def query(self, subject: np.ndarray, predicate: np.ndarray) -> Optional[np.ndarray]:
        """Query: (subject ⊗ predicate) → object."""
        fact_key = np.logical_xor(subject, predicate).astype(np.int8)
        key_hash = hash(fact_key.tobytes())
        if key_hash in self.facts:
            return self.facts[key_hash]
        return None


class ProceduralMemory:
    """
    Stores action sequences: state ⊗ action → next_state.
    Serves as world model.
    """

    def __init__(self, dim: int = 2000, n_actions: int = 10, seed: int = 42):
        self.dim = dim
        self.n_actions = n_actions
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # World model: P(s'|s, a) = W_pred · [s; a]
        self.W_pred = np.zeros((dim, dim + n_actions))

    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """Predict next state given current state and action."""
        # One-hot action
        action_vec = np.zeros(self.n_actions)
        action_vec[action] = 1.0

        # Concatenate state and action
        sa = np.concatenate([state.astype(np.float64), action_vec])

        # Predict
        prediction = self.W_pred @ sa
        return (prediction > 0.5).astype(np.int8)

    def train_step(self, state: np.ndarray, action: int, next_state: np.ndarray, lr: float = 0.1):
        """Train world model on single transition."""
        # One-hot action
        action_vec = np.zeros(self.n_actions)
        action_vec[action] = 1.0

        # Concatenate
        sa = np.concatenate([state.astype(np.float64), action_vec])

        # Predict
        prediction = self.W_pred @ sa
        error = next_state.astype(np.float64) - prediction

        # Update
        delta = lr * np.outer(error, sa)
        self.W_pred += delta


# =============================================================================
# Level 3: Planning & Meta-Learning
# =============================================================================

class PlanningModule:
    """
    Uses world model for internal simulation.
    Process: start_state → predict_state_1 → ... → predict_state_k
    Evaluates predicted states against goals.
    """

    def __init__(self, procedural_memory: ProceduralMemory, horizon: int = 10, seed: int = 42):
        self.memory = procedural_memory
        self.horizon = horizon
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def plan(self, current_state: np.ndarray, goal: np.ndarray, n_actions: int = 10) -> List[int]:
        """Plan action sequence to reach goal state."""
        best_sequence = []
        best_score = -1

        # Try multiple random action sequences
        for _ in range(100):
            state = current_state.copy()
            sequence = []

            for _ in range(self.horizon):
                # Sample random action
                action = self.rng.integers(0, n_actions)
                sequence.append(action)

                # Predict next state
                state = self.memory.predict(state, action)

            # Evaluate
            score = self._similarity(state, goal)
            if score > best_score:
                best_score = score
                best_sequence = sequence

        return best_sequence

    def _similarity(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Cosine similarity between two SDRs."""
        dot = np.sum(s1 * s2)
        norms = np.sum(s1) * np.sum(s2)
        if norms == 0:
            return 0.0
        return dot / norms


class MetaLearner:
    """
    Adjusts learning rates based on prediction error.
    High error → increase plasticity.
    Low error → decrease plasticity.
    """

    def __init__(self, target_error: float = 0.1):
        self.target_error = target_error
        self.base_lr = 0.01

    def get_lr(self, prediction_error: float) -> float:
        """Compute adaptive learning rate."""
        return self.base_lr * np.tanh(abs(prediction_error) / self.target_error)


# =============================================================================
# Full HAPN Network
# =============================================================================

class HAPNNetwork:
    """
    Complete Hierarchical Attractor Predictive Network.
    Integrates all components into a unified system.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 2000,
        n_classes: int = 10,
        sparsity: float = 0.05,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.sparsity = sparsity
        self.seed = seed

        # Level 1: Perception & Action
        self.vision_encoder = VisionEncoder(input_dim, hidden_dim, sparsity, seed)
        self.action_decoder = ActionDecoder(hidden_dim, n_classes, seed)

        # Level 2: Memory Systems
        self.episodic_memory = EpisodicMemory(hidden_dim, sparsity, seed=seed)
        self.semantic_memory = SemanticMemory(hidden_dim, seed=seed)
        self.procedural_memory = ProceduralMemory(hidden_dim, n_classes, seed=seed)

        # Level 3: Planning & Meta-Learning
        self.planning = PlanningModule(self.procedural_memory, horizon=5, seed=seed)
        self.meta_learner = MetaLearner()

        # Statistics
        self.stats = {
            'n_trained': 0,
            'n_stored': 0,
            'n_plans': 0,
        }

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to SDR."""
        return self.vision_encoder.encode(x)

    def predict(self, x: np.ndarray) -> int:
        """Predict class label for input."""
        sdr = self.encode(x)
        return self.action_decoder.decode(sdr)

    def train_unsupervised(self, X: np.ndarray, epochs: int = 5):
        """Unsupervised pretraining of vision encoder."""
        print(f"Unsupervised pretraining ({epochs} epochs)...")
        n_samples = len(X)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            total_similarity = 0.0

            for i in indices:
                sdr = self.encode(X[i])
                # Store as attractor
                self.episodic_memory.attractor.store(sdr)
                total_similarity += self.episodic_memory.attractor.stored_patterns[-1].mean()

            self.stats['n_stored'] = self.episodic_memory.attractor.n_stored
            print(f"  Epoch {epoch + 1}: stored {self.stats['n_stored']} patterns")

    def train_supervised(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        """Train action decoder on labeled data."""
        print(f"Supervised training ({epochs} epochs)...")
        n_samples = len(X)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            correct = 0

            for i in indices:
                sdr = self.encode(X[i])
                label = int(y[i])

                # Train action decoder
                self.action_decoder.train_step(sdr, label)

                # Check accuracy
                pred = self.action_decoder.decode(sdr)
                if pred == label:
                    correct += 1

            if (epoch + 1) % 10 == 0:
                acc = correct / n_samples
                print(f"  Epoch {epoch + 1}: acc={acc:.2%}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate accuracy."""
        correct = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            if pred == y[i]:
                correct += 1
        accuracy = correct / len(X)
        return {"accuracy": accuracy, "correct": correct, "total": len(X)}

    def plan_action_sequence(self, current_state: np.ndarray, goal: np.ndarray) -> List[int]:
        """Plan action sequence to reach goal."""
        self.stats['n_plans'] += 1
        return self.planning.plan(current_state, goal, self.n_classes)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HAPN — Hierarchical Attractor Predictive Network")
    print("=" * 60)

    # Create network
    net = HAPNNetwork(
        input_dim=784,
        hidden_dim=2000,
        n_classes=10,
        sparsity=0.05,
        seed=42,
    )
    print(f"Architecture: {net.input_dim} -> {net.hidden_dim} -> {net.n_classes}")
    print(f"Sparsity: {net.sparsity} ({int(net.hidden_dim * net.sparsity)} active units)")

    # Generate simple synthetic data
    rng = np.random.default_rng(42)
    X = rng.random((500, 784))
    y = rng.integers(0, 10, 500)

    # Train
    net.train_unsupervised(X, epochs=3)
    net.train_supervised(X, y, epochs=30)

    # Evaluate on held-out test data
    X_test = rng.random((100, 784))
    y_test = rng.integers(0, 10, 100)
    print("\nEvaluating (held-out)...")
    results = net.evaluate(X_test, y_test)
    print(f"  Accuracy: {results['accuracy']:.2%}")

    print("\nHAPN test complete.")
