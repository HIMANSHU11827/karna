"""
RAIE — Real-time AI Everything: Unified Neural Architecture
=============================================================

Merges perception (PEN) and memory (HAPN) into one system:

Architecture:
- Sparse Random Binary Init (SRBI): weight init with sparse +1/-1 values
- k-Winners-Take-All (KWTA): sparse activation, competitive learning
- Sparse Hebbian with Decay (SHD): local learning rule, no backprop
- Iterative Attractor Convergence (IAC): memory retrieval via dynamics
- XOR Binding: semantic memory composition
- Complementary Learning: fast episodic + slow semantic systems

Goal: MNIST >95% accuracy through sparse, dynamic representations.
"""

import numpy as np
from typing import Optional


# ============================================================
# SRBI: Sparse Random Binary Initialization
# ============================================================

def srbi_init(rows: int, cols: int, sparsity: float = 0.1, seed: int = 42) -> np.ndarray:
    """Sparse Random Binary Initialization.

    Creates a binary matrix where non-zero entries are +1 or -1,
    with specified sparsity (fraction of non-zero entries).

    Args:
        rows: Number of rows
        cols: Number of columns
        sparsity: Fraction of non-zero entries (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        Binary matrix of shape (rows, cols) with values in {+1, -1, 0}
    """
    rng = np.random.default_rng(seed)
    # Create mask of non-zero entries
    mask = rng.random((rows, cols)) < sparsity
    # Assign +1 or -1 to non-zero entries
    signs = rng.choice([-1.0, 1.0], size=(rows, cols))
    return signs * mask


# ============================================================
# SHD: Sparse Hebbian with Decay
# ============================================================

def shd_update(
    weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    decay: float = 0.001,
) -> np.ndarray:
    """Sparse Hebbian with Decay update rule.

    Δw_ij = η * y_j * x_i - λ * w_ij

    Combines Hebbian strengthening (correlated activity) with
    exponential decay (prevents unbounded growth, maintains sparsity).

    Args:
        weights: Current weight matrix (input_dim, output_dim)
        x: Input vector (input_dim,)
        y: Output vector (output_dim,)
        learning_rate: Hebbian learning rate (η)
        decay: Decay rate (λ)

    Returns:
        Updated weight matrix
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    # Hebbian term: outer product of input and output
    hebbian = np.outer(x, y)
    # Decay term: shrink existing weights
    decay_term = decay * weights
    # Update
    new_weights = weights + learning_rate * hebbian - decay_term
    # Sparsify: keep only top-k weights per column (maintain sparsity)
    k = max(1, int(new_weights.shape[0] * 0.1))
    for col in range(new_weights.shape[1]):
        col_vals = np.abs(new_weights[:, col])
        threshold = np.sort(col_vals)[-k] if k < len(col_vals) else 0
        new_weights[:, col] = np.where(col_vals >= threshold, new_weights[:, col], 0)
    return new_weights


# ============================================================
# KWTA: k-Winners-Take-All
# ============================================================

def kwta(x: np.ndarray, k: int = 10) -> np.ndarray:
    """k-Winners-Take-All activation.

    Activates exactly k neurons with highest input values.
    All other neurons are suppressed to zero.

    Args:
        x: Input vector or matrix (batch_size, dim)
        k: Number of winners to select

    Returns:
        Sparse activation with exactly k non-zero values per row
    """
    if x.ndim == 1:
        result = np.zeros_like(x)
        if k >= len(x):
            return np.maximum(x, 0)
        top_k_idx = np.argsort(x)[-k:]
        result[top_k_idx] = np.maximum(x[top_k_idx], 0)
        return result
    else:
        # Batch
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            if k >= x.shape[1]:
                result[i] = np.maximum(x[i], 0)
            else:
                top_k_idx = np.argsort(x[i])[-k:]
                result[i, top_k_idx] = np.maximum(x[i, top_k_idx], 0)
        return result


# ============================================================
# IAC: Iterative Attractor Convergence
# ============================================================

def iterative_attractor(
    cue: np.ndarray,
    memory: np.ndarray,
    n_steps: int = 10,
    threshold: float = 0.0,
) -> np.ndarray:
    """Iterative Attractor Convergence for memory retrieval.

    Recalls a stored memory pattern by iteratively updating
    the cue through the memory matrix until convergence.

    Args:
        cue: Initial cue vector (dim,)
        memory: Memory matrix (dim, dim)
        n_steps: Number of iterations
        threshold: Activation threshold

    Returns:
        Converged pattern
    """
    state = cue.copy()
    for _ in range(n_steps):
        # Update: state = sign(memory @ state)
        activation = memory @ state
        state = np.where(activation > threshold, 1.0,
                        np.where(activation < -threshold, -1.0, 0.0))
    return state


# ============================================================
# XOR Binding for Semantic Memory
# ============================================================

def xor_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR binding for semantic composition.

    Binds two vectors via XOR for compositional memory.
    For binary vectors (+1/-1): same → -1, different → +1.
    Used to create associations between concepts.

    Args:
        a: First vector (+1/-1 values)
        b: Second vector (+1/-1 values)

    Returns:
        Bound vector (+1/-1 values)
    """
    return np.sign(-np.multiply(a, b))


def xor_unbind(bound: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR unbinding — recovers original vector from bound pair.

    Args:
        bound: Bound vector (+1/-1 values)
        b: Known vector (+1/-1 values)

    Returns:
        Recovered original vector (+1/-1 values)
    """
    return np.sign(-np.multiply(bound, b))


# ============================================================
# Complementary Learning System
# ============================================================

class ComplementaryMemory:
    """Dual-system complementary learning.

    Fast episodic system: rapid one-shot learning
    Slow semantic system: gradual consolidation

    Args:
        input_dim: Input dimension
        episodic_dim: Episodic memory dimension
        semantic_dim: Semantic memory dimension
    """

    def __init__(self, input_dim: int, episodic_dim: int, semantic_dim: int):
        self.input_dim = input_dim
        self.episodic_dim = episodic_dim
        self.semantic_dim = semantic_dim

        # Episodic: fast, sparse, high capacity
        self.episodic_weights = srbi_init(input_dim, episodic_dim, sparsity=0.05)

        # Semantic: slow, dense, consolidated
        self.semantic_weights = srbi_init(input_dim, semantic_dim, sparsity=0.1)

    def fast_learn(self, x: np.ndarray, learning_rate: float = 0.1) -> None:
        """Fast learning into episodic memory."""
        x = np.atleast_1d(x)
        # Sparse Hebbian update
        activation = kwta(x @ self.episodic_weights, k=max(1, self.episodic_dim // 20))
        self.episodic_weights = shd_update(
            self.episodic_weights, x, activation,
            learning_rate=learning_rate, decay=0.001
        )

    def slow_learn(self, x: np.ndarray, learning_rate: float = 0.01) -> None:
        """Slow learning into semantic memory (consolidation)."""
        x = np.atleast_1d(x)
        # Consolidated from episodic
        semantic_input = x @ self.episodic_weights
        activation = kwta(x @ self.semantic_weights, k=max(1, self.semantic_dim // 20))
        self.semantic_weights = shd_update(
            self.semantic_weights, x, activation,
            learning_rate=learning_rate, decay=0.0001
        )

    def fast_recall(self, cue: np.ndarray) -> np.ndarray:
        """Recall from episodic memory.

        Projects cue into episodic space, then retrieves.
        """
        cue = np.atleast_1d(cue)
        # Project to episodic space
        episodic_activation = kwta(cue @ self.episodic_weights, k=max(1, self.episodic_dim // 20))
        # Reconstruct back to input space
        reconstructed = episodic_activation @ self.episodic_weights.T
        return reconstructed

    def slow_recall(self, cue: np.ndarray) -> np.ndarray:
        """Recall from semantic memory.

        Projects cue into semantic space, then retrieves.
        """
        cue = np.atleast_1d(cue)
        # Project to semantic space
        semantic_activation = kwta(cue @ self.semantic_weights, k=max(1, self.semantic_dim // 20))
        # Reconstruct back to input space
        reconstructed = semantic_activation @ self.semantic_weights.T
        return reconstructed


# ============================================================
# RAIE Network
# ============================================================

class RAIENetwork:
    """Real-time AI Everything — Unified Network.

    Combines:
    - Multimodal encoders with KWTA
    - IAC-based memory
    - XOR binding for semantics
    - Complementary learning (episodic + semantic)
    - SHD learning rule throughout

    Args:
        input_dim: Input dimension (784 for MNIST)
        encoder_dims: Hidden layer dimensions
        memory_dim: Memory dimension
        output_dim: Number of classes
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int],
        memory_dim: int,
        output_dim: int,
    ):
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.memory_dim = memory_dim
        self.output_dim = output_dim

        # Encoder layers with SRBI initialization
        self.encoder_weights = []
        prev_dim = input_dim
        for dim in encoder_dims:
            self.encoder_weights.append(srbi_init(prev_dim, dim, sparsity=0.1))
            prev_dim = dim

        # Memory layer (IAC)
        self.memory_weights = srbi_init(encoder_dims[-1], memory_dim, sparsity=0.1)

        # Output layer
        self.output_weights = srbi_init(memory_dim, output_dim, sparsity=0.1)

        # Complementary memory system
        self.complementary = ComplementaryMemory(
            input_dim=input_dim,
            episodic_dim=memory_dim,
            semantic_dim=memory_dim,
        )

        # Biases
        self.encoder_biases = [np.zeros(d) for d in encoder_dims]
        self.memory_bias = np.zeros(memory_dim)
        self.output_bias = np.zeros(output_dim)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input through encoder layers with KWTA."""
        current = x
        for i, (W, b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            activation = current @ W + b
            current = kwta(activation, k=max(1, int(W.shape[1] * 0.1)))
        return current

    def memory(self, encoded: np.ndarray) -> np.ndarray:
        """Process through memory layer."""
        activation = encoded @ self.memory_weights + self.memory_bias
        return kwta(activation, k=max(1, int(self.memory_dim * 0.1)))

    def output(self, memory_out: np.ndarray) -> np.ndarray:
        """Compute output logits."""
        return memory_out @ self.output_weights + self.output_bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Normalize input
        x = x - 0.5  # Center around zero

        # Encode
        encoded = np.zeros((x.shape[0], self.encoder_dims[-1]))
        for i in range(x.shape[0]):
            encoded[i] = self.encode(x[i])

        # Memory
        memory_out = np.zeros((x.shape[0], self.memory_dim))
        for i in range(x.shape[0]):
            memory_out[i] = self.memory(encoded[i])

        # Output
        logits = self.output(memory_out)

        return logits if logits.shape[0] > 1 else logits[0]

    def learn(self, x: np.ndarray, target: Optional[np.ndarray] = None) -> None:
        """Learn from input using SHD rule."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            if target is not None:
                target = np.atleast_1d(target)

        for i in range(x.shape[0]):
            xi = x[i] - 0.5  # Center
            ti = target[i] if target is not None else None

            # Forward pass
            encoded = self.encode(xi)
            memory_out = self.memory(encoded)
            logits = self.output(memory_out)

            # SHD update for output layer
            if ti is not None:
                self.output_weights = shd_update(
                    self.output_weights, memory_out, ti,
                    learning_rate=0.01, decay=0.001
                )

            # SHD update for memory layer
            self.memory_weights = shd_update(
                self.memory_weights, encoded, memory_out,
                learning_rate=0.01, decay=0.001
            )

            # SHD update for encoder layers (backprop-free)
            prev_output = xi
            for j, W in enumerate(self.encoder_weights):
                activation = prev_output @ W + self.encoder_biases[j]
                sparse_act = kwta(activation, k=max(1, int(W.shape[1] * 0.1)))
                self.encoder_weights[j] = shd_update(
                    self.encoder_weights[j], prev_output, sparse_act,
                    learning_rate=0.01, decay=0.001
                )
                prev_output = sparse_act

            # Complementary learning
            self.complementary.fast_learn(xi)

    def train_epochs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        verbose: bool = True,
    ) -> list[float]:
        """Train for multiple epochs."""
        losses = []
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(len(X))
            epoch_loss = 0.0

            for idx in indices:
                x = X[idx]
                # One-hot encode target
                target = np.zeros(self.output_dim)
                target[y[idx]] = 1.0

                # Forward
                logits = self.forward(x)
                loss = np.mean((logits - target) ** 2)
                epoch_loss += loss

                # Learn
                self.learn(x, target)

            avg_loss = epoch_loss / len(X)
            losses.append(avg_loss)

            if verbose:
                print(f"  Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            logits = self.forward(X[i])
            predictions[i] = np.argmax(logits)
        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate accuracy and loss."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)

        # Compute loss
        losses = []
        for i in range(len(X)):
            logits = self.forward(X[i])
            target = np.zeros(self.output_dim)
            target[y[i]] = 1.0
            losses.append(np.mean((logits - target) ** 2))

        return {"accuracy": accuracy, "loss": np.mean(losses)}


# ============================================================
# Demo
# ============================================================

def main():
    """Run RAIE demo on MNIST."""
    print("=" * 60)
    print("  RAIE Network — MNIST Training")
    print("=" * 60)
    print()

    from mnist_loader import load_mnist

    X_train, y_train, X_test, y_test = load_mnist()

    # Subset for speed
    n_train = 5000
    n_test = 1000
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_test]
    y_test = y_test[:n_test]

    print(f"Training: {n_train} samples, Test: {n_test} samples")

    # Create network
    net = RAIENetwork(
        input_dim=784,
        encoder_dims=[512, 256],
        memory_dim=128,
        output_dim=10,
    )

    print(f"Architecture: 784 -> 512 -> 256 -> 128 -> 10")
    print()

    # Train
    losses = net.train_epochs(X_train, y_train, epochs=20, verbose=True)

    # Evaluate
    results = net.evaluate(X_test, y_test)

    print()
    print("=" * 60)
    print("  Final Results")
    print("=" * 60)
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Test Loss: {results['loss']:.4f}")
    print(f"  Loss History: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print()


if __name__ == "__main__":
    main()
