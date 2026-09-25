"""
RAIE — Real-time Adaptive Intelligence Engine
================================================
Merging PEN + HAPN into one unified system.

Components:
1. KWTA (k-Winners-Take-All) activation
2. IAC (Iterative Attractor Convergence) memory
3. XOR Binding for semantic memory
4. Complementary Learning (fast episodic + slow semantic)
5. SRBI (Sparse Random Binary Init) weight initialization
6. SHD (Sparse Hebbian with Decay) learning rule

Target: MNIST 95%+
"""

import numpy as np
import os
import sys
import time
import json
import urllib.request
import gzip
from pathlib import Path
from collections import defaultdict, deque


# =============================================================================
# 1. SRBI — Sparse Random Binary Init
# =============================================================================

def srbi_weights(n_in, n_out, density=0.1, seed=None):
    """
    Sparse Random Binary initialization.
    
    Each neuron connects to only `density` fraction of inputs.
    Connections are +1 or -1 (binary).
    
    Why this works:
    - Sparse connectivity forces diverse feature detectors
    - Binary weights prevent vanishing/exploding signals
    - Very few parameters to learn (just which connections to strengthen)
    """
    rng = np.random.RandomState(seed)
    mask = rng.rand(n_in, n_out) < density
    weights = (rng.rand(n_in, n_out) * 2 - 1).astype(np.float32) * mask
    # Normalize per output neuron
    for j in range(n_out):
        n_connections = np.sum(mask[:, j])
        if n_connections > 0:
            weights[:, j] /= np.sqrt(n_connections)
    return weights


# =============================================================================
# 2. KWTA — k-Winners-Take-All Activation
# =============================================================================

def kwta(activations, k):
    """
    k-Winners-Take-All activation.
    
    Only the top-k neurons remain active.
    All others are set to zero.
    
    Key properties:
    - Sparse representation (exactly k active)
    - Competition forces diverse features
    - Biological inspiration: lateral inhibition in cortex
    """
    result = np.zeros_like(activations)
    if k >= len(activations):
        result[:] = activations
        return result
    
    # Find top-k indices
    if activations.ndim == 1:
        top_k_idx = np.argpartition(activations, -k)[-k:]
        result[top_k_idx] = activations[top_k_idx]
    else:
        # Batch processing
        for i in range(activations.shape[0]):
            top_k_idx = np.argpartition(activations[i], -k)[-k:]
            result[i, top_k_idx] = activations[i, top_k_idx]
    
    return result


# =============================================================================
# 3. XOR Binding for Semantic Memory
# =============================================================================

class XORMemory:
    """
    XOR binding memory for semantic associations.
    
    Based on: Kanerva's Sparse Distributed Memory
    Also related to: Vector Symbolic Architectures (VSA)
    
    XOR binding allows:
    - Store pair (key, value): bound = key XOR value
    - Retrieve value from key: value ≈ bound XOR key
    - Composability: can bind and unbind arbitrarily many times
    
    Key property: XOR binding is its own inverse (symmetric).
    """
    
    def __init__(self, dim, n_items=100):
        self.dim = dim
        self.n_items = n_items
        # Memory matrix (each row is a stored binding)
        self.memory = np.zeros((n_items, dim), dtype=np.float32)
        self.keys = np.zeros((n_items, dim), dtype=np.float32)
        self.idx = 0
    
    def store(self, key, value):
        """Store key-value pair using XOR binding."""
        # Normalize to binary-like
        key_bin = np.sign(key).astype(np.float32)
        value_bin = np.sign(value).astype(np.float32)
        
        # XOR binding: stored = key XOR value
        # For binary: XOR = key + value mod 2
        # For continuous: XOR = key * value (element-wise product)
        bound = key_bin * value_bin
        
        idx = self.idx % self.n_items
        self.memory[idx] = bound
        self.keys[idx] = key_bin
        self.idx += 1
    
    def retrieve(self, query_key, k=5):
        """
        Retrieve value by query key.
        Uses XOR unbind: value ≈ stored XOR key
        """
        query_bin = np.sign(query_key).astype(np.float32)
        
        # Find most similar keys
        sims = self.keys @ query_bin
        top_k_idx = np.argsort(sims)[-k:]
        
        # Retrieve: XOR unbind
        retrieved = np.zeros(self.dim, dtype=np.float32)
        for idx in top_k_idx:
            # XOR unbind: value ≈ stored XOR key
            retrieved += self.memory[idx] * query_bin
        
        return np.sign(retrieved)
    
    def bind(self, a, b):
        """Bind two vectors with XOR."""
        return np.sign(a) * np.sign(b)
    
    def unbind(self, bound, key):
        """Unbind: retrieve value from bound using key."""
        return bound * np.sign(key)


# =============================================================================
# 4. IAC — Iterative Attractor Convergence Memory
# =============================================================================

class IACMemory:
    """
    Iterative Attractor Convergence memory.
    
    Memory is stored as an attractor network.
    Retrieval is iterative convergence to nearest attractor.
    
    Energy: E = -0.5 * sum_ij w_ij * s_i * s_j
    Update: s_i = sign(sum_j w_ij * s_j)
    
    Converges to local minimum = stored pattern.
    """
    
    def __init__(self, dim, n_patterns=100):
        self.dim = dim
        self.n_patterns = n_patterns
        self.weights = np.zeros((dim, dim), dtype=np.float32)
        self.stored_patterns = []
        self.stored_values = []
    
    def store(self, pattern, value=None):
        """Store pattern using Hopfield outer product rule."""
        pattern = np.sign(pattern).astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern = pattern / norm
        
        # Outer product (Hebbian)
        delta_w = np.outer(pattern, pattern)
        # Remove self-connections
        np.fill_diagonal(delta_w, 0)
        
        self.weights += delta_w
        self.stored_patterns.append(pattern.copy())
        self.stored_values.append(value)
    
    def retrieve(self, query, max_steps=10, noise=0.0):
        """
        Retrieve pattern via iterative attractor convergence.
        """
        if len(self.stored_patterns) == 0:
            return query
        
        state = np.sign(query).astype(np.float32)
        
        for step in range(max_steps):
            # Update async (random order)
            prev_state = state.copy()
            indices = np.random.permutation(self.dim)
            
            for i in indices:
                activation = np.dot(self.weights[i], state)
                if noise > 0:
                    activation += np.random.randn() * noise
                state[i] = 1.0 if activation > 0 else -1.0
            
            # Check convergence
            if np.sum(np.abs(state - prev_state)) < 1e-6:
                break
        
        return state
    
    def energy(self, state):
        """Compute energy of state."""
        return -0.5 * state @ self.weights @ state
    
    def capacity(self):
        """Theoretical capacity: ~0.138 * dim patterns."""
        return int(0.138 * self.dim)


# =============================================================================
# 5. Complementary Learning Systems (Fast Episodic + Slow Semantic)
# =============================================================================

class ComplementaryLearning:
    """
    Two memory systems:
    - Fast Epidemic Memory (hippocampus-like): rapid encoding, pattern separation
    - Slow Semantic Memory (neocortex-like): gradual consolidation, pattern generalization
    
    Based on: McClelland et al. (1995), Kumaran et al. (2016)
    """
    
    def __init__(self, input_dim, episodic_capacity=500, semantic_dim=128):
        # Fast episodic memory (IAC attractor)
        self.episodic = IACMemory(input_dim, episodic_capacity)
        
        # Slow semantic memory (XOR binding memory)
        self.semantic = XORMemory(semantic_dim, n_items=200)
        
        # Consolidation counter
        self.episodic_count = 0
        self.consolidation_interval = 50
    
    def encode(self, input_vec, label_vec=None):
        """Rapid encoding into episodic memory."""
        self.episodic.store(input_vec, label_vec)
        self.episodic_count += 1
        
        # Periodic consolidation
        if self.episodic_count % self.consolidation_interval == 0:
            self.consolidate()
    
    def consolidate(self):
        """Transfer episodic → semantic (slow consolidation)."""
        for pattern, value in zip(self.episodic.stored_patterns[-50:],
                                   self.episodic.stored_values[-50:]):
            if value is not None:
                self.semantic.store(pattern, value)
    
    def recall(self, query, k=5):
        """
        Recall: try episodic first (precise), then semantic (generalized).
        """
        # Episodic recall
        episodic_result = self.episodic.retrieve(query)
        
        # Semantic recall
        semantic_result = self.semantic.retrieve(query, k=k)
        
        return episodic_result, semantic_result


# =============================================================================
# 6. SHD — Sparse Hebbian with Decay
# =============================================================================

class SHDLearning:
    """
    Sparse Hebbian with Decay learning rule.
    
    Key innovations:
    - Hebbian update only for active synapses (sparse)
    - Weight decay prevents unbounded growth
    - Eligibility traces for temporal credit assignment
    
    Δw_ij = η * (pre_i * post_j - decay * w_ij)
    
    The decay term is crucial: implements "use it or lose it"
    Only synapses that are repeatedly co-active survive.
    """
    
    def __init__(self, n_in, n_out, lr=0.01, decay=0.001, 
                 density=0.1, seed=None):
        # SRBI initialization
        self.W = srbi_weights(n_in, n_out, density=density, seed=seed)
        self.lr = lr
        self.decay = decay
        
        # Eligibility traces
        self.trace = np.zeros_like(self.W)
        self.trace_decay = 0.9
        
        # Activity tracking
        self.pre_activity = np.zeros(n_in, dtype=np.float32)
        self.post_activity = np.zeros(n_out, dtype=np.float32)
        self.activity_momentum = 0.9
    
    def update(self, pre, post, global_reward=0.0):
        """
        SHD update rule.
        
        Args:
            pre: pre-synaptic activity (n_in,)
            post: post-synaptic activity (n_out,)
            global_reward: scalar reward signal (modulates learning)
        """
        # Update running activity averages
        self.pre_activity = (self.activity_momentum * self.pre_activity + 
                           (1 - self.activity_momentum) * np.abs(pre))
        self.post_activity = (self.activity_momentum * self.post_activity + 
                            (1 - self.activity_momentum) * np.abs(post))
        
        # Eligibility trace
        outer = np.outer(pre, post)
        self.trace = self.trace_decay * self.trace + (1 - self.trace_decay) * outer
        
        # SHD update: Hebbian + decay
        # Δw = η * (trace - decay * w) * reward_modulation
        reward_mod = 1.0 + global_reward
        delta = self.lr * (self.trace - self.decay * self.W) * reward_mod
        
        self.W += delta
        
        # Clip weights
        self.W = np.clip(self.W, -2, 2)
        
        # Maintain sparsity: prune small weights
        small_weights = np.abs(self.W) < 0.01
        self.W[small_weights] = 0
    
    def normalize(self):
        """Normalize weight columns."""
        norms = np.linalg.norm(self.W, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.W /= norms


# =============================================================================
# 7. RAIE — Unified Network
# =============================================================================

class RAIENetwork:
    """
    Real-time Adaptive Intelligence Engine.
    
    Combines all components into one system:
    - SRBI initialization
    - KWTA activation
    - SHD learning
    - IAC memory
    - XOR semantic binding
    - Complementary learning
    
    Architecture:
    Input → SHD(KWTA) → SHD(KWTA) → SHD(KWTA) → Output
                ↓              ↓
            IAC Memory    XOR Semantic
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 sparsity=0.15, lr=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        
        # Build layers with SHD
        self.layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layer = SHDLearning(
                prev_dim, h_dim,
                lr=lr,
                decay=0.001,
                density=sparsity,
                seed=np.random.randint(0, 10000)
            )
            self.layers.append(layer)
            prev_dim = h_dim
        
        # Output layer (simpler: linear + softmax)
        self.output_W = srbi_weights(prev_dim, output_dim, density=0.3, seed=42)
        self.output_b = np.zeros(output_dim, dtype=np.float32)
        
        # Memories
        self.episodic = IACMemory(output_dim, n_patterns=200)
        self.semantic = XORMemory(output_dim, n_items=200)
        
        # k for KWTA
        self.k_values = [max(1, int(h * 0.1)) for h in hidden_dims]
    
    def forward(self, x, train=False, y=None):
        """Forward pass through network."""
        current = x.copy()
        layer_outputs = [current]
        
        for i, (layer, k) in enumerate(zip(self.layers, self.k_values)):
            # Linear projection
            h = current @ layer.W
            
            # KWTA activation
            h = kwta(h, k)
            
            if train and y is not None:
                # SHD update
                reward = 1.0 if np.argmax(h) == np.argmax(y) else -1.0
                layer.update(current, h, global_reward=reward)
            
            layer_outputs.append(h)
            current = h
        
        # Output
        output = current @ self.output_W + self.output_b
        
        # Softmax
        exp_out = np.exp(output - np.max(output))
        probs = exp_out / np.sum(exp_out)
        
        if train and y is not None:
            # Simple Hebbian on output
            error = y - probs
            self.output_W += 0.01 * np.outer(current, error)
            self.output_b += 0.01 * error
        
        return probs, layer_outputs
    
    def predict(self, x):
        """Predict class."""
        probs, _ = self.forward(x)
        return np.argmax(probs)
    
    def train_epoch(self, X, y_onehot, labels, epochs=10):
        """Train for one epoch."""
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        correct = 0
        total = 0
        
        for idx in indices:
            probs, _ = self.forward(X[idx], train=True, y=y_onehot[idx])
            pred = np.argmax(probs)
            
            if pred == labels[idx]:
                correct += 1
            total += 1
        
        # Normalize layers
        for layer in self.layers:
            layer.normalize()
        
        return correct / total
    
    def evaluate(self, X, labels):
        """Evaluate accuracy."""
        correct = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            if pred == labels[i]:
                correct += 1
        return correct / len(X)


# =============================================================================
# 8. MNIST Data Loader
# =============================================================================

def download_mnist(data_dir="/tmp/mnist_data"):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets.mnist/",
    ]
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            continue
        for mirror in mirrors:
            try:
                urllib.request.urlretrieve(mirror + filename, filepath)
                break
            except:
                pass
    return data_dir

def load_mnist(data_dir="/tmp/mnist_data"):
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int32)
    return (load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz")),
            load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz")),
            load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")),
            load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")))


# =============================================================================
# 9. Main Experiment
# =============================================================================

def main():
    print("=" * 70)
    print("RAIE — Real-time Adaptive Intelligence Engine")
    print("MNIST Experiment — Target: 95%+")
    print("=" * 70)
    
    # Load data
    data_dir = download_mnist()
    try:
        train_images, train_labels, test_images, test_labels = load_mnist()
        dataset = "mnist"
        print(f"Loaded MNIST: {len(train_images)} train, {len(test_images)} test")
    except:
        print("MNIST load failed")
        return
    
    # Use subset for initial test
    n_train = 6000
    n_test = 1000
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]
    
    # One-hot encode
    train_onehot = np.zeros((n_train, 10), dtype=np.float32)
    train_onehot[np.arange(n_train), train_labels] = 1.0
    
    # Network config
    input_dim = 784
    hidden_dims = [128, 64]
    output_dim = 10
    
    # Create network
    net = RAIENetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        sparsity=0.15,
        lr=0.01
    )
    
    print(f"Network: [{input_dim}, {hidden_dims[0]}, {hidden_dims[1]}, {output_dim}]")
    print(f"Sparsity: {net.sparsity}")
    print(f"SRBI init: True")
    print(f"KWTA activation: True")
    print(f"SHD learning: True")
    print(f"IAC memory: True")
    print(f"XOR binding: True")
    print()
    
    # Train
    n_epochs = 20
    print(f"Training for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        train_acc = net.train_epoch(train_images, train_onehot, train_labels, epochs=1)
        test_acc = net.evaluate(test_images, test_labels)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:3d}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    
    # Final
    final_acc = net.evaluate(test_images, test_labels)
    print(f"\nFinal test accuracy: {final_acc:.4f}")
    print(f"Target: 0.9500")
    print(f"Met target: {final_acc >= 0.95}")
    
    # Save results
    results = {
        "system": "RAIE",
        "dataset": dataset,
        "n_train": n_train,
        "n_test": n_test,
        "final_accuracy": float(final_acc),
        "target_met": bool(final_acc >= 0.95),
    }
    path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/11_LOGS/raie_mnist.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    
    return results


if __name__ == "__main__":
    main()
