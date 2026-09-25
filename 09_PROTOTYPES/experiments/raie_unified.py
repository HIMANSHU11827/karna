"""AGI Research Lab - RAIE: Real-time Adaptive Intelligence Engine
=================================================================

Unified architecture merging PEN + HAPN into one system.

Components:
1. KWTA (k-Winners-Take-All) activation
2. Iterative Attractor Convergence (IAC) memory
3. XOR binding for semantic memory
4. Complementary Learning (fast episodic + slow semantic)
5. SRBI (Sparse Random Binary Init) initialization
6. SHD (Sparse Hebbian with Decay) learning rule

Target: MNIST 95%+
"""

import numpy as np
import os
import sys
import time
import json
import hashlib
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional, Tuple


# =============================================================================
# 1. SRBI - Sparse Random Binary Initialization
# =============================================================================

class SRBI:
    """
    Sparse Random Binary Initialization.
    
    Instead of dense Gaussian init, uses sparse binary patterns:
    - Each neuron connects to only k inputs (sparse connectivity)
    - Weights are +1/-1 (binary)
    - Biological plausibility: cortical synapses are sparse and binary-like
    - Reduces initial interference between features
    """
    
    @staticmethod
    def init_weights(n_out, n_in, sparsity=0.1, binary=True):
        """Initialize sparse random binary weights."""
        W = np.zeros((n_out, n_in), dtype=np.float32)
        k = max(1, int(n_in * sparsity))
        
        for i in range(n_out):
            # Random k indices
            indices = np.random.choice(n_in, size=k, replace=False)
            if binary:
                vals = np.random.choice([-1, 1], size=k).astype(np.float32)
            else:
                vals = np.random.randn(k).astype(np.float32)
            W[i, indices] = vals
        
        return W
    
    @staticmethod
    def init_adaptive_thresholds(n_neurons, target_sparsity=0.1):
        """Initialize adaptive thresholds for KWTA."""
        # Set so that approximately target_sparsity neurons are active
        # Start with low thresholds, let them adapt
        return np.zeros(n_neurons, dtype=np.float32)


# =============================================================================
# 2. KWTA - k-Winners-Take-All Activation
# =============================================================================

class KWTA:
    """
    k-Winners-Take-All activation.
    
    Replaces ReLU with competitive activation:
    - Only top-k neurons fire (sparse activity)
    - Adaptive thresholds (automatically adjust k per layer)
    - Biological: lateral inhibition in cortical columns
    
    Benefits over ReLU:
    - Sparse representations (less interference)
    - Energy efficient
    - More features can be learned
    - Naturally handles noise
    """
    
    def __init__(self, k_ratio=0.1, adaptive=True, bias=0.01):
        self.k_ratio = k_ratio
        self.adaptive = adaptive
        self.bias = bias
        self.thresholds = None
    
    def __call__(self, x, k=None):
        """
        Apply KWTA activation.
        
        Args:
            x: input array (batch, features) or (features,)
            k: number of winners (if None, use k_ratio * n_features)
        """
        orig_ndim = x.ndim
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_features = x.shape[-1]
        if k is None:
            k = max(1, int(n_features * self.k_ratio))
        
        # Add small bias to break ties
        x = x + self.bias * np.random.randn(*x.shape).astype(np.float32) * 0.01
        
        # Find top-k
        if k == 1:
            # Argmax (winner-takes-all)
            winners = np.argmax(x, axis=-1)
            out = np.zeros_like(x)
            for i in range(x.shape[0]):
                out[i, winners[i]] = x[i, winners[i]]
        else:
            # Top-k
            top_k_idx = np.argsort(x, axis=-1)[:, -k:]
            out = np.zeros_like(x)
            for i in range(x.shape[0]):
                out[i, top_k_idx[i]] = x[i, top_k_idx[i]]
        
        if orig_ndim == 1:
            out = out.reshape(-1)
        
        return out
    
    def adapt_thresholds(self, x, target_sparsity=0.1):
        """Adaptive threshold adjustment."""
        if self.thresholds is None:
            self.thresholds = np.zeros(x.shape[-1], dtype=np.float32)
        
        # Simple: adjust based on current activity
        active = (x > self.thresholds).astype(np.float32)
        current_sparsity = np.mean(active)
        
        # Increase threshold if too many active, decrease if too few
        if current_sparsity > target_sparsity:
            self.thresholds += 0.01
        else:
            self.thresholds -= 0.01


# =============================================================================
# 3. XOR Binding for Semantic Memory
# =============================================================================

class XORBinding:
    """
    XOR binding for semantic memory.
    
    Based on Vector Symbolic Architectures (VSA) / Hyperdimensional Computing.
    
    Operations:
    - Bind: XOR two vectors to create association
    - Unbind: XOR bound vector with one component to retrieve other
    - Bundle: Add vectors to create a set/sequence
    
    Properties:
    - XOR is its own inverse: a XOR b XOR b = a
    - XOR preserves distance (orthogonal inputs -> orthogonal outputs)
    - Efficient implementation for binary vectors
    """
    
    def __init__(self, dim=64):
        self.dim = dim
    
    def bind(self, a, b):
        """Bind two vectors via XOR (for binary) or multiplicative (for real)."""
        if np.issubdtype(a.dtype, np.integer) or np.all((a == 0) | (a == 1)):
            # Binary XOR
            return (a.astype(np.int32) ^ b.astype(np.int32)).astype(np.float32)
        else:
            # Real-valued: use element-wise multiplication with sign
            return a * b
    
    def unbind(self, bound, a):
        """Unbind: retrieve b from bound = a XOR b."""
        if np.issubdtype(bound.dtype, np.integer) or np.all((bound == 0) | (bound == 1)):
            return (bound.astype(np.int32) ^ a.astype(np.int32)).astype(np.float32)
        else:
            return bound * np.sign(a)
    
    def bundle(self, vectors):
        """Bundle multiple vectors via addition (with clipping)."""
        result = np.zeros(vectors[0].shape, dtype=np.float32)
        for v in vectors:
            result += v
        # Clip to prevent runaway
        result = np.clip(result, -1, 1)
        return result
    
    def random_vector(self, sparsity=0.5):
        """Generate a random binary vector."""
        v = (np.random.rand(self.dim) < sparsity).astype(np.float32)
        return v
    
    def similarity(self, a, b):
        """Cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# =============================================================================
# 4. Iterative Attractor Convergence (IAC) Memory
# =============================================================================

class IACMemory:
    """
    Iterative Attractor Convergence Memory.
    
    Based on Hopfield networks + modern dense associative memory.
    
    Operations:
    - Store: Hebbian outer product
    - Retrieve: Iterative energy minimization (attractor convergence)
    - Capacity: O(n) for n neurons (much better than classical Hopfield O(n/log n))
    
    Key features:
    - Iterative retrieval converges to stored patterns
    - Handles noisy/incomplete inputs (pattern completion)
    - Can store both images and labels
    - Online learning (store new patterns without retraining)
    """
    
    def __init__(self, dim, beta=1.0):
        self.dim = dim
        self.beta = beta
        self.stored_patterns = []
        self.stored_labels = []
        self.W = np.zeros((dim, dim), dtype=np.float32)
        self.n_stored = 0
    
    def store(self, pattern, label=None):
        """Store a pattern via Hebbian outer product."""
        # Normalize
        p = pattern.flatten().astype(np.float32)
        norm = np.linalg.norm(p)
        if norm > 0:
            p = p / norm
        
        # Hebbian update (outer product)
        self.W += np.outer(p, p)
        self.stored_patterns.append(p)
        self.stored_labels.append(label)
        self.n_stored += 1
        
        # Normalize weight matrix (prevent unbounded growth)
        self.W /= self.n_stored
    
    def retrieve(self, query, steps=5):
        """
        Retrieve stored pattern via iterative energy minimization.
        
        Uses modern Hopfield energy:
        E = -β * log(Σ exp(β * pattern_i^T * query))
        
        Converges to closest stored pattern (attractor dynamics).
        """
        # Normalize query
        q = query.flatten().astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        
        # Iterative convergence
        for _ in range(steps):
            # Compute similarities
            similarities = self.W @ q
            
            # Softmax (modern Hopfield)
            exp_sim = np.exp(self.beta * similarities - np.max(self.beta * similarities))
            weights = exp_sim / np.sum(exp_sim)
            
            # Update query (retrieval)
            q = self.W.T @ weights
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
        
        return q
    
    def retrieve_with_label(self, query, steps=5):
        """Retrieve pattern and associated label."""
        retrieved = self.retrieve(query, steps)
        
        # Find closest stored pattern
        best_sim = -1
        best_label = None
        for i, pattern in enumerate(self.stored_patterns):
            sim = np.dot(retrieved, pattern)
            if sim > best_sim:
                best_sim = sim
                best_label = self.stored_labels[i]
        
        return retrieved, best_label, best_sim
    
    def energy(self, state):
        """Compute energy of a state (should decrease during retrieval)."""
        return -0.5 * state @ self.W @ state


# =============================================================================
# 5. Complementary Learning Systems (Fast Episodic + Slow Semantic)
# =============================================================================

class ComplementaryLearning:
    """
    Complementary Learning Systems (CLS) theory.
    
    Based on: McClelland, McNaughton, O'Reilly (1995); Kumaran et al. (2016)
    
    Two systems:
    - Fast learning (Episodic): Hippocampus-like, stores specific experiences
    - Slow learning (Semantic): Neocortex-like, extracts statistical regularities
    
    Interaction:
    - Episodic stores new experiences immediately
    - Semantic learns from episodic replay (consolidation)
    - Semantic provides prior knowledge for new learning
    """
    
    def __init__(self, input_dim, episodic_capacity=1000, semantic_lr=0.001):
        self.input_dim = input_dim
        self.episodic_capacity = episodic_capacity
        self.semantic_lr = semantic_lr
        
        # Episodic system (fast, high capacity, specific)
        self.episodic = IACMemory(input_dim, beta=2.0)
        
        # Semantic system (slow, statistical, general)
        self.semantic_W = np.random.randn(input_dim, input_dim).astype(np.float32) * 0.01
        self.semantic_mean = np.zeros(input_dim, dtype=np.float32)
        self.semantic_var = np.ones(input_dim, dtype=np.float32)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=100)
        self.replay_interval = 10  # Consolidate every N experiences
    
    def experience(self, x, label=None, context=None):
        """
        Process a new experience.
        
        1. Store in episodic (immediate)
        2. Accumulate for semantic consolidation
        3. Periodically consolidate episodic -> semantic
        """
        x = x.flatten().astype(np.float32)
        
        # 1. Episodic storage
        self.episodic.store(x, label)
        
        # 2. Add to replay buffer
        self.replay_buffer.append({
            'pattern': x,
            'label': label,
            'context': context,
            'time': time.time()
        })
        
        # 3. Update running statistics (semantic)
        self.semantic_mean = 0.99 * self.semantic_mean + 0.01 * x
        self.semantic_var = 0.99 * self.semantic_var + 0.01 * (x - self.semantic_mean) ** 2
        
        # 4. Periodic consolidation
        if len(self.replay_buffer) % self.replay_interval == 0:
            self._consolidate()
    
    def _consolidate(self):
        """
        Consolidate episodic -> semantic.
        
        Replay recent experiences to update semantic weights.
        This is the "sleep" or "rest" phase of learning.
        """
        if len(self.replay_buffer) < 5:
            return
        
        # Sample from replay buffer
        n_replay = min(20, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), n_replay, replace=False)
        
        for idx in indices:
            experience = self.replay_buffer[idx]
            x = experience['pattern']
            
            # Hebbian update on semantic weights
            # PCA-like: learn covariance structure
            self.semantic_W += self.semantic_lr * (np.outer(x, x) - self.semantic_W)
            
            # Keep weights bounded
            self.semantic_W = np.clip(self.semantic_W, -1, 1)
    
    def recall(self, query, steps=5):
        """
        Recall using both systems.
        
        1. Episodic: find similar past experience
        2. Semantic: provide statistical prior
        3. Combine: use both for robust retrieval
        """
        query = query.flatten().astype(np.float32)
        
        # Episodic retrieval
        episodic_pattern, episodic_label, episodic_sim = \
            self.episodic.retrieve_with_label(query, steps)
        
        # Semantic retrieval (linear projection)
        semantic_pattern = self.semantic_W @ query
        semantic_pattern = np.clip(semantic_pattern, -1, 1)
        
        # Combine (weighted by episodic similarity)
        weight = min(episodic_sim, 1.0)
        combined = weight * episodic_pattern + (1 - weight) * semantic_pattern
        
        return {
            'combined': combined,
            'episodic': episodic_pattern,
            'semantic': semantic_pattern,
            'label': episodic_label,
            'similarity': episodic_sim
        }
    
    def get_stats(self):
        return {
            'episodic_stored': self.episodic.n_stored,
            'replay_buffer_size': len(self.replay_buffer),
            'semantic_weight_norm': float(np.linalg.norm(self.semantic_W)),
            'semantic_mean_norm': float(np.linalg.norm(self.semantic_mean)),
        }


# =============================================================================
# 6. SHD - Sparse Hebbian with Decay
# =============================================================================

class SHDLearning:
    """
    Sparse Hebbian with Decay learning rule.
    
    Features:
    - Sparse activation (only k neurons active)
    - Hebbian update for active neurons
    - Weight decay (prevents unbounded growth)
    - Eligibility traces (temporal credit assignment)
    
    Δw = η * pre * post - λ * w * (post²) - decay * w
    
    Biological basis:
    - Hebbian: neurons that fire together wire together
    - Decay: synapses weaken without use (forgetting)
    - Eligibility: marks synapses for future updates
    """
    
    def __init__(self, n_out, n_in, lr=0.01, decay=0.001, sparsity=0.1):
        self.n_out = n_out
        self.n_in = n_in
        self.lr = lr
        self.decay = decay
        self.sparsity = sparsity
        
        # SRBI initialization
        self.W = SRBI.init_weights(n_out, n_in, sparsity=sparsity, binary=False)
        
        # Eligibility traces
        self.eligibility = np.zeros_like(self.W)
        self.trace_decay = 0.9
        
        # KWTA activation
        self.kwta = KWTA(k_ratio=sparsity)
    
    def forward(self, x):
        """
        Forward pass with SHD activation.
        
        Returns sparse activation.
        """
        # Linear projection
        h = self.W @ x
        # KWTA (sparse activation)
        h = self.kwta(h)
        return h
    
    def learn(self, x, global_reward=0.0):
        """
        SHD learning rule.
        
        Args:
            x: input vector
            global_reward: neuromodulatory signal (dopamine-like)
        """
        # Forward pass
        h = self.forward(x)
        
        # Compute eligibility trace
        self.eligibility = (self.trace_decay * self.eligibility + 
                           np.outer(h, x))
        
        # Hebbian update with eligibility trace
        delta_hebbian = self.lr * self.eligibility
        
        # Weight decay (Oja-style stabilization)
        delta_decay = -self.decay * self.W * (h ** 2)[:, None]
        
        # Global modulation (reward-guided plasticity)
        if global_reward != 0:
            modulation = 1.0 + global_reward * 0.1
            delta_hebbian *= modulation
        
        # Apply updates
        self.W += delta_hebbian + delta_decay
        
        # Stabilization
        self.W = np.clip(self.W, -2, 2)
        
        # Maintain sparsity (prune small weights)
        small = np.abs(self.W) < 0.01
        self.W[small] *= 0.9
        
        return h
    
    def get_weight_stats(self):
        return {
            'mean': float(np.mean(self.W)),
            'std': float(np.std(self.W)),
            'sparsity': float(np.mean(self.W == 0)),
            'max': float(np.max(self.W)),
            'min': float(np.min(self.W)),
        }


# =============================================================================
# RAIE - Unified Architecture
# =============================================================================

class RAIE:
    """
    Real-time Adaptive Intelligence Engine.
    
    Unified architecture combining:
    1. KWTA activation
    2. IAC memory
    3. XOR binding
    4. Complementary Learning
    5. SRBI initialization
    6. SHD learning
    
    Architecture:
    Input -> SHD Layer 1 (KWTA) -> SHD Layer 2 (KWTA) -> Output
                ↓                      ↓
            IAC Memory 1          IAC Memory 2
                ↓                      ↓
            Complementary Learning System
                ↓
            XOR Binding for Semantics
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, sparsity=0.1):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.sparsity = sparsity
        
        # SHD layers
        self.layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layer = SHDLearning(prev_dim, h_dim, lr=0.01, sparsity=sparsity)
            self.layers.append(layer)
            prev_dim = h_dim
        
        # Output layer (simple linear + softmax)
        self.output_W = SRBI.init_weights(output_dim, prev_dim, sparsity=0.3, binary=False)
        self.output_b = np.zeros(output_dim, dtype=np.float32)
        
        # IAC memory at each hidden layer
        self.memories = [IACMemory(h_dim, beta=2.0) for h_dim in hidden_dims]
        
        # Complementary learning
        self.cls = ComplementaryLearning(input_dim)
        
        # XOR binding
        self.xor_bind = XORBinding(dim=input_dim)
        
        # Class prototypes (for classification)
        self.class_prototypes = {}
        self.class_counts = defaultdict(int)
    
    def forward(self, x):
        """Forward pass through SHD layers."""
        current = x.flatten().astype(np.float32)
        
        # Normalize input
        norm = np.linalg.norm(current)
        if norm > 0:
            current = current / norm
        
        activations = [current]
        
        for layer in self.layers:
            h = layer.forward(current)
            activations.append(h)
            current = h
        
        # Output
        logits = self.output_W @ current + self.output_b
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs, activations
    
    def learn(self, x, target_class, reward=0.0):
        """
        Learning step.
        
        1. Forward pass
        2. SHD learning on each layer
        3. IAC memory storage
        4. Complementary learning (episodic + semantic)
        5. Class prototype update
        """
        x = x.flatten().astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
        
        # Forward pass
        probs, activations = self.forward(x)
        
        # SHD learning on each layer
        current = x
        for i, layer in enumerate(self.layers):
            h = layer.learn(current, global_reward=reward)
            current = h
            
            # Store in IAC memory
            self.memories[i].store(h, target_class)
        
        # Output layer learning (perceptron/Hebbian hybrid)
        target_vec = np.zeros(self.output_dim, dtype=np.float32)
        target_vec[target_class] = 1.0
        error = target_vec - probs
        self.output_W += 0.01 * np.outer(error, activations[-1])
        self.output_b += 0.01 * error
        
        # Complementary learning
        self.cls.experience(x, target_class)
        
        # Class prototype update
        if target_class not in self.class_prototypes:
            self.class_prototypes[target_class] = activations[-1].copy()
            self.class_counts[target_class] = 0
        else:
            # Running average
            self.class_prototypes[target_class] = (
                0.9 * self.class_prototypes[target_class] + 0.1 * activations[-1]
            )
        self.class_counts[target_class] += 1
        
        return probs
    
    def predict(self, x):
        """Predict class."""
        probs, _ = self.forward(x)
        return np.argmax(probs), probs
    
    def predict_with_memory(self, x):
        """Predict using memory-based retrieval."""
        # Forward pass
        probs, activations = self.forward(x)
        fwd_pred = np.argmax(probs)
        
        # Memory-based prediction (top layer)
        top_layer = activations[-1]
        memory = self.memories[-1]
        if memory.n_stored > 0:
            retrieved, mem_label, mem_sim = memory.retrieve_with_label(top_layer, steps=3)
            if mem_label is not None and mem_sim > 0.5:
                return mem_label, probs, 'memory'
        
        return fwd_pred, probs, 'forward'
    
    def summary(self):
        total_params = sum(
            layer.W.size for layer in self.layers
        ) + self.output_W.size + self.output_b.size
        
        return {
            'name': 'RAIE',
            'layers': [self.input_dim] + self.hidden_dims + [self.output_dim],
            'total_params': total_params,
            'sparsity': self.sparsity,
            'components': ['KWTA', 'IAC', 'XOR', 'CLS', 'SRBI', 'SHD'],
        }


# =============================================================================
# MNIST Experiment
# =============================================================================

def download_mnist(data_dir="/tmp/mnist_data"):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ]
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            continue
        for mirror in mirrors:
            try:
                import urllib.request
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


def run_mnist_experiment(epochs=5, n_train=6000, n_test=1000,
                         hidden_dims=[128, 64], sparsity=0.1):
    """Run MNIST experiment with RAIE."""
    print("=" * 70)
    print("RAIE - MNIST Experiment")
    print("=" * 70)
    
    # Load data
    print("Loading MNIST...")
    data_dir = download_mnist()
    try:
        train_images, train_labels, test_images, test_labels = load_mnist()
        print(f"  Real MNIST: {len(train_images)} train, {len(test_images)} test")
    except Exception as e:
        print(f"  Failed: {e}, generating synthetic")
        rng = np.random.RandomState(42)
        train_images = rng.rand(6000, 784).astype(np.float32) * 0.5 + 0.2
        train_labels = rng.randint(0, 10, 6000)
        test_images = rng.rand(1000, 784).astype(np.float32) * 0.5 + 0.2
        test_labels = rng.randint(0, 10, 1000)
    
    # Subset
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]
    
    # Create RAIE
    raie = RAIE(
        input_dim=784,
        hidden_dims=hidden_dims,
        output_dim=10,
        sparsity=sparsity
    )
    
    summary = raie.summary()
    print(f"\nArchitecture: {summary['name']}")
    print(f"Layers: {summary['layers']}")
    print(f"Params: {summary['total_params']}")
    print(f"Sparsity: {summary['sparsity']}")
    print(f"Components: {summary['components']}")
    
    results = {'train_acc': [], 'test_acc': [], 'loss': [], 'time': []}
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        start = time.time()
        
        # Train
        train_correct = 0
        train_total = 0
        
        indices = np.arange(n_train)
        np.random.shuffle(indices)
        
        for idx in indices:
            probs = raie.learn(train_images[idx], int(train_labels[idx]))
            pred = np.argmax(probs)
            if pred == train_labels[idx]:
                train_correct += 1
            train_total += 1
        
        train_acc = train_correct / train_total
        
        # Test
        test_correct = 0
        test_total = 0
        for i in range(n_test):
            pred, _ = raie.predict(test_images[i])
            if pred == test_labels[i]:
                test_correct += 1
            test_total += 1
        
        test_acc = test_correct / test_total
        
        elapsed = time.time() - start
        
        results['train_acc'].append(float(train_acc))
        results['test_acc'].append(float(test_acc))
        results['time'].append(float(elapsed))
        
        print(f"  Train acc: {train_acc:.4f}")
        print(f"  Test acc:  {test_acc:.4f}")
        print(f"  Time:      {elapsed:.2f}s")
        
        # Memory stats
        cls_stats = raie.cls.get_stats()
        print(f"  Episodic stored: {cls_stats['episodic_stored']}")
        print(f"  Replay buffer: {cls_stats['replay_buffer_size']}")
    
    final_acc = results['test_acc'][-1]
    print(f"\n{'=' * 70}")
    print(f"FINAL TEST ACCURACY: {final_acc:.4f}")
    print(f"{'=' * 70}")
    
    return results, raie


def main():
    results, raie = run_mnist_experiment(
        epochs=5,
        n_train=6000,
        n_test=1000,
        hidden_dims=[128, 64],
        sparsity=0.1
    )
    
    # Save
    path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/11_LOGS/raie_mnist_results.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")
    
    return results


if __name__ == "__main__":
    main()
