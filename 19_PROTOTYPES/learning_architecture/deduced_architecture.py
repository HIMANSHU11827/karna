#!/usr/bin/env python3
"""
Neural Architecture Deduced from First Principles for AGI Assistant
====================================================================

This module contains a custom neural network architecture deduced from
fundamental requirements for AGI, NOT copied from existing work.

Core deductions:
1. Sparse distributed representations → high capacity, pattern separation
2. Hierarchical temporal prediction → learn to predict world across timescales
3. Local learning + global modulation → avoids catastrophic forgetting, enables credit assignment
4. Graph-structured composition → explicit relational reasoning
5. Energy-based inference → principled, convergent dynamics

Unique contributions:
- Multi-timescale Hebbian traces (fast eligibility + slow weights)
- Structural plasticity (learns graph topology, not just weights)
- Contrastive learning rule derived from energy principle
- Explicit memory consolidation (working → episodic → semantic)
"""

import numpy as np
import time
import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set
from dataclasses import dataclass, field


# =============================================================================
# Deduction 1: Sparse Distributed Representations
# =============================================================================

class SparseCode:
    """
    Deduction: AGI needs high capacity and pattern separation.
    
    Mathematical basis: A k-sparse code in d dimensions has capacity
    C(d,k) = d!/(k!(d-k)!) possible codes. For d=1000, k=10: ~10^23 codes.
    This is vastly higher than dense representations.
    
    Properties:
    - Only k neurons active at any time (k << d)
    - Activations are non-negative (rectified)
    - L1 regularization enforces sparsity
    """
    
    def __init__(self, dim: int, sparsity: int, seed: int = 42):
        self.dim = dim
        self.sparsity = sparsity
        self.rng = np.random.RandomState(seed)
        
        # Sparse code vector
        self.code = np.zeros(dim, dtype=np.float32)
        
        # Dictionary for encoding (will be learned)
        self.dictionary = self.rng.randn(dim, dim).astype(np.float32) * 0.1
        norms = np.linalg.norm(self.dictionary, axis=0, keepdims=True)
        self.dictionary /= np.maximum(norms, 1e-6)
    
    def encode(self, x: np.ndarray, n_steps: int = 10, lr: float = 0.1,
               threshold: float = 0.1) -> np.ndarray:
        """
        Encode input into sparse code via ISTA (Iterative Shrinkage-Thresholding).
        
        Solves: min_c ||x - Dc||² + λ||c||₁
        """
        self.code = np.zeros(self.dim, dtype=np.float32)
        
        for _ in range(n_steps):
            # Reconstruction
            reconstruction = self.dictionary @ self.code
            error = x - reconstruction
            
            # Gradient step
            gradient = self.dictionary.T @ error
            self.code += lr * gradient
            
            # Soft thresholding (L1 proximal operator)
            self.code = np.sign(self.code) * np.maximum(np.abs(self.code) - threshold, 0)
            
            # Hard k-sparsity constraint
            if np.count_nonzero(self.code) > self.sparsity:
                top_k = np.argsort(np.abs(self.code))[-self.sparsity:]
                mask = np.zeros_like(self.code)
                mask[top_k] = 1.0
                self.code *= mask
        
        return self.code.copy()
    
    def decode(self) -> np.ndarray:
        """Decode sparse code back to input space."""
        return self.dictionary @ self.code
    
    def learn_dictionary(self, x: np.ndarray, lr: float = 0.001):
        """
        Learn dictionary atoms via Hebbian outer product.
        ΔD = η * c ⊗ (x - Dc)
        """
        reconstruction = self.decode()
        error = x - reconstruction
        self.dictionary += lr * np.outer(self.code, error)
        # Normalize atoms
        norms = np.linalg.norm(self.dictionary, axis=0, keepdims=True)
        self.dictionary /= np.maximum(norms, 1e-6)


# =============================================================================
# Deduction 2: Multi-Timescale Hebbian Traces
# =============================================================================

class MultiTimescaleSynapse:
    """
    Deduction: AGI needs both fast adaptation and long-term stability.
    
    Mathematical basis:
    - Fast trace (eligibility): τ_fast * de/dt = -e + pre * post
    - Slow weights: Δw = η * e * m(t)
      where m(t) is a global modulation signal (attention/reward)
    
    This gives:
    - Fast: working memory, one-shot learning
    - Slow: long-term knowledge, stable skills
    
    The separation avoids catastrophic forgetting:
    fast traces decay quickly unless consolidated by slow weights.
    """
    
    def __init__(self, pre_dim: int, post_dim: int, seed: int = 42):
        self.pre_dim = pre_dim
        self.post_dim = post_dim
        self.rng = np.random.RandomState(seed)
        
        # Slow weights (long-term)
        self.W_slow = self.rng.randn(post_dim, pre_dim).astype(np.float32) * 0.1
        
        # Fast weights (working memory)
        self.W_fast = np.zeros((post_dim, pre_dim), dtype=np.float32)
        
        # Eligibility traces
        self.E = np.zeros((post_dim, pre_dim), dtype=np.float32)
        
        # Timescales
        self.tau_slow = 1000.0   # slow time constant
        self.tau_fast = 10.0     # fast time constant
        self.tau_E = 50.0        # eligibility trace time constant
        
        # Learning rates
        self.lr_slow = 0.0001
        self.lr_fast = 0.01
        
        # Oja stabilization
        self.oja_gamma = 0.001
    
    def forward(self, pre: np.ndarray) -> np.ndarray:
        """Compute post = W_slow @ pre + W_fast @ pre."""
        return self.W_slow @ pre + self.W_fast @ pre
    
    def update_eligibility(self, pre: np.ndarray, post: np.ndarray):
        """
        Update eligibility trace: τ_E * dE/dt = -E + pre ⊗ post
        """
        outer = np.outer(post, pre)
        dE = (-self.E + outer) / self.tau_E
        self.E += dE
    
    def learn(self, pre: np.ndarray, post: np.ndarray, 
              global_modulation: float = 1.0, reward: float = 0.0):
        """
        Multi-timescale learning rule.
        
        1. Fast weights: Hebbian with decay
           ΔW_fast = lr_fast * (E - W_fast)
        
        2. Slow weights: consolidated by global signal
           ΔW_slow = lr_slow * E * modulation * (1 + reward)
        """
        # Update eligibility
        self.update_eligibility(pre, post)
        
        # Fast weight update (working memory)
        self.W_fast += self.lr_fast * (self.E - self.W_fast) / self.tau_fast
        
        # Slow weight update (long-term consolidation)
        # Only consolidate when global modulation is active
        if abs(global_modulation) > 0.1:
            self.W_slow += self.lr_slow * self.E * global_modulation * (1 + reward)
            
            # Oja stabilization on slow weights
            post_sq = post[:, None] ** 2 if post.ndim > 0 else post ** 2
            decay = self.oja_gamma * self.W_slow * post_sq
            self.W_slow -= decay
    
    def consolidate(self):
        """
        Explicit consolidation: transfer fast to slow.
        Called during "sleep" or when modulation is sustained.
        """
        self.W_slow += 0.1 * self.W_fast
        self.W_fast *= 0.9  # partial decay
    
    def reset_fast(self):
        """Clear working memory."""
        self.W_fast = np.zeros((self.post_dim, self.pre_dim), dtype=np.float32)
        self.E = np.zeros((self.post_dim, self.pre_dim), dtype=np.float32)


# =============================================================================
# Deduction 3: Graph-Structured Neurons with Structural Plasticity
# =============================================================================

@dataclass
class GraphNeuron:
    """
    Deduction: AGI needs explicit relational reasoning, not just flat vectors.
    
    A graph neuron is a node in a dynamic graph:
    - Each neuron has a state vector
    - Edges are learned via structural plasticity
    - Clustering forms "concept assemblies"
    
    Properties:
    - Edges form when neurons co-activate repeatedly
    - Edges decay without co-activation (use it or lose it)
    - Graph topology determines information flow
    """
    dim: int
    neuron_id: int
    state: np.ndarray = None
    activation: float = 0.0
    edges: Dict[int, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.state is None:
            self.state = np.zeros(self.dim, dtype=np.float32)


class GraphAssembly:
    """
    A cluster of strongly connected neurons representing a concept.
    
    Deduction: Concepts are assemblies of neurons (Hebb, 1949).
    Assembly formation: neurons that fire together, wire together.
    Assembly activation: pattern completion from partial cues.
    """
    
    def __init__(self, neuron_ids: Set[int], assembly_id: int):
        self.neuron_ids = neuron_ids
        self.assembly_id = assembly_id
        self.activation = 0.0
        self.stability = 0.0  # how consolidated this assembly is
    
    def compute_activation(self, neuron_activations: Dict[int, float]) -> float:
        """Average activation of member neurons."""
        acts = [neuron_activations.get(nid, 0.0) for nid in self.neuron_ids]
        self.activation = np.mean(acts) if acts else 0.0
        return self.activation


class StructuralPlasticityGraph:
    """
    Graph network with structural plasticity.
    
    Deduction: The brain's topology is not fixed.
    New synapses form, old ones prune. This is essential for:
    - Lifelong learning (new concepts)
    - Pruning (efficiency)
    - Credit assignment (which connections matter)
    """
    
    def __init__(self, n_neurons: int, dim: int, seed: int = 42):
        self.n_neurons = n_neurons
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        
        # Neurons
        self.neurons = [
            GraphNeuron(dim=dim, neuron_id=i)
            for i in range(n_neurons)
        ]
        
        # Assemblies (learned clusters)
        self.assemblies: List[GraphAssembly] = []
        self.next_assembly_id = 0
        
        # Sparse connectivity matrix
        self.adjacency = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        
        # Structural plasticity parameters
        self.formation_threshold = 0.5   # co-activation needed to form edge
        self.prune_threshold = 0.01      # edges below this decay
        self.formation_rate = 0.01
        self.prune_rate = 0.001
        self.max_edges_per_neuron = 20
    
    def activate(self, inputs: Dict[int, np.ndarray], n_steps: int = 5):
        """
        Activate the network.
        
        Deduction: Activation spreads through the graph.
        Neurons with stronger input activate more.
        Lateral inhibition implements competition.
        """
        # Set input neuron states
        for nid, inp in inputs.items():
            if nid < self.n_neurons:
                self.neurons[nid].state = inp
                self.neurons[nid].activation = np.linalg.norm(inp)
        
        # Spread activation through graph
        for _ in range(n_steps):
            new_activations = np.zeros(self.n_neurons, dtype=np.float32)
            
            for i, neuron in enumerate(self.neurons):
                # Sum inputs from connected neurons
                input_sum = 0.0
                for j in range(self.n_neurons):
                    if self.adjacency[i, j] > 0:
                        input_sum += self.adjacency[i, j] * self.neurons[j].activation
                
                # Update with lateral inhibition
                total_activation = neuron.activation + input_sum
                # Lateral inhibition: subtract mean of neighbors
                neighbor_acts = [
                    self.neurons[j].activation
                    for j in range(self.n_neurons)
                    if self.adjacency[i, j] > 0 and j != i
                ]
                inhibition = np.mean(neighbor_acts) if neighbor_acts else 0.0
                
                new_activations[i] = max(0.0, total_activation - 0.5 * inhibition)
            
            # Update neurons
            for i, neuron in enumerate(self.neurons):
                neuron.activation = new_activations[i]
        
        # Update assemblies
        neuron_acts = {n.neuron_id: n.activation for n in self.neurons}
        for assembly in self.assemblies:
            assembly.compute_activation(neuron_acts)
    
    def structural_update(self):
        """
        Structural plasticity: form and prune edges.
        
        Deduction: Topology should reflect statistics of co-activation.
        Edges between frequently co-active neurons strengthen.
        Unused edges prune.
        """
        activations = np.array([n.activation for n in self.neurons])
        
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i == j:
                    continue
                
                coactivation = activations[i] * activations[j]
                
                if coactivation > self.formation_threshold:
                    # Form or strengthen edge
                    if self.adjacency[i, j] == 0:
                        # New edge
                        edge_count = np.count_nonzero(self.adjacency[i])
                        if edge_count < self.max_edges_per_neuron:
                            self.adjacency[i, j] = self.formation_rate * coactivation
                    else:
                        # Strengthen existing edge
                        self.adjacency[i, j] += self.formation_rate * coactivation
                else:
                    # Prune/decay edge
                    self.adjacency[i, j] *= (1 - self.prune_rate)
                    if self.adjacency[i, j] < self.prune_threshold:
                        self.adjacency[i, j] = 0.0
    
    def form_assembly(self, neuron_ids: Optional[Set[int]] = None) -> Optional[GraphAssembly]:
        """
        Form a new assembly from highly active neurons.
        
        Deduction: Concepts form when neurons co-activate.
        An assembly is a stable cluster that can be reactivated.
        """
        if neuron_ids is None:
            # Find highly active neurons
            threshold = np.percentile([n.activation for n in self.neurons], 90)
            neuron_ids = {
                n.neuron_id for n in self.neurons
                if n.activation > threshold
            }
        
        if len(neuron_ids) < 2:
            return None
        
        assembly = GraphAssembly(neuron_ids, self.next_assembly_id)
        self.next_assembly_id += 1
        self.assemblies.append(assembly)
        return assembly
    
    def recall_assembly(self, assembly_id: int, cue_neurons: Set[int]) -> np.ndarray:
        """
        Pattern completion: recall full assembly from partial cue.
        
        Deduction: Assemblies are attractor states.
        Partial activation spreads to complete the pattern.
        """
        assembly = None
        for a in self.assemblies:
            if a.assembly_id == assembly_id:
                assembly = a
                break
        
        if assembly is None:
            return np.zeros(self.n_neurons, dtype=np.float32)
        
        # Activate cue neurons
        activations = np.zeros(self.n_neurons, dtype=np.float32)
        for nid in cue_neurons:
            if nid < self.n_neurons:
                activations[nid] = 1.0
        
        # Spread through graph
        for _ in range(10):
            activations = self.adjacency @ activations + activations
            activations = np.maximum(0, activations)
            # Normalize
            norm = np.linalg.norm(activations)
            if norm > 0:
                activations /= norm
        
        return activations


# =============================================================================
# Deduction 4: Energy-Based Inference
# =============================================================================

class EnergyBasedLayer:
    """
    Deduction: Inference should minimize an energy function.
    
    Mathematical basis:
    E(x, h) = ||x - f(h)||² + λ||h||₁ + γ||h - h_prev||²
    
    Terms:
    - ||x - f(h)||²: reconstruction error (prediction)
    - λ||h||₁: sparsity prior
    - γ||h - h_prev||²: temporal continuity (smoothness)
    
    Inference: gradient descent on E
    Learning: contrastive (increase prob of observed, decrease of inferred)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rng = np.random.RandomState(seed)
        
        # Forward weights (generative): hidden → input
        self.W_gen = self.rng.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        
        # Inference weights (recognition): input → hidden
        self.W_inf = self.rng.randn(hidden_dim, input_dim).astype(np.float32) * 0.1
        
        # Hidden state
        self.h = np.zeros(hidden_dim, dtype=np.float32)
        self.h_prev = np.zeros(hidden_dim, dtype=np.float32)
        
        # Energy function parameters
        self.sparsity_lambda = 0.1
        self.temporal_gamma = 0.5
        
        # Inference step size
        self.dt = 0.1
    
    def energy(self, x: np.ndarray) -> float:
        """Compute energy E(x, h)."""
        reconstruction = self.W_gen @ self.h
        pred_error = np.sum((x - reconstruction) ** 2)
        sparsity = self.sparsity_lambda * np.sum(np.abs(self.h))
        temporal = self.temporal_gamma * np.sum((self.h - self.h_prev) ** 2)
        return pred_error + sparsity + temporal
    
    def inference(self, x: np.ndarray, n_steps: int = 10):
        """
        Inference: gradient descent on energy.
        dh/dt = -∂E/∂h
        """
        for _ in range(n_steps):
            # Reconstruction error gradient
            reconstruction = self.W_gen @ self.h
            error = x - reconstruction
            grad_pred = -2 * self.W_gen.T @ error
            
            # Sparsity gradient (subgradient of L1)
            grad_sparse = self.sparsity_lambda * np.sign(self.h)
            
            # Temporal continuity gradient
            grad_temporal = 2 * self.temporal_gamma * (self.h - self.h_prev)
            
            # Total gradient
            grad = grad_pred + grad_sparse + grad_temporal
            
            # Update
            self.h -= self.dt * grad
            # ReLU
            self.h = np.maximum(0, self.h)
    
    def generate(self) -> np.ndarray:
        """Generate input from hidden state."""
        return self.W_gen @ self.h
    
    def learn_contrastive(self, x_pos: np.ndarray, x_neg: np.ndarray,
                          lr: float = 0.001):
        """
        Contrastive learning rule.
        
        Deduction: Learning should increase probability of observed data
        and decrease probability of inferred (negative) samples.
        
        ΔW ∝ x_pos @ h_pos^T - x_neg @ h_neg^T
        """
        # Positive phase
        self.inference(x_pos, n_steps=5)
        h_pos = self.h.copy()
        
        # Negative phase
        self.inference(x_neg, n_steps=5)
        h_neg = self.h.copy()
        
        # Weight update
        outer_pos = np.outer(x_pos, h_pos)
        outer_neg = np.outer(x_neg, h_neg)
        self.W_gen += lr * (outer_pos - outer_neg)
        
        # Update inference weights (Hebbian)
        self.W_inf += lr * (h_pos @ x_pos.T - h_neg @ x_neg.T)
    
    def temporal_update(self):
        """Store current state as previous for next timestep."""
        self.h_prev = self.h.copy()


# =============================================================================
# Deduction 5: Explicit Memory Consolidation
# =============================================================================

class MemoryConsolidationSystem:
    """
    Deduction: AGI needs explicit memory systems with consolidation.
    
    Based on memory systems in the brain:
    - Working memory: fast, limited capacity, transient
    - Episodic memory: experiences, associative
    - Semantic memory: facts, concepts, stable
    
    Consolidation: transfer from working → episodic → semantic
    """
    
    def __init__(self, working_capacity: int = 7, episodic_capacity: int = 1000,
                 semantic_capacity: int = 10000, dim: int = 64, seed: int = 42):
        self.working_capacity = working_capacity
        self.episodic_capacity = episodic_capacity
        self.semantic_capacity = semantic_capacity
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        
        # Working memory: circular buffer
        self.working_memory: List[np.ndarray] = []
        
        # Episodic memory: list of (key, value, context, timestamp)
        self.episodic_memory: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        
        # Semantic memory: key-value store (like a hash table)
        self.semantic_memory: Dict[str, np.ndarray] = {}
        
        # Synaptic weights for indexing
        self.episodic_keys = None  # Will be allocated when first episode stored
        self.episodic_values = None
        
        # Consolidation counter
        self.steps_since_consolidation = 0
        self.consolidation_interval = 100
    
    def perceive(self, item: np.ndarray, context: Optional[np.ndarray] = None):
        """Add item to working memory."""
        if context is None:
            context = np.zeros(self.dim, dtype=np.float32)
        
        self.working_memory.append(item.copy())
        
        # Enforce capacity limit
        if len(self.working_memory) > self.working_capacity:
            oldest = self.working_memory.pop(0)
            # Move to episodic memory
            self._store_episode(oldest, context)
        
        # Check consolidation
        self.steps_since_consolidation += 1
        if self.steps_since_consolidation >= self.consolidation_interval:
            self.consolidate()
            self.steps_since_consolidation = 0
    
    def _store_episode(self, item: np.ndarray, context: np.ndarray):
        """Store an episode in episodic memory."""
        # Generate key from item
        key = item.copy()
        timestamp = time.time()
        
        # Store
        self.episodic_memory.append((key, item.copy(), context, timestamp))
        
        # Enforce capacity
        if len(self.episodic_memory) > self.episodic_capacity:
            # Consolidate oldest to semantic
            oldest = self.episodic_memory.pop(0)
            self._consolidate_to_semantic(oldest[1], oldest[2])
        
        # Rebuild index matrix
        n = len(self.episodic_memory)
        self.episodic_keys = np.array([e[0] for e in self.episodic_memory])
        self.episodic_values = np.array([e[1] for e in self.episodic_memory])
    
    def _consolidate_to_semantic(self, item: np.ndarray, context: np.ndarray):
        """Consolidate an episode into semantic memory."""
        # Use hash of content as key
        key = hashlib.md5(item.tobytes()).hexdigest()[:16]
        self.semantic_memory[key] = item.copy()
    
    def recall(self, query: np.ndarray, n: int = 5) -> List[np.ndarray]:
        """Recall similar memories using cosine similarity."""
        if not self.episodic_memory:
            return []
        
        # Compute similarities
        query_norm = query / (np.linalg.norm(query) + 1e-6)
        keys_norm = self.episodic_keys / (np.linalg.norm(self.episodic_keys, axis=1, keepdims=True) + 1e-6)
        similarities = keys_norm @ query_norm
        
        # Return top-n
        top_indices = np.argsort(similarities)[-n:][::-1]
        return [self.episodic_values[i] for i in top_indices]
    
    def remember_fact(self, key: str, value: np.ndarray):
        """Store a fact in semantic memory."""
        self.semantic_memory[key] = value.copy()
    
    def recall_fact(self, key: str) -> Optional[np.ndarray]:
        """Recall a fact from semantic memory."""
        return self.semantic_memory.get(key)
    
    def consolidate(self):
        """
        Consolidation: transfer frequent/recent episodes to semantic.
        
        Deduction: During "rest", the brain replays important episodes
        and transfers them to long-term storage.
        """
        if not self.episodic_memory:
            return
        
        # Consolidate most recent 10%
        n_consolidate = max(1, len(self.episodic_memory) // 10)
        for i in range(n_consolidate):
            episode = self.episodic_memory[-(i+1)]
            self._consolidate_to_semantic(episode[1], episode[2])
    
    def stats(self) -> Dict[str, int]:
        return {
            "working": len(self.working_memory),
            "episodic": len(self.episodic_memory),
            "semantic": len(self.semantic_memory),
        }


# =============================================================================
# Full Deduced Architecture: Integration
# =============================================================================

class DeducedAGIArchitecture:
    """
    Full AGI architecture deduced from first principles.
    
    Integrates:
    1. Sparse coding for input representation
    2. Energy-based layers for inference
    3. Multi-timescale synapses for learning
    4. Structural plasticity graph for composition
    5. Memory consolidation for lifelong learning
    
    This is NOT a transformer, NOT a predictive coding network,
    NOT a Hopfield network. This is a new architecture deduced
    specifically for this project's AGI requirements.
    """
    
    def __init__(self, input_dim: int, layer_dims: List[int],
                 sparsity: int = 10, seed: int = 42):
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.sparsity = sparsity
        
        # Input sparse coding
        self.input_sparse = SparseCode(input_dim, sparsity=sparsity, seed=seed)
        
        # Energy-based layers
        self.layers: List[EnergyBasedLayer] = []
        prev_dim = input_dim
        for dim in layer_dims:
            layer = EnergyBasedLayer(prev_dim, dim, seed=seed)
            self.layers.append(layer)
            prev_dim = dim
        
        # Multi-timescale synapses between layers
        self.synapses: List[MultiTimescaleSynapse] = []
        prev_dim = input_dim
        for dim in layer_dims:
            syn = MultiTimescaleSynapse(prev_dim, dim, seed=seed)
            self.synapses.append(syn)
            prev_dim = dim
        
        # Structural plasticity graph
        total_neurons = sum(layer_dims)
        self.graph = StructuralPlasticityGraph(total_neurons, dim=32, seed=seed)
        
        # Memory system
        self.memory = MemoryConsolidationSystem(
            working_capacity=7,
            episodic_capacity=1000,
            semantic_capacity=10000,
            dim=layer_dims[-1] if layer_dims else 64,
            seed=seed
        )
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def forward(self, x: np.ndarray, n_inference_steps: int = 5) -> List[np.ndarray]:
        """
        Full forward pass through the architecture.
        
        1. Sparse encode input
        2. Pass through energy-based layers (inference)
        3. Activate structural graph
        """
        # Sparse encode
        sparse_input = self.input_sparse.encode(x, n_steps=n_inference_steps)
        
        # Energy-based inference
        current = sparse_input
        activations = []
        
        for i, layer in enumerate(self.layers):
            layer.inference(current, n_steps=n_inference_steps)
            acts = layer.h.copy()
            activations.append(acts)
            current = acts
            
            # Multi-timescale synapse forward
            if i < len(self.synapses):
                current = self.synapses[i].forward(current)
        
        # Activate structural graph (top layer only)
        if activations:
            graph_input = {i: activations[-1][i % len(activations[-1])] 
                          for i in range(min(len(activations[-1]), self.graph.n_neurons))}
            self.graph.activate(graph_input, n_steps=3)
        
        # Store in memory
        if activations:
            self.memory.perceive(activations[-1])
        
        return activations
    
    def learn(self, x: np.ndarray, reward: float = 0.0,
              global_modulation: float = 1.0, n_neg_samples: int = 5):
        """
        Full learning cycle.
        
        1. Forward pass
        2. Multi-timescale learning
        3. Contrastive learning on layers
        4. Structural plasticity
        5. Input dictionary learning
        """
        # Forward pass
        activations = self.forward(x)
        
        # Multi-timescale learning
        current = x.copy()
        for i, (layer, synapse) in enumerate(zip(self.layers, self.synapses)):
            if i < len(activations):
                h = activations[i]
                synapse.learn(current, h, global_modulation, reward)
                current = h
        
        # Contrastive learning on each layer
        for i, layer in enumerate(self.layers):
            # Generate negative sample
            x_neg = self.rng.randn(layer.input_dim).astype(np.float32) * 0.1
            layer.temporal_update()
            layer.learn_contrastive(current, x_neg, lr=0.001)
            current = layer.h
        
        # Structural plasticity
        self.graph.structural_update()
        
        # Input dictionary learning
        self.input_sparse.learn_dictionary(x)
    
    def generate(self, n_steps: int = 10) -> np.ndarray:
        """
        Generate output from top-layer state.
        
        Deduction: Generation is the reverse of inference.
        Start from top-layer prior, decode downward.
        """
        # Start with top layer state
        if not self.layers:
            return np.zeros(self.input_dim, dtype=np.float32)
        
        current = self.layers[-1].h.copy()
        
        # Decode through layers (reverse order)
        for layer in reversed(self.layers):
            current = layer.W_gen @ current
            current = np.maximum(0, current)  # ReLU
        
        # Decode sparse code
        reconstruction = self.input_sparse.dictionary @ current[:self.input_dim]
        return reconstruction
    
    def remember(self, key: str, value: np.ndarray):
        """Store a fact in semantic memory."""
        self.memory.remember_fact(key, value)
    
    def recall(self, key: str) -> Optional[np.ndarray]:
        """Recall a fact."""
        return self.memory.recall_fact(key)
    
    def consolidate(self):
        """Explicit memory consolidation."""
        self.memory.consolidate()
        for synapse in self.synapses:
            synapse.consolidate()
    
    def summary(self) -> Dict[str, Any]:
        return {
            "name": "Deduced AGI Architecture (DAGI)",
            "deductions": [
                "Sparse distributed representations",
                "Energy-based inference",
                "Multi-timescale Hebbian traces",
                "Structural plasticity graph",
                "Memory consolidation system",
            ],
            "input_dim": self.input_dim,
            "layer_dims": self.layer_dims,
            "total_neurons": sum(self.layer_dims),
            "total_parameters": sum(
                l.input_dim * l.hidden_dim * 2 for l in self.layers
            ) + self.input_dim ** 2,
            "backprop": False,
            "transformers": False,
            "attention": False,
            "learning": "Multi-timescale Hebbian + Contrastive",
            "inference": "Energy minimization",
            "memory": "Working + Episodic + Semantic with consolidation",
        }


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark():
    """Benchmark the deduced architecture."""
    print("=" * 70)
    print("DEDUCTED AGI ARCHITECTURE — BENCHMARK")
    print("=" * 70)
    
    # Small scale for quick test
    arch = DeducedAGIArchitecture(
        input_dim=64,
        layer_dims=[32, 16, 8],
        sparsity=5,
        seed=42
    )
    
    x = np.random.randn(64).astype(np.float32)
    
    # Warmup
    arch.forward(x)
    arch.learn(x)
    
    # Benchmark forward
    n = 100
    start = time.perf_counter()
    for _ in range(n):
        arch.forward(x)
    fwd_ms = (time.perf_counter() - start) / n * 1000
    
    # Benchmark learn
    start = time.perf_counter()
    for _ in range(n):
        arch.learn(x)
    learn_ms = (time.perf_counter() - start) / n * 1000
    
    print(f"\nSmall scale (64 → 32 → 16 → 8):")
    print(f"  Forward: {fwd_ms:.2f}ms")
    print(f"  Learn:   {learn_ms:.2f}ms")
    
    # MNIST scale projection
    print(f"\nMNIST scale projection (784 → 256 → 64 → 10):")
    arch_mnist = DeducedAGIArchitecture(
        input_dim=784,
        layer_dims=[256, 64, 10],
        sparsity=10,
        seed=42
    )
    x_mnist = np.random.randn(784).astype(np.float32)
    
    start = time.perf_counter()
    arch_mnist.forward(x_mnist)
    fwd_mnist = (time.perf_counter() - start) * 1000
    
    start = time.perf_counter()
    arch_mnist.learn(x_mnist)
    learn_mnist = (time.perf_counter() - start) * 1000
    
    print(f"  Forward: {fwd_mnist:.2f}ms")
    print(f"  Learn:   {learn_mnist:.2f}ms")
    
    epoch_s = learn_mnist * 60000 / 1000
    print(f"  1 epoch: {epoch_s:.1f}s ({epoch_s/60:.1f} min)")
    print(f"  100 epochs: {epoch_s * 100 / 3600:.1f} hours")
    
    # Memory
    import tracemalloc
    tracemalloc.start()
    arch2 = DeducedAGIArchitecture(784, [256, 64, 10], sparsity=10)
    for _ in range(5):
        arch2.learn(x_mnist)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n  Peak memory (MNIST scale): {peak/1024/1024:.2f} MB")
    
    # Summary
    print(f"\n{json.dumps(arch_mnist.summary(), indent=2)}")


if __name__ == "__main__":
    run_benchmark()
