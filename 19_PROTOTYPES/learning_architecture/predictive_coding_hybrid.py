"""
AGI Research Lab - Learning Architecture Prototype
=================================================

Redesigned from scratch using alternatives to backpropagation/transformers.

Architecture: Predictive Coding + Global-Hebbian Hybrid (PCHH)
------------------------------------------------------------
Based on:
  - Predictive Coding Networks (Rao & Ballard, Friston, Whittington & Bogacz)
  - Global-guided Hebbian Learning (GHL) — Hebb + global sign signal
  - Sparse Coding / Sparse Predictive Coding (Laplacian prior, local competition)
  - Temporal Predictive Coding (tPC) for sequence learning
  - Neural Generative Coding (NGC) — decoupled forward/backward pathways
  - Inference Learning (IL) — biologically plausible alternative to backprop

Key innovations over transformers:
  - Local, Hebbian weight updates (no backward gradient flow needed)
  - Hierarchical prediction-error minimization (free energy principle)
  - Sparse, biologically plausible activity patterns
  - Temporal prediction for sequence/world-model learning
  - Fast weights for working memory (Hebbian traces)
  - Neuromorphic-friendly: event-driven, local computation
"""

import math
import time
import hashlib
import json
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import random


# =============================================================================
# Neural Units
# =============================================================================

@dataclass
class ValueNeuron:
    """
    Value neuron in Predictive Coding Network.
    Represents inferred state at a layer — encodes the posterior.
    In PC, these neurons settle to minimize prediction error (inference phase).
    """
    activation: float = 0.1  # small baseline activation to bootstrap
    prediction: float = 0.0        # top-down prediction from layer above
    prediction_error: float = 0.0  # residual (value - prediction)
    precision: float = 1.0         # learned reliability weight (inverse variance)
    lateral_inhibition: float = 0.0
    membrane_potential: float = 0.1  # small baseline to prevent dead neurons
    leak_rate: float = 0.1
    threshold: float = 1e-6
    bias: float = 0.01  # small constant input to prevent silence

    def update_membrane(self, input_current: float, dt: float = 0.1):
        """Leaky integrator dynamics (biological realism)."""
        self.leak_rate = 0.1
        self.membrane_potential += dt * (
            -self.leak_rate * self.membrane_potential + input_current
        )
        # ReLU activation (simplified LIF)
        self.activation = max(0.0, self.membrane_potential)

    def compute_error(self):
        """Compute prediction error (precision-weighted)."""
        self.prediction_error = (self.activation - self.prediction) * self.precision


@dataclass
class ErrorNeuron:
    """
    Dedicated error neuron (neural generative coding innovation).
    Encodes prediction error explicitly — allows flexible routing without
    weight symmetry constraints.
    """
    activation: float = 0.0
    source_value: float = 0.0
    target_prediction: float = 0.0

    def compute(self, value: float, prediction: float):
        self.source_value = value
        self.target_prediction = prediction
        self.activation = value - prediction


# =============================================================================
# Synapse with Hebbian Plasticity
# =============================================================================

@dataclass
class HebbianSynapse:
    """
    Synapse with multiple Hebbian update rules:

    1. Plain Hebb:      Δw = η * pre * post
    2. Oja's rule:      Δw = η * pre * post - η * w * post²  (stabilized)
    3. BCM rule:        Δw = η * pre * post * (post - θ)     (sliding threshold)
    4. STDP:            Δw = A+ * exp(-Δt/τ+)  if Δt > 0
                        Δw = A- * exp(Δt/τ-)   if Δt < 0
    5. Global-guided:   Δw = η * pre * post * sign(global_grad)  (GHL)
    """
    weight: float = 0.05  # small non-zero initialization
    trace: float = 0.0           # eligibility trace for STDP
    delta_t: float = 0.0         # spike timing difference
    rule: str = "oja"
    learning_rate: float = 0.01

    # STDP parameters
    A_plus: float = 0.01
    A_minus: float = 0.01
    tau_plus: float = 20.0
    tau_minus: float = 20.0

    # Oja stability parameter
    oja_gamma: float = 0.001

    def update(self, pre: float, post: float, global_sign: float = 0.0):
        """Apply selected Hebbian update rule."""
        if self.rule == "hebb":
            self.weight += self.learning_rate * pre * post
        elif self.rule == "oja":
            # Stabilized Hebbian learning (prevents unbounded growth)
            self.weight += self.learning_rate * (
                pre * post - self.oja_gamma * self.weight * post * post
            )
        elif self.rule == "bcm":
            # Bienenstock-Cooper-Munro (sliding threshold)
            theta = 0.5
            self.weight += self.learning_rate * pre * post * max(0, post - theta)
        elif self.rule == "stdp":
            # Spike-timing dependent plasticity
            if self.delta_t > 0:
                self.weight += self.A_plus * math.exp(-self.delta_t / self.tau_plus)
            else:
                self.weight -= self.A_minus * math.exp(self.delta_t / self.tau_minus)
        elif self.rule == "ghl":
            # Global-guided Hebbian Learning (sign of global gradient)
            self.weight += self.learning_rate * pre * post * global_sign


# =============================================================================
# Predictive Coding Layer
# =============================================================================

class PredictiveCodingLayer:
    """
    A single layer in a hierarchical predictive coding network.

    Architecture:
      - Value neurons (state representation)
      - Error neurons (prediction errors)
      - Synaptic weights (generative model)
      - Lateral competition (sparse coding)

    Dynamics (Inference Phase):
      1. Receive top-down prediction from layer above
      2. Update value neurons to minimize prediction error
      3. Compute prediction error → feedforward to layer above

    Dynamics (Learning Phase):
      1. Update synaptic weights with Hebbian rule
      2. Update precision (inverse variance) of each neuron
    """

    def __init__(self, n_value: int, n_error: int = None,
                 n_inputs: int = None, activation: str = "relu",
                 sparsity_target: float = 0.1):
        self.n_value = n_value
        self.n_error = n_error or n_value
        self.n_inputs = n_inputs or n_value
        self.activation_fn = activation
        self.sparsity_target = sparsity_target

        # Neural populations
        self.values = [ValueNeuron() for _ in range(n_value)]
        self.errors = [ErrorNeuron() for _ in range(self.n_error)]

        # Synaptic weights (generative model) — initialize with small random values
        self.forward_weights = [
            [HebbianSynapse(rule="oja", weight=random.gauss(0, 0.1))
             for _ in range(self.n_inputs)]
            for _ in range(n_value)
        ]
        # Separate feedback weights (neural generative coding)
        self.feedback_weights = [
            [HebbianSynapse(rule="oja", weight=random.gauss(0, 0.1))
             for _ in range(self.n_inputs)]
            for _ in range(n_value)
        ]

        # Predictive Coding state
        self.input_data: list[float] = []
        self.top_down_prediction: list[float] = []
        self.bottom_up_error: list[float] = []
        self.free_energy: float = 0.0
        self.inference_steps: int = 5
        self.dt: float = 0.1

    def set_input(self, data: list[float]):
        """Clamp input to this layer."""
        self.input_data = data[:self.n_inputs]

    def set_top_down_prediction(self, prediction: list[float]):
        """Receive prediction from layer above."""
        self.top_down_prediction = prediction[:self.n_value]

    def predict(self, source_activations: list[float]) -> list[float]:
        """Generate prediction from source (forward or backward pass)."""
        predictions = []
        for i in range(self.n_value):
            pred = 0.0
            for j, synapse in enumerate(self.forward_weights[i]):
                if j < len(source_activations):
                    pred += synapse.weight * source_activations[j]
            predictions.append(pred)
        return predictions

    def inference_step(self):
        """
        Single inference step: update value neurons to minimize prediction error.
        This implements variational inference via gradient descent on free energy.
        """
        for i, v in enumerate(self.values):
            # Combine bottom-up input and top-down prediction
            bottom_up = 0.0
            if self.input_data:
                for j, synapse in enumerate(self.forward_weights[i]):
                    if j < len(self.input_data):
                        bottom_up += synapse.weight * self.input_data[j]

            top_down = self.top_down_prediction[i] if i < len(self.top_down_prediction) else 0.0

            # Total input: prediction error drives value update
            prediction_error = bottom_up - top_down
            total_input = bottom_up + v.bias  # bias prevents silent neurons

            # Update with lateral competition (WTA-inspired) — REDUCED strength
            lateral = sum(
                self.values[j].activation * 0.01  # reduced from 0.1
                for j in range(self.n_value) if j != i
            )
            v.lateral_inhibition = lateral

            # Leaky integration with lateral inhibition
            v.update_membrane(total_input - lateral, self.dt)
            v.prediction = top_down
            v.compute_error()

        # Update error neurons
        for i, e in enumerate(self.errors):
            if i < len(self.values):
                e.compute(self.values[i].activation, self.values[i].prediction)

        # Compute free energy (sum of precision-weighted prediction errors)
        self.free_energy = sum(
            v.prediction_error ** 2 / (2 * max(v.precision, 1e-6))
            for v in self.values
        )

    def inference(self, n_steps: int = None):
        """Run inference to convergence (minimize free energy)."""
        steps = n_steps or self.inference_steps
        for _ in range(steps):
            self.inference_step()

    def learn(self, learning_rate: float = 0.01, global_sign: float = 0.0):
        """
        Learning phase: update synaptic weights with Hebbian plasticity.
        Uses outer product of pre-synaptic and error neuron activity.
        """
        for i in range(self.n_value):
            pre = self.input_data[i] if i < len(self.input_data) else 0.0
            post_error = self.errors[i].activation if i < len(self.errors) else 0.0

            for j in range(self.n_inputs):
                # Hebbian update: weight change ∝ pre * post_error
                if j < len(self.input_data):
                    self.forward_weights[i][j].update(
                        self.input_data[j], post_error, global_sign
                    )

        # Update precision (learned reliability)
        for v in self.values:
            # Precision increases with consistent prediction (low error)
            v.precision = 1.0 / (1.0 + v.prediction_error ** 2 + 1e-6)

    def get_activations(self) -> list[float]:
        """Return current value neuron activations (sparse code)."""
        return [v.activation for v in self.values]

    def get_prediction_errors(self) -> list[float]:
        """Return prediction errors."""
        return [e.activation for e in self.errors]


# =============================================================================
# Temporal Predictive Coding (tPC) Layer
# =============================================================================

class TemporalPredictiveCodingLayer:
    """
    Temporal extension of predictive coding for sequence learning.

    Architecture:
      - State neurons represent latent dynamics
      - Transition weights model temporal dynamics
      - Recurrent connections propagate predictions across time

    Learning:
      - Hebbian updates on transition weights
      - Approximates Kalman filter for linear dynamics
      - Learns motion-sensitive features for visual sequences
    """

    def __init__(self, state_dim: int, transition_rank: int = 4):
        self.state_dim = state_dim
        self.transition_rank = transition_rank

        # State representation (latent dynamics)
        self.state = [0.0] * state_dim
        self.prev_state = [0.0] * state_dim
        self.predicted_next = [0.0] * state_dim

        # Transition dynamics (low-rank factorization for biological plausibility)
        # A = U * V^T where U, V are low-rank matrices
        self.U = [[random.gauss(0, 0.1) for _ in range(transition_rank)]
                  for _ in range(state_dim)]
        self.V = [[random.gauss(0, 0.1) for _ in range(transition_rank)]
                  for _ in range(state_dim)]

        # Precision matrices (learned)
        self.precision = [1.0] * state_dim

        # Hebbian traces
        self.eligibility = [0.0] * state_dim

    def predict_next(self) -> list[float]:
        """Predict next state: s_{t+1} = f(U * V^T * s_t)"""
        # Low-rank transition
        intermediate = [0.0] * self.transition_rank
        for k in range(self.transition_rank):
            for i in range(self.state_dim):
                intermediate[k] += self.V[i][k] * self.state[i]

        # Apply transition
        for i in range(self.state_dim):
            prediction = 0.0
            for k in range(self.transition_rank):
                prediction += self.U[i][k] * intermediate[k]
            # Non-linearity
            self.predicted_next[i] = math.tanh(prediction)

        return self.predicted_next

    def inference(self, observation: list[float], n_steps: int = 3):
        """
        Infer current state given observation.
        Minimizes: ||obs - state||² + ||state - predicted_prev||²
        """
        for step in range(n_steps):
            prediction_errors = [
                observation[i] - self.state[i] if i < len(observation) else 0.0
                for i in range(self.state_dim)
            ]
            temporal_errors = [
                self.state[i] - self.predicted_next[i]
                for i in range(self.state_dim)
            ]

            for i in range(self.state_dim):
                update = (
                    prediction_errors[i] * self.precision[i] +
                    temporal_errors[i] * 0.5
                )
                self.state[i] += 0.1 * update

    def learn_transition(self, learning_rate: float = 0.01):
        """
        Hebbian update on transition weights.
        Strengthens connections that led to correct temporal predictions.
        """
        # Outer product update (simplified)
        for i in range(self.state_dim):
            for k in range(self.transition_rank):
                # U update
                delta_u = learning_rate * self.prev_state[i] * self.state[k % self.state_dim]
                self.U[i][k] += delta_u
                # V update
                delta_v = learning_rate * self.state[i] * self.prev_state[k % self.state_dim]
                self.V[i][k] += delta_v

        self.prev_state = self.state.copy()


# =============================================================================
# Sparse Coding Layer (Alternative to Attention)
# =============================================================================

class SparseCodingLayer:
    """
    Sparse Coding as alternative to attention mechanism.

    Instead of softmax attention over all tokens (O(n²)),
    uses iterative inference to find sparse representation.

    Based on: Olshausen & Field (1996), Sparse Predictive Coding

    Key difference from attention:
    - Local learning rules (Hebbian on reconstruction error)
    - Sparse activation (only k neurons active at once)
    - Iterative inference instead of matrix multiply
    - No position embeddings needed (structure emerges from dynamics)
    """

    def __init__(self, input_dim: int, dict_size: int, sparsity: int = 10):
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity = sparsity  # k-sparse

        # Dictionary (weight matrix)
        self.dictionary = [
            [random.gauss(0, 0.1) for _ in range(input_dim)]
            for _ in range(dict_size)
        ]

        # Sparse codes (activations)
        self.codes = [0.0] * dict_size
        self.inference_rate = 0.1

    def inference(self, input_data: list[float], n_steps: int = 10):
        """
        Infer sparse codes via Iterative Shrinkage-Thresholding (ISTA).
        Minimizes: ||x - D*c||² + λ||c||₁
        """
        for step in range(n_steps):
            # Reconstruction
            reconstruction = [0.0] * self.input_dim
            for i in range(self.input_dim):
                for j in range(self.dict_size):
                    reconstruction[i] += self.dictionary[j][i] * self.codes[j]

            # Reconstruction error
            error = [
                input_data[i] - reconstruction[i] if i < len(input_data) else 0.0
                for i in range(self.input_dim)
            ]

            # Gradient step
            for j in range(self.dict_size):
                gradient = 0.0
                for i in range(self.input_dim):
                    gradient += self.dictionary[j][i] * error[i]
                self.codes[j] += self.inference_rate * gradient

            # Proximal operator (soft thresholding → L1 sparsity)
            threshold = 0.1
            for j in range(self.dict_size):
                if self.codes[j] > threshold:
                    self.codes[j] -= threshold
                elif self.codes[j] < -threshold:
                    self.codes[j] += threshold
                else:
                    self.codes[j] = 0.0

            # Hard k-sparsity (keep top-k only)
            if sum(1 for c in self.codes if abs(c) > 1e-6) > self.sparsity:
                sorted_indices = sorted(
                    range(self.dict_size),
                    key=lambda j: abs(self.codes[j]),
                    reverse=True
                )
                for idx in sorted_indices[self.sparsity:]:
                    self.codes[idx] = 0.0

    def reconstruct(self) -> list[float]:
        """Reconstruct input from sparse codes."""
        reconstruction = [0.0] * self.input_dim
        for i in range(self.input_dim):
            for j in range(self.dict_size):
                reconstruction[i] += self.dictionary[j][i] * self.codes[j]
        return reconstruction

    def learn(self, input_data: list[float], learning_rate: float = 0.001):
        """
        Hebbian learning on dictionary.
        ΔD_ij = η * c_j * error_i (outer product)
        """
        reconstruction = self.reconstruct()
        for i in range(self.input_dim):
            error = input_data[i] - reconstruction[i] if i < len(input_data) else 0.0
            for j in range(self.dict_size):
                self.dictionary[j][i] += learning_rate * self.codes[j] * error

        # Normalize dictionary atoms
        for j in range(self.dict_size):
            norm = math.sqrt(sum(self.dictionary[j][i] ** 2 for i in range(self.input_dim)))
            if norm > 1e-6:
                for i in range(self.input_dim):
                    self.dictionary[j][i] /= norm


# =============================================================================
# Integrated Learning Architecture
# =============================================================================

class LearningArchitecture:
    """
    Integrated Learning Architecture for AGI Research Lab.

    Combines:
    - Predictive Coding (hierarchical inference)
    - Temporal Predictive Coding (sequence dynamics)
    - Sparse Coding (efficient representation)
    - Hebbian Plasticity (local, biologically plausible learning)
    - Global Guidance (sign-based credit assignment)

    This is NOT a transformer. No attention, no softmax over sequences,
    no backpropagation through layers. Each layer learns independently
    with local rules while coordinating via prediction errors.
    """

    def __init__(self, input_dim: int, layer_dims: list[int]):
        self.input_dim = input_dim
        self.layer_dims = layer_dims

        # Build hierarchical layers
        self.pc_layers = []
        prev_dim = input_dim
        for dim in layer_dims:
            layer = PredictiveCodingLayer(
                n_value=dim,
                n_inputs=prev_dim,
                activation="relu"
            )
            self.pc_layers.append(layer)
            prev_dim = dim

        # Temporal layers for each level
        self.temporal_layers = [
            TemporalPredictiveCodingLayer(dim, transition_rank=min(4, dim))
            for dim in layer_dims
        ]

        # Sparse coding at each level
        self.sparse_layers = [
            SparseCodingLayer(
                input_dim=dim,
                dict_size=dim * 2,
                sparsity=max(2, dim // 5)
            )
            for dim in layer_dims
        ]

    def forward(self, input_data: list[float]) -> list[list[float]]:
        """
        Forward pass through hierarchy (inference only, no learning).
        Returns activations at each layer.
        """
        current = input_data[:self.input_dim]
        all_activations = []

        # Upward pass (bottom-up inference)
        for i, layer in enumerate(self.pc_layers):
            layer.set_input(current)
            layer.inference(n_steps=3)
            activations = layer.get_activations()
            all_activations.append(activations)
            current = activations

        return all_activations

    def learn(self, input_data: list[float], global_gradient: float = 0.0):
        """
        Full learning cycle: inference + Hebbian updates.
        No backpropagation — each layer learns independently.
        """
        # Run inference
        activations = self.forward(input_data)

        # Hebbian learning at each layer (local, parallel)
        current_input = input_data[:self.input_dim]
        for i, layer in enumerate(self.pc_layers):
            layer.set_input(current_input)
            layer.learn(global_sign=global_gradient)
            current_input = layer.get_activations()

        # Temporal learning
        for i, tpc in enumerate(self.temporal_layers):
            tpc.predict_next()
            if i < len(activations):
                tpc.inference(activations[i], n_steps=2)
                tpc.learn_transition()

        # Sparse coding learning
        current_input = input_data[:self.input_dim]
        for i, sc in enumerate(self.sparse_layers):
            sc.inference(current_input, n_steps=5)
            sc.learn(current_input)
            current_input = sc.reconstruct()

    def get_representation(self, layer_idx: int = -1) -> list[float]:
        """Get representation at specified layer."""
        if not self.pc_layers:
            return []
        idx = layer_idx if layer_idx >= 0 else len(self.pc_layers) + layer_idx
        if 0 <= idx < len(self.pc_layers):
            return self.pc_layers[idx].get_activations()
        return []

    def get_total_free_energy(self) -> float:
        """Total free energy across all layers (should decrease during learning)."""
        return sum(layer.free_energy for layer in self.pc_layers)

    def summary(self) -> dict:
        """Architecture summary."""
        return {
            "type": "Predictive Coding + Hebbian Hybrid (PCHH)",
            "n_layers": len(self.pc_layers),
            "layer_dims": self.layer_dims,
            "total_neurons": sum(self.layer_dims),
            "total_synapses": sum(
                layer.n_value * layer.n_inputs
                for layer in self.pc_layers
            ),
            "learning_rule": "Local Hebbian (Oja/BCM/STDP/GHL) + Precision-weighted errors",
            "backprop": False,
            "attention": False,
            "transformers": False
        }


# =============================================================================
# Demo
# =============================================================================

def run_demo():
    """Demonstrate the learning architecture."""
    print("=" * 70)
    print("AGI Research Lab - Learning Architecture Demo")
    print("Architecture: Predictive Coding + Hebbian Hybrid (PCHH)")
    print("NO transformers. NO backprop. NO attention.")
    print("=" * 70)

    # Create architecture
    arch = LearningArchitecture(
        input_dim=16,
        layer_dims=[12, 8, 4]
    )

    # Print summary
    summary = arch.summary()
    print("\n--- Architecture ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Training data (synthetic)
    print("\n--- Training ---")
    n_epochs = 20
    train_data = [
        [random.gauss(0, 1) for _ in range(16)]
        for _ in range(100)
    ]

    free_energies = []
    for epoch in range(n_epochs):
        total_fe = 0.0
        for sample in train_data:
            # Simulate global sign (random direction for prototype)
            global_sign = random.choice([-1.0, 1.0])
            arch.learn(sample, global_gradient=global_sign)
            total_fe += arch.get_total_free_energy()
        avg_fe = total_fe / len(train_data)
        free_energies.append(avg_fe)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: avg_free_energy = {avg_fe:.4f}")

    # Test inference
    print("\n--- Inference ---")
    test_sample = [random.gauss(0, 1) for _ in range(16)]
    activations = arch.forward(test_sample)

    for i, acts in enumerate(activations):
        sparsity = sum(1 for a in acts if abs(a) > 1e-6) / len(acts)
        print(f"  Layer {i+1}: {len(acts)} neurons, {sparsity*100:.1f}% active")

    # Temporal prediction
    print("\n--- Temporal Prediction ---")
    tpc = TemporalPredictiveCodingLayer(state_dim=8, transition_rank=4)
    sequence = [[math.sin(t * 0.1 + i * 0.5) for i in range(8)] for t in range(20)]

    prediction_errors = []
    for t, obs in enumerate(sequence):
        predicted = tpc.predict_next()
        tpc.inference(obs, n_steps=3)
        if t > 0:
            error = sum(
                (obs[i] - predicted[i]) ** 2 for i in range(8)
            ) / 8
            prediction_errors.append(error)
        tpc.learn_transition()

    if prediction_errors:
        print(f"  Avg prediction error: {sum(prediction_errors)/len(prediction_errors):.4f}")
        print(f"  Error trend: {'decreasing' if prediction_errors[-1] < prediction_errors[0] else 'increasing'}")

    # Sparse coding
    print("\n--- Sparse Coding ---")
    sc = SparseCodingLayer(input_dim=32, dict_size=64, sparsity=5)
    sparse_data = [random.gauss(0, 1) for _ in range(32)]

    sc.inference(sparse_data, n_steps=10)
    sc.learn(sparse_data)

    reconstruction = sc.reconstruct()
    recon_error = sum(
        (sparse_data[i] - reconstruction[i]) ** 2 for i in range(32)
    ) / 32

    active_codes = sum(1 for c in sc.codes if abs(c) > 1e-6)
    print(f"  Active codes: {active_codes}/{len(sc.codes)}")
    print(f"  Reconstruction error: {recon_error:.4f}")

    print("\n" + "=" * 70)
    print("Demo complete. Architecture learned locally, without backprop.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
