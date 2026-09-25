"""
AGI Research Lab - MNIST Training Pipeline
==========================================

Phase 2: Train PCHH (Predictive Coding + Hebbian Hybrid) on MNIST
- No backpropagation
- No transformers
- Local Hebbian updates only
- NumPy only (no PyTorch/TF)

Architecture:
  Input (784) -> PC1 (128) -> PC2 (64) -> PC3 (32) -> Output (10)

Training protocol:
  1. Clamp input image to layer 0
  2. Clamp target label to output layer
  3. Run inference (minimize free energy)
  4. Update weights with Hebbian rule (outer product of pre and post)
  5. Repeat

Baseline: Same architecture trained with backprop (for comparison).
"""

import numpy as np
import os
import sys
import time
import json
import urllib.request
import gzip
from pathlib import Path

# =============================================================================
# MNIST Data Loader
# =============================================================================

def download_mnist(data_dir="/tmp/mnist_data"):
    """Download MNIST dataset."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(base_url + filename, filepath)
                print(f"  Saved to {filepath}")
            except Exception as e:
                print(f"  Failed: {e}")
                return None
    
    return data_dir


def load_mnist(data_dir="/tmp/mnist_data"):
    """Load MNIST data from disk."""
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            # Skip magic number and dimensions
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data.astype(np.int32)
    
    train_images = load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_labels = load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_images = load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    
    return train_images, train_labels, test_images, test_labels


def generate_synthetic_mnist(n_samples=1000, seed=42):
    """Generate synthetic digit-like data for offline testing."""
    rng = np.random.RandomState(seed)
    images = rng.rand(n_samples, 784).astype(np.float32) * 0.5 + 0.2
    labels = rng.randint(0, 10, size=n_samples)
    # Add some structure per class
    for i in range(n_samples):
        cls = labels[i]
        # Add class-specific blob pattern
        row_start = (cls // 3) * 8
        col_start = (cls % 3) * 8
        for r in range(row_start, min(row_start + 12, 28)):
            for c in range(col_start, min(col_start + 12, 28)):
                images[i, r * 28 + c] += 0.3
    images = np.clip(images, 0, 1)
    return images, labels


# =============================================================================
# Predictive Coding Layer (Production Version)
# =============================================================================

class PCLayer:
    """
    Predictive Coding Layer with Hebbian learning.
    
    Each layer receives:
      - Bottom-up input (from layer below)
      - Top-down prediction (from layer above)
    
    Each layer outputs:
      - Value activations (sparse representation)
      - Prediction errors (residuals)
    
    Learning: Local Hebbian update on weights
      ΔW = η * error * input^T
    """

    def __init__(self, n_inputs, n_outputs, learning_rate=0.01,
                 n_iterations=5, sparsity=0.15):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.sparsity = sparsity
        
        # Weight matrix (generative model)
        self.W = np.random.randn(n_inputs, n_outputs).astype(np.float32) * 0.1
        # Precision (inverse variance) per output neuron
        self.precision = np.ones(n_outputs, dtype=np.float32)
        # Bias
        self.bias = np.zeros(n_outputs, dtype=np.float32)
        
        # State
        self.values = np.zeros(n_outputs, dtype=np.float32)
        self.predictions = np.zeros(n_outputs, dtype=np.float32)
        self.errors = np.zeros(n_outputs, dtype=np.float32)
        
    def set_input(self, bottom_up):
        """Set bottom-up input from layer below."""
        self.bottom_up = bottom_up
        
    def set_prediction(self, top_down):
        """Set top-down prediction from layer above."""
        self.predictions = top_down
        
    def infer(self, bottom_up, top_down):
        """
        Inference phase: minimize prediction error.
        Updates value activations to minimize:
          F = ||bottom_up - W @ values||^2 + precision * ||values - top_down||^2
        """
        self.bottom_up = bottom_up
        self.predictions = top_down
        
        # Initialize values (can be warm-started)
        self.values = np.zeros(self.n_outputs, dtype=np.float32)
        
        dt = 0.1
        for iteration in range(self.n_iterations):
            # Predict bottom-up from current values
            predicted_input = self.W @ self.values
            
            # Bottom-up error
            bu_error = bottom_up - predicted_input
            
            # Top-down error
            td_error = top_down - self.values
            
            # Gradient of free energy w.r.t. values
            # dF/dv = -W^T @ bu_error - precision * td_error
            grad = -self.W.T @ bu_error - self.precision * td_error
            
            # Update values (gradient descent on free energy)
            self.values -= dt * grad
            
            # ReLU (non-negative activations)
            self.values = np.maximum(self.values, 0)
            
            # Sparse k-sparsity: keep top-k only
            k = max(1, int(self.n_outputs * self.sparsity))
            top_k_idx = np.argsort(self.values)[-k:]
            sparse_values = np.zeros_like(self.values)
            sparse_values[top_k_idx] = self.values[top_k_idx]
            self.values = sparse_values
            
            # Normalize to prevent runaway
            norm = np.linalg.norm(self.values)
            if norm > 10:
                self.values *= 10 / norm
        
        # Compute final errors
        self.errors = self.values - self.predictions
        
        return self.values
    
    def learn(self, bottom_up, values, errors, global_sign=0.0):
        """
        Hebbian learning: update weights.
        
        ΔW = η * errors * bottom_up^T
        
        For GHL (global-guided): multiply by global_sign
        """
        # Outer product update
        # errors shape: (n_outputs,)
        # bottom_up shape: (n_inputs,)
        delta_W = np.outer(bottom_up, errors) * self.learning_rate
        
        # Apply global sign if GHL
        if global_sign != 0:
            delta_W *= global_sign
        
        self.W += delta_W
        
        # Oja-style stabilization (prevent unbounded growth)
        self.W *= 0.999
        
        # Clip weights
        self.W = np.clip(self.W, -2, 2)
        
        # Update precision (learned reliability)
        prediction_error_sq = np.mean(errors ** 2)
        self.precision = 1.0 / (1.0 + prediction_error_sq + 1e-6)
        
        return delta_W
    
    def predict_up(self, values):
        """Predict layer below's activity."""
        return self.W @ values
    
    def predict_down(self, top_down_activity):
        """Predict this layer's activity from layer above."""
        return self.W.T @ top_down_activity


# =============================================================================
# PCHH Network
# =============================================================================

class PCHHNetwork:
    """
    Predictive Coding + Hebbian Hybrid Network.
    
    Architecture: Stack of PC layers.
    Output layer is clamped to target during training.
    """

    def __init__(self, layer_sizes, learning_rate=0.01, n_iterations=5):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = PCLayer(
                n_inputs=layer_sizes[i],
                n_outputs=layer_sizes[i + 1],
                learning_rate=learning_rate,
                n_iterations=n_iterations
            )
            self.layers.append(layer)
        self.n_layers = len(self.layers)
        
    def forward(self, input_data, target=None, learn_flag=True, global_sign=0.0):
        """
        Forward pass through hierarchy.
        
        Args:
            input_data: Input vector (batch_size, input_dim) or (input_dim,)
            target: Target vector for output layer (clamped during training)
            learn_flag: Whether to update weights
            global_sign: Global gradient sign for GHL
            
        Returns:
            activations: List of activations at each layer
        """
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        batch_size = input_data.shape[0]
        activations = [input_data]  # Layer 0 is input
        
        # Upward pass (bottom-up inference)
        for i, layer in enumerate(self.layers):
            bottom_up = activations[-1]
            
            # Top-down prediction (from layer above, if exists)
            if i < len(self.layers) - 1:
                # No top-down for first pass (will be refined)
                top_down = np.zeros(layer.n_outputs)
            else:
                # Output layer: clamp to target
                if target is not None:
                    top_down = target
                else:
                    top_down = np.zeros(layer.n_outputs)
            
            # Infer values for this layer
            layer_activations = []
            for b in range(batch_size):
                bu = bottom_up[b] if bottom_up.ndim > 1 else bottom_up
                vals = layer.infer(bu, top_down)
                layer_activations.append(vals)
            
            layer_activations = np.array(layer_activations)
            activations.append(layer_activations)
            
            if learn_flag:
                # Learn this layer
                errors = layer.errors
                for b in range(batch_size):
                    bu = bottom_up[b] if bottom_up.ndim > 1 else bottom_up
                    vals = layer_activations[b]
                    layer.learn(bu, vals, errors, global_sign)
        
        return activations
    
    def predict(self, input_data):
        """Predict output (no learning)."""
        activations = self.forward(input_data, target=None, learn_flag=False)
        return activations[-1]
    
    def get_total_free_energy(self):
        """Total free energy across all layers."""
        total_fe = 0
        for layer in self.layers:
            total_fe += np.sum(layer.errors ** 2)
        return total_fe


# =============================================================================
# Backprop Baseline (for comparison)
# =============================================================================

class BackpropBaseline:
    """Same architecture as PCHH but trained with backprop."""

    def __init__(self, layer_sizes, learning_rate=0.01):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]).astype(np.float32)
            w *= np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(w)
            self.biases.append(np.zeros(layer_sizes[i+1], dtype=np.float32))
        self.lr = learning_rate

    def forward(self, x):
        """Forward pass."""
        activations = [x]
        pre_activations = []
        for w, b in zip(self.weights, self.biases):
            z = activations[-1] @ w + b
            pre_activations.append(z)
            a = np.maximum(z, 0)  # ReLU
            activations.append(a)
        return activations, pre_activations

    def backward(self, activations, pre_activations, target):
        """Backward pass (backprop)."""
        batch_size = target.shape[0]
        # MSE loss
        output = activations[-1]
        loss = np.mean((output - target) ** 2)
        
        # Output layer gradient
        delta = 2 * (output - target) / batch_size
        
        weight_grads = []
        bias_grads = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Weight gradient
            grad_w = activations[i].T @ delta
            grad_b = np.sum(delta, axis=0)
            weight_grads.insert(0, grad_w)
            bias_grads.insert(0, grad_b)
            
            # Propagate gradient
            if i > 0:
                delta = delta @ self.weights[i].T
                # ReLU derivative
                delta *= (pre_activations[i-1] > 0).astype(np.float32)
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * weight_grads[i]
            self.biases[i] -= self.lr * bias_grads[i]
        
        return loss

    def train_step(self, x, y):
        """Single training step."""
        activations, pre_activations = self.forward(x)
        loss = self.backward(activations, pre_activations, y)
        return loss

    def predict(self, x):
        """Predict output."""
        activations, _ = self.forward(x)
        return activations[-1]


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, train_images, train_labels, batch_size=32,
                epochs=5, is_pchh=True, log_prefix=""):
    """Train for one epoch."""
    n_samples = len(train_images)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    total_loss = 0
    n_batches = 0
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        
        images = train_images[batch_idx]
        labels = train_labels[batch_idx]
        
        # One-hot encode labels
        targets = np.zeros((len(labels), 10), dtype=np.float32)
        for i, label in enumerate(labels):
            targets[i, label] = 1.0
        
        if is_pchh:
            # PCHH training: clamp target to output layer
            for b in range(len(images)):
                global_sign = np.random.choice([-1.0, 1.0])
                model.forward(images[b], target=targets[b], learn_flag=True,
                             global_sign=global_sign)
            total_loss += model.get_total_free_energy()
        else:
            # Backprop training
            loss = model.train_step(images, targets)
            total_loss += loss
        
        n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def evaluate(model, test_images, test_labels, is_pchh=True):
    """Evaluate model accuracy."""
    correct = 0
    total = len(test_images)
    
    for i in range(total):
        image = test_images[i]
        label = test_labels[i]
        
        if is_pchh:
            output = model.predict(image)
        else:
            output = model.predict(image.reshape(1, -1))
        
        predicted = np.argmax(output)
        if predicted == label:
            correct += 1
    
    return correct / total


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run full MNIST experiment."""
    print("=" * 70)
    print("AGI Research Lab - Phase 2: MNIST Training")
    print("Architecture: PCHH (Predictive Coding + Hebbian Hybrid)")
    print("Baseline: Same architecture with backprop")
    print("=" * 70)
    
    # Try to load MNIST, fall back to synthetic
    data_dir = download_mnist()
    if data_dir and all(
        os.path.exists(os.path.join(data_dir, f))
        for f in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                  "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    ):
        print("Loading MNIST...")
        train_images, train_labels, test_images, test_labels = load_mnist()
        print(f"  Train: {len(train_images)} images")
        print(f"  Test: {len(test_images)} images")
        dataset = "mnist"
    else:
        print("MNIST download failed, using synthetic data...")
        train_images, train_labels = generate_synthetic_mnist(6000, seed=42)
        test_images, test_labels = generate_synthetic_mnist(1000, seed=43)
        dataset = "synthetic"
        print(f"  Synthetic train: {len(train_images)} images")
        print(f"  Synthetic test: {len(test_images)} images")
    
    # Network architecture
    input_dim = 784
    hidden1 = 128
    hidden2 = 64
    hidden3 = 32
    output_dim = 10
    layer_sizes = [input_dim, hidden1, hidden2, hidden3, output_dim]
    
    # Training config
    epochs = 5
    batch_size = 32
    lr = 0.001
    
    # Initialize models
    pchh_model = PCHHNetwork(layer_sizes, learning_rate=lr, n_iterations=3)
    bp_model = BackpropBaseline(layer_sizes, learning_rate=lr)
    
    results = {
        "dataset": dataset,
        "architecture": layer_sizes,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "pchh": {"train_loss": [], "test_acc": [], "time": []},
        "backprop": {"train_loss": [], "test_acc": [], "time": []}
    }
    
    print(f"\nArchitecture: {layer_sizes}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print()
    
    # Training
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        
        # PCHH
        start = time.time()
        pchh_loss = train_epoch(pchh_model, train_images, train_labels,
                                batch_size, 1, is_pchh=True)
        pchh_time = time.time() - start
        pchh_acc = evaluate(pchh_model, test_images, test_labels, is_pchh=True)
        
        results["pchh"]["train_loss"].append(float(pchh_loss))
        results["pchh"]["test_acc"].append(float(pchh_acc))
        results["pchh"]["time"].append(float(pchh_time))
        
        print(f"  PCHH    : loss={pchh_loss:.4f}, acc={pchh_acc:.4f}, time={pchh_time:.2f}s")
        
        # Backprop
        start = time.time()
        bp_loss = train_epoch(bp_model, train_images, train_labels,
                             batch_size, 1, is_pchh=False)
        bp_time = time.time() - start
        bp_acc = evaluate(bp_model, test_images, test_labels, is_pchh=False)
        
        results["backprop"]["train_loss"].append(float(bp_loss))
        results["backprop"]["test_acc"].append(float(bp_acc))
        results["backprop"]["time"].append(float(bp_time))
        
        print(f"  Backprop: loss={bp_loss:.4f}, acc={bp_acc:.4f}, time={bp_time:.2f}s")
        print()
    
    # Save results
    results_path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/11_LOGS/mnist_phase2_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"PCHH     - Final acc: {results['pchh']['test_acc'][-1]:.4f}, "
          f"Total time: {sum(results['pchh']['time']):.2f}s")
    print(f"Backprop - Final acc: {results['backprop']['test_acc'][-1]:.4f}, "
          f"Total time: {sum(results['backprop']['time']):.2f}s")
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_experiment()
