"""
AGI Research Lab - MNIST Training Pipeline (Fixed)
===================================================
- Download from alternative source
- Debug PCHH learning rule
- Test shallow network first
"""

import numpy as np
import os
import time
import json
import urllib.request
import gzip
from pathlib import Path


def download_mnist_fixed(data_dir="/tmp/mnist_data"):
    """Download MNIST from alternative mirrors."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Try multiple mirrors
    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]
    
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"  {filename} already exists")
            continue
        
        downloaded = False
        for mirror in mirrors:
            url = mirror + filename
            try:
                print(f"  Trying {url}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"    Saved to {filepath}")
                downloaded = True
                break
            except Exception as e:
                print(f"    Failed: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        if not downloaded:
            print(f"  Could not download {filename}")
            return None
    
    return data_dir


def load_mnist(data_dir="/tmp/mnist_data"):
    """Load MNIST data."""
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
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


# =============================================================================
# Simplified PCHH Layer (debugged)
# =============================================================================

class SimplePCLayer:
    """
    Simplified Predictive Coding Layer.
    
    Key insight: For credit assignment in deep networks, we need
    feedback about the global error. Options:
    1. Feedback weights (target propagation style)
    2. Global sign signal (GHL style)
    3. Local error only (works for shallow networks)
    """

    def __init__(self, n_inputs, n_outputs, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.lr = learning_rate
        
        # Weight matrix
        self.W = np.random.randn(n_inputs, n_outputs).astype(np.float32) * 0.1
        self.bias = np.zeros(n_outputs, dtype=np.float32)
        
        # State
        self.values = np.zeros(n_outputs, dtype=np.float32)
        self.input_cache = None
        self.error_cache = None
        
    def forward(self, bottom_up):
        """Forward pass: predict from input."""
        self.input_cache = bottom_up
        # Linear prediction
        prediction = bottom_up @ self.W + self.bias
        # ReLU
        self.values = np.maximum(prediction, 0)
        return self.values
    
    def learn_hebbian(self, global_error_sign=0.0):
        """
        Hebbian learning with optional global error sign.
        
        For shallow networks: local error is sufficient.
        For deep networks: need global_error_sign for credit assignment.
        """
        if self.input_cache is None:
            return
        
        # Local error (for output layer) or global sign
        if global_error_sign != 0:
            # GHL: use global sign to guide local Hebbian
            # Approximate: error ≈ global_sign * ones
            error = np.ones(self.n_outputs) * global_error_sign
        else:
            # For output layer: use actual prediction error
            error = self.error_cache if self.error_cache is not None else np.zeros(self.n_outputs)
        
        # Hebbian update: ΔW = η * input * error^T
        delta_W = np.outer(self.input_cache, error) * self.lr
        self.W += delta_W
        
        # Oja stabilization
        self.W *= 0.999
        self.W = np.clip(self.W, -2, 2)
        
        return delta_W
    
    def learn_with_target(self, target):
        """
        Learn with explicit target (for output layer).
        error = target - prediction
        """
        if self.input_cache is None:
            return
        
        error = target - self.values
        self.error_cache = error
        
        # Hebbian update
        delta_W = np.outer(self.input_cache, error) * self.lr
        self.W += delta_W
        self.bias += error * self.lr * 0.1
        
        # Stabilize
        self.W = np.clip(self.W, -2, 2)
        
        return np.mean(error ** 2)


# =============================================================================
# Shallow PCHH Network (1 hidden layer)
# =============================================================================

class ShallowPCHH:
    """
    Shallow PCHH: Input -> Hidden -> Output
    Only 1 hidden layer (easier to train with local rules).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.hidden = SimplePCLayer(input_dim, hidden_dim, lr)
        self.output = SimplePCLayer(hidden_dim, output_dim, lr)
        
    def forward(self, x, learn_flag=True, target=None):
        """Forward pass."""
        h = self.hidden.forward(x)
        o = self.output.forward(h)
        
        if learn_flag and target is not None:
            # Output layer: learn with target
            output_loss = self.output.learn_with_target(target)
            
            # Hidden layer: use output error as global signal
            # Approximate: propagate error sign through feedback
            output_error = target - o
            global_sign = np.sign(np.mean(output_error))
            self.hidden.learn_hebbian(global_error_sign=global_sign)
            
            return o, output_loss
        
        return o, 0
    
    def predict(self, x):
        """Predict output class."""
        h = self.hidden.forward(x)
        o = self.output.forward(h)
        return np.argmax(o)


# =============================================================================
# Backprop Baseline (shallow)
# =============================================================================

class ShallowBackprop:
    """Same architecture, trained with backprop."""

    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(output_dim, dtype=np.float32)
        self.lr = lr

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(self.z1, 0)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(self.z2, 0)
        return self.a2

    def backward(self, x, target):
        """Backprop."""
        batch_size = x.shape[0] if x.ndim > 1 else 1
        
        # Output error
        error = self.a2 - target
        loss = np.mean(error ** 2)
        
        # Gradients
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        
        dW2 = self.a1.T @ error / batch_size
        db2 = np.mean(error, axis=0)
        
        delta1 = (error @ self.W2.T) * (self.z1 > 0).astype(np.float32)
        dW1 = x.T @ delta1 / batch_size
        db1 = np.mean(delta1, axis=0)
        
        # Update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        return loss

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h = np.maximum(x @ self.W1 + self.b1, 0)
        o = np.maximum(h @ self.W2 + self.b2, 0)
        return np.argmax(o, axis=-1)


# =============================================================================
# Training
# =============================================================================

def train_model(model, train_images, train_labels, test_images, test_labels,
                epochs=10, batch_size=32, is_pchh=True):
    """Train a model."""
    n_samples = len(train_images)
    losses = []
    accs = []
    
    for epoch in range(epochs):
        # Shuffle
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        total_loss = 0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            images = train_images[batch_idx]
            labels = train_labels[batch_idx]
            
            # One-hot
            targets = np.zeros((len(labels), 10), dtype=np.float32)
            for i, label in enumerate(labels):
                targets[i, label] = 1.0
            
            if is_pchh:
                # PCHH: sample-by-sample
                epoch_loss = 0
                for b in range(len(images)):
                    _, loss = model.forward(images[b], learn_flag=True, target=targets[b])
                    epoch_loss += loss
                total_loss += epoch_loss / len(images)
            else:
                # Backprop: batch
                output = model.forward(images)
                loss = model.backward(images, targets)
                total_loss += loss
            
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Evaluate
        correct = 0
        for i in range(len(test_images)):
            pred = model.predict(test_images[i])
            if isinstance(pred, np.ndarray):
                pred = pred.item()
            if pred == test_labels[i]:
                correct += 1
        acc = correct / len(test_images)
        
        losses.append(float(avg_loss))
        accs.append(float(acc))
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.4f}")
    
    return losses, accs


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("AGI Research Lab - Phase 2: MNIST (Debugged)")
    print("=" * 70)
    
    # Download MNIST
    print("\nDownloading MNIST...")
    data_dir = download_mnist_fixed()
    
    if data_dir:
        print("Loading MNIST...")
        train_images, train_labels, test_images, test_labels = load_mnist()
        print(f"  Train: {len(train_images)}, Test: {len(test_images)}")
        dataset = "mnist"
    else:
        print("Download failed, generating synthetic data...")
        rng = np.random.RandomState(42)
        train_images = rng.rand(6000, 784).astype(np.float32) * 0.5 + 0.2
        train_labels = rng.randint(0, 10, 6000)
        test_images = rng.rand(1000, 784).astype(np.float32) * 0.5 + 0.2
        test_labels = rng.randint(0, 10, 1000)
        dataset = "synthetic"
        print(f"  Synthetic: {len(train_images)} train, {len(test_images)} test")
    
    # Use subset for speed
    n_train = min(6000, len(train_images))
    n_test = min(1000, len(test_images))
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]
    
    # Config
    input_dim = 784
    hidden_dim = 64
    output_dim = 10
    epochs = 10
    batch_size = 32
    lr = 0.001
    
    print(f"\nConfig: [{input_dim}, {hidden_dim}, {output_dim}], "
          f"epochs={epochs}, lr={lr}")
    print()
    
    # Train PCHH
    print("--- PCHH (Hebbian) ---")
    pchh = ShallowPCHH(input_dim, hidden_dim, output_dim, lr=lr)
    pchh_losses, pchh_accs = train_model(
        pchh, train_images, train_labels, test_images, test_labels,
        epochs, batch_size, is_pchh=True
    )
    
    # Train Backprop
    print("\n--- Backprop ---")
    bp = ShallowBackprop(input_dim, hidden_dim, output_dim, lr=lr)
    bp_losses, bp_accs = train_model(
        bp, train_images, train_labels, test_images, test_labels,
        epochs, batch_size, is_pchh=False
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"PCHH     - Final acc: {pchh_accs[-1]:.4f}")
    print(f"Backprop - Final acc: {bp_accs[-1]:.4f}")
    
    # Save results
    results = {
        "dataset": dataset,
        "config": {"input_dim": input_dim, "hidden_dim": hidden_dim,
                   "output_dim": output_dim, "epochs": epochs, "lr": lr},
        "pchh": {"losses": pchh_losses, "accs": pchh_accs},
        "backprop": {"losses": bp_losses, "accs": bp_accs}
    }
    results_path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/11_LOGS/mnist_phase2_debug.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    main()
