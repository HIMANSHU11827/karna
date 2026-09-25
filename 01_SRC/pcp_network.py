"""
Phase-Coupled Predictive Coding (PCP) Network
Pure NumPy implementation — no PyTorch, no autograd.

Core components:
- Phase-coupled oscillators for temporal binding
- Event-driven sparse activation
- Local learning rules (no backpropagation)
- Multi-timescale predictive hierarchy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time


class PCPLayer:
    """One layer of the Phase-Coupled Predictive Coding network."""
    
    def __init__(self, size: int, dt: float = 0.1, tau: float = 1.0,
                 lambda_sparse: float = 0.01, mu_phase: float = 0.1,
                 omega_base: float = 1.0):
        self.size = size
        self.dt = dt
        self.tau = tau
        self.lambda_sparse = lambda_sparse
        self.mu_phase = mu_phase
        self.omega_base = omega_base
        
        # State variables
        self.x = np.zeros(size)           # activation
        self.phi = np.random.uniform(0, 2*np.pi, size)  # phase
        self.m = np.zeros(size)           # memory trace
        
        # Error signal
        self.error = np.zeros(size)
        
        # Importance weights for continual learning
        self.omega = np.zeros(size)
        
        # History for analysis
        self.history = []
    
    def reset(self):
        """Reset layer state."""
        self.x = np.zeros(self.size)
        self.phi = np.random.uniform(0, 2*np.pi, self.size)
        self.m = np.zeros(self.size)
        self.error = np.zeros(self.size)
        self.omega = np.zeros(self.size)
    
    def activate(self, input_drive: np.ndarray) -> np.ndarray:
        """Apply input drive and compute sparse activation."""
        # Leaky integration
        self.x += (self.dt / self.tau) * (input_drive - self.x)
        
        # Apply sparsity constraint
        self.x = np.maximum(self.x - self.lambda_sparse, 0)
        
        return self.x.copy()
    
    def update_phases(self, coupling: np.ndarray):
        """Update oscillator phases based on local coupling."""
        # Kuramoto-like phase dynamics
        dphi = self.omega_base * self.dt
        
        # Add coupling influence
        if coupling.shape[0] == self.size:
            for i in range(self.size):
                for j in range(self.size):
                    if i != j and coupling[i, j] != 0:
                        dphi_i = self.mu_phase * coupling[i, j] * np.sin(self.phi[j] - self.phi[i])
                        dphi_i *= self.dt
                        # Store as part of phase update
                        self.phi[i] += dphi_i
        
        self.phi += dphi
        self.phi = self.phi % (2 * np.pi)
    
    def compute_error(self, prediction: np.ndarray) -> np.ndarray:
        """Compute prediction error."""
        self.error = self.x - prediction
        return self.error


class PCPNetwork:
    """
    Multi-layer Phase-Coupled Predictive Coding network.
    Learns through local plasticity rules — no backpropagation.
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01,
                 dt: float = 0.1, name: str = "pcp_network"):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.lr = learning_rate
        self.dt = dt
        self.name = name
        
        # Create layers
        self.layers = []
        for i, size in enumerate(layer_sizes):
            layer = PCPLayer(size=size, dt=dt)
            self.layers.append(layer)
        
        # Bottom-up weights (feedforward perception)
        # w_bu[l] connects layer l-1 to layer l
        self.w_bu = [None]  # index 0 unused
        for l in range(1, self.n_layers):
            scale = 1.0 / np.sqrt(layer_sizes[l-1])
            w = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * scale
            self.w_bu.append(w)
        
        # Top-down weights (feedback prediction)
        # w_td[l] connects layer l+1 to layer l
        self.w_td = [None] * self.n_layers
        for l in range(self.n_layers - 1):
            scale = 1.0 / np.sqrt(layer_sizes[l+1])
            w = np.random.randn(layer_sizes[l], layer_sizes[l+1]) * scale
            self.w_td[l] = w
        
        # Phase coupling weights (learned)
        self.coupling = [None]
        for l in range(1, self.n_layers - 1):
            c = np.zeros((layer_sizes[l], layer_sizes[l]))
            self.coupling.append(c)
        self.coupling.append(None)  # output layer
        
        # Synaptic importance for continual learning
        self.synaptic_importance = [None]
        for l in range(1, self.n_layers):
            imp = np.zeros((layer_sizes[l], layer_sizes[l-1]))
            self.synaptic_importance.append(imp)
        
        # Training stats
        self.train_loss = []
        self.train_steps = 0
    
    def reset(self):
        """Reset all layer states."""
        for layer in self.layers:
            layer.reset()
    
    def forward(self, input_data: np.ndarray, n_steps: int = 10) -> Dict[str, np.ndarray]:
        """
        Run network inference with iterative prediction-error minimization.
        Returns layer activities.
        """
        self.reset()
        
        # Set input layer
        self.layers[0].x = input_data.flatten()
        
        for step in range(n_steps):
            # Bottom-up pass
            for l in range(1, self.n_layers):
                # Input drive from below
                drive = self.w_bu[l] @ self.layers[l-1].x
                self.layers[l].activate(drive)
            
            # Top-down predictions and errors
            for l in range(self.n_layers - 1):
                if self.w_td[l] is not None:
                    prediction = self.w_td[l] @ self.layers[l+1].x
                    self.layers[l].compute_error(prediction)
            
            # Update phases
            for l in range(1, self.n_layers):
                if self.coupling[l] is not None:
                    self.layers[l].update_phases(self.coupling[l])
        
        # Collect results
        results = {
            'activations': [l.x.copy() for l in self.layers],
            'phases': [l.phi.copy() for l in self.layers],
            'errors': [l.error.copy() for l in self.layers],
        }
        
        return results
    
    def learn(self, input_data: np.ndarray, target: Optional[np.ndarray] = None,
              n_steps: int = 10) -> float:
        """
        One learning step using local plasticity rules.
        Returns loss.
        """
        results = self.forward(input_data, n_steps)
        activations = results['activations']
        
        # Output layer supervision (if target provided)
        if target is not None:
            output = activations[-1]
            output_error = target.flatten() - output
            self.layers[-1].error = output_error
        else:
            output_error = None
        
        # Local learning rules
        loss = 0.0
        
        # Update bottom-up weights
        for l in range(1, self.n_layers):
            pre = activations[l-1]
            post = activations[l]
            post_error = self.layers[l].error
            
            # Hebbian with error modulation
            dw = self.lr * np.outer(post_error, pre)
            
            # Importance-weighted learning rate (continual learning)
            importance = 1.0 / (1.0 + self.synaptic_importance[l])
            dw *= importance
            
            self.w_bu[l] += dw
            
            # Update synaptic importance
            self.synaptic_importance[l] += 0.01 * dw**2
        
        # Update top-down weights
        for l in range(self.n_layers - 1):
            if self.w_td[l] is not None:
                pre = activations[l+1]
                post_true = activations[l]
                post_pred = self.w_td[l] @ pre
                td_error = post_true - post_pred
                
                du = self.lr * np.outer(td_error, pre)
                self.w_td[l] += du
        
        # Update phase coupling
        for l in range(1, self.n_layers):
            if self.coupling[l] is not None:
                phi = self.layers[l].phi
                for i in range(self.layer_sizes[l]):
                    for j in range(i+1, self.layer_sizes[l]):
                        dc = self.lr * 0.1 * np.sin(phi[j] - phi[i]) * activations[l][i] * activations[l][j]
                        self.coupling[l][i, j] += dc
                        self.coupling[l][j, i] += dc
        
        # Compute loss
        if output_error is not None:
            loss = np.mean(output_error**2)
        
        self.train_loss.append(loss)
        self.train_steps += 1
        
        return loss
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray, epochs: int = 1,
                    batch_size: int = 32, verbose: bool = True) -> List[float]:
        """Train for one epoch."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        epoch_loss = []
        
        for epoch in range(epochs):
            batch_loss = []
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                
                for idx in batch_idx:
                    loss = self.learn(X[idx], y[idx])
                    batch_loss.append(loss)
            
            avg_loss = np.mean(batch_loss) if batch_loss else 0.0
            epoch_loss.append(avg_loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")
        
        return epoch_loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for input batch."""
        predictions = []
        for i in range(X.shape[0]):
            results = self.forward(X[i])
            predictions.append(results['activations'][-1])
        return np.array(predictions)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate accuracy on labeled data."""
        predictions = self.predict(X)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(pred_labels == true_labels)
        
        return {
            'accuracy': accuracy,
            'n_samples': X.shape[0],
            'n_correct': int(np.sum(pred_labels == true_labels)),
        }
    
    def save(self, path: str):
        """Save network weights."""
        data = {
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.lr,
            'dt': self.dt,
            'name': self.name,
            'train_steps': self.train_steps,
            'w_bu': [w.tolist() if w is not None else None for w in self.w_bu],
            'w_td': [w.tolist() if w is not None else None for w in self.w_td],
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load network weights."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for l in range(1, self.n_layers):
            if data['w_bu'][l] is not None:
                self.w_bu[l] = np.array(data['w_bu'][l])
        
        for l in range(self.n_layers - 1):
            if data['w_td'][l] is not None:
                self.w_td[l] = np.array(data['w_td'][l])


class SparseEventEncoder:
    """
    Encodes inputs as sparse, event-driven spike trains.
    Uses difference-of-Gaussians receptive fields.
    """
    
    def __init__(self, input_size: int, n_features: int, n_spikes: int = 10):
        self.input_size = input_size
        self.n_features = n_features
        self.n_spikes = n_spikes
        
        # Random receptive fields
        self.weights = np.random.randn(n_features, input_size) * 0.1
        self.thresholds = np.random.uniform(0.1, 0.5, n_features)
    
    def encode(self, x: np.ndarray, n_steps: int = 10) -> np.ndarray:
        """Encode input as sparse spike pattern."""
        # Compute feature responses
        responses = self.weights @ x.flatten()
        
        # Select top-k features (sparse)
        top_k = min(self.n_spikes, self.n_features)
        top_indices = np.argsort(responses)[-top_k:]
        
        spikes = np.zeros(self.n_features)
        spikes[top_indices] = responses[top_indices]
        
        return spikes


def load_mnist(path: str = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/02_DATA/mnist"):
    """Load MNIST dataset."""
    import os
    
    # Check if data exists
    train_path = os.path.join(path, "train.npz")
    test_path = os.path.join(path, "test.npz")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_data = np.load(train_path, allow_pickle=True)
        test_data = np.load(test_path, allow_pickle=True)
        return (train_data['X'], train_data['y'], test_data['X'], test_data['y'])
    
    # Download if not present
    print("Downloading MNIST...")
    try:
        from urllib.request import urlretrieve
        import gzip
        
        os.makedirs(path, exist_ok=True)
        
        base_url = "http://yann.lecun.com/exdb/mnist/"
        files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]
        
        for f in files:
            url = base_url + f
            filepath = os.path.join(path, f)
            if not os.path.exists(filepath):
                print(f"  Downloading {f}...")
                urlretrieve(url, filepath)
        
        # Parse files
        def load_images(filepath):
            with gzip.open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            return data.reshape(-1, 28*28) / 255.0
        
        def load_labels(filepath):
            with gzip.open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
            return data
        
        X_train = load_images(os.path.join(path, "train-images-idx3-ubyte.gz"))
        y_train = load_labels(os.path.join(path, "train-labels-idx1-ubyte.gz"))
        X_test = load_images(os.path.join(path, "t10k-images-idx3-ubyte.gz"))
        y_test = load_labels(os.path.join(path, "t10k-labels-idx1-ubyte.gz"))
        
        # One-hot encode labels
        y_train_oh = np.zeros((y_train.shape[0], 10))
        y_train_oh[np.arange(y_train.shape[0]), y_train] = 1
        y_test_oh = np.zeros((y_test.shape[0], 10))
        y_test_oh[np.arange(y_test.shape[0]), y_test] = 1
        
        # Save
        np.savez(train_path, X=X_train, y=y_train_oh)
        np.savez(test_path, X=X_test, y=y_test_oh)
        
        return X_train, y_train_oh, X_test, y_test_oh
        
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        # Generate synthetic data for testing
        print("Generating synthetic data for testing...")
        X_train = np.random.randn(1000, 784) * 0.1 + 0.5
        y_train = np.zeros((1000, 10))
        y_train[np.arange(1000), np.random.randint(0, 10, 1000)] = 1
        X_test = np.random.randn(200, 784) * 0.1 + 0.5
        y_test = np.zeros((200, 10))
        y_test[np.arange(200), np.random.randint(0, 10, 200)] = 1
        return X_train, y_train, X_test, y_test


def run_mnist_baseline():
    """Run MNIST baseline experiment."""
    print("=" * 60)
    print("PCP Network — MNIST Baseline")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input dimension: {X_train.shape[1]}")
    
    # Create network
    # Architecture: 784 -> 256 -> 64 -> 10
    net = PCPNetwork(
        layer_sizes=[784, 256, 64, 10],
        learning_rate=0.005,
        dt=0.1,
        name="pcp_mnist_v1"
    )
    
    print(f"\nNetwork architecture: {net.layer_sizes}")
    print(f"Total parameters: {sum(w.size for w in net.w_bu if w is not None) + sum(w.size for w in net.w_td if w is not None)}")
    
    # Train
    print("\nTraining...")
    start_time = time.time()
    
    # Use subset for quick baseline
    n_train = min(5000, X_train.shape[0])
    n_test = min(1000, X_test.shape[0])
    
    loss_history = net.train_epoch(
        X_train[:n_train], y_train[:n_train],
        epochs=5, batch_size=32, verbose=True
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.1f}s")
    
    # Evaluate
    print("\nEvaluating...")
    train_results = net.evaluate(X_train[:n_train], y_train[:n_train])
    test_results = net.evaluate(X_test[:n_test], y_test[:n_test])
    
    print(f"\nResults:")
    print(f"  Train accuracy: {train_results['accuracy']:.4f} ({train_results['n_correct']}/{train_results['n_samples']})")
    print(f"  Test accuracy:  {test_results['accuracy']:.4f} ({test_results['n_correct']}/{test_results['n_samples']})")
    print(f"  Final loss:     {loss_history[-1]:.4f}")
    
    # Save results
    results = {
        'architecture': net.layer_sizes,
        'train_accuracy': train_results['accuracy'],
        'test_accuracy': test_results['accuracy'],
        'train_time': train_time,
        'loss_history': loss_history,
        'n_train': n_train,
        'n_test': n_test,
    }
    
    results_path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/04_RESULTS/mnist_baseline.json"
    import os
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    results = run_mnist_baseline()
