"""
AGI Research Lab - MNIST Training Pipeline
==========================================
PCHH Architecture: Predictive Coding + Hebbian Hybrid

NO backprop. NO autograd. NO PyTorch.
Pure NumPy + local Hebbian learning rules.

Architecture:
- Input: 784 (28x28 MNIST pixels)
- PC Layer 1: 256 neurons (ReLU, Oja's rule)
- PC Layer 2: 128 neurons (ReLU, BCM rule)
- PC Layer 3: 64 neurons (ReLU, GHL)
- Output: 10 neurons (softmax prediction)

Training:
- Inference phase: minimize prediction error (free energy)
- Learning phase: Hebbian weight updates (local, per-layer)
- Global guidance: sign of global reward signal (dopamine-like)
"""

import numpy as np
import time
import json
import pickle
import gzip
import urllib.request
import os
from pathlib import Path

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# =============================================================================
# MNIST Data Loader (No external dependencies)
# =============================================================================

def download_mnist(path='mnist_data'):
    """Download MNIST dataset if not present."""
    base = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    Path(path).mkdir(exist_ok=True)
    downloaded = {}
    for key, fname in files.items():
        filepath = os.path.join(path, fname)
        if not os.path.exists(filepath):
            print(f'Downloading {fname}...')
            urllib.request.urlretrieve(base + fname, filepath)
        downloaded[key] = filepath
    return downloaded


def load_mnist(path='mnist_data'):
    """Load MNIST dataset into numpy arrays."""
    try:
        files = download_mnist(path)
    except Exception as e:
        print(f'Download failed: {e}')
        print('Generating synthetic data for testing...')
        return generate_synthetic_mnist()
    
    def load_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            # Skip magic number and dimensions
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0
    
    def load_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data.astype(np.int32)
    
    return {
        'train_images': load_images(files['train_images']),
        'train_labels': load_labels(files['train_labels']),
        'test_images': load_images(files['test_images']),
        'test_labels': load_labels(files['test_labels'])
    }


def generate_synthetic_mnist(n_train=1000, n_test=200):
    """Generate synthetic MNIST-like data for offline testing."""
    print('Generating synthetic MNIST-like data...')
    rng = np.random.RandomState(42)
    return {
        'train_images': rng.randn(n_train, 784).astype(np.float32) * 0.3 + 0.5,
        'train_labels': rng.randint(0, 10, n_train).astype(np.int32),
        'test_images': rng.randn(n_test, 784).astype(np.float32) * 0.3 + 0.5,
        'test_labels': rng.randint(0, 10, n_test).astype(np.int32)
    }


def one_hot(labels, n_classes=10):
    """One-hot encode labels."""
    one_hot = np.zeros((len(labels), n_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


# =============================================================================
# PCHH Components (from predictive_coding_hybrid.py)
# =============================================================================

class ValueNeuron:
    """Value neuron with leaky integrator dynamics."""
    __slots__ = ['activation', 'prediction', 'prediction_error', 'precision',
                 'membrane_potential', 'bias', 'leak_rate']
    
    def __init__(self, n):
        self.activation = np.full(n, 0.1, dtype=np.float32)
        self.prediction = np.zeros(n, dtype=np.float32)
        self.prediction_error = np.zeros(n, dtype=np.float32)
        self.precision = np.ones(n, dtype=np.float32)
        self.membrane_potential = np.full(n, 0.1, dtype=np.float32)
        self.bias = np.full(n, 0.01, dtype=np.float32)
        self.leak_rate = 0.1


class ErrorNeuron:
    """Error neuron encoding prediction error."""
    __slots__ = ['activation']
    
    def __init__(self, n):
        self.activation = np.zeros(n, dtype=np.float32)


class HebbianSynapse:
    """Synapse with multiple Hebbian update rules."""
    
    def __init__(self, n_out, n_in, rule='oja'):
        self.rule = rule
        # Small random initialization (critical for breaking symmetry)
        self.weight = np.random.randn(n_out, n_in).astype(np.float32) * 0.1
        self.learning_rate = 0.01
        self.oja_gamma = 0.001
        
    def update(self, pre, post, global_sign=0.0):
        """Apply Hebbian update rule."""
        # pre shape: (n_in,)
        # post shape: (n_out,)
        if self.rule == 'oja':
            # Oja's rule: Δw = η * (pre * post - w * post²)
            post_sq = post ** 2  # (n_out,)
            delta = self.learning_rate * (
                np.outer(post, pre) - self.oja_gamma * self.weight * post_sq[:, None]
            )
        elif self.rule == 'bcm':
            # BCM rule: Δw = η * pre * post * (post - θ)
            theta = 0.5
            delta = self.learning_rate * np.outer(post * np.maximum(post - theta, 0), pre)
        elif self.rule == 'ghl':
            # Global-guided Hebbian: Δw = η * pre * post * sign(global_grad)
            delta = self.learning_rate * np.outer(post, pre) * global_sign
        else:  # plain hebb
            delta = self.learning_rate * np.outer(post, pre)
        
        self.weight += delta
        # Clip weights to prevent explosion
        np.clip(self.weight, -5.0, 5.0, out=self.weight)


class PCLayer:
    """
    Predictive Coding Layer.
    
    Inference: minimize prediction error (free energy)
    Learning: Hebbian weight updates (local, parallel)
    """
    
    def __init__(self, n_in, n_out, rule='oja', n_inference_steps=5):
        self.n_in = n_in
        self.n_out = n_out
        self.n_inference_steps = n_inference_steps
        self.dt = 0.1
        
        # Populations
        self.values = ValueNeuron(n_out)
        self.errors = ErrorNeuron(n_out)
        
        # Weights
        self.forward_weights = HebbianSynapse(n_out, n_in, rule=rule)
        
        # Track free energy
        self.free_energy = 0.0
        
    def set_input(self, x):
        """Set bottom-up input."""
        self.input = x
        
    def set_top_down_prediction(self, pred):
        """Set top-down prediction from layer above."""
        self.values.prediction = pred
        
    def inference(self):
        """Run inference to minimize prediction error."""
        for step in range(self.n_inference_steps):
            # Bottom-up input
            bottom_up = self.forward_weights.weight @ self.input  # (n_out,)
            
            # Top-down prediction
            top_down = self.values.prediction
            
            # Prediction error
            prediction_error = bottom_up - top_down
            
            # Update membrane potential (leaky integrator)
            total_input = bottom_up + self.values.bias
            self.values.membrane_potential += self.dt * (
                -self.values.leak_rate * self.values.membrane_potential + total_input
            )
            
            # ReLU activation
            self.values.activation = np.maximum(self.values.membrane_potential, 0)
            
            # Update prediction error
            self.values.prediction_error = prediction_error
            self.errors.activation = prediction_error
            
        # Compute free energy (precision-weighted prediction error)
        self.free_energy = np.sum(
            prediction_error ** 2 / (2 * np.maximum(self.values.precision, 1e-6))
        )
        
    def learn(self, global_sign=0.0):
        """Local Hebbian learning."""
        self.forward_weights.update(
            self.input,
            self.errors.activation,
            global_sign
        )
        
        # Update precision (learned reliability)
        self.values.precision = 1.0 / (1.0 + self.values.prediction_error ** 2 + 1e-6)
        
    def get_activations(self):
        """Get current activations."""
        return self.values.activation.copy()


class PCHHNetwork:
    """
    Multi-layer Predictive Coding + Hebbian Hybrid Network.
    
    Architecture:
    Input → PC1 → PC2 → PC3 → Output
    
    Key design choices:
    - Each layer learns independently (no backprop through layers)
    - Global sign signal guides all layers simultaneously (dopamine-like)
    - Sparse activity emerges naturally from lateral competition
    """
    
    def __init__(self, layer_sizes, rules=None):
        """
        Args:
            layer_sizes: list of layer dimensions [784, 256, 128, 64, 10]
            rules: list of Hebbian rules per layer
        """
        if rules is None:
            rules = ['oja', 'bcm', 'ghl', 'oja']
        
        assert len(layer_sizes) >= 2
        assert len(rules) == len(layer_sizes) - 1
        
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        # Build layers
        self.layers = []
        for i in range(self.n_layers):
            layer = PCLayer(
                n_in=layer_sizes[i],
                n_out=layer_sizes[i+1],
                rule=rules[i]
            )
            self.layers.append(layer)
        
        # Output classification weights (simple linear readout)
        self.output_weights = np.random.randn(layer_sizes[-1], 10).astype(np.float32) * 0.01
        self.output_bias = np.zeros(10, dtype=np.float32)
        
    def forward(self, x):
        """
        Forward pass through hierarchy.
        Bottom-up inference at each layer.
        """
        current = x
        
        # Set input at first layer
        self.layers[0].set_input(current)
        self.layers[0].inference()
        current = self.layers[0].get_activations()
        
        # Propagate upward
        for i in range(1, self.n_layers):
            self.layers[i].set_input(current)
            # Top-down prediction from above (simplified: zeros initially)
            self.layers[i].set_top_down_prediction(np.zeros(self.layer_sizes[i+1]))
            self.layers[i].inference()
            current = self.layers[i].get_activations()
        
        # Output classification
        logits = current @ self.output_weights + self.output_bias
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs, current
    
    def learn(self, x, target, reward_signal):
        """
        Full learning cycle for one sample.
        
        Args:
            x: input (784,)
            target: one-hot target (10,)
            reward_signal: scalar global reward (used for GHL)
        """
        # Forward pass
        probs, top_activations = self.forward(x)
        
        # Compute global sign (simplified: reward * prediction_correctness)
        predicted_class = np.argmax(probs)
        target_class = np.argmax(target)
        correct = float(predicted_class == target_class)
        global_sign = reward_signal * (1.0 if correct else -1.0)
        
        # Learning at each layer (Hebbian, local)
        for layer in self.layers:
            layer.learn(global_sign=global_sign)
        
        # Update output weights (simple Hebbian on top layer)
        # This is equivalent to perceptron learning with Hebbian flavor
        target_one_hot = target.astype(np.float32)
        error = target_one_hot - probs
        top_acts = top_activations
        self.output_weights += 0.001 * np.outer(top_acts, error)
        self.output_bias += 0.001 * error
        
        return probs
    
    def get_total_free_energy(self):
        return sum(layer.free_energy for layer in self.layers)
    
    def summary(self):
        total_params = sum(
            layer.forward_weights.weight.size for layer in self.layers
        )
        total_params += self.output_weights.size + self.output_bias.size
        return {
            'type': 'PCHH Network',
            'layers': self.layer_sizes,
            'n_layers': self.n_layers,
            'total_params': total_params,
            'backprop': False,
            'attention': False,
            'transformers': False,
            'learning': 'Local Hebbian + Global Sign'
        }


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(net, images, labels, epoch, batch_size=32, n_samples=None):
    """Train one epoch."""
    if n_samples is None:
        n_samples = len(images)
    else:
        n_samples = min(n_samples, len(images))
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        
        for idx in batch_indices:
            x = images[idx]
            target = one_hot([labels[idx]], 10)[0]
            
            # Learn
            probs = net.learn(x, target, reward_signal=1.0)
            
            # Track metrics
            predicted = np.argmax(probs)
            target_cls = labels[idx]
            
            if predicted == target_cls:
                correct += 1
            total += 1
            
            # Cross-entropy loss
            loss = -np.log(max(probs[int(target_cls)], 1e-8))
            total_loss += loss
    
    elapsed = time.time() - start_time
    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    
    return accuracy, avg_loss, elapsed


def evaluate(net, images, labels, n_samples=None):
    """Evaluate on test set."""
    if n_samples is None:
        n_samples = len(images)
    else:
        n_samples = min(n_samples, len(images))
    
    correct = 0
    total = 0
    total_loss = 0.0
    
    for i in range(n_samples):
        x = images[i]
        target_cls = labels[i]
        
        probs, _ = net.forward(x)
        predicted = np.argmax(probs)
        
        if predicted == target_cls:
            correct += 1
        total += 1
        
        loss = -np.log(max(probs[int(target_cls)], 1e-8))
        total_loss += loss
    
    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    
    return accuracy, avg_loss


def run_mnist_experiment(epochs=5, n_train=1000, n_test=200, 
                         layer_sizes=None, use_synthetic=False):
    """
    Run MNIST experiment with PCHH.
    
    Args:
        epochs: number of training epochs
        n_train: number of training samples (use small for testing)
        n_test: number of test samples
        layer_sizes: layer dimensions
        use_synthetic: if True, use synthetic data (no download)
    """
    if layer_sizes is None:
        layer_sizes = [784, 128, 64, 32, 10]
    
    print('='*70)
    print('PCHH MNIST Experiment')
    print('='*70)
    
    # Load data
    if use_synthetic:
        data = generate_synthetic_mnist(n_train, n_test)
    else:
        try:
            data = load_mnist()
        except:
            data = generate_synthetic_mnist(n_train, n_test)
    
    train_images = data['train_images'][:n_train]
    train_labels = data['train_labels'][:n_train]
    test_images = data['test_images'][:n_test]
    test_labels = data['test_labels'][:n_test]
    
    print(f'Training samples: {len(train_images)}')
    print(f'Test samples: {len(test_images)}')
    
    # Create network
    net = PCHHNetwork(layer_sizes=layer_sizes)
    summary = net.summary()
    print(f'\nNetwork: {summary["type"]}')
    print(f'Layers: {summary["layers"]}')
    print(f'Total params: {summary["total_params"]}')
    print(f'Backprop: {summary["backprop"]}')
    print(f'Attention: {summary["attention"]}')
    print(f'Transformers: {summary["transformers"]}')
    print(f'Learning: {summary["learning"]}')
    
    # Training
    print(f'\n--- Training ({epochs} epochs) ---')
    history = []
    
    for epoch in range(epochs):
        train_acc, train_loss, elapsed = train_epoch(
            net, train_images, train_labels, epoch, n_samples=n_train
        )
        
        test_acc, test_loss = evaluate(net, test_images, test_labels, n_samples=n_test)
        
        fe = net.get_total_free_energy()
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'free_energy': float(fe),
            'time': elapsed
        })
        
        print(f'Epoch {epoch+1:3d} | '
              f'Train: {train_acc:.3f} ({train_loss:.3f}) | '
              f'Test: {test_acc:.3f} ({test_loss:.3f}) | '
              f'FE: {fe:.2f} | '
              f'{elapsed:.1f}s')
    
    # Final results
    final_test_acc, final_test_loss = evaluate(net, test_images, test_labels, n_samples=n_test)
    
    print(f'\n--- Final Results ---')
    print(f'Test accuracy: {final_test_acc:.3f}')
    print(f'Test loss: {final_test_loss:.3f}')
    print(f'Free energy: {net.get_total_free_energy():.2f}')
    
    # Save results
    results = {
        'config': {
            'architecture': 'PCHH',
            'layers': layer_sizes,
            'epochs': epochs,
            'n_train': n_train,
            'n_test': n_test,
            'backprop': False,
            'attention': False,
            'transformers': False,
        },
        'summary': summary,
        'history': history,
        'final_test_accuracy': final_test_acc,
        'final_test_loss': final_test_loss,
    }
    
    return results, net


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Quick test with synthetic data
    print('Running PCHH MNIST experiment...')
    print('(Using synthetic data for initial test)')
    
    results, net = run_mnist_experiment(
        epochs=3,
        n_train=500,
        n_test=100,
        layer_sizes=[784, 64, 32, 10],
        use_synthetic=True
    )
    
    # Save results
    output_path = Path('06_EXPERIMENTS/mnist_pchh/results_synthetic.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved to {output_path}')
