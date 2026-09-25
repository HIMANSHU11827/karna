"""
AGI Research Lab — Learned Vision Encoder
Uses Hebbian learning to train the encoder weights so similar inputs
activate similar neurons in the hidden layer.
"""

import numpy as np
from pathlib import Path
import json


class LearnedVisionEncoder:
    """
    A vision encoder that LEARNS its projection weights.
    
    Learning rule (simplified self-organizing):
    - For each input, find the top-k most active hidden neurons
    - Strengthen connections from active input neurons to active hidden neurons
    - Weaken connections from inactive input neurons to active hidden neurons
    - This creates a mapping where similar inputs activate similar hidden neurons
    
    After encoding, k-WTA is applied to get a sparse SDR.
    """
    
    def __init__(self, input_dim: int = 784, output_dim: int = 2000, sparsity: float = 0.05, seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.k = max(1, int(output_dim * sparsity))
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Projection weights (these get LEARNED)
        self.projection = self.rng.normal(0, 0.05, (output_dim, input_dim)) / np.sqrt(input_dim)
        self.bias = np.zeros(output_dim)
        
        # Running mean activity for homeostasis
        self.mean_activity = np.zeros(output_dim)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to SDR."""
        x = x / (np.linalg.norm(x) + 1e-8)
        activations = self.projection @ x + self.bias
        # k-WTA
        result = np.zeros(self.output_dim, dtype=np.int8)
        top_k = np.argpartition(activations, -self.k)[-self.k:]
        result[top_k] = 1
        return result
    
    def learn(self, x: np.ndarray, lr: float = 0.01):
        """
        Learn from this input.
        
        Rule: 
        - Find top-k active hidden neurons
        - Strengthen: w_j += lr * (x - w_j) for top-k neurons j
        - Weaken: w_j -= lr * 0.01 * w_j for inactive neurons j (forgetting)
        """
        x = x / (np.linalg.norm(x) + 1e-8)
        activations = self.projection @ x + self.bias
        
        # Find top-k
        top_k = np.argpartition(activations, -self.k)[-self.k:]
        
        # Create mask
        mask = np.zeros(self.output_dim)
        mask[top_k] = 1.0
        
        # Hebbian update for top-k (move toward input)
        self.projection[top_k] += lr * (np.outer(np.ones(self.k), x) - self.projection[top_k])
        
        # Homeostasis
        self.mean_activity = 0.99 * self.mean_activity + 0.01 * mask
        excess = self.mean_activity - self.sparsity
        self.bias -= lr * 0.1 * excess
        
        # Anti-Hebbian for inactive (forgetting)
        inactive = ~mask.astype(bool)
        self.projection[inactive] *= (1 - lr * 0.001)
        
        # Normalize
        norms = np.linalg.norm(self.projection, axis=1, keepdims=True)
        self.projection = self.projection / (norms + 1e-8) * np.sqrt(self.input_dim) * 0.05


class LearnedHAPN:
    """
    HAPN with a LEARNED vision encoder.
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 2000, n_classes: int = 10, sparsity: float = 0.05, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.sparsity = sparsity
        self.k = max(1, int(hidden_dim * sparsity))
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Learned encoder
        self.encoder = LearnedVisionEncoder(input_dim, hidden_dim, sparsity, seed)
        
        # Classifier
        self.classifier_weights = np.zeros((n_classes, hidden_dim))
        self.classifier_bias = np.zeros(n_classes)
        
        # Stats
        self.n_learned = 0
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.encoder.encode(x)
    
    def predict(self, x: np.ndarray) -> int:
        sdr = self.encode(x)
        logits = self.classifier_weights @ sdr.astype(np.float64) + self.classifier_bias
        return int(np.argmax(logits))
    
    def train_encoder(self, X: np.ndarray, epochs: int = 20, verbose: bool = True):
        """Train the encoder unsupervised."""
        if verbose:
            print(f"Training encoder ({epochs} epochs)...")
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        for epoch in range(epochs):
            self.rng.shuffle(indices)
            total_sparsity = 0.0
            
            for i in indices:
                x = X[i].astype(np.float64)
                self.encoder.learn(x, lr=0.01)
                self.n_learned += 1
                total_sparsity += self.encoder.mean_activity.mean()
            
            if verbose and (epoch + 1) % 5 == 0:
                mean_sparsity = total_sparsity / n_samples
                print(f"  Epoch {epoch + 1}: mean_activity={mean_sparsity:.4f}")
    
    def train_classifier(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, verbose: bool = True):
        """Train the classifier."""
        if verbose:
            print(f"Training classifier ({epochs} epochs)...")
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        for epoch in range(epochs):
            self.rng.shuffle(indices)
            total_loss = 0.0
            correct = 0
            
            for i in indices:
                sdr = self.encode(X[i])
                label = int(y[i])
                
                # Forward
                logits = self.classifier_weights @ sdr.astype(np.float64) + self.classifier_bias
                exp = np.exp(logits - logits.max())
                probs = exp / exp.sum()
                
                loss = -np.log(probs[label] + 1e-8)
                total_loss += loss
                
                if np.argmax(probs) == label:
                    correct += 1
                
                # Gradient
                dlogits = probs.copy()
                dlogits[label] -= 1
                self.classifier_weights -= 0.1 * np.outer(dlogits, sdr.astype(np.float64))
                self.classifier_bias -= 0.1 * dlogits
            
            if verbose and (epoch + 1) % 10 == 0:
                acc = correct / n_samples
                print(f"  Epoch {epoch + 1}: loss={total_loss / n_samples:.4f}, acc={acc:.2%}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == y[i])
        accuracy = correct / len(X)
        return {"accuracy": accuracy, "correct": correct, "total": len(X)}


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42):
    """Generate more complex synthetic data with structure."""
    rng = np.random.default_rng(seed)
    
    def make_digit(digit: int, size: int = 784) -> np.ndarray:
        img = np.zeros(size)
        # Use a 28x28 grid
        if digit == 0:
            for i in range(28):
                img[i * 28 + 5] = 1
                img[i * 28 + 22] = 1
            img[5:23] = 1
            img[27*28 + 5:27*28 + 23] = 1
        elif digit == 1:
            for i in range(28):
                img[i * 28 + 14] = 1
        elif digit == 2:
            for i in range(14):
                img[i * 28 + 14 + i] = 1
            for i in range(14, 28):
                img[i * 28 + 27 - i] = 1
        elif digit == 3:
            for i in range(28):
                img[i * 28 + 20] = 1
        elif digit == 4:
            for i in range(28):
                img[i * 28 + 10] = 1
            img[14 * 28:14 * 28 + 25] = 1
        elif digit == 5:
            for i in range(28):
                img[i * 28 + 14] = 1
            img[5 * 28 + 5:5 * 28 + 20] = 1
        elif digit == 6:
            img[::3] = 0.8
        elif digit == 7:
            img[::4] = 0.9
        elif digit == 8:
            img[::5] = 0.7
        elif digit == 9:
            img[::7] = 0.6
        
        # Add moderate noise
        img += rng.normal(0, 0.05, size)
        img = np.clip(img, 0, 1)
        return img
    
    X, y = [], []
    samples_per_class = n_samples // 10
    for digit in range(10):
        for _ in range(samples_per_class):
            X.append(make_digit(digit))
            y.append(digit)
    
    X = np.array(X)
    y = np.array(y)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:]


if __name__ == "__main__":
    print("=" * 60)
    print("Learned HAPN — Hebbian Encoder + Linear Classifier")
    print("=" * 60)
    
    # Data
    X_train, y_train, X_test, y_test = generate_synthetic_data(2000, seed=42)
    print(f"Data: {len(X_train)} train, {len(X_test)} test")
    
    # Model
    net = LearnedHAPN(input_dim=784, hidden_dim=2000, n_classes=10, sparsity=0.05)
    print(f"Architecture: 784 -> 2000 -> 10")
    
    # Train
    net.train_encoder(X_train, epochs=20)
    net.train_classifier(X_train, y_train, epochs=50)
    
    # Evaluate on both train and test
    print("\nEvaluating...")
    train_results = net.evaluate(X_train[:200], y_train[:200])
    test_results = net.evaluate(X_test, y_test)
    print(f"  Train: {train_results['accuracy']:.2%} ({train_results['correct']}/{train_results['total']})")
    print(f"  Test:  {test_results['accuracy']:.2%} ({test_results['correct']}/{test_results['total']})")
    
    print("\nDone.")
