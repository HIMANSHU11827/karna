"""
AGI Research Lab — Hybrid Approach: Hebbian Feature Extraction + Supervised Classifier
Pivot from pure Hebbian to a hybrid that can actually reach 95% MNIST.
"""

import numpy as np
from pathlib import Path
import json
import time
from typing import List, Tuple, Optional, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent))
from hebbian_layers import HebbianLayer, PredictiveCodingLayer, HebbianSoftmaxClassifier


class LogisticRegression:
    """Simple logistic regression classifier using gradient descent."""
    
    def __init__(self, input_dim: int, n_classes: int, learning_rate: float = 0.1):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = learning_rate
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_dim + n_classes))
        self.weights = np.random.uniform(-limit, limit, (n_classes, input_dim))
        self.bias = np.zeros(n_classes)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Softmax probabilities."""
        z = self.weights @ x + self.bias
        z = z - z.max()  # numerical stability
        exp = np.exp(z)
        return exp / exp.sum()
    
    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))
    
    def train_step(self, x: np.ndarray, label: int) -> float:
        """Single gradient descent step. Returns loss."""
        x = x.astype(np.float64)
        x = x / (np.linalg.norm(x) + 1e-8)
        
        probs = self.forward(x)
        
        # Cross-entropy loss
        loss = -np.log(probs[label] + 1e-8)
        
        # Gradient
        one_hot = np.zeros(self.n_classes)
        one_hot[label] =1.0
        delta = probs - one_hot
        
        self.weights -= self.lr * np.outer(delta, x)
        self.bias -= self.lr * delta
        
        return loss
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == y[i])
        return correct / len(X)


class BCMRule:
    """
    Bienenstock-Cooper-Munro (BCM) learning rule.
    More sophisticated than pure Hebbian — includes a sliding threshold
    that makes neurons compete and develop selectivity.
    
    Δw = lr * y * (y - threshold) * x
    threshold = running_average(y^2)
    """
    
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (output_dim, input_dim))
        self.bias = np.zeros(output_dim)
        
        # BCM sliding threshold (per neuron)
        self.threshold = np.ones(output_dim)
        self.threshold_decay = 0.99
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.weights @ x + self.bias
        return np.maximum(0, z)
    
    def update(self, x: np.ndarray, y: np.ndarray):
        """BCM update rule."""
        # Update threshold (running average of y^2)
        self.threshold = self.threshold_decay * self.threshold + (1 - self.threshold_decay) * (y ** 2)
        
        # BCM rule: delta_w = lr * y * (y - threshold) * x
        delta = self.lr * y * (y - self.threshold)
        delta_w = np.outer(delta, x)
        self.weights += delta_w
        
        # Weight normalization
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        self.weights = self.weights / norms * np.sqrt(self.input_dim)
        
        # Anti-Hebbian decay
        self.weights *= 0.9999


class HybridNetwork:
    """
    Hybrid network: Hebbian/BCM feature extraction + supervised classifier.
    
    Architecture:
    1. Unsupervised feature extractor (Hebbian or BCM layers)
    2. Supervised classifier (logistic regression) trained on extracted features
    
    This separates the two problems:
    - Hebbian/BCM learns good feature representations without labels
    - Logistic regression learns the mapping from features to labels
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        classifier_type: str = "logistic",
        learning_rate: float = 0.01,
        feature_lr: float = 0.01,
        seed: int = 42,
    ):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.feature_lr = feature_lr
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Build feature extractor layers
        self.feature_layers: List[HebbianLayer] = []
        for i in range(len(layer_sizes) - 1):
            layer = HebbianLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i + 1],
                learning_rate=feature_lr,
                inhibition_strength=0.1,
                homeostasis_target=0.1,
                weight_bound=1.0,
                seed=seed + i,
            )
            self.feature_layers.append(layer)
        
        # Classifier
        if classifier_type == "logistic":
            self.classifier = LogisticRegression(
                input_dim=layer_sizes[-1],
                n_classes=10,
                learning_rate=learning_rate,
            )
        elif classifier_type == "bcm_classifier":
            self.classifier = HebbianSoftmaxClassifier(
                feature_dim=layer_sizes[-1],
                n_classes=10,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract features through Hebbian layers."""
        current = x.astype(np.float64)
        current = current / (np.linalg.norm(current) + 1e-8)
        for layer in self.feature_layers:
            current = layer.forward(current)
        return current
    
    def pretrain_unsupervised(self, X: np.ndarray, epochs: int = 10, verbose: bool = True):
        """Unsupervised pretraining of feature layers."""
        if verbose:
            print(f"\nUnsupervised pretraining ({epochs} epochs)...")
        
        n_samples = len(X)
        for epoch in range(epochs):
            indices = self.rng.permutation(n_samples)
            total_sparsity = 0.0
            
            for i in indices:
                x = X[i].astype(np.float64)
                x = x / (np.linalg.norm(x) + 1e-8)
                
                current = x
                for layer in self.feature_layers:
                    y = layer.forward(current)
                    layer.update(current, y)
                    current = y
                
                total_sparsity += self.feature_layers[-1].sparsity
            
            if verbose:
                mean_sparsity = total_sparsity / n_samples
                print(f"  Epoch {epoch + 1}: sparsity={mean_sparsity:.2%}")
    
    def train_classifier(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, verbose: bool = True):
        """Train supervised classifier on extracted features."""
        if verbose:
            print(f"\nTraining classifier ({epochs} epochs)...")
        
        n_samples = len(X)
        for epoch in range(epochs):
            indices = self.rng.permutation(n_samples)
            total_loss = 0.0
            
            for i in indices:
                x = X[i]
                label = int(y[i])
                
                # Extract features
                features = self.extract_features(x)
                
                # Train classifier
                if isinstance(self.classifier, LogisticRegression):
                    loss = self.classifier.train_step(features, label)
                    total_loss += loss
                elif isinstance(self.classifier, HebbianSoftmaxClassifier):
                    self.classifier.update(features, label)
            
            if verbose and epoch % 10 == 0:
                acc = self.classifier.accuracy(
                    [self.extract_features(X[i]) for i in range(min(200, n_samples))],
                    y[:min(200, n_samples)]
                )
                print(f"  Epoch {epoch}: loss={total_loss / n_samples:.4f}, acc={acc:.2%}")
    
    def predict(self, x: np.ndarray) -> int:
        features = self.extract_features(x)
        if isinstance(self.classifier, LogisticRegression):
            return self.classifier.predict(features)
        elif isinstance(self.classifier, HebbianSoftmaxClassifier):
            return self.classifier.predict(features)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == y[i])
        accuracy = correct / len(X)
        return {"accuracy": accuracy, "correct": correct, "total": len(X)}


def generate_synthetic_mnist(n_samples: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate more complex synthetic digit-like data."""
    rng = np.random.default_rng(seed)
    
    def make_digit(digit: int, size: int = 784) -> np.ndarray:
        img = np.zeros(size)
        
        # More complex patterns — horizontal and vertical bars at different positions
        if digit == 0:
            # Rectangle outline
            for i in range(28):
                img[i * 28 + 5] = 1.0
                img[i * 28 + 22] = 1.0
            for j in range(5, 23):
                img[j] = 1.0
                img[27 * 28 + j] = 1.0
        elif digit == 1:
            for i in range(28):
                img[i * 28 + 14] = 1.0
            img[0:5] = 0.5  # top serif
        elif digit == 2:
            for i in range(14):
                img[i * 28 + 14 + i] = 1.0
            for i in range(14):
                img[(14 + i) * 28 + 14 - i] = 1.0
        elif digit == 3:
            for i in range(28):
                img[i * 28 + 20] = 1.0
            img[5 * 28:5 * 28 + 20] = 0.5
            img[20 * 28:20 * 28 + 20] = 0.5
        elif digit == 4:
            for i in range(15):
                img[i * 28 + 10] = 1.0
            img[15 * 28:15 * 28 + 25] = 1.0
            for i in range(15, 28):
                img[i * 28 + 20] = 1.0
        elif digit == 5:
            img[5 * 28:5 * 28 + 20] = 1.0
            for i in range(5, 14):
                img[i * 28 + 5] = 1.0
            img[14 * 28:14 * 28 + 20] = 1.0
            for i in range(14, 22):
                img[i * 28 + 22] = 1.0
        elif digit == 6:
            for i in range(28):
                img[i * 28 + 10] = 1.0
            img[5 * 28 + 10:5 * 28 + 25] = 1.0
            for i in range(5, 20):
                img[i * 28 + 24] = 1.0
        elif digit == 7:
            img[5 * 28:5 * 28 + 25] = 1.0
            for i in range(5, 28):
                img[i * 28 + 25 - (i - 5)] = 1.0
        elif digit == 8:
            for i in [5, 14, 23]:
                img[i * 28 + 8:i * 28 + 20] = 1.0
            for i in range(6, 14):
                img[i * 28 + 8] = 1.0
                img[i * 28 + 19] = 1.0
            for i in range(15, 23):
                img[i * 28 + 8] = 1.0
                img[i * 28 + 19] = 1.0
        elif digit == 9:
            for i in range(28):
                img[i * 28 + 18] = 1.0
            img[5 * 28 + 8:5 * 28 + 18] = 1.0
            for i in range(5, 20):
                img[i * 28 + 8] = 1.0
        
        # Add noise
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
    print("Hybrid Network: Hebbian + Logistic Regression")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating synthetic data...")
    X_train, y_train, X_test, y_test = generate_synthetic_mnist(2000)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Create hybrid network
    net = HybridNetwork(
        layer_sizes=[784, 256, 100],
        classifier_type="logistic",
        learning_rate=0.1,
        feature_lr=0.01,
    )
    
    # Phase 1: Unsupervised pretraining
    net.pretrain_unsupervised(X_train, epochs=20)
    
    # Phase 2: Supervised classifier training
    net.train_classifier(X_train, y_train, epochs=100)
    
    # Evaluate
    print("\nEvaluating...")
    results = net.evaluate(X_test, y_test)
    print(f"  Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    
    # Save
    output_path = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/20_OUTPUT/hybrid_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "accuracy": results["accuracy"],
        "config": {
            "layers": net.layer_sizes,
            "classifier": "logistic_regression",
        }
    }
    Path(output_path).write_text(json.dumps(state, indent=2))
    print(f"\nResults saved to {output_path}")
