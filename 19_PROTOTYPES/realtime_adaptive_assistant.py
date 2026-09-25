"""
AGI Research Lab — Real-Time Adaptive Assistant Core
Phase 3 Pivot: From batch training to online, real-time learning.

Key differences from traditional LLM approach:
- No batch training — learns from each interaction immediately
- No static model — weights update on every forward pass
- No perfect-prompt requirement — adapts to messy input
- Multimodal — processes text + vision features simultaneously
- Memory accumulates — every interaction shapes future behavior
"""

import numpy as np
from pathlib import Path
import json
import time
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent))
from hebbian_layers import HebbianLayer, HebbianSoftmaxClassifier


# =============================================================================
# Real-Time Multimodal Input Processing
# =============================================================================

class TextEncoder:
    """
    Real-time text encoder using character-level hashing.
    No pre-trained embeddings — learns vocabulary online.
    """
    
    def __init__(self, embedding_dim: int = 128, max_features: int = 5000):
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.embeddings = np.random.randn(max_features, embedding_dim) * 0.1
        self.next_id = 0
        self.usage_count = np.zeros(max_features)
        
    def _hash_token(self, token: str) -> int:
        """Hash a token to an embedding index."""
        if token not in self.vocab:
            if self.next_id < self.max_features:
                self.vocab[token] = self.next_id
                self.next_id += 1
            else:
                # Fallback: hash modulo
                return hash(token) % self.max_features
        return self.vocab[token]
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to a fixed-size vector."""
        # Normalize: lowercase, strip, split
        tokens = text.lower().strip().split()
        
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        # Average embedding of all tokens
        vectors = []
        for token in tokens:
            idx = self._hash_token(token)
            vectors.append(self.embeddings[idx])
            self.usage_count[idx] += 1
        
        return np.mean(vectors, axis=0)
    
    def update_embedding(self, token: str, target: np.ndarray, lr: float = 0.01):
        """Online update: move embedding toward target."""
        if token in self.vocab:
            idx = self.vocab[token]
            self.embeddings[idx] += lr * (target - self.embeddings[idx])


class VisionEncoder:
    """
    Real-time vision encoder using online PCA-like feature extraction.
    No pre-trained CNN — learns visual features online.
    """
    
    def __init__(self, input_size: int = 784, feature_dim: int = 128):
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.mean = np.zeros(input_size)
        self.components = np.random.randn(feature_dim, input_size) * 0.01
        self.n_samples = 0
        
    def encode(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image."""
        x = image.flatten().astype(np.float64)
        x = x / (np.linalg.norm(x) + 1e-8)
        
        # Center
        if self.n_samples > 0:
            x = x - self.mean
        
        # Project
        features = self.components @ x
        features = np.maximum(0, features)  # ReLU
        return features
    
    def update(self, image: np.ndarray, lr: float = 0.001):
        """Online update of visual features."""
        x = image.flatten().astype(np.float64)
        x = x / (np.linalg.norm(x) + 1e-8)
        
        # Update running mean
        self.n_samples += 1
        self.mean = self.mean * (1 - 1.0 / self.n_samples) + x / self.n_samples
        
        # Hebbian update on components
        features = self.encode(image)
        delta = lr * np.outer(features, x)
        self.components += delta
        
        # Normalize
        norms = np.linalg.norm(self.components, axis=1, keepdims=True)
        self.components = self.components / (norms + 1e-8)


# =============================================================================
# Online Learning Memory
# =============================================================================

class OnlineMemory:
    """
    Memory that updates on every interaction.
    Stores patterns and their outcomes for future reference.
    """
    
    def __init__(self, capacity: int = 1000, feature_dim: int = 256):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.patterns = np.zeros((capacity, feature_dim))
        self.labels = np.zeros(capacity, dtype=np.int32)
        self.metadata: List[Dict[str, Any]] = []
        self.timestamps = np.zeros(capacity)
        self.confidence = np.zeros(capacity)
        self.size = 0
        
    def store(self, features: np.ndarray, label: int, metadata: Optional[Dict] = None):
        """Store a pattern with its label."""
        idx = self.size % self.capacity
        
        self.patterns[idx] = features
        self.labels[idx] = label
        self.timestamps[idx] = time.time()
        self.confidence[idx] = 0.5  # Initial confidence
        self.metadata.append(metadata or {})
        
        if self.size < self.capacity:
            self.size += 1
    
    def query(self, features: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find k nearest patterns.
        Returns: (patterns, labels, similarities)
        """
        if self.size == 0:
            return np.zeros((0, self.feature_dim)), np.zeros(0), np.zeros(0)
        
        # Cosine similarity
        norms = np.linalg.norm(self.patterns[:self.size], axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = self.patterns[:self.size] / norms
        
        query_norm = features / (np.linalg.norm(features) + 1e-8)
        similarities = normalized @ query_norm
        
        # Top-k
        k = min(k, self.size)
        top_k = np.argsort(similarities)[-k:][::-1]
        
        return self.patterns[top_k], self.labels[top_k], similarities[top_k]
    
    def update_confidence(self, idx: int, correct: bool):
        """Update confidence based on correctness."""
        if idx < self.capacity:
            if correct:
                self.confidence[idx] = min(1.0, self.confidence[idx] + 0.1)
            else:
                self.confidence[idx] = max(0.0, self.confidence[idx] - 0.2)


# =============================================================================
# Real-Time Adaptive Classifier
# =============================================================================

class AdaptiveClassifier:
    """
    Classifier that adapts in real-time.
    Uses online nearest-neighbor + Hebbian prototype learning.
    """
    
    def __init__(self, feature_dim: int, n_classes: int = 10):
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        
        # Prototypes (one per class)
        self.prototypes = np.zeros((n_classes, feature_dim))
        self.counts = np.zeros(n_classes)
        
        # Online logistic regression weights
        self.weights = np.zeros((n_classes, feature_dim))
        self.bias = np.zeros(n_classes)
        
        # Learning rate (adaptive)
        self.lr = 0.01
        
    def predict(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict class and confidence."""
        # Cosine similarity to prototypes
        norms = np.linalg.norm(self.prototypes, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        proto_sims = (self.prototypes @ features) / norms.flatten()
        
        # Linear classifier
        logits = self.weights @ features + self.bias
        
        # Combine
        combined = proto_sims + logits
        probs = self._softmax(combined)
        
        return int(np.argmax(probs)), probs
    
    def update(self, features: np.ndarray, label: int, lr: float = 0.01):
        """Online update from a single example."""
        # Update prototype (running average)
        self.counts[label] += 1
        alpha = 1.0 / self.counts[label]
        self.prototypes[label] = (1 - alpha) * self.prototypes[label] + alpha * features
        
        # Update logistic regression
        pred, probs = self.predict(features)
        
        # Gradient
        dlogits = probs.copy()
        dlogits[label] -= 1
        
        self.weights -= lr * np.outer(dlogits, features)
        self.bias -= lr * dlogits
        
        # Return whether prediction was correct
        return pred == label
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = x - x.max()
        exp = np.exp(x)
        return exp / (exp.sum() + 1e-8)


# =============================================================================
# Real-Time Adaptive Assistant
# =============================================================================

class RealTimeAdaptiveAssistant:
    """
    The core system: processes input, learns, responds, adapts — all in real-time.
    
    Architecture:
    1. Multimodal encoder (text + vision)
    2. Online memory (stores every interaction)
    3. Adaptive classifier (learns from every interaction)
    4. Response generator (adapts based on learned patterns)
    """
    
    def __init__(self, text_dim: int = 128, vision_dim: int = 128, n_classes: int = 10):
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.feature_dim = text_dim + vision_dim
        
        # Encoders
        self.text_encoder = TextEncoder(embedding_dim=text_dim)
        self.vision_encoder = VisionEncoder(input_size=784, feature_dim=vision_dim)
        
        # Memory
        self.memory = OnlineMemory(capacity=5000, feature_dim=self.feature_dim)
        
        # Classifier
        self.classifier = AdaptiveClassifier(self.feature_dim, n_classes)
        
        # Interaction history
        self.interaction_count = 0
        self.correct_count = 0
        self.recent_accuracy = 0.0
        
        # Pattern tracking
        self.user_patterns: Dict[str, Any] = {}
        self.correction_history: List[Dict] = []
        
    def process(
        self,
        text: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        label: Optional[int] = None,
        learn: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single interaction in real-time.
        
        Args:
            text: Text input (can be messy, misspelled, etc.)
            image: Image input (flattened 784 for MNIST)
            label: Ground truth label (if available)
            learn: Whether to learn from this interaction
            
        Returns:
            Dict with prediction, confidence, and metadata
        """
        start_time = time.time()
        
        # 1. Encode inputs
        text_features = np.zeros(self.text_dim)
        vision_features = np.zeros(self.vision_dim)
        
        if text is not None:
            text_features = self.text_encoder.encode(text)
        
        if image is not None:
            vision_features = self.vision_encoder.encode(image)
        
        # 2. Combine features
        features = np.concatenate([text_features, vision_features])
        features = features / (np.linalg.norm(features) + 1e-8)
        
        # 3. Predict
        prediction, confidence = self.classifier.predict(features)
        max_confidence = float(np.max(confidence))
        
        # 4. Query memory for similar patterns
        if self.memory.size > 0:
            mem_patterns, mem_labels, mem_sims = self.memory.query(features, k=3)
            if len(mem_labels) > 0:
                # Weighted vote from memory
                mem_votes = np.zeros(self.classifier.n_classes)
                for i, sim in enumerate(mem_sims):
                    mem_votes[int(mem_labels[i])] += sim
                
                # Combine classifier + memory
                combined = confidence + 0.5 * mem_votes / (mem_votes.sum() + 1e-8)
                prediction = int(np.argmax(combined))
                max_confidence = float(np.max(combined))
        
        # 5. Learn (if label provided)
        correct = None
        if learn and label is not None:
            # Update classifier
            correct = self.classifier.update(features, label)
            
            # Update memory
            self.memory.store(features, label, metadata={
                "text": text,
                "prediction": prediction,
                "correct": correct,
            })
            
            # Update encoders
            if text is not None:
                # Update text embeddings toward correct class prototype
                target = self.classifier.prototypes[label]
                for token in text.lower().split():
                    self.text_encoder.update_embedding(token, target[:self.text_dim], lr=0.001)
            
            if image is not None:
                self.vision_encoder.update(image, lr=0.0001)
            
            # Track accuracy
            self.interaction_count += 1
            if correct:
                self.correct_count += 1
            self.recent_accuracy = self.correct_count / self.interaction_count
        
        # 6. Build response
        processing_time = time.time() - start_time
        
        result = {
            "prediction": prediction,
            "confidence": max_confidence,
            "processing_time_ms": processing_time * 1000,
            "interaction_count": self.interaction_count,
            "accuracy": self.recent_accuracy,
            "memory_size": self.memory.size,
        }
        
        if label is not None:
            result["correct"] = correct
            result["label"] = label
        
        return result
    
    def chat(self, user_input: str) -> str:
        """
        Adaptive chat: handles messy input, learns from corrections.
        """
        # Process the input
        result = self.process(text=user_input, learn=False)
        
        # Simple response generation based on learned patterns
        if result["confidence"] > 0.7:
            response = f"I understand. (confidence: {result['confidence']:.2f})"
        elif result["confidence"] > 0.4:
            response = f"I think I get it, but I'm not sure. Can you clarify?"
        else:
            response = "I'm still learning. Could you help me understand?"
        
        return response
    
    def correct(self, user_input: str, correct_label: int):
        """
        Learn from a correction — the user tells us the right answer.
        This is how the system adapts.
        """
        result = self.process(text=user_input, label=correct_label, learn=True)
        
        if result["correct"]:
            return "Got it! I'll remember that."
        else:
            return f"Thanks for the correction. I predicted {result['prediction']}, but the correct answer is {correct_label}."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "interactions": self.interaction_count,
            "accuracy": self.recent_accuracy,
            "memory_size": self.memory.size,
            "vocab_size": self.text_encoder.next_id,
            "prototypes_learned": int((self.classifier.counts > 0).sum()),
        }


# =============================================================================
# Demo: Real-Time Adaptive Learning
# =============================================================================

def demo_realtime_learning():
    """Demonstrate real-time adaptive learning."""
    print("=" * 60)
    print("Real-Time Adaptive Assistant — Demo")
    print("=" * 60)
    
    assistant = RealTimeAdaptiveAssistant(
        text_dim=64,
        vision_dim=64,
        n_classes=10,
    )
    
    # Simulate a stream of interactions
    print("\nSimulating real-time interactions...")
    
    # Generate some synthetic data
    rng = np.random.default_rng(42)
    
    # Phase 1: Initial learning (first 50 interactions)
    print("\n--- Phase 1: Initial Learning (50 interactions) ---")
    for i in range(50):
        # Generate synthetic image
        image = rng.rand(784).astype(np.float64)
        image = image / image.max()
        
        # Generate synthetic text
        text = f"sample text {i}"
        
        # Label
        label = i % 10
        
        result = assistant.process(text=text, image=image, label=label, learn=True)
        
        if (i + 1) % 10 == 0:
            print(f"  Interaction {i+1}: accuracy={result['accuracy']:.2%}, "
                  f"confidence={result['confidence']:.2f}, "
                  f"time={result['processing_time_ms']:.1f}ms")
    
    # Phase 2: Test without learning (next 20 interactions)
    print("\n--- Phase 2: Testing (20 interactions, no learning) ---")
    correct = 0
    for i in range(20):
        image = rng.rand(784).astype(np.float64)
        image = image / image.max()
        text = f"test sample {i}"
        label = i % 10
        
        result = assistant.process(text=text, image=image, label=label, learn=False)
        if result["prediction"] == label:
            correct += 1
    
    print(f"  Test accuracy: {correct / 20:.2%}")
    
    # Phase 3: Show stats
    print("\n--- System Statistics ---")
    stats = assistant.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Phase 4: Chat demo
    print("\n--- Chat Demo ---")
    test_inputs = [
        "hello world",
        "what is this",
        "I need help",
        "classify this image",
    ]
    
    for text in test_inputs:
        response = assistant.chat(text)
        print(f"  User: '{text}' -> Assistant: '{response}'")
    
    # Phase 5: Correction demo
    print("\n--- Correction Demo ---")
    response = assistant.correct("this is a cat", 3)
    print(f"  {response}")
    response = assistant.correct("this is a dog", 5)
    print(f"  {response}")
    
    print("\nDemo complete.")


if __name__ == "__main__":
    demo_realtime_learning()
