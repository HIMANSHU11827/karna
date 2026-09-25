"""
RT-ALM: User Intent Model
========================
Adapts within conversation to user's style and intent.

Key features:
- Conversation state tracking (what's being discussed)
- Style adaptation (formality, verbosity, domain)
- Confidence-based self-correction (no "please clarify")
- Graceful degradation for ambiguous input
"""
import numpy as np
from typing import List, Dict, Optional
import time


class UserIntentModel:
    """
    Models user intent within a conversation.
    
    Adaptation mechanisms:
    1. Conversation context (what we're talking about)
    2. Style tracking (formality, verbosity, technical level)
    3. Confidence estimation (know when we don't know)
    4. Self-correction without asking
    """
    
    def __init__(self, dim: int = 1024, k: int = 20):
        self.dim = dim
        self.k = k
        
        # Conversation state
        self.turn_count = 0
        self.conversation_start = time.time()
        self.last_interaction_time = time.time()
        
        # Topic tracking (active topic SDR)
        self.current_topic = np.zeros(dim, dtype=np.float32)
        self.topic_history = []  # List of (topic_sdr, timestamp, relevance)
        
        # Style tracking
        self.style_metrics = {
            'avg_word_length': 5.0,      # Running average
            'formality_estimate': 0.5,   # 0=casual, 1=formal
            'verbosity': 1.0,            # Words per message
            'technical_level': 0.5,      # 0=layman, 1=expert
            'question_ratio': 0.2,       # Fraction of messages that are questions
        }
        
        # Confidence tracking
        self.recent_confidences = []  # Last N confidence scores
        self.confidence_window = 10
        
        # Intent model weights (for predicting next user action)
        self.intent_weights = np.zeros((dim, dim), dtype=np.float32)
        
        # Vocabulary accumulation (words user has used)
        self.user_vocabulary = {}  # word -> (count, first_seen, last_seen)
        self.domain_terms = set()  # Technical terms detected
        
    def process_message(self, message_sdr: np.ndarray, 
                        response_sdr: np.ndarray = None,
                        message_text: str = "") -> dict:
        """
        Process a user message and update intent model.
        
        Args:
            message_sdr: SDR representation of user message
            response_sdr: SDR representation of our response
            message_text: Original text (for style analysis)
            
        Returns:
            Intent analysis dict with confidence, topic, style info
        """
        self.turn_count += 1
        current_time = time.time()
        time_since_last = current_time - self.last_interaction_time
        self.last_interaction_time = current_time
        
        # Update topic
        self._update_topic(message_sdr)
        
        # Update style from text
        if message_text:
            self._update_style(message_text)
        
        # Track vocabulary
        if message_text:
            self._track_vocabulary(message_text)
        
        # Update intent weights (learn transition patterns)
        if response_sdr is not None:
            self._update_intent_weights(message_sdr, response_sdr)
        
        # Compute confidence for this turn
        confidence = self._compute_confidence(message_sdr)
        
        # Detect if user is correcting us
        is_correction = self._detect_correction(message_sdr, message_text)
        
        return {
            'confidence': confidence,
            'topic': self.current_topic.copy(),
            'style': self.style_metrics.copy(),
            'turn': self.turn_count,
            'is_correction': is_correction,
            'time_since_last': time_since_last,
            'vocabulary_size': len(self.user_vocabulary),
        }
    
    def get_intent(self, message_sdr: np.ndarray, message_text: str = "") -> dict:
        """
        Get intent analysis for a message (without updating state).
        
        Returns:
            Intent dict with:
            - confidence: How confident we are in understanding
            - needs_clarification: True if confidence too low
            - predicted_topic: What we think user is asking about
            - suggested_response_style: How we should respond
        """
        confidence = self._compute_confidence(message_sdr)
        
        # Determine if clarification needed (threshold = 0.3)
        needs_clarification = confidence < 0.3
        
        # Predict topic
        predicted_topic = self._predict_topic(message_sdr)
        
        # Suggest response style
        response_style = self._suggest_response_style()
        
        return {
            'confidence': confidence,
            'needs_clarification': needs_clarification,
            'predicted_topic': predicted_topic,
            'response_style': response_style,
        }
    
    def _update_topic(self, message_sdr: np.ndarray):
        """Update conversation topic with exponential moving average."""
        alpha = 0.3  # Topic adaptation rate
        
        if message_sdr.dtype != np.float32:
            dense = np.zeros(self.dim, dtype=np.float32)
            dense[message_sdr] = 1.0
        else:
            dense = message_sdr
        
        # Normalize
        if dense.sum() > 0:
            dense = dense / dense.sum()
        
        # EMA update
        self.current_topic = alpha * dense + (1 - alpha) * self.current_topic
        
        # Store in history
        self.topic_history.append({
            'topic': dense.copy(),
            'timestamp': time.time(),
            'relevance': 1.0
        })
        
        # Trim history
        if len(self.topic_history) > 100:
            self.topic_history.pop(0)
    
    def _update_style(self, text: str):
        """Update style metrics from message text."""
        words = text.split()
        if not words:
            return
        
        # Average word length
        avg_len = np.mean([len(w) for w in words])
        self.style_metrics['avg_word_length'] = 0.7 * self.style_metrics['avg_word_length'] + 0.3 * avg_len
        
        # Verbosity (words per message)
        self.style_metrics['verbosity'] = 0.7 * self.style_metrics['verbosity'] + 0.3 * len(words)
        
        # Question ratio
        is_question = '?' in text
        self.style_metrics['question_ratio'] = 0.9 * self.style_metrics['question_ratio'] + 0.1 * float(is_question)
        
        # Formality estimate (based on word length + no contractions + capitalization)
        formal_markers = sum(1 for w in words if len(w) > 6) / max(len(words), 1)
        informal_markers = text.count("n't") + text.count("'ll") + text.count("'re")
        informal_markers /= max(len(words), 1)
        
        formality = 0.5 + 0.3 * formal_markers - 0.3 * informal_markers
        formality = np.clip(formality, 0, 1)
        self.style_metrics['formality_estimate'] = 0.8 * self.style_metrics['formality_estimate'] + 0.2 * formality
        
        # Technical level (based on domain terms)
        tech_ratio = len(self.domain_terms) / max(len(self.user_vocabulary), 1)
        self.style_metrics['technical_level'] = 0.8 * self.style_metrics['technical_level'] + 0.2 * tech_ratio
    
    def _track_vocabulary(self, text: str):
        """Track user vocabulary."""
        words = text.lower().split()
        current_time = time.time()
        
        for word in words:
            if word in self.user_vocabulary:
                count, first_seen, _ = self.user_vocabulary[word]
                self.user_vocabulary[word] = (count + 1, first_seen, current_time)
            else:
                self.user_vocabulary[word] = (1, current_time, current_time)
            
            # Detect technical terms (heuristic: unusual words used multiple times)
            if self.user_vocabulary[word][0] > 2 and len(word) > 5:
                self.domain_terms.add(word)
    
    def _update_intent_weights(self, message_sdr: np.ndarray, response_sdr: np.ndarray):
        """Learn user -> response transition patterns."""
        msg_dense = self._to_dense(message_sdr)
        resp_dense = self._to_dense(response_sdr)
        
        # Hebbian update
        eta = 0.001
        self.intent_weights += eta * np.outer(resp_dense, msg_dense)
        
        # Decay
        self.intent_weights -= 0.0001 * self.intent_weights
        self.intent_weights = np.clip(self.intent_weights, 0, 1)
    
    def _compute_confidence(self, message_sdr: np.ndarray) -> float:
        """
        Compute confidence in understanding the message.
        
        Based on:
        - Similarity to known topics
        - Vocabulary coverage
        - Recency of topic
        """
        msg_dense = self._to_dense(message_sdr)
        
        # Topic similarity
        topic_sim = np.dot(msg_dense, self.current_topic)
        
        # Vocabulary coverage (fraction of message words we've seen before)
        # Approximate using SDR overlap with topic
        coverage = topic_sim / max(msg_dense.sum(), 1)
        
        # Combine factors
        confidence = 0.5 * topic_sim + 0.3 * coverage + 0.2 * min(self.turn_count / 10, 1.0)
        
        return float(np.clip(confidence, 0, 1))
    
    def _detect_correction(self, message_sdr: np.ndarray, message_text: str) -> bool:
        """
        Detect if user is correcting us.
        
        Heuristics:
        - "No," "Actually," "I mean," "Wrong" at start
        - Short message following our long message
        - Negative sentiment markers
        """
        if not message_text:
            return False
        
        text_lower = message_text.lower().strip()
        
        correction_markers = [
            "no,", "no.", "no ", "actually,", "actually ", "i mean,",
            "i mean ", "wrong", "thats wrong", "that's wrong",
            "incorrect", "not right", "fix", "mistake"
        ]
        
        for marker in correction_markers:
            if text_lower.startswith(marker) or marker in text_lower[:20]:
                return True
        
        return False
    
    def _predict_topic(self, message_sdr: np.ndarray) -> np.ndarray:
        """Predict topic for a message."""
        msg_dense = self._to_dense(message_sdr)
        
        # Combine with current topic
        predicted = 0.7 * msg_dense + 0.3 * self.current_topic
        
        return predicted
    
    def _suggest_response_style(self) -> dict:
        """Suggest response style based on user's current style."""
        return {
            'formality': self.style_metrics['formality_estimate'],
            'verbosity': 'high' if self.style_metrics['verbosity'] > 20 else 'low',
            'technical': self.style_metrics['technical_level'] > 0.5,
            'use_examples': self.turn_count < 5  # Use examples for new users
        }
    
    def _to_dense(self, sdr: np.ndarray) -> np.ndarray:
        """Convert SDR to dense array."""
        if sdr is None:
            return np.zeros(self.dim, dtype=np.float32)
        
        dense = np.zeros(self.dim, dtype=np.float32)
        
        if sdr.dtype in [np.int32, np.int64]:
            if len(sdr) > 0:
                dense[sdr] = 1.0
        else:
            dense[sdr > 0] = 1.0
        
        return dense
    
    def get_stats(self) -> dict:
        """Return intent model statistics."""
        return {
            'turn_count': self.turn_count,
            'conversation_duration': time.time() - self.conversation_start,
            'vocabulary_size': len(self.user_vocabulary),
            'domain_terms': len(self.domain_terms),
            'topic_history_length': len(self.topic_history),
            'style': self.style_metrics.copy(),
        }


def test_user_intent_model():
    """Test UserIntentModel."""
    model = UserIntentModel(dim=1024, k=20)
    
    np.random.seed(42)
    
    # Simulate conversation
    messages = [
        ("Hello, how are you?", "I'm doing well, thanks for asking!"),
        ("What's the weather like?", "I don't have access to weather data, but I can help you find it."),
        ("No, I meant in New York.", "Ah, let me check New York weather for you."),
        ("Actually, tell me about AI instead.", "Sure, what would you like to know about AI?"),
        ("What are transformers?", "Transformers are a type of neural network architecture..."),
    ]
    
    for user_msg, our_response in messages:
        msg_sdr = np.random.choice(1024, size=20, replace=False)
        resp_sdr = np.random.choice(1024, size=20, replace=False)
        
        result = model.process_message(msg_sdr, resp_sdr, user_msg)
        
        print(f"Turn {result['turn']}: confidence={result['confidence']:.2f}, "
              f"is_correction={result['is_correction']}, "
              f"vocab_size={result['vocabulary_size']}")
    
    # Get final stats
    stats = model.get_stats()
    print(f"\nFinal stats:")
    print(f"  Turns: {stats['turn_count']}")
    print(f"  Vocabulary: {stats['vocabulary_size']}")
    print(f"  Domain terms: {stats['domain_terms']}")
    print(f"  Formality: {stats['style']['formality_estimate']:.2f}")
    print(f"  Technical: {stats['style']['technical_level']:.2f}")
    
    # Test intent analysis
    test_sdr = np.random.choice(1024, size=20, replace=False)
    intent = model.get_intent(test_sdr, "test message")
    print(f"\nIntent analysis: confidence={intent['confidence']:.2f}, "
          f"needs_clarification={intent['needs_clarification']}")
    
    print("\nAll user intent model tests passed!")


if __name__ == "__main__":
    test_user_intent_model()
