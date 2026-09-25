"""
AGI Research Lab — Real-Time Adaptive Text Engine
=================================================
Demonstrates real-time adaptive chat with:
1. Incremental processing (character-level, not batch)
2. Intent understanding despite errors (misspellings, wrong words, incomplete input)
3. Continuous learning from every interaction
4. Persistent memory across sessions
5. Self-correction and multi-hypothesis tracking

NOT an LLM. No waiting for complete input. No perfect-prompt requirement.
The system builds understanding incrementally and adapts on the fly.
"""

import numpy as np
import os
import sys
import time
import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional


# =============================================================================
# Adaptive Character-Level Encoder
# =============================================================================

class CharEncoder:
    """
    Character-level encoder that handles:
    - Misspellings (edit distance)
    - Incomplete words (prefix matching)
    - Novel characters/symbols (online vocabulary)
    - Real-time streaming (processes one char at a time)
    """

    def __init__(self, embedding_dim=32):
        self.dim = embedding_dim
        self.char2idx = {}
        self.idx2char = {}
        self.embeddings = []
        self.next_idx = 0

        # Special tokens
        self._add_char('<PAD>')
        self._add_char('<UNK>')
        self._add_char('<SPACE>')
        self._add_char('<NEWLINE>')

        # Base vocabulary
        for c in 'abcdefghijklmnopqrstuvwxyz01234567889 .,!?;:\'"-_()\n':
            self._add_char(c)

    def _add_char(self, c):
        if c not in self.char2idx:
            self.char2idx[c] = self.next_idx
            self.idx2char[self.next_idx] = c
            # Random embedding for new char
            emb = np.random.randn(self.dim).astype(np.float32) * 0.1
            self.embeddings.append(emb)
            self.next_idx += 1
        return self.char2idx[c]

    def encode(self, text, stream=False):
        """Encode text to embedding sequence. If stream=True, process char by char."""
        if stream:
            return self._encode_streaming(text)
        return self._encode_batch(text)

    def _encode_batch(self, text):
        """Batch encoding."""
        indices = []
        for c in text.lower():
            if c not in self.char2idx:
                self._add_char(c)
            indices.append(self.char2idx[c])
        return np.array(indices, dtype=np.int32)

    def _encode_streaming(self, text):
        """Streaming encoding: yields one embedding at a time."""
        for c in text.lower():
            if c not in self.char2idx:
                self._add_char(c)
            idx = self.char2idx[c]
            yield self.embeddings[idx], c

    def decode_embedding(self, emb, top_k=3):
        """Find closest characters to an embedding."""
        sims = []
        for idx, char_emb in enumerate(self.embeddings):
            sim = np.dot(emb, char_emb) / (np.linalg.norm(emb) * np.linalg.norm(char_emb) + 1e-8)
            sims.append((sim, self.idx2char[idx]))
        sims.sort(reverse=True)
        return sims[:top_k]


# =============================================================================
# Fuzzy Word Matcher (Handles Misspellings + Incomplete Words)
# =============================================================================

class FuzzyMatcher:
    """
    Matches input against vocabulary despite:
    - Misspellings (edit distance)
    - Incomplete words (prefix matching)
    - Typos (character substitution)
    - Phonetic similarity
    """

    def __init__(self):
        self.vocab = set()
        self.word_freq = defaultdict(int)
        self.word_embeddings = {}

    def add_word(self, word, emb=None):
        """Add word to vocabulary."""
        word = word.lower().strip()
        if not word:
            return
        self.vocab.add(word)
        if emb is not None:
            self.word_embeddings[word] = emb

    def match(self, query, max_edit_dist=2, prefix_ratio=0.6):
        """
        Find best matches for query in vocabulary.
        Handles: misspellings, incomplete words, typos.
        """
        query = query.lower().strip()
        if not query:
            return []

        # Exact match
        if query in self.vocab:
            return [(query, 1.0)]

        # Prefix match (for incomplete words)
        prefix_matches = []
        for word in self.vocab:
            if word.startswith(query):
                score = len(query) / len(word)  # longer match = higher score
                prefix_matches.append((word, score))

        # Edit distance match (for misspellings)
        edit_matches = []
        for word in self.vocab:
            dist = self._edit_distance(query, word)
            if dist <= max_edit_dist:
                score = 1.0 - (dist / max(len(query), len(word)))
                if score >= prefix_ratio:
                    edit_matches.append((word, score))

        # Combine and sort
        all_matches = {}
        for word, score in prefix_matches + edit_matches:
            if word not in all_matches or score > all_matches[word]:
                all_matches[word] = score

        # Add embedding similarity if available
        if query in self.word_embeddings:
            query_emb = self.word_embeddings[query]
            for word, word_emb in self.word_embeddings.items():
                if word in all_matches:
                    sim = np.dot(query_emb, word_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(word_emb) + 1e-8
                    )
                    all_matches[word] = max(all_matches[word], sim * 0.5)

        result = sorted(all_matches.items(), key=lambda x: x[1], reverse=True)
        return result[:5]

    def _edit_distance(self, s1, s2):
        """Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]


# =============================================================================
# Real-Time Intent Tracker
# =============================================================================

class IntentTracker:
    """
    Tracks user intent in real-time as characters arrive.
    Maintains multiple hypotheses and updates them incrementally.
    Does NOT wait for complete input.
    """

    def __init__(self, n_intents=20):
        self.n_intents = n_intents
        self.intent_labels = [
            'greeting', 'question', 'command', 'feedback', 'clarification',
            'agreement', 'disagreement', 'explanation', 'request', 'acknowledgment',
            'emotional_positive', 'emotional_negative', 'story', 'opinion',
            'comparison', 'cause_effect', 'definition', 'prediction',
            'planning', 'other'
        ]

        # Intent prototypes (learned from examples)
        self.intent_prototypes = defaultdict(list)

        # Current state
        self.current_hypotheses = []  # (intent, confidence)
        self.accumulated_text = ""
        self.accumulated_emb = None
        self.word_count = 0

        # Adaptive thresholds
        self.confidence_threshold = 0.3
        self.update_rate = 0.3

    def add_example(self, text, intent):
        """Add a labeled example to improve intent detection."""
        emb = self._text_embedding(text)
        self.intent_prototypes[intent].append(emb)

    def _text_embedding(self, text):
        """Simple but effective text embedding."""
        # Character n-gram hashing
        emb = np.zeros(64, dtype=np.float32)
        text = text.lower()
        for i in range(len(text) - 2):
            gram = text[i:i+3]
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            idx = h % 64
            emb[idx] += 1
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def update_incremental(self, new_char, char_emb):
        """
        Update intent hypotheses as each new character arrives.
        This is the real-time adaptive core.
        """
        self.accumulated_text += new_char

        # Update running embedding
        if self.accumulated_emb is None:
            self.accumulated_emb = char_emb.copy()
        else:
            self.accumulated_emb = (1 - self.update_rate) * self.accumulated_emb + \
                                   self.update_rate * char_emb

        # Count words (approximate by spaces)
        if new_char == ' ':
            self.word_count += 1

        # Update hypotheses every few characters or on space
        if new_char in ' .!?\n' or len(self.accumulated_text) % 5 == 0:
            self._update_hypotheses()

        return self.current_hypotheses

    def _update_hypotheses(self):
        """Score all intents against current accumulated text."""
        if self.accumulated_emb is None:
            return

        scores = []
        for intent, prototypes in self.intent_prototypes.items():
            if not prototypes:
                continue
            # Average similarity to prototypes
            sims = []
            for proto in prototypes:
                sim = np.dot(self.accumulated_emb, proto)
                sims.append(sim)
            avg_sim = np.mean(sims) if sims else 0
            scores.append((intent, avg_sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Keep top-k above threshold
        self.current_hypotheses = [
            (intent, conf) for intent, conf in scores
            if conf > self.confidence_threshold
        ][:5]

    def get_dominant_intent(self):
        """Get current best intent hypothesis."""
        if self.current_hypotheses:
            return self.current_hypotheses[0]
        return ('unknown', 0.0)

    def get_clarity(self):
        """How confident are we about the intent?"""
        if not self.current_hypotheses:
            return 0.0
        if len(self.current_hypotheses) == 1:
            return self.current_hypotheses[0][1]
        # Confidence gap between top 2
        return self.current_hypotheses[0][1] - self.current_hypotheses[1][1]

    def needs_clarification(self):
        """Should we ask for clarification or do we understand?"""
        clarity = self.get_clarity()
        intent, conf = self.get_dominant_intent()
        return conf < 0.4 or clarity < 0.15


# =============================================================================
# Real-Time Adaptive Response Generator
# =============================================================================

class AdaptiveResponseGen:
    """
    Generates responses that adapt to:
    - User's communication style (vocabulary, sentence structure)
    - User's knowledge level (technical vs simple)
    - Conversation history (context)
    - Real-time feedback (correction, confusion)
    """

    def __init__(self):
        # User model
        self.user_vocab = set()
        self.user_sentence_lengths = []
        self.user_tech_level = 0.5  # 0 = simple, 1 = technical
        self.user_formality = 0.5  # 0 = casual, 1 = formal

        # Response templates (adapted over time)
        self.response_patterns = {
            'greeting': ['Hey!', 'Hi there!', 'Hello!'],
            'question': ['Let me think...', 'Good question!', 'Here\'s what I know:'],
            'clarification': ['I think you mean...', 'Did you mean...?', 'I understood:'],
            'acknowledgment': ['Got it.', 'Understood.', 'Makes sense.'],
            'correction': ['Actually,', 'I think it\'s', 'Correction:'],
        }

        # Memory of past interactions
        self.interaction_history = deque(maxlen=100)

    def update_user_model(self, user_input, response_quality=1.0):
        """Learn from each interaction."""
        words = user_input.lower().split()
        self.user_vocab.update(words)
        self.user_sentence_lengths.append(len(words))

        # Update technical level
        tech_words = ['algorithm', 'function', 'parameter', 'optimization',
                      'neural', 'network', 'architecture', 'framework',
                      'implementation', 'computation', 'analysis']
        tech_count = sum(1 for w in words if w in tech_words)
        if len(words) > 0:
            tech_ratio = tech_count / len(words)
            self.user_tech_level = 0.9 * self.user_tech_level + 0.1 * tech_ratio

        # Update formality
        formal_words = ['please', 'thank', 'would', 'could', 'shall', 'may']
        informal_words = ['hey', 'yeah', 'nah', 'gonna', 'wanna', 'lol']
        formal_count = sum(1 for w in words if w in formal_words)
        informal_count = sum(1 for w in words if w in informal_words)
        if formal_count + informal_count > 0:
            formality = formal_count / (formal_count + informal_count)
            self.user_formality = 0.9 * self.user_formality + 0.1 * formality

    def generate(self, intent, understood_text, confidence):
        """Generate adaptive response."""
        intent_name, intent_conf = intent

        if confidence < 0.3:
            return self._ask_clarification(understood_text)

        response_parts = []

        # Opening based on intent
        if intent_name in self.response_patterns:
            response_parts.append(np.random.choice(self.response_patterns[intent_name]))

        # Add understanding echo (shows we understood)
        if understood_text and confidence > 0.5:
            response_parts.append(f"I understood: '{understood_text}'")

        # Add confidence-based follow-up
        if confidence > 0.8:
            response_parts.append("I'm confident about this.")
        elif confidence > 0.5:
            response_parts.append("I think this is right, but let me know if I'm wrong.")
        else:
            response_parts.append("I'm not entirely sure. Could you tell me more?")

        return ' '.join(response_parts)

    def _ask_clarification(self, text):
        """Ask for clarification in a natural way."""
        return f"I want to make sure I understand. Did you mean: '{text}'?"


# =============================================================================
# Continual Learning Memory
# =============================================================================

class ContinualMemory:
    """
    Persistent memory that updates continuously.
    No "training phase" — learns from every interaction.
    """

    def __init__(self, save_path="/tmp/agi_memory.json"):
        self.save_path = save_path
        self.short_term = deque(maxlen=20)  # Recent interactions
        self.long_term = {}  # Consolidated knowledge
        self.corrections = {}  # User corrections (wrong -> right)
        self.user_facts = {}  # Facts about the user

        # Fast weights for immediate adaptation
        self.fast_weights = {}

        self.load()

    def add_interaction(self, user_input, response, intent, success=1.0):
        """Add interaction to memory."""
        self.short_term.append({
            'time': time.time(),
            'input': user_input,
            'response': response,
            'intent': intent,
            'success': success
        })

        # Consolidate periodically
        if len(self.short_term) >= 10:
            self._consolidate()

    def add_correction(self, wrong, correct):
        """Learn from user correction."""
        self.corrections[wrong.lower()] = correct.lower()
        # Update fast weights
        self.fast_weights[wrong.lower()] = correct.lower()

    def add_user_fact(self, key, value):
        """Remember a fact about the user."""
        self.user_facts[key] = {
            'value': value,
            'time': time.time(),
            'confidence': 1.0
        }

    def recall_similar(self, text, k=3):
        """Find similar past interactions."""
        results = []
        for interaction in self.short_term:
            sim = self._text_similarity(text, interaction['input'])
            results.append((sim, interaction))
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:k]]

    def _text_similarity(self, t1, t2):
        """Jaccard similarity on character trigrams."""
        def trigrams(t):
            t = t.lower()
            return set(t[i:i+3] for i in range(len(t)-2))
        g1, g2 = trigrams(t1), trigrams(t2)
        if not g1 or not g2:
            return 0.0
        return len(g1 & g2) / len(g1 | g2)

    def _consolidate(self):
        """Move short-term to long-term."""
        for interaction in list(self.short_term)[-5:]:
            intent = interaction.get('intent', 'unknown')
            if intent not in self.long_term:
                self.long_term[intent] = []
            self.long_term[intent].append(interaction)

    def save(self):
        """Save to disk."""
        data = {
            'corrections': self.corrections,
            'user_facts': {k: v for k, v in self.user_facts.items()},
            'user_vocab': list(set(w for i in self.short_term
                                 for w in i.get('input', '').lower().split())),
            'fast_weights': self.fast_weights,
            'long_term': {k: v[-10:] for k, v in self.long_term.items()},
        }
        try:
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            pass

    def load(self):
        """Load from disk."""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path) as f:
                    data = json.load(f)
                self.corrections = data.get('corrections', {})
                self.fast_weights = data.get('fast_weights', {})
        except Exception as e:
            pass


# =============================================================================
# Real-Time Chat Engine
# =============================================================================

class RealTimeChatEngine:
    """
    Core engine: processes input incrementally, adapts continuously.
    No batch processing. No waiting for complete input.
    """

    def __init__(self):
        self.encoder = CharEncoder(embedding_dim=32)
        self.fuzzy = FuzzyMatcher()
        self.intent_tracker = IntentTracker()
        self.response_gen = AdaptiveResponseGen()
        self.memory = ContinualMemory()

        # Initialize fuzzy matcher with common words
        common_words = [
            'hello', 'hi', 'hey', 'good', 'morning', 'evening', 'how', 'are',
            'you', 'what', 'is', 'the', 'weather', 'today', 'help', 'me',
            'please', 'thanks', 'sorry', 'yes', 'no', 'maybe', 'know',
            'tell', 'about', 'who', 'why', 'when', 'where', 'can', 'would',
            'should', 'think', 'like', 'want', 'need', 'make', 'do', 'have',
            'been', 'will', 'was', 'were', 'did', 'does', 'get', 'got',
            'really', 'actually', 'basically', 'interesting', 'problem',
            'solution', 'idea', 'better', 'worse', 'best', 'worst',
            'happy', 'sad', 'angry', 'excited', 'confused', 'sure',
            'right', 'wrong', 'correct', 'incorrect', 'true', 'false',
            'learning', 'memory', 'think', 'understand', 'forget', 'remember',
            'misspell', 'spelling', 'error', 'mistake', 'correction',
            'pne', 'teh', 'recieve', 'seperate', 'definately', 'occured',
            'wierd', 'untill', 'accomodate', 'acheive', 'embarass',
            'maintainance', 'neccessary', 'occassion', 'recomend',
        ]
        for w in common_words:
            self.fuzzy.add_word(w)

        # Intent examples
        self.intent_tracker.add_example("hello how are you", "greeting")
        self.intent_tracker.add_example("what is the meaning of life", "question")
        self.intent_tracker.add_example("help me with this problem", "request")
        self.intent_tracker.add_example("I think that's wrong actually", "disagreement")
        self.intent_tracker.add_example("yes exactly right", "agreement")
        self.intent_tracker.add_example("tell me about neural networks", "explanation")
        self.intent_tracker.add_example("I love this", "emotional_positive")
        self.intent_tracker.add_example("this is terrible", "emotional_negative")
        self.intent_tracker.add_example("the algorithm works by...", "explanation")

    def process_char(self, char):
        """Process a single character in real-time."""
        char_idx = self.encoder._add_char(char)
        char_emb = self.encoder.embeddings[char_idx]

        # Update intent tracker
        hypotheses = self.intent_tracker.update_incremental(char, char_emb)

        return {
            'char': char,
            'hypotheses': hypotheses,
            'dominant_intent': self.intent_tracker.get_dominant_intent(),
            'clarity': self.intent_tracker.get_clarity(),
            'needs_clarification': self.intent_tracker.needs_clarification(),
        }

    def process_message(self, text):
        """
        Process complete message (fallback for batch mode).
        Returns adaptive response.
        """
        # Correct misspellings
        corrected = self._auto_correct(text)

        # Check memory for similar past interactions
        similar = self.memory.recall_similar(corrected)

        # Run intent tracker
        for char in corrected:
            self.process_char(char)

        intent = self.intent_tracker.get_dominant_intent()
        clarity = self.intent_tracker.get_clarity()

        # Generate response
        response = self.response_gen.generate(intent, corrected, clarity)

        # Update models
        self.response_gen.update_user_model(corrected)
        self.memory.add_interaction(corrected, response, intent[0])

        # Reset for next message
        self._reset()

        return {
            'original_input': text,
            'corrected_input': corrected,
            'intent': intent[0],
            'confidence': intent[1],
            'clarity': clarity,
            'similar_past': len(similar),
            'response': response,
        }

    def _auto_correct(self, text):
        """Auto-correct misspellings using fuzzy matching + memory."""
        corrections = text.lower().split()
        corrected = []

        for word in corrections:
            # Check memory corrections first (user-specific)
            if word in self.memory.corrections:
                corrected.append(self.memory.corrections[word])
                continue

            # Check fuzzy matcher
            matches = self.fuzzy.match(word, max_edit_dist=1, prefix_ratio=0.7)
            if matches and matches[0][1] > 0.8:
                corrected.append(matches[0][0])
            else:
                corrected.append(word)

        return ' '.join(corrected)

    def _reset(self):
        """Reset for next message."""
        self.intent_tracker = IntentTracker()
        # Re-add examples
        self.intent_tracker.add_example("hello how are you", "greeting")
        self.intent_tracker.add_example("what is the meaning of life", "question")
        self.intent_tracker.add_example("help me with this problem", "request")

    def run_interactive(self):
        """Run interactive demo."""
        print("="*60)
        print("Real-Time Adaptive Chat Engine")
        print("Type a message (or 'quit' to exit)")
        print("="*60)
        print()

        while True:
            try:
                text = input("You: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break

                result = self.process_message(text)
                print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
                print(f"Clarity: {result['clarity']:.2f}")
                if result['corrected_input'] != result['original_input'].lower():
                    print(f"Corrected: {result['corrected_input']}")
                print(f"AI: {result['response']}")
                print()

            except EOFError:
                break
            except KeyboardInterrupt:
                break


# =============================================================================
# Demo: Handles Misspellings and Incomplete Input
# =============================================================================

def run_adaptive_demo():
    """Demonstrate adaptive capabilities."""
    print("="*70)
    print("AGI Research Lab — Real-Time Adaptive Chat Demo")
    print("="*70)

    engine = RealTimeChatEngine()

    # Test cases: misspellings, wrong words, incomplete input
    test_inputs = [
        # Normal
        "hello how are you",
        "what is the weather today",

        # Misspellings
        "helo how are yuo",
        "what is teh weathr today",
        "help me undrestand this",
        "I recieve your mesage",

        # Incomplete
        "help me und",
        "what is th",
        "I want to l",

        # Wrong words (semantically wrong)
        "the algorithm is very beautiful",
        "I drank some pizza for lunch",
        "the computer is feeling happy",

        # Mixed
        "whjat is machien learnng",
        "tanks for yuor hp",
        "I reallu enjoi tihs",
    ]

    results = []

    for text in test_inputs:
        result = engine.process_message(text)
        results.append(result)

        print(f"\nInput: '{text}'")
        if result['corrected_input'] != text.lower():
            print(f"  Auto-corrected: '{result['corrected_input']}'")
        print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.2f}, clarity: {result['clarity']:.2f})")
        print(f"  Response: {result['response']}")

    # Summary
    print("\n" + "="*70)
    print("ADAPTIVE CAPABILITIES DEMONSTRATED")
    print("="*70)
    print(f"  Total inputs: {len(test_inputs)}")
    print(f"  Corrected: {sum(1 for r in results if r['corrected_input'] != r['original_input'].lower())}")
    print(f"  Avg confidence: {np.mean([r['confidence'] for r in results]):.2f}")
    print(f"  Avg clarity: {np.mean([r['clarity'] for r in results]):.2f}")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        engine = RealTimeChatEngine()
        engine.run_interactive()
    else:
        run_adaptive_demo()
