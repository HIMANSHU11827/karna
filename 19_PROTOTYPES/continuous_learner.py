"""
AGI Research Lab — Continuous Learning Module
Learns from every interaction in real-time, adapts instantly.
"""

import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict


class ContinuousLearner:
    """
    Learns continuously from every interaction.
    No batch training — adapts instantly.
    """

    def __init__(self, data_dir: str = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/06_MEMORY"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Learned patterns
        self.patterns: Dict[str, int] = defaultdict(int)
        self.user_preferences: Dict[str, Any] = {}
        self.response_outcomes: List[Dict[str, Any]] = []

        # Load existing patterns
        self._load_patterns()

    def _load_patterns(self):
        """Load previously learned patterns."""
        path = self.data_dir / "learned_patterns.json"
        if path.exists():
            data = json.loads(path.read_text())
            self.patterns = defaultdict(int, data.get('patterns', {}))
            self.user_preferences = data.get('preferences', {})

    def save_patterns(self):
        """Save learned patterns."""
        path = self.data_dir / "learned_patterns.json"
        path.write_text(json.dumps({
            'patterns': dict(self.patterns),
            'preferences': self.user_preferences,
            'last_updated': time.time(),
        }, indent=2))

    def learn(self, key: str, value: Any = 1):
        """Learn a pattern."""
        self.patterns[key] += value
        self.save_patterns()

    def get_pattern(self, key: str) -> int:
        """Get pattern frequency."""
        return self.patterns.get(key, 0)

    def get_top_patterns(self, n: int = 10) -> List[tuple]:
        """Get most common patterns."""
        return sorted(self.patterns.items(), key=lambda x: x[1], reverse=True)[:n]

    def record_outcome(self, interaction: Dict[str, Any], success: bool):
        """Record whether a response was successful."""
        self.response_outcomes.append({
            **interaction,
            'success': success,
            'timestamp': time.time(),
        })

    def get_success_rate(self) -> float:
        """Get overall success rate."""
        if not self.response_outcomes:
            return 0.0
        successes = sum(1 for r in self.response_outcomes if r['success'])
        return successes / len(self.response_outcomes)


class AdaptiveInterface:
    """
    Adapts to user's communication style and preferences.
    Learns patterns like:
    - Prefers short vs long responses
    - Technical level (beginner/expert)
    - Common phrases and abbreviations
    - Time of day patterns
    """

    def __init__(self):
        self.learner = ContinuousLearner()
        self.response_lengths: List[int] = []
        self.interaction_times: List[float] = []

    def record_interaction(self, user_input: str, response: str, feedback: Optional[str] = None):
        """Record an interaction and learn from it."""
        # Learn patterns from user input
        for word in user_input.lower().split():
            self.learner.learn(f"word_{word}")

        # Learn from phrases (bigrams)
        words = user_input.lower().split()
        for i in range(len(words) - 1):
            self.learner.learn(f"phrase_{words[i]}_{words[i+1]}")

        # Track response length
        self.response_lengths.append(len(response))

        # Track interaction time
        self.interaction_times.append(time.time())

        # Learn from explicit feedback
        if feedback:
            self.learner.learn(f"feedback_{feedback}")

        # Determine response length preference
        if len(self.response_lengths) > 10:
            avg_len = sum(self.response_lengths[-10:]) / 10
            if avg_len < 50:
                self.learner.user_preferences['response_length'] = 'short'
            elif avg_len < 150:
                self.learner.user_preferences['response_length'] = 'medium'
            else:
                self.learner.user_preferences['response_length'] = 'long'

    def get_response_length_preference(self) -> str:
        """Get preferred response length."""
        return self.learner.user_preferences.get('response_length', 'medium')

    def get_common_topics(self, n: int = 5) -> List[str]:
        """Get most common topics."""
        word_patterns = [(k, v) for k, v in self.learner.patterns.items() if k.startswith('word_')]
        top = sorted(word_patterns, key=lambda x: x[1], reverse=True)[:n]
        return [k.replace('word_', '') for k, v in top]

    def get_stats(self) -> Dict[str, Any]:
        """Get interface stats."""
        return {
            'total_interactions': len(self.response_lengths),
            'response_length_pref': self.get_response_length_preference(),
            'common_topics': self.get_common_topics(),
            'learned_patterns': len(self.learner.patterns),
            'success_rate': self.learner.get_success_rate(),
        }
