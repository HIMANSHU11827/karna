"""
RT-ALM: Real-Time Adaptive Language Model
==========================================
SDR Encoder Module

Encodes text input to Sparse Distributed Representations (SDRs).
Based on Kanerva (1988) Sparse Distributed Memory.

Honest attribution:
- SDR representation: Pentti Kanerva, "Sparse Distributed Memory" (1988)
- Character n-gram encoding: Standard technique in hyperdimensional computing
"""

import numpy as np
from typing import List, Set


class SDREncoder:
    """
    Encodes text to SDRs via character n-gram hashing.
    
    Each n-gram is hashed to a position in the SDR vector.
    The SDR is binary (0/1) with exactly k bits active.
    
    Properties:
    - Deterministic: same input always produces same SDR
    - Locality-preserving: similar inputs have similar SDRs
    - Fixed sparsity: exactly k active bits per encoding
    """
    
    def __init__(self, dim: int = 10000, k: int = 200, ngram_size: int = 3, seed: int = 42):
        self.dim = dim
        self.k = k
        self.ngram_size = ngram_size
        self.rng = np.random.RandomState(seed)
        
        # Hash function: each character gets a random vector
        # We use multiple hash functions for n-gram combination
        self.char_vectors = {}
        
        # Pre-compute common characters
        for c in 'abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:\'"-_\n':
            self.char_vectors[c] = self.rng.randint(0, 2, dim).astype(np.int8)
    
    def _get_char_vector(self, c: str) -> np.ndarray:
        """Get or create random vector for character."""
        if c not in self.char_vectors:
            self.char_vectors[c] = self.rng.randint(0, 2, self.dim).astype(np.int8)
        return self.char_vectors[c]
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to SDR via character n-gram hashing.
        
        1. Generate n-grams from text
        2. For each n-gram, compute hash positions
        3. Union all active bits
        4. Keep only top-k active bits (enforce sparsity)
        """
        text = text.lower()
        sdr = np.zeros(self.dim, dtype=np.int8)
        
        # Generate character n-grams
        ngrams = []
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.append(text[i:i + self.ngram_size])
        
        if not ngrams:
            return sdr
        
        # For each n-gram, compute hash and activate bits
        active_indices = set()
        for ngram in ngrams:
            # Hash n-gram to k positions
            positions = self._hash_ngram(ngram)
            active_indices.update(positions)
        
        # Union: activate all positions from all n-grams
        if active_indices:
            indices = np.array(list(active_indices), dtype=np.int32)
            sdr[indices] = 1
        
        # Enforce exact sparsity: keep top-k
        if np.sum(sdr) > self.k:
            # Randomly select k active bits
            active_positions = np.where(sdr == 1)[0]
            if len(active_positions) > self.k:
                selected = self.rng.choice(active_positions, self.k, replace=False)
                sdr[:] = 0
                sdr[selected] = 1
        
        return sdr
    
    def _hash_ngram(self, ngram: str) -> List[int]:
        """
        Hash n-gram to k positions in SDR.
        Uses character vectors combined via XOR.
        """
        # Combine character vectors via XOR
        combined = np.zeros(self.dim, dtype=np.int8)
        for c in ngram:
            combined ^= self._get_char_vector(c)
        
        # Activate top-k positions from combined vector
        positions = np.argsort(combined)[-self.k:]
        return positions.tolist()
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts to SDR matrix."""
        sdrs = np.zeros((len(texts), self.dim), dtype=np.int8)
        for i, text in enumerate(texts):
            sdrs[i] = self.encode(text)
        return sdrs
    
    def similarity(self, sdr1: np.ndarray, sdr2: np.ndarray) -> float:
        """Compute overlap similarity between two SDRs."""
        return float(np.sum(sdr1 & sdr2)) / self.k
    
    def union(self, sdr1: np.ndarray, sdr2: np.ndarray) -> np.ndarray:
        """Union (OR) of two SDRs."""
        return (sdr1 | sdr2).astype(np.int8)


class TextEncoder:
    """
    Higher-level text encoder that handles:
    - Tokenization (word-level)
    - Word-level SDR encoding
    - Sentence-level SDR aggregation
    """
    
    def __init__(self, dim: int = 10000, k: int = 200, seed: int = 42):
        self.dim = dim
        self.k = k
        self.ngram_encoder = SDREncoder(dim, k, ngram_size=3, seed=seed)
        
        # Word-level encoder: different seed for word-level features
        self.word_encoder = SDREncoder(dim, k, ngram_size=5, seed=seed + 1)
    
    def encode_word(self, word: str) -> np.ndarray:
        """Encode a single word to SDR."""
        return self.word_encoder.encode(word)
    
    def encode_text(self, text: str, level: str = 'sentence') -> np.ndarray:
        """
        Encode text at specified level.
        
        Args:
            text: Input text
            level: 'char', 'word', or 'sentence'
        """
        if level == 'char':
            return self.ngram_encoder.encode(text)
        
        # Word-level encoding
        words = text.lower().split()
        if not words:
            return np.zeros(self.dim, dtype=np.int8)
        
        if level == 'word':
            return self.encode_word(words[0])
        
        # Sentence level: union of word encodings
        sdr = np.zeros(self.dim, dtype=np.int8)
        for word in words:
            word_sdr = self.encode_word(word)
            sdr = (sdr | word_sdr).astype(np.int8)
        
        # Enforce sparsity
        active = np.where(sdr == 1)[0]
        if len(active) > self.k:
            selected = np.random.RandomState(42).choice(active, self.k, replace=False)
            sdr[:] = 0
            sdr[selected] = 1
        
        return sdr
    
    def batch_encode(self, texts: List[str], level: str = 'sentence') -> np.ndarray:
        """Encode multiple texts."""
        sdrs = np.zeros((len(texts), self.dim), dtype=np.int8)
        for i, text in enumerate(texts):
            sdrs[i] = self.encode_text(text, level)
        return sdrs


if __name__ == "__main__":
    # Demo
    encoder = TextEncoder(dim=10000, k=200)
    
    texts = [
        "hello how are you",
        "hello how are you",  # identical
        "helo how are you",   # typo
        "help me please",     # different
        "the weather is nice", # completely different
    ]
    
    print("SDR Encoding Demo")
    print("=" * 60)
    
    sdrs = []
    for text in texts:
        sdr = encoder.encode_text(text)
        sdrs.append(sdr)
        print(f"'{text}' -> {np.sum(sdr)} active bits")
    
    print("\nSimilarity Matrix:")
    print("-" * 60)
    for i in range(len(sdrs)):
        for j in range(len(sdrs)):
            sim = encoder.word_encoder.similarity(sdrs[i], sdrs[j])
            print(f"  {sim:.2f}", end="")
        print()
    
    # Show that similar texts have higher similarity
    print("\nSimilar text similarity:")
    print(f"  identical: {encoder.word_encoder.similarity(sdrs[0], sdrs[1]):.3f}")
    print(f"  typo:      {encoder.word_encoder.similarity(sdrs[0], sdrs[2]):.3f}")
    print(f"  different: {encoder.word_encoder.similarity(sdrs[0], sdrs[3]):.3f}")
