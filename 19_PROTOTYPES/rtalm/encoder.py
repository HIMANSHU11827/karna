"""
RT-ALM: Character n-gram SDR Encoder
Handles misspellings via character-level n-gram hashing.
"""
import hashlib
import numpy as np
from sdr import SDR


class CharacterSDREncoder:
    """Encode text as SDR using character n-grams."""
    
    def __init__(self, N=1024, k=20, ngram_size=3):
        self.N = N
        self.k = k
        self.ngram_size = ngram_size
    
    def _hash_ngram(self, ngram):
        """Hash a character n-gram to k bit positions."""
        h = hashlib.md5(ngram.encode('utf-8')).digest()
        seed = int.from_bytes(h[:4], 'little')
        rng = np.random.RandomState(seed)
        return rng.choice(self.N, self.k, replace=False)
    
    def encode(self, text):
        """Encode a word/phrase as SDR."""
        sdr = SDR(self.N, self.k)
        bits = np.zeros(self.N, dtype=bool)
        
        # Add start/end tokens
        padded = f"#{text.lower()}#"
        
        # Slide n-gram window
        for i in range(len(padded) - self.ngram_size + 1):
            ngram = padded[i:i + self.ngram_size]
            positions = self._hash_ngram(ngram)
            bits[positions] = True
        
        sdr.bits = bits
        return sdr
    
    def encode_sequence(self, text):
        """Encode a sequence of words as list of SDRs."""
        words = text.lower().split()
        return [self.encode(w) for w in words]
    
    def similarity(self, text_a, text_b):
        """Compute similarity between two strings."""
        sdr_a = self.encode(text_a)
        sdr_b = self.encode(text_b)
        return sdr_a.similarity(sdr_b)
