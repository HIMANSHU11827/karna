#!/usr/bin/env python3
"""
RT-ALM: Real-Time Adaptive Language Model
==========================================
Built from design docs. Profiled for real-time compliance.

Architecture:
  Text → Char SDR Encoder → Spatial Pooler (k-WTA) → Attractor Memory →
  Temporal Pooler (XOR) → Decoder → Text
"""

import numpy as np
import time
import json
import hashlib
from collections import deque
from typing import Dict, List, Optional, Tuple, Any


# =============================================================================
# 1. SDR (Sparse Distributed Representation)
# =============================================================================

class SDR:
    """Sparse Distributed Representation with N bits, k active."""
    
    N = 1024  # dimension
    K = 20    # active bits (2% sparsity)
    
    def __init__(self, N: int = 1024, k: int = 20):
        self.N = N
        self.k = k
        self.bits = np.zeros(N, dtype=np.bool_)
    
    def randomize(self, rng: np.random.RandomState):
        """Set k random bits."""
        self.bits = np.zeros(self.N, dtype=np.bool_)
        active = rng.choice(self.N, self.k, replace=False)
        self.bits[active] = True
        return self
    
    def overlap(self, other: 'SDR') -> int:
        """Count shared active bits."""
        return int(np.sum(self.bits & other.bits))
    
    def cosine_similarity(self, other: 'SDR') -> float:
        """Cosine similarity between SDRs."""
        ov = self.overlap(other)
        if self.k == 0 or other.k == 0:
            return 0.0
        return ov / np.sqrt(self.k * other.k)
    
    def union(self, other: 'SDR') -> 'SDR':
        """OR of two SDRs."""
        result = SDR(self.N, self.k)
        result.bits = self.bits | other.bits
        return result
    
    def xor_bind(self, other: 'SDR') -> 'SDR':
        """XOR binding."""
        result = SDR(self.N, self.k)
        result.bits = self.bits ^ other.bits
        return result


# =============================================================================
# 2. Character-Level SDR Encoder
# =============================================================================

class CharacterSDREncoder:
    """Encodes words to SDRs using character n-grams."""
    
    def __init__(self, N: int = 1024, k: int = 20, window: int = 3, seed: int = 42):
        self.N = N
        self.k = k
        self.window = window
        self.rng = np.random.RandomState(seed)
        
        # Pre-compute random SDRs for each byte value
        self.byte_sdrs = {}
        for i in range(256):
            sdr = SDR(N, k)
            sdr.randomize(self.rng)
            self.byte_sdrs[i] = sdr.bits
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to sparse vector via character n-gram hashing."""
        text = text.lower().strip()
        if not text:
            return np.zeros(self.N, dtype=np.bool_)
        
        # Union of character SDRs in sliding window
        result = np.zeros(self.N, dtype=np.bool_)
        padded = f"#{text}#"
        
        for i in range(len(padded) - self.window + 1):
            ngram = padded[i:i + self.window]
            # Hash n-gram to byte indices
            h = int(hashlib.md5(ngram.encode()).hexdigest()[:8], 16)
            for j in range(self.window):
                byte_idx = (h >> (j * 8)) & 0xFF
                result |= self.byte_sdrs[byte_idx]
        
        return result


# =============================================================================
# 3. Spatial Pooler (k-WTA)
# =============================================================================

class SpatialPooler:
    """k-Winners-take-all spatial pooling."""
    
    def __init__(self, N: int = 1024, k: int = 20, seed: int = 42):
        self.N = N
        self.k = k
        self.thresholds = np.ones(N, dtype=np.float32) * 0.5
        self.rng = np.random.RandomState(seed)
        self.eta_theta = 0.001
    
    def compete(self, input_bits: np.ndarray) -> np.ndarray:
        """Select top k bits."""
        activity = input_bits.astype(np.float32)
        
        # Apply thresholds
        activity = np.where(activity >= self.thresholds, activity, 0.0)
        
        # k-WTA: keep only top k
        if np.count_nonzero(activity) > self.k:
            top_k = np.argsort(activity)[-self.k:]
            result = np.zeros(self.N, dtype=np.bool_)
            result[top_k] = True
            return result
        else:
            return activity > 0
    
    def adapt_thresholds(self, activity: np.ndarray, target: float = 0.02):
        """ATM: adapt thresholds to maintain target sparsity."""
        actual = np.mean(activity > 0)
        self.thresholds += self.eta_theta * (actual - target)
        self.thresholds = np.clip(self.thresholds, 0.01, 0.99)


# =============================================================================
# 4. Attractor Memory (SHD Rule)
# =============================================================================

class AttractorMemory:
    """Hopfield-like attractor network with Sparse Hebbian + Decay."""
    
    def __init__(self, N: int = 1024, k: int = 20, seed: int = 42):
        self.N = N
        self.k = k
        self.W = np.zeros((N, N), dtype=np.float32)
        self.eta = 0.01
        self.lam = 0.001
        self.T = 5  # fixed iterations for bounded latency
    
    def converge(self, query_bits: np.ndarray) -> np.ndarray:
        """Iterate to nearest attractor (bounded T iterations)."""
        state = query_bits.astype(np.float32).copy()
        
        for _ in range(self.T):
            # Compute input to each neuron
            input_vec = self.W @ state
            
            # k-WTA: top k become active
            inp_copy = input_vec.copy()
            top_k = np.argsort(inp_copy)[-self.k:]
            new_state = np.zeros(self.N, dtype=np.float32)
            new_state[top_k] = 1.0
            
            # Check convergence
            if np.allclose(state, new_state, atol=0.01):
                break
            
            state = new_state
        
        return state > 0
    
    def store(self, query_bits: np.ndarray, target_bits: np.ndarray):
        """SHD learning rule."""
        # Hebbian: outer product
        hebbian = np.outer(target_bits.astype(np.float32), 
                          query_bits.astype(np.float32))
        
        # Decay: weaken all weights
        decay = self.lam * self.W
        
        # Update (only active connections change meaningfully)
        self.W += self.eta * (hebbian - decay)
        self.W = np.clip(self.W, 0, 1)
    
    def recall(self, query_bits: np.ndarray) -> np.ndarray:
        """Store and recall in one step."""
        converged = self.converge(query_bits)
        self.store(query_bits, converged)
        return converged


# =============================================================================
# 5. Temporal Pooler (XOR Binding + SPU)
# =============================================================================

class TemporalPooler:
    """XOR binding-based temporal pooler with Sparse Predictive Update."""
    
    def __init__(self, N: int = 1024, k: int = 20, seed: int = 42):
        self.N = N
        self.k = k
        self.W = np.zeros((N, N), dtype=np.float32)
        self.eta = 0.01
    
    def bind(self, sdr_a: np.ndarray, sdr_b: np.ndarray) -> np.ndarray:
        """XOR binding of two SDRs."""
        return sdr_a ^ sdr_b
    
    def learn_transition(self, current: np.ndarray, next_sdr: np.ndarray):
        """SPU: learn transition."""
        # Transition = current XOR next
        transition = current ^ next_sdr
        
        # Store: current ⊕ transition → next
        self.W += self.eta * np.outer(
            next_sdr.astype(np.float32),
            transition.astype(np.float32)
        )
    
    def predict_next(self, current: np.ndarray) -> np.ndarray:
        """Predict next state from current."""
        input_vec = self.W @ current.astype(np.float32)
        
        # Keep top k
        top_k = np.argsort(input_vec)[-self.k:]
        result = np.zeros(self.N, dtype=np.bool_)
        result[top_k] = True
        return result


# =============================================================================
# 6. Real-Time Adaptive Language Model (RT-ALM)
# =============================================================================

class RTALM:
    """Complete Real-Time Adaptive Language Model."""
    
    def __init__(self, N: int = 1024, k: int = 20, seed: int = 42):
        self.N = N
        self.k = k
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Components
        self.encoder = CharacterSDREncoder(N, k, seed=seed)
        self.spatial_pooler = SpatialPooler(N, k, seed=seed)
        self.attractor = AttractorMemory(N, k, seed=seed)
        self.temporal = TemporalPooler(N, k, seed=seed)
        
        # State
        self.vocabulary: Dict[str, np.ndarray] = {}  # word → SDR
        self.episodic_memory: List[Tuple[np.ndarray, str, float]] = []  # (SDR, response, timestamp)
        self.user_patterns: Dict[str, int] = {}
        
        # Context
        self.context: deque = deque(maxlen=20)
        self.last_sdr: Optional[np.ndarray] = None
    
    def process(self, text: str, learn: bool = True) -> np.ndarray:
        """Process input text through full pipeline."""
        # 1. Encode
        text_sdr = self.encoder.encode(text)
        
        # 2. Spatial pool
        pooled = self.spatial_pooler.compete(text_sdr)
        self.spatial_pooler.adapt_thresholds(pooled)
        
        # 3. Attractor converge
        recalled = self.attractor.recall(pooled)
        
        # 4. Temporal
        if self.last_sdr is not None:
            self.temporal.learn_transition(self.last_sdr, recalled)
        
        self.last_sdr = recalled
        
        # 5. Learn vocabulary
        if learn:
            words = text.lower().split()
            for word in words:
                if word not in self.vocabulary:
                    word_sdr = self.encoder.encode(word)
                    self.vocabulary[word] = word_sdr
        
        return recalled
    
    def generate(self, prompt: str, max_words: int = 10) -> str:
        """Generate text from prompt."""
        prompt_sdr = self.process(prompt, learn=False)
        
        current = prompt_sdr
        generated = []
        
        for _ in range(max_words):
            # Predict next
            next_sdr = self.temporal.predict_next(current)
            
            # Decode to word (closest in vocabulary)
            best_word = None
            best_sim = 0.3  # threshold
            
            for word, word_sdr in self.vocabulary.items():
                sim = self._cosine_sim(next_sdr, word_sdr)
                if sim > best_sim:
                    best_sim = sim
                    best_word = word
            
            if best_word is None:
                break
            
            generated.append(best_word)
            current = next_sdr
        
        return " ".join(generated)
    
    def respond(self, text: str, learn: bool = True) -> str:
        """Generate response for input text."""
        # Process input
        text_sdr = self.process(text, learn=learn)
        
        # Find similar episodic memory
        if self.episodic_memory:
            best_response = None
            best_sim = 0.5  # threshold
            
            for mem_sdr, response, ts in self.episodic_memory[-100:]:  # recent 100
                sim = self._cosine_sim(text_sdr, mem_sdr)
                if sim > best_sim:
                    best_sim = sim
                    best_response = response
            
            if best_response:
                return best_response
        
        # Default response
        return "I understand. Could you tell me more?"
    
    def store_interaction(self, user_input: str, response: str):
        """Store interaction in episodic memory."""
        input_sdr = self.encoder.encode(user_input)
        self.episodic_memory.append((input_sdr, response, time.time()))
    
    def learn_from_feedback(self, user_input: str, correct_response: str, positive: bool):
        """Learn from explicit feedback."""
        input_sdr = self.encoder.encode(user_input)
        
        if positive:
            # Store in episodic memory
            self.store_interaction(user_input, correct_response)
            self.user_patterns[user_input] = self.user_patterns.get(user_input, 0) + 1
        else:
            # Find and update
            for i, (mem_sdr, resp, ts) in enumerate(self.episodic_memory):
                sim = self._cosine_sim(input_sdr, mem_sdr)
                if sim > 0.7:
                    # Replace with correct response
                    self.episodic_memory[i] = (mem_sdr, correct_response, ts)
                    break
            else:
                # Not found, add new
                self.store_interaction(user_input, correct_response)
    
    def consolidate(self):
        """Memory consolidation: reinforce frequent patterns."""
        for pattern, count in self.user_patterns.items():
            if count > 2:
                # Reinforce in attractor memory
                pattern_sdr = self.encoder.encode(pattern)
                self.attractor.store(pattern_sdr, pattern_sdr)
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two binary vectors."""
        a_bool = a.astype(np.bool_)
        b_bool = b.astype(np.bool_)
        ov = np.sum(a_bool & b_bool)
        if np.sum(a_bool) == 0 or np.sum(b_bool) == 0:
            return 0.0
        return ov / np.sqrt(np.sum(a_bool) * np.sum(b_bool))
    
    def stats(self) -> Dict[str, Any]:
        return {
            "vocabulary_size": len(self.vocabulary),
            "episodic_memory_size": len(self.episodic_memory),
            "user_patterns": len(self.user_patterns),
        }


# =============================================================================
# Profiling Suite
# =============================================================================

def profile_component(name: str, fn, n: int = 1000, warmup: int = 100) -> Dict[str, float]:
    """Profile a component with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Timed runs
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    times.sort()
    return {
        "name": name,
        "n": n,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": times[0],
        "max_ms": times[-1],
        "p50_ms": times[int(n * 0.50)],
        "p90_ms": times[int(n * 0.90)],
        "p95_ms": times[int(n * 0.95)],
        "p99_ms": times[int(n * 0.99)],
    }


def profile_memory(model: RTALM, sizes: List[int]) -> List[Dict]:
    """Profile memory usage at different scales."""
    import tracemalloc
    
    results = []
    for n in sizes:
        tracemalloc.start()
        
        # Create fresh model
        m = RTALM(N=1024, k=20, seed=42)
        
        # Process n interactions
        for i in range(n):
            m.process(f"Message number {i} for testing", learn=True)
            m.respond(f"Response to message {i}", learn=False)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results.append({
            "n_interactions": n,
            "current_kb": current / 1024,
            "peak_kb": peak / 1024,
            "vocab_size": len(m.vocabulary),
            "episodic_size": len(m.episodic_memory),
        })
    
    return results


def profile_throughput(model: RTALM, n: int = 1000) -> Dict:
    """Profile throughput under load."""
    start = time.perf_counter()
    
    for i in range(n):
        model.process(f"Message {i} for throughput test", learn=True)
        model.respond(f"Response {i}", learn=False)
    
    total = time.perf_counter() - start
    
    return {
        "total_requests": n,
        "total_time_s": total,
        "requests_per_second": n / total,
        "avg_latency_ms": total / n * 1000,
    }


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("=" * 70)
    print("RT-ALM: Real-Time Adaptive Language Model — BENCHMARKS")
    print("=" * 70)
    
    model = RTALM(N=1024, k=20, seed=42)
    
    # Pre-train with some interactions
    print("\n[Setup] Pre-training with 100 interactions...")
    for i in range(100):
        model.process(f"Hello world this is message {i}", learn=True)
        model.respond(f"Response to message {i}", learn=False)
        model.store_interaction(f"Message {i}", f"Response {i}")
    
    print(f"  Vocabulary: {len(model.vocabulary)} words")
    print(f"  Episodic: {len(model.episodic_memory)} interactions")
    
    # =========================================================================
    # 1. Component-level latency
    # =========================================================================
    print("\n" + "─" * 70)
    print("1. COMPONENT-LEVEL LATENCY")
    print("─" * 70)
    
    # 1a. Encoder
    def encode_fn():
        model.encoder.encode("hello world test message")
    
    enc_stats = profile_component("Encoder", encode_fn, n=5000)
    print(f"\n  Encoder (char SDR):")
    print(f"    P50={enc_stats['p50_ms']:.3f}ms  P95={enc_stats['p95_ms']:.3f}ms  "
          f"P99={enc_stats['p99_ms']:.3f}ms")
    
    # 1b. Spatial Pooler
    test_input = model.encoder.encode("test input")
    def spatial_fn():
        model.spatial_pooler.compete(test_input)
    
    spa_stats = profile_component("Spatial Pooler", spatial_fn, n=5000)
    print(f"\n  Spatial Pooler (k-WTA):")
    print(f"    P50={spa_stats['p50_ms']:.3f}ms  P95={spa_stats['p95_ms']:.3f}ms  "
          f"P99={spa_stats['p99_ms']:.3f}ms")
    
    # 1c. Attractor Memory
    pooled = model.spatial_pooler.compete(test_input)
    def attractor_fn():
        model.attractor.recall(pooled)
    
    att_stats = profile_component("Attractor", attractor_fn, n=2000)
    print(f"\n  Attractor Memory (SHD, T={model.attractor.T} iterations):")
    print(f"    P50={att_stats['p50_ms']:.3f}ms  P95={att_stats['p95_ms']:.3f}ms  "
          f"P99={att_stats['p99_ms']:.3f}ms")
    
    # 1d. Temporal Pooler
    recalled = model.attractor.recall(pooled)
    def temporal_fn():
        model.temporal.predict_next(recalled)
    
    tem_stats = profile_component("Temporal", temporal_fn, n=5000)
    print(f"\n  Temporal Pooler (XOR):")
    print(f"    P50={tem_stats['p50_ms']:.3f}ms  P95={tem_stats['p95_ms']:.3f}ms  "
          f"P99={tem_stats['p99_ms']:.3f}ms")
    
    # 1e. Full process
    def process_fn():
        model.process("hello world test", learn=True)
    
    proc_stats = profile_component("Full Process", process_fn, n=2000)
    print(f"\n  Full Process (encode→pool→attractor→temporal+learn):")
    print(f"    P50={proc_stats['p50_ms']:.3f}ms  P95={proc_stats['p95_ms']:.3f}ms  "
          f"P99={proc_stats['p99_ms']:.3f}ms")
    
    # 1f. Respond (with retrieval)
    def respond_fn():
        model.respond("hello world", learn=False)
    
    resp_stats = profile_component("Respond", respond_fn, n=1000)
    print(f"\n  Respond (process + retrieval):")
    print(f"    P50={resp_stats['p50_ms']:.3f}ms  P95={resp_stats['p95_ms']:.3f}ms  "
          f"P99={resp_stats['p99_ms']:.3f}ms")
    
    # 1g. Learning update
    def learn_fn():
        model.learn_from_feedback("test input", "correct response", True)
    
    learn_stats = profile_component("Learning Update", learn_fn, n=2000)
    print(f"\n  Learning Update (feedback-based):")
    print(f"    P50={learn_stats['p50_ms']:.3f}ms  P95={learn_stats['p95_ms']:.3f}ms  "
          f"P99={learn_stats['p99_ms']:.3f}ms")
    
    # =========================================================================
    # 2. Memory scaling
    # =========================================================================
    print("\n" + "─" * 70)
    print("2. MEMORY USAGE SCALING")
    print("─" * 70)
    
    mem_results = profile_memory(model, [100, 500, 1000])
    for r in mem_results:
        print(f"  N={r['n_interactions']:4d}:  "
              f"Current={r['current_kb']:.0f}KB  Peak={r['peak_kb']:.0f}KB  "
              f"Vocab={r['vocab_size']:4d}  Episodic={r['episodic_size']:4d}")
    
    # =========================================================================
    # 3. Throughput
    # =========================================================================
    print("\n" + "─" * 70)
    print("3. THROUGHPUT UNDER LOAD")
    print("─" * 70)
    
    # Fresh model for throughput test
    model2 = RTALM(N=1024, k=20, seed=42)
    for i in range(50):
        model2.process(f"pretrain {i}", learn=True)
    
    thr_stats = profile_throughput(model2, n=500)
    print(f"  Total: {thr_stats['total_requests']} requests")
    print(f"  Time: {thr_stats['total_time_s']:.2f}s")
    print(f"  Throughput: {thr_stats['requests_per_second']:.0f} req/s")
    print(f"  Avg latency: {thr_stats['avg_latency_ms']:.2f}ms")
    
    # =========================================================================
    # 4. Bottleneck identification
    # =========================================================================
    print("\n" + "─" * 70)
    print("4. BOTTLENECK IDENTIFICATION")
    print("─" * 70)
    
    all_stats = [
        ("Encoder", enc_stats),
        ("Spatial Pooler", spa_stats),
        ("Attractor Memory", att_stats),
        ("Temporal Pooler", tem_stats),
        ("Full Process", proc_stats),
        ("Respond", resp_stats),
        ("Learning", learn_stats),
    ]
    
    print(f"\n  {'Component':<20s} {'P50':>8s} {'P95':>8s} {'P99':>8s} {'Budget':>8s} {'Status':>8s}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    
    budgets = {
        "Encoder": 10.0,
        "Spatial Pooler": 5.0,
        "Attractor Memory": 10.0,
        "Temporal Pooler": 5.0,
        "Full Process": 25.0,
        "Respond": 50.0,
        "Learning": 1.0,
    }
    
    for name, stats in all_stats:
        budget = budgets.get(name, 50.0)
        p99 = stats['p99_ms']
        status = "PASS" if p99 < budget else "FAIL"
        print(f"  {name:<20s} {stats['p50_ms']:>7.2f}ms {stats['p95_ms']:>7.2f}ms "
              f"{stats['p99_ms']:>7.2f}ms {budget:>7.1f}ms {status:>8s}")
    
    # =========================================================================
    # 5. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_p99 = sum(s['p99_ms'] for _, s in all_stats)
    total_budget = sum(budgets.values())
    print(f"\n  Total P99 (all components): {total_p99:.2f}ms")
    print(f"  Total budget: {total_budget:.1f}ms")
    print(f"  Margin: {total_budget - total_p99:.2f}ms")
    
    # The real end-to-end is process + respond
    e2e_p99 = proc_stats['p99_ms'] + resp_stats['p99_ms']
    print(f"\n  End-to-end P99 (process + respond): {e2e_p99:.2f}ms")
    print(f"  Budget: 50ms")
    print(f"  Margin: {50 - e2e_p99:.2f}ms")
    print(f"  Status: {'PASS' if e2e_p99 < 50 else 'FAIL'}")
    
    # Throughput
    print(f"\n  Throughput: {thr_stats['requests_per_second']:.0f} req/s")
    print(f"  Memory: {mem_results[-1]['peak_kb']:.0f}KB at {mem_results[-1]['n_interactions']} interactions")
    
    # Novelty check
    print(f"\n  Vocabulary learned: {len(model.vocabulary)} words (from text only)")
    print(f"  Episodic memories: {len(model.episodic_memory)} interactions")
    print(f"  User patterns: {len(model.user_patterns)} patterns")
    
    print("\n" + "=" * 70)
    print("Benchmarks complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_full_benchmark()
