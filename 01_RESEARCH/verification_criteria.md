# Verification Criteria — Real-Time Adaptive Language Model

**Date:** 2026-09-23  
**Author:** verifier  
**Purpose:** Define measurable, falsifiable requirements for the design team. Every claim must be testable.

---

## 1. Core Claims → Verifiable Requirements

### Claim 1: "Generates and understands text"
**Verification requirement:**
- BLEU score > 20 on language modeling benchmark (or perplexity < 100 on held-out text)
- Next-token prediction accuracy > 60% on conversational text
- Coherence: human eval or automated metric (e.g., entity consistency across 5+ turns)

### Claim 2: "Learns continuously from conversation"
**Verification requirement:**
- After 10 interactions with a user on topic X, accuracy on X-related queries improves by > 10 percentage points
- No separate training epoch — weight updates happen during inference
- Learning is measurable: weights change after every interaction (delta > 0)

### Claim 3: "Handles misspellings and messy input naturally"
**Verification requirement:**
- Given input with 20% character-level noise (swaps, deletions, insertions), intent recovery rate > 85%
- No "please clarify" or "I didn't understand" responses
- Self-correction happens within the same response (no re-prompting)

### Claim 4: "Gets better the more you use it"
**Verification requirement:**
- Same user, 5 sessions over 5 days: accuracy on repeated query types improves monotonically
- Improvement curve: positive slope over time (not flat, not negative)
- No catastrophic forgetting: old query types don't degrade

### Claim 5: "No retraining cycles — adapts in real-time"
**Verification requirement:**
- Weight update latency < 1ms per sample (CPU)
- No batch training, no epochs, no "training mode" vs "inference mode"
- System is always learning, always serving

---

## 2. Architecture Constraints (from FINAL DIRECTION)

| Constraint | Verification method |
|------------|---------------------|
| No transformers | Code review: no attention, no softmax over sequences, no multi-head attention |
| No backpropagation | Code review: no chain rule, no gradient descent through layers |
| No existing templates (Hopfield, Krotov, BCM, etc.) | Literature comparison: novel weight update rule, not a renamed existing method |
| NumPy only | Import check: no PyTorch, no TensorFlow, no JAX |
| CPU-only | Runtime check: no CUDA, no GPU calls |
| Neuromorphic-ready | Design review: sparse, event-driven, local computation |

---

## 3. Novelty Requirements

The design must include at least ONE of:
1. **Novel weight update rule** — not Hebbian, not Oja, not BCM, not STDP. A new formula derived from first principles.
2. **Novel architecture topology** — not feedforward, not recurrent, not residual. A new connectivity pattern.
3. **Novel representation** — not embeddings, not one-hot, not sparse codes. A new way to encode information.
4. **Novel learning objective** — not MSE, not cross-entropy, not free energy. A new loss function derived from our project's goals.

**Verification:** The design document must explicitly state what is novel and cite why existing methods were rejected.

---

## 4. Real-Time Requirements

| Metric | Target | How to measure |
|--------|--------|----------------|
| Inference latency | < 10ms per token | `time.time()` around forward pass |
| Learning update | < 1ms per sample | `time.time()` around weight update |
| Throughput | > 100 tokens/sec | Tokens generated / wall-clock time |
| Memory growth | < 1MB per 1000 interactions | `psutil` or `tracemalloc` |
| Real-time factor | < 1.0 | Processing time / input duration |

---

## 5. Continual Learning Requirements

| Metric | Target | How to measure |
|--------|--------|----------------|
| Split-MNIST ACC | > 85% | Standard Split-MNIST protocol, single-headed |
| Split-MNIST BWT | > 0 (no forgetting) | $R_{i,T} - R_{i,i} > 0$ |
| Forward transfer | > 5% | Accuracy on new task > random init |
| Samples to adapt | < 50 | New pattern learned within 50 examples |
| No replay buffer | 0 stored samples | Code review: no storage of past inputs |

---

## 6. Interaction Quality Requirements

| Test | Target | Method |
|------|--------|--------|
| Misspelling recovery | > 85% intent accuracy | Inject 20% char noise into queries |
| Contradiction handling | Self-corrects within 1 turn | Give contradictory info, check if system resolves |
| Context retention | > 90% after 10 turns | Reference something said 10 turns ago |
| Style adaptation | Matches user's formality | User is casual → system becomes casual |
| Silence handling | Doesn't hallucinate | Empty input → appropriate response, not fabrication |

---

## 7. What "Done" Looks Like

A design is complete when:
1. Mathematical formulas are written down (not just described)
2. The weight update rule is specified as a closed-form equation
3. The architecture diagram shows information flow
4. Every claim in the design has a corresponding verification test
5. The team can implement it in < 500 lines of NumPy

A design is NOT complete when:
1. It says "we'll use X" without specifying how X works
2. It references "standard techniques" without naming them
3. It claims "real-time" without measuring latency
4. It claims "learns continuously" without specifying the update rule
5. It's a renamed existing method (Hopfield, Krotov, etc.)

---

## 8. Red Flags — Automatic Fail

Any of these in the design → immediate rejection:
- "Attention mechanism" or "self-attention"
- "Backpropagation" or "gradient descent"
- "Transformer" or "BERT" or "GPT"
- "Oja's rule" or "BCM" or "STDP" or "Hebbian"
- "Hopfield network" or "Krotov network"
- "Predictive coding" (existing framework)
- "Sparse coding" (existing framework)
- Any reference to "improving" or "extending" an existing method

---

*verifier — the quality gate. If you can't test it, you haven't designed it.*
