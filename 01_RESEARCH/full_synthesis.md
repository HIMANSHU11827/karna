# SYNTHESIS: All Studies Merged

## What We Learned

### Transformers
- Self-attention provides content-based addressing but is O(n²)
- Weights frozen after pre-training — no real-time learning
- Feed-forward layers are dense and fixed
- **Problem:** Cannot learn continuously, too slow for real-time

### CNNs
- Great for spatial feature extraction via convolution
- No temporal modeling, no state, no working memory
- **Problem:** Feedforward only, no recurrence, no online learning

### RNNs/LSTMs
- Sequential processing with state
- Gating (LSTM) solves vanishing gradients but is inherently sequential
- **Problem:** Cannot parallelize, slow for long sequences

### Hebbian Learning
- Oja's rule does PCA, not classification (~20% MNIST)
- BCM adds sliding threshold but still not discriminative
- **Problem:** No error signal, cannot do discriminative learning

### Hopfield Networks
- Energy-based associative memory
- Krotov-Hopfield achieves 97.8% MNIST with L^p Hebbian
- **Problem:** Shallow credit assignment, limited capacity

### SDRs (Sparse Distributed Representations)
- Robust to noise, composable via XOR binding
- Kanerva's SDM provides content-addressable memory
- **Reuse:** We adopt SDRs as our core representation

### Predictive Coding
- Hierarchical error signals drive local learning
- Rao & Ballard: top-down predictions, bottom-up errors
- **Reuse:** We adopt predictive coding framework

### Continual Learning
- Catastrophic forgetting is the central challenge
- EWC, replay buffers all require batch training
- **Problem:** No good solution for single-example online learning

### Multimodal Systems
- CLIP/Whisper use Transformers + backprop (all banned)
- Fusion strategies: early, intermediate, late
- **Problem:** Need from-scratch fusion with local learning

### Real-Time Systems
- Streaming ML: SGD variants, concept drift detection
- Latency/throughput tradeoffs governed by Little's law
- **Key insight:** Target <50ms requires capped iterations and sparse ops

---

## What We Reuse (Known, Public Domain)

| Component | Source |
|-----------|--------|
| k-WTA activation | Maass (2000) |
| SDR representation | Kanerva (1988) |
| XOR binding | Plate (1995), VSA literature |
| Attractor dynamics | Hopfield (1982) |
| Predictive coding | Rao & Ballard (1999) |
| Eligibility traces | Sutton (1988) |
| Complementary learning | McClelland et al. (1995) |

---

## What Must Be Genuinely New

1. **Online Discriminative Learning Rule** — Learn from single examples in real-time
2. **Multimodal SDR Fusion** — Unified sparse representation across modalities
3. **User Intent Model** — Adapt to user style and preferences continuously
4. **Real-time predictive coding** — Hierarchical errors without batch training

---

## Architecture Decision: RT-ALM (Real-Time Adaptive Language Model)

Based on all studies, our system uses:

1. **Character n-gram SDR encoder** — Handles misspellings naturally
2. **4-level hierarchical predictive coding** — All sparse, all online
3. **Episodic + semantic memory** — Complementary learning systems
4. **Online discriminative learning** — Eligibility traces + error-driven updates
5. **Response via retrieval + perturbation** — Not autoregressive generation
6. **User pattern model** — Continuously updated

---

## Next Step

Write `04_ARCHITECTURE/FINAL_DESIGN.md` with full implementation spec.
