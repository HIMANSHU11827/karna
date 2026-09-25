# RNNs/LSTMs Study

**Status**: Research — Sequential Processing Architectures
**Output**: `01_RESEARCH/rnns_study.md`
**Scope**: RNNs, LSTMs, GRUs — gating, vanishing gradients, speed limitations

---

## 1. Sequential Processing

### 1.1 What RNNs Do

Recurrent Neural Networks process sequences by maintaining a hidden state that captures information from previous time steps:

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
```

At each time step t, the hidden state h_t depends on:
- Current input x_t
- Previous hidden state h_{t-1} (the "memory")

### 1.2 Why This Matters for AGI

Real-time AGI must process streaming input — text arrives token by token, audio arrives sample by sample, video arrives frame by frame. RNNs are designed for exactly this.

### 1.3 The Sequential Bottleneck

The critical limitation: **h_t depends on h_{t-1}**. This means:
- Cannot parallelize across time steps
- GPU sits idle waiting for sequential computation
- Training time scales linearly with sequence length
- For sequence of length T, need T sequential operations

---

## 2. Gating Mechanisms (LSTM/GRU)

### 2.1 LSTM (Hochreiter & Schmidhuber, 1997)

Long Short-Term Memory introduced a separate cell state that acts as a "conveyor belt" for information:

**Three gates:**

| Gate | Formula | Purpose |
|---|---|---|
| **Forget gate** | f_t = σ(W_f * x_t + U_f * h_{t-1}) | What to erase from memory |
| **Input gate** | i_t = σ(W_i * x_t + U_i * h_{t-1}) | What new info to write |
| **Output gate** | o_t = σ(W_o * x_t + U_o * h_{t-1}) | What to expose as output |

**Cell state update (the key innovation):**

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

Where c̃_t = tanh(W_c * x_t + U_c * h_{t-1})

**Why this matters**: The cell state is updated by **addition**, not multiplication. Gradients flow through addition without decay.

### 2.2 GRU (Cho et al., 2014)

Gated Recurrent Unit — simpler, fewer parameters:

| Gate | Formula | Purpose |
|---|---|---|
| **Update gate** | z_t = σ(W_z * x_t + U_z * h_{t-1}) | Balance old vs new |
| **Reset gate** | r_t = σ(W_r * x_t + U_r * h_{t-1}) | How much past to use |

```
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### 2.3 Why Gates Work

Gating creates a **gradient highway**: information can flow across hundreds of steps without the repeated multiplications that cause vanishing gradients.

---

## 3. Vanishing Gradient Problem

### 3.1 The Math

In Backpropagation Through Time (BPTT), the gradient at time T must travel back to time t through a product of Jacobians:

```
∂L/∂h_t = ∂L/∂h_T * ∏(k=t+1 to T) ∂h_k/∂h_{k-1}
```

Each Jacobian contains the recurrent weight matrix W_hh and the derivative of tanh (bounded by 1).

### 3.2 Why Gradients Vanish

If the largest eigenvalue of W_hh is less than 1, the product shrinks exponentially:

| Spectral Radius | After 100 steps | Result |
|---|---|---|
| 0.9 | 0.9^100 ≈ 0.0000266 | Vanished |
| 1.0 | 1.0^100 = 1 | Preserved (ideal but unstable) |
| 1.1 | 1.1^100 ≈ 13,780 | Exploded |

**The tanh derivative makes it worse**: tanh'(x) ∈ [0, 1], and in practice averages ~0.6. So even with spectral radius 1.0, effective gain is ~0.77 per step. Over 19 steps: 0.77^19 ≈ 0.0062.

### 3.3 Practical Limits

| Sequence Length | Vanilla RNN | LSTM/GRU |
|---|---|---|
| < 20 tokens | Usually works | Works |
| 20-50 tokens | Struggles | Works |
| 50-100 tokens | Fails | Works |
| 100+ tokens | Fails | Degrades gradually |

### 3.4 Why LSTM Solves It

The LSTM cell state is updated by **addition**, not multiplication:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

The gradient through the cell state is simply the forget gate value (a scalar between 0 and 1), not a full matrix multiplication. When forget gate ≈ 1, gradients pass through unchanged.

**Key insight**: Addition preserves gradient magnitude. Multiplication shrinks it.

---

## 4. Why RNNs Are Slow

### 4.1 The Sequential Bottleneck

RNNs process sequences strictly left-to-right:

```
h_1 → h_2 → h_3 → ... → h_T
```

Each step depends on the previous one. This means:
- **Cannot parallelize** across time steps
- **GPU sits idle** — most silicon waits for sequential computation
- **Training time** scales linearly with sequence length

### 4.2 Concrete Numbers

For sequence length 2,048, batch 64, hidden size 1,024:

| Architecture | Operations | Parallelism |
|---|---|---|
| **RNN** | 2,048 sequential steps × 67M MACs each | None — 2,048 dependent kernel launches |
| **Transformer** | 1 step × 137B MACs | Full — all positions processed simultaneously |

The RNN does 2,048 small matmuls that don't saturate the GPU. The Transformer does 1 massive matmul that the hardware is designed for.

### 4.3 Why This Killed RNNs

Between 2012-2017, the winning move in ML became "train larger models on more data using more accelerators." RNNs cannot convert bigger budgets into better models because their budget is spent on sequential latency, not arithmetic.

---

## 5. Comparison to Transformers

| Property | RNN/LSTM | Transformer |
|---|---|---|
| Sequential processing | Yes — O(T) steps | No — O(1) parallel |
| Long-range dependencies | Limited (even LSTM degrades) | Direct attention, O(1) path |
| Training speed | Slow (sequential) | Fast (parallel) |
| Inference speed | O(1) per token | O(T) per token (growing KV cache) |
| Memory usage | O(1) state | O(T) KV cache |
| Online/streaming | Natural fit | Needs full sequence |
| Biological plausibility | Higher (recurrence) | Lower (attention) |

### 5.1 Where RNNs Still Win

- **Online/streaming**: RNNs process tokens as they arrive. Transformers need the full sequence.
- **Constant memory**: RNN state is fixed-size. Transformer KV cache grows with sequence.
- **Low latency**: RNN inference is O(1) per token. Transformer is O(T) per token.

### 5.2 Where Transformers Win

- **Training speed**: Parallel processing is 10-100x faster
- **Long-range dependencies**: Direct attention connects any two positions
- **Scalability**: Can train on massive datasets

---

## 6. Implications for Our Architecture

### 6.1 What We Should Learn from RNNs

1. **Gating is essential**: Forget/input/output gates solve vanishing gradients. Our architecture should incorporate gating.
2. **Additive memory**: The LSTM cell state (addition, not multiplication) is the key to gradient flow. Our dual-memory system should use additive updates.
3. **Sequential processing has value**: For real-time streaming, RNNs are still the right structure. Transformers need full sequences.

### 6.2 What We Should Avoid

1. **Pure sequential bottleneck**: Don't make the entire network sequential. Use parallel attention where possible.
2. **Vanishing gradients**: Don't use pure multiplicative recurrence. Use additive memory paths.
3. **Fixed hidden state**: Don't compress all history into one vector. Use external memory.

### 6.3 Hybrid Approach

Our R-APN architecture should combine:
- **RNN-style gating** for streaming processing (from LSTM)
- **Transformer-style attention** for long-range dependencies
- **External memory** (attractor network) instead of fixed hidden state
- **Additive memory updates** (from LSTM cell state) for gradient flow

---

## 7. Key Takeaways

1. **RNNs process sequences naturally** but are slow due to sequential dependency
2. **LSTM/GRU gating solves vanishing gradients** via additive cell state updates
3. **The sequential bottleneck is the real killer** — not vanishing gradients (LSTM fixed that)
4. **Transformers replaced RNNs** because parallelism > sequential processing at scale
5. **RNNs still win for online/streaming** — constant memory, low latency
6. **Our architecture should use gating + attention + external memory** — best of both worlds

---

## 8. References

1. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
2. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. *arXiv:1406.1078*.
3. Chung, J., et al. (2014). Empirical Evaluation of Gated Recurrent Neural Networks. *arXiv:1412.3555*.
4. Gers, F.A., et al. (1999). Learning to Forget: Continual Prediction with LSTM. *Neural Computation*.
5. Bengio, Y., et al. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE TNN*.
6. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
7. Gu, J., et al. (2016). Gated Orthogonal Recurrent Units. *arXiv:1706.02761*.
8. Zaremba, W., et al. (2014). Recurrent Neural Network Regularization. *arXiv:1409.2329*.
