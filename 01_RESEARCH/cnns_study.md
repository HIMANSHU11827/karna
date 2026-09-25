# CNNs Study — Analysis
*By planner*

---

## 1. Convolution Operation

**Definition:** A convolution applies a small filter (kernel) across an input signal, computing a dot product at each position.

$$\text{Output}(i,j) = \sum_{m}\sum_{n} \text{Input}(i+m, j+n) \cdot \text{Kernel}(m, n)$$

**Key properties:**
- **Local connectivity:** Each output unit depends only on a small local region of the input (receptive field)
- **Weight sharing:** The same kernel is applied across all positions → translation invariance
- **Multiple filters:** Different kernels detect different features (edges, textures, patterns)

**Why it works for images:**
- Images have strong spatial locality — nearby pixels are correlated
- Features are translation-invariant — a cat in the corner is still a cat
- Hierarchical composition — edges → textures → parts → objects

---

## 2. Feature Extraction Hierarchy

CNNs build a hierarchy through stacked convolution + pooling layers:

| Level | Features | Receptive Field |
|-------|----------|-----------------|
| Early | Edges, colors, textures | 3-11 pixels |
| Middle | Shapes, patterns, object parts | 30-100 pixels |
| Deep | Object categories, scenes | Full image |

**Hierarchy mechanism:**
1. **Convolution** detects local features
2. **Non-linearity** (ReLU) creates selectivity
3. **Pooling** increases receptive field, reduces spatial extent
4. **Stacking** composes low-level → high-level features

**This is similar to our PEN architecture's hierarchy, but:**
- CNNs use dense activations (not sparse)
- CNNs use backpropagation (not local learning)
- CNNs are feedforward (no temporal dynamics)

---

## 3. Why CNNs Fail at Temporal Tasks

### 3.1 No Temporal Dynamics

CNNs process static spatial patterns. They have no notion of time.

- **No recurrent connections:** Output at time t doesn't depend on time t-1
- **No state:** The network has no memory of what it processed before
- **Fixed receptive field:** Sees one moment, not a sequence

### 3.2 No Online Learning

CNNs require backpropagation through the entire network:
- Needs batches of data
- Needs fixed architecture
- Needs gradient flow through time (BPTT) for sequences
- Cannot learn from a single example

### 3.3 Catastrophic Forgetting

Fine-tuning a CNN on new data destroys old knowledge:
- All weights updated simultaneously
- No mechanism to protect important weights
- No complementary learning systems

### 3.4 No Working Memory

CNNs have no mechanism to:
- Hold information over time
- Selectively update information
- Manipulate information (reasoning)

### 3.5 Fixed Architecture

CNNs have fixed layer count and filter sizes:
- Cannot grow new neurons for new concepts
- Cannot rewire connections based on experience
- Architecture is frozen after training

---

## 4. Key Limitations

| Limitation | Impact | Our Solution |
|------------|--------|--------------|
| Dense activations | Interference, poor capacity | k-WTA sparse activation |
| Backpropagation | Batch-only, not local | Eligibility traces + global signal |
| Feedforward only | No temporal dynamics | Recurrent attractor dynamics |
| Fixed architecture | Cannot adapt online | Dynamic growth via episodic memory |
| No working memory | Cannot reason | Attractor states as persistent activity |
| Catastrophic forgetting | Cannot learn continually | Complementary learning systems |
| No world model | Cannot predict or plan | Procedural memory as transition model |
| Single modality | No multimodal fusion | Unified SDR representation |

---

## 5. What We Can Reuse from CNNs

| Component | How We Adapt |
|-----------|--------------|
| Local receptive fields | Random projection with local patches |
| Hierarchical composition | Our 4-level hierarchy (already sparse) |
| Weight sharing | Shared random projections across positions |
| Pooling | k-WTA as learned pooling |

---

*End of CNNs Study — planner*
