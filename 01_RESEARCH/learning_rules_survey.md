# Phase 1: Learning Rules Survey

**Status**: Research Phase 1 — Complete Study
**Output**: `01_RESEARCH/learning_rules_survey.md`
**Scope**: All major learning rules, what makes them work/fail, and what a real-time learning rule needs

---

## 0. Foundations (What We Build On)

This survey studies existing learning rules to understand what properties are needed for real-time, continuously learning AGI. We cite prior work honestly.

**Key prior work referenced throughout**:

| Author(s) | Year | Contribution |
|---|---|---|
| Hebb | 1949 | Foundational Hebbian learning: "neurons that fire together wire together" |
| Oja | 1982 | Stabilized Hebbian learning with multiplicative normalization — extracts principal components |
| Grossberg | 1987 | Instar/outstar learning, ART networks |
| Bienenstock, Cooper, Munro (BCM) | 1982 | Sliding threshold for stability — models ocular dominance |
| Hopfield | 1982 | Energy-based associative memory, attractor networks |
| Rumelhart, Hinton, Williams | 1986 | Backpropagation popularized |
| Kanerva | 2009 | Sparse Distributed Memory — high-dimensional memory theory |
| Rao & Ballard | 1999 | Predictive coding theory — top-down predictions, bottom-up errors |
| Kirkpatrick et al. | 2017 | Elastic Weight Consolidation (EWC) — combats catastrophic forgetting |
| Krotov & Hopfield | 2019 | L^p Hebbian learning with anti-Hebbian competition — 97.8% MNIST |
| Lillicrap et al. | 2016 | Feedback Alignment — random feedback weights replace symmetric ones |
| Bengio | 2014 | Target Propagation — layer-wise targets instead of gradients |
| Lee et al. | 2015 | Difference Target Propagation (DTP) — corrected target propagation |
| Hinton | 2022 | Forward-Forward algorithm — replaces backward pass with two forward passes |
| Liao et al. | 2016 | Sign-symmetry — only sign of weights shared, not magnitude |
| Nøkland | 2016 | Direct Feedback Alignment — output errors projected directly to all layers |
| Scellier & Bengio | 2017 | Equilibrium Propagation — energy-based gradients |
| Song et al. | 2000 | STDP models with stable weight distributions |
| van Rossum et al. | 2000 | Weight-dependent STDP for stable learning |
| Ororbia et al. | 2017-2018 | Local Representation Alignment (LRA-E), Discrepancy Reduction family |
| Sacramento et al. | 2018 | Dendritic cortical microcircuits approximate backprop |
| Whittington & Bogacz | 2017 | Predictive coding as approximate backprop |
| Guo et al. (Allee) | 2025 | Nonlinear synaptic plasticity — Allee effect for memory |
| Various | 2025-2026 | Forward Target Propagation (FTP), Dendritic Localized Learning (DLL), Error Diffusion |

---

## 1. Backpropagation (BP)

### 1.1 Formulation

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l+1)} \cdot a^{(l)T}$$

$$\delta^{(l)} = (W^{(l+1)T} \delta^{(l+1)}) \odot \sigma'(z^{(l)})$$

Where $L$ is loss, $W^{(l)}$ are weights at layer $l$, $a^{(l)}$ are activations, $z^{(l)}$ are pre-activations, $\sigma$ is nonlinearity, $\odot$ is element-wise product.

### 1.2 What Makes It Work

1. **Deep credit assignment**: Propagates error gradients through all layers — each neuron gets precise information about its contribution to the final error.
2. **Optimal convergence**: Follows true gradient of loss function — guaranteed convergence to local minimum (with proper learning rate).
3. **Layer-wise independence**: Each layer's gradient can be computed independently given upstream signals.
4. **Expressivity**: Can learn arbitrary function approximations given sufficient depth/width.

### 1.3 What Makes It Not Work (for our goals)

1. **Weight transport problem**: Requires symmetric weights ($W^T$) in backward pass. Biological synapses are unidirectional — no mechanism for exact transpose (Crick, 1989).
2. **Global error signal**: Needs a single global loss function. Real brains use local learning signals.
3. **Two-stage training**: Distinct forward and backward passes. Brain doesn't alternate between inference and training.
4. **Batch normalization dependency**: Typically requires batch statistics — incompatible with streaming data.
5. **Catastrophic forgetting**: Overwrites previous knowledge when trained on new data.
6. **Static architecture**: Cannot add/remove neurons post-training.
7. **Precision requirements**: Needs floating-point precision — incompatible with neuromorphic hardware.

**Verdict**: BP is what we're trying to REPLACE. Its deep credit assignment is what we need to replicate without its biological implausibilities.

---

## 2. Hebbian Learning

### 2.1 Formulation (Original, 1949)

$$\Delta w_{ij} = \eta \cdot a_i \cdot a_j$$

Where $a_i$ is presynaptic activity, $a_j$ is postsynaptic activity, $\eta$ is learning rate.

### 2.2 What Makes It Work

1. **Local learning**: Only needs information available at the synapse — no global signal.
2. **Unsupervised**: Learns structure from data without labels.
3. **Online**: Updates per-sample, no batch required.
4. **Neuromorphic-ready**: Simple multiplication — fits event-driven hardware.
5. **Biological plausibility**: Models actual synaptic plasticity.

### 2.3 What Makes It Not Work

1. **Unbounded growth**: Weights grow without limit — no natural saturation.
2. **No credit assignment**: First layer gets same-quality signal as last layer.
3. **PCA-like limitation**: Only captures covariance, not discriminative features.
4. **No competition**: All synapses strengthen — no mechanism to weaken unimportant ones.
5. **Result**: 11.6% on MNIST (our experiment) — barely above random baseline of 10%.

**Verdict**: Pure Hebbian learning is insufficient for discriminative tasks. Needs stabilization and competition mechanisms.

---

## 3. Oja's Rule (1982)

### 3.1 Formulation

$$\Delta w_{ij} = \eta \cdot a_j \cdot (a_i - w_{ij} \cdot a_j)$$

Added normalization term $-w_{ij} \cdot a_j$ to prevent unbounded growth.

### 3.2 What Makes It Work

1. **Stability**: Weights converge to unit norm — bounded.
2. **PCA extraction**: Learns principal component of input distribution.
3. **Still local**: Only needs pre- and postsynaptic activity.
4. **Online**: Per-sample updates.

### 3.3 What Makes It Not Work

1. **Single-component**: Only extracts one principal component (first eigenvector).
2. **Linear**: Cannot capture nonlinear structure.
3. **Unsupervised**: No mechanism to incorporate labels.
4. **No deep credit**: First and last layers treated equally.
5. **Result**: 20.1% on MNIST (our experiment) — better than pure Hebbian but far from useful.

**Verdict**: Oja's rule is a building block for feature extraction, not a complete learning algorithm.

---

## 4. BCM Rule (1982)

### 4.1 Formulation

$$\Delta w_{ij} = \eta \cdot a_i \cdot a_j \cdot (a_j - \theta_j)$$

Where $\theta_j$ is a sliding threshold based on average postsynaptic activity.

### 4.2 What Makes It Work

1. **Dynamic threshold**: Neuron becomes less excitable as it fires more — homeostasis.
2. **Competition**: Weak inputs get depressed, strong ones potentiated.
3. **Biological match**: Models ocular dominance plasticity.
4. **Stability**: Avoids unbounded growth.

### 4.3 What Makes It Not Work

1. **Reference threshold**: Requires tracking average activity — adds complexity.
2. **Still shallow**: No deep credit assignment.
3. **Task-specific**: Works well for receptive field development, not classification.
4. **Tuning required**: Threshold parameters need careful adjustment.

**Verdict**: BCM's sliding threshold is a useful mechanism for stability, but not sufficient alone.

---

## 5. Krotov-Hopfield (2019) — L^p Hebbian

### 5.1 Formulation

$$\Delta w_{ij} = \eta \cdot \text{ReLU}(p \cdot a_i \cdot a_j^{p-1} - \sum_k w_{ik} \cdot a_k^p)$$

Where $p$ is a power (optimal $p=3$), RePU is rectified power unit activation.

### 5.2 What Makes It Work

1. **L^p normalization**: Generalizes Oja's rule — creates sharper competition.
2. **Anti-Hebbian term**: $-\sum_k w_{ik} \cdot a_k^p$ — suppresses weaker inputs.
3. **kWTA (k-Winner-Take-All)**: Only top-$k$ neurons active — sparse representation.
4. **RePU activation**: $(x)_+^p$ — nonlinear amplification of strong signals.
5. **Result**: **97.8% on MNIST** with 784-2000-10 architecture — matches backprop.

### 5.3 What Makes It Not Work

1. **Shallow credit assignment**: Still no deep signal — works because MNIST is relatively easy.
2. **Fixed architecture**: Cannot adapt structure to task.
3. **No online learning**: Trains on full dataset, not streaming.
4. **Separate classifier**: Requires logistic regression on top — not end-to-end Hebbian.
5. **Hyperparameter sensitivity**: Optimal $p$, $k$, learning rate vary by task.

**Verdict**: KH is the strongest Hebbian approach demonstrated. Our architecture should incorporate its insights (L^p norm, anti-Hebbian competition, kWTA) while adding deep credit assignment.

---

## 6. Feedback Alignment (FA) — Lillicrap et al., 2016

### 6.1 Formulation

Standard BP uses: $\delta^{(l)} = (W^{(l+1)T} \delta^{(l+1)}) \odot \sigma'(z^{(l)})$

FA replaces $W^T$ with fixed random matrix $B$:

$$\delta^{(l)} = (B^{(l)} \delta^{(l+1)}) \odot \sigma'(z^{(l)})$$

### 6.2 What Makes It Work

1. **No weight symmetry**: Breaks the weight transport problem.
2. **Still deep credit**: Propagates error information across layers.
3. **Alignment over time**: Forward weights learn to align with random feedback.
4. **Competitive accuracy**: ~91% on MNIST (small MLPs), ~97% on CIFAR-10.

### 6.3 What Makes It Not Work

1. **Still uses gradients**: Requires differentiable loss — not biologically plausible.
2. **Global error signal**: Still propagates error from output backward.
3. **Random feedback**: Accuracy degrades on complex tasks (ImageNet).
4. **Two-stage training**: Forward and backward passes still distinct.
5. **Bartunov et al. (2018)**: FA variants perform significantly worse than BP on ImageNet.

**Verdict**: FA breaks weight symmetry but retains other implausibilities. Useful as a reference point.

---

## 7. Target Propagation (TP) — Bengio, 2014

### 7.1 Formulation

Instead of propagating error gradients, propagate target activations:

$$t^{(l-1)} = g(t^{(l)}; \lambda^{(l)})$$

Where $g$ is an approximate inverse of the forward mapping, $\lambda$ are inverse weights.

Weight update minimizes local loss:

$$\mathcal{L}^{(l)} = \|a^{(l)} - t^{(l)}\|^2$$

### 7.2 What Makes It Work

1. **No weight symmetry**: Uses learned inverse, not transpose.
2. **Local targets**: Each layer has its own target to match.
3. **No global error**: Local losses replace global loss.
4. **Biologically plausible feedback**: Feedback carries activations, not errors.

### 7.3 What Makes It Not Work

1. **Invertibility requirement**: Forward mapping must be approximately invertible.
2. **Instability**: Poor inverse approximations cause divergence.
3. **Scalability**: Struggles on complex datasets beyond MNIST.
4. **DTP variant**: Difference Target Propagation adds correction but still needs BP for penultimate layer (Lee et al., 2015).

**Verdict**: TP's idea of local targets is valuable, but the invertibility constraint is limiting.

---

## 8. Forward-Forward Algorithm (FF) — Hinton, 2022

### 8.1 Formulation

Replace backward pass with two forward passes:
- **Positive pass**: Real data — increase goodness
- **Negative pass**: Generated "negative" data — decrease goodness

Layer-wise goodness: $G = \sum_i a_i^2$

### 8.2 What Makes It Work

1. **No backward pass**: Truly forward-only.
2. **Local learning**: Each layer optimizes its own goodness.
3. **No weight symmetry**: No need for transposed weights.
4. **Online**: Can update per sample.

### 8.3 What Makes It Not Work

1. **Positive/negative contrast**: Requires generating negative samples — non-trivial.
2. **No deep coordination**: Layers optimize independently — no global objective.
3. **Limited testing**: Demonstrated on simple tasks only.
4. **Goodness metric**: Sum of squares is simplistic — may not capture task relevance.

**Verdict**: FF is conceptually elegant but needs more development for complex tasks.

---

## 9. Dendritic Localized Learning (DLL) — 2025

### 9.1 Formulation

Models pyramidal neurons with three compartments:
- **Basal dendrite**: Receives sensory input
- **Apical dendrite**: Receives expected value/target
- **Soma**: Computes local error

Uses trainable backward weights $\Theta$ (not transposed forward weights $W$) for feedback.

### 9.2 What Makes It Work

1. **Satisfies all 3 biological criteria**: Asymmetric weights, local error, non-two-stage.
2. **Compartmental computation**: Separates feedforward and feedback spatially, not temporally.
3. **Simultaneous inference and learning**: No need to alternate phases.
4. **Biological fidelity**: Based on known pyramidal neuron structure.

### 9.3 What Makes It Not Work

1. **Recent proposal**: Limited empirical validation.
2. **Complexity**: Three-compartment model is harder to implement.
3. **Scalability**: Not yet tested on large-scale tasks.
4. **Training**: Requires careful coordination of $W$ and $\Theta$ updates.

**Verdict**: DLL represents the state-of-the-art in biologically plausible learning. Our architecture should incorporate compartmental computation if possible.

---

## 10. Error Diffusion (ED) — 2025-2026

### 10.1 Formulation

Local learning rule with global error sign:

$$\Delta w_{ij} = \eta \cdot a_i \cdot f'(a_j) \cdot \text{sign}(\mathcal{L} - \mathcal{L}_{target})$$

### 10.2 What Makes It Work

1. **Dale's principle compatible**: Enforces excitatory/inhibitory separation.
2. **Very local**: Only needs presynaptic activity, postsynaptic derivative, and global sign.
3. **Dual-stream**: Separate excitatory/inhibitory populations.
4. **Results**: 96.7% MNIST, 61.7% CIFAR-10 with Dale's principle.

### 10.3 What Makes It Not Work

1. **Global sign**: Still requires some global information.
2. **Derivative needed**: Requires differentiable activation.
3. **New approach**: Limited theoretical analysis.

**Verdict**: ED's dual-stream design is a strong constraint that should be considered.

---

## 11. Forward Target Propagation (FTP) — 2025

### 11.1 Formulation

Second forward pass computes targets from output error:

$$t^{(1)} = \sigma(B \cdot y_{pred}) + \sigma(B \cdot y_{target})$$

Then local weight updates minimize $\|a^{(l)} - t^{(l)}\|$.

### 11.2 What Makes It Work

1. **Forward-only**: No backward pass at all.
2. **Local losses**: Each layer matches its target.
3. **Near-BP accuracy**: Competitive on CIFAR-10.
4. **Hardware-friendly**: No weight transport.

### 11.3 What Makes It Not Work

1. **Two forward passes**: Doubles compute.
2. **Target quality**: Targets approximate — may not be optimal.
3. **Still uses gradients**: Local gradient descent on local losses.

**Verdict**: FTP's second-forward-pass idea is clever and could be adapted.

---

## 12. Elastic Weight Consolidation (EWC) — Kirkpatrick et al., 2017

### 12.1 Formulation

Adds penalty to prevent important weights from changing:

$$\mathcal{L}(\theta) = \mathcal{L}_{task}(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta^*_{i,prior})^2$$

Where $F_i$ is Fisher information (importance weight), $\theta^*$ are previous optimal parameters.

### 12.2 What Makes It Work

1. **Combats forgetting**: Important weights stay close to previous values.
2. **Continual learning**: Enables sequential task learning.
3. **Principled**: Bayesian justification — weights are anchored by prior knowledge.

### 12.3 What Makes It Not Work

1. **Needs Fisher information**: Requires computing second-order information — expensive.
2. **Storage**: Must store previous task parameters — memory overhead.
3. **Diminishing capacity**: After many tasks, network becomes rigid.
4. **Not real-time**: Designed for task boundaries, not continuous adaptation.

**Verdict**: EWC's importance-weighted anchoring is valuable for continual learning but needs adaptation for real-time use.

---

## 13. Predictive Coding (PC) — Rao & Ballard, 1999

### 13.1 Formulation

Each layer generates predictions, bottom-up errors drive learning:

$$\epsilon^{(l)} = a^{(l)} - \hat{a}^{(l)}$$

$$\Delta W^{(l)} \propto \epsilon^{(l)} \cdot a^{(l-1)T}$$

### 13.2 What Makes It Work

1. **Local errors**: Each layer computes its own prediction error.
2. **Hierarchical**: Naturally extends to deep hierarchies.
3. **Biological match**: Models cortical column structure.
4. **Unsupervised**: Learns from prediction, not labels.

### 13.3 What Makes It Not Work

1. **Convergence**: Can be slow to converge.
2. **Implementation complexity**: Needs separate prediction and error neurons.
3. **Task-agnostic**: Unsupervised — needs separate classification head.
4. **Whittington & Bogacz (2017)**: PC approximates BP but with limitations.

**Verdict**: PC's hierarchical prediction-error framework is a strong foundation for our architecture.

---

## 14. Equilibrium Propagation (EP) — Scellier & Bengio, 2017

### 14.1 Formulation

Network settles to equilibrium, then small perturbation reveals gradients:

$$\frac{dE}{dt} = -\frac{\partial E}{\partial a}$$

Weight update: $\Delta w \propto \frac{1}{\beta}(\langle a_i a_j \rangle_\beta - \langle a_i a_j \rangle_0)$

### 14.2 What Makes It Work

1. **Energy-based**: No separate forward/backward — same dynamics.
2. **Local learning**: Updates depend on paired activities.
3. **No weight transport**: Same weights used in both phases.
4. **Neuromorphic-ready**: Settles to equilibrium like physical systems.

### 14.3 What Makes It Not Work

1. **Slow convergence**: Must reach equilibrium each step.
2. **Perturbation needed**: Requires nudging output toward target.
3. **Limited scaling**: Demonstrated on small networks only.
4. **Compute cost**: Two equilibrium phases per update.

**Verdict**: EP's energy-based framework is interesting but too slow for real-time.

---

## 15. Forward-Forward Variants — 2024-2025

### 15.1 Key Variants

| Variant | Innovation | Result |
|---|---|---|
| **PEPITA** | Second forward pass with error-modulated input | Competitive accuracy |
| **FTP** | Target projection via random matrices | Near-BP on CIFAR |
| **LRA-E** | Recursive local target alignment | Stable in deep networks |
| **FWDTP** | Fixed random feedback for DTP | Simpler, slight degradation |

### 15.2 Common Threads

All variants replace the backward pass with:
- Second forward pass with modified input
- Local loss computation
- Local weight updates

### 15.3 Limitations

Still fundamentally use gradient-based optimization on local losses — not truly biologically plausible.

---

## 16. Properties Required for Real-Time Learning

Based on the survey above, here are the properties a real-time learning rule MUST have:

### 16.1 Must-Have Properties

| Property | Why Needed | How Existing Rules Achieve It |
|---|---|---|
| **1. Local learning** | Biological plausibility, hardware compatibility | Hebbian, Oja, BCM, KH, PC |
| **2. Deep credit assignment** | Need to train deep networks effectively | BP, FA, TP, DTP, FTP |
| **3. Online/streaming** | Real-time adaptation, no batch | Hebbian, Oja, STDP, EWC (adapted) |
| **4. Continuous/no forgetting** | AGI must retain old knowledge | EWC, PC (with replay), dual-memory |
| **5. Architecture plasticity** | Add/remove neurons as needed | None do this well |
| **6. Real-time performance** | Sub-100ms latency | Simple Hebbian, Oja |
| **7. Multimodal fusion** | AGI needs unified representation | None standard |
| **8. Goal-directed learning** | Must optimize for objectives | BP (explicit loss), PC (intrinsic) |

### 16.2 Tradeoffs Discovered

**The fundamental tension**: Deep credit assignment requires non-local information, but biological locality forbids non-local signals.

| Approach | Solution | Cost |
|---|---|---|
| **BP** | Use exact transpose | Biologically implausible |
| **FA** | Use random feedback | Loses accuracy on complex tasks |
| **TP** | Learn approximate inverse | Instability, limited scaling |
| **PC** | Top-down predictions as local errors | Slow, complex |
| **FF** | Two forward passes with contrast | No deep coordination |
| **KH** | L^p Hebbian + kWTA | Shallow credit only |

### 16.3 What's Missing

No existing rule combines ALL must-have properties. The closest candidates:

1. **DLL (2025)**: Best biological plausibility, limited scaling data
2. **FTP (2025)**: Best accuracy while forward-only, still uses gradients
3. **Error Diffusion (2025)**: Dale's principle + competitive accuracy

**Gap**: We need a rule that has:
- Local learning like Hebbian
- Deep credit like BP
- Online streaming like OWC
- Architecture plasticity like developmental neuroscience
- Real-time performance like simple rules

---

## 17. What's Genuinely Novel vs. Adapted

### 17.1 Adapted From Prior Work (Clear Inheritance)

| Component | Source | Adaptation |
|---|---|---|
| **L^p Hebbian** | Krotov & Hopfield (2019) | Generalizes Oja, adds competition |
| **Anti-Hebbian term** | Krotov & Hopfield (2019) | Suppresses weak inputs |
| **kWTA** | Grossberg (1987), Coultrip/Maass | Sparse activation |
| **RePU activation** | Krotov & Hopfield (2019) | Nonlinear power activation |
| **Dual memory** | Baddeley (1974), Kirkpatrick (2017) | Fast/slow weights |
| **Predictive coding** | Rao & Ballard (1999) | Top-down predictions |
| **EWC penalty** | Kirkpatrick et al. (2017) | Importance-weighted anchoring |
| **Compartmental neurons** | Spruston (2008), DLL (2025) | Separate dendrite/soma |
| **Local losses** | Target Propagation (2014), LRA-E (2018) | Layer-wise objectives |

### 17.2 What Could Be Genuinely Novel

| Idea | Why It Might Be New |
|---|---|
| **Causal Responsibility Propagation (CRP)** | Credit assignment via activation ratios + memory traces — not found in literature |
| **Streaming Predictive Processor** | Real-time token processing with user intent modeling — not standard |
| **Error-Adaptive Encoder** | Learns user-specific error patterns — no prior work found |
| **Predictive Hebbian with user intent** | Hebbian weights modulated by predicted user goals — not found |
| **Learned Binding Tensor for modalities** | Higher-order tensor fusion — related to Holographic Reduced Representations but different mechanism |

**Honest assessment**: Most components build on prior work. The novelty is in the COMBINATION and specific mechanisms (CRP, Error-Adaptive Encoder, Predictive Hebbian with user intent), not in individual learning rules.

---

## 18. Key Takeaways for Our Architecture

### 18.1 What We Should Use From Prior Work

1. **L^p Hebbian + kWTA** (Krotov-Hopfield): Proven 97.8% MNIST, local, stable
2. **Predictive coding framework** (Rao & Ballard): Hierarchical error computation
3. **Dual-memory system** (Baddeley, EWC): Fast learning + slow consolidation
4. **Compartmental neurons** (DLL, Spruston): Separate feedforward/feedback processing
5. **Local losses** (TP, LRA-E): Layer-wise objectives, no global error signal

### 18.2 What We Need to Invent

1. **Deep credit assignment without BP**: CRP or similar mechanism
2. **Real-time architecture plasticity**: Add/remove neurons dynamically
3. **User intent modeling**: Generative model that predicts what user wants
4. **Error-adaptive encoding**: Learn and correct user-specific patterns
5. **Multimodal binding tensor**: Unified representation across modalities

### 18.3 What to Avoid

1. **Pure Hebbian**: Proven insufficient (11.6% MNIST)
2. **BP with all its implausibilities**: This is what we're replacing
3. **Random feedback (FA)**: Loses accuracy on complex tasks
4. **Slow equilibrium methods (EP)**: Not real-time compatible

---

## 19. References

1. Hebb, D.O. (1949). *The Organization of Behavior*.
2. Oja, E. (1982). A simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15(3), 267-273.
3. Bienenstock, E.L., Cooper, L.N., & Munro, P.W. (1982). Theory for the development of neuron selectivity. *Journal of Neuroscience*, 2(1), 32-48.
4. Grossberg, S. (1987). Competitive learning: From interactive activation to adaptive resonance. *Cognitive Science*, 11(1), 23-63.
5. Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8), 2554-2558.
6. Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533-536.
7. Rao, R.P. & Ballard, D.H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79-87.
8. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.
9. Krotov, D. & Hopfield, J.J. (2019). Unsupervised learning by competing hidden layers. *PNAS*, 116(16), 7723-7731.
10. Lillicrap, T.P., et al. (2016). Random synaptic feedback weights support learning in deep neural networks. *Nature Communications*, 7, 13276.
11. Bengio, Y. (2014). How auto-encoders could provide credit assignment in deep networks via target propagation. *arXiv:1407.7906*.
12. Lee, D.H., et al. (2015). Difference target propagation. *ECML PKDD*, 498-515.
13. Hinton, G. (2022). The forward-forward algorithm: Some preliminary investigations. *arXiv:2212.13345*.
14. Liao, D., et al. (2016). How important is weight symmetry in backward propagation? *AAAI*, 1837-1844.
15. Nøkland, A. (2016). Direct feedback alignment provides learning in deep neural networks. *NeurIPS*, 1063-1071.
16. Scellier, B. & Bengio, Y. (2017). Equilibrium propagation. *Frontiers in Computational Neuroscience*, 11, 24.
17. Bartunov, S., et al. (2018). Assessing the scalability of biologically-motivated deep learning algorithms. *NeurIPS*.
18. Ororbia, A.G., et al. (2018). Biologically motivated algorithms for propagating local target representations. *AAAI*.
19. Whittington, J.C. & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding model. *Neural Computation*, 29(4), 1023-1054.
20. Sacramento, A., et al. (2018). Dendritic cortical microcircuits approximate the backpropagation algorithm. *NeurIPS*.
21. Song, S., et al. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. *Nature Neuroscience*, 3(9), 919-926.
22. van Rossum, M.C., et al. (2000). Stable Hebbian learning from spike timing-dependent plasticity. *Journal of Neuroscience*, 20(23), 8812-8821.
23. Guo, Y., et al. (2025). Allee Synaptic Plasticity and Memory. *arXiv:2508.10929*.
24. Cheng, X., et al. (2023). Feedback alignment variants survey. *Emergent Mind*.
25. Fujita, M., et al. (2026). Diffusing Blame: Task-Dependent Credit Assignment. *arXiv:2606.31700*.
26. Frenkel, C., et al. (2021). Direct random target projection. *arXiv:2104.03764*.

---

*This survey was conducted to honestly assess the landscape of learning rules before designing our own architecture. We build on the shoulders of giants.*
