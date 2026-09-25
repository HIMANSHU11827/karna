# Predictive Coding — Deep Research Survey

## 1. Original Rao & Ballard Model (1999)

### Core Hypothesis

The brain processes information by constantly generating predictions and computing prediction errors. Only the **error** (unpredicted information) propagates up the cortical hierarchy. This is efficient: if a prediction is correct, no further processing is needed.

### Mathematical Formulation

For layer $l$:
- **Prediction**: $\hat{a}_i^{(l)} = \sum_j W_{ij}^{(l)} a_j^{(l+1)}$
- **Prediction error**: $\epsilon_i^{(l)} = a_i^{(l)} - \hat{a}_i^{(l)}$
- **State update**: $\frac{da_i^{(l)}}{dt} = -\epsilon_i^{(l)} + \sum_j W_{ji}^{(l+1)} \epsilon_j^{(l+1)}$

Layer $l$ receives:
1. Its own prediction error (from below)
2. Weighted prediction errors from above (top-down)

The system minimizes prediction error at all levels simultaneously.

### Key Properties

1. **Generative model**: Higher layers generate predictions for lower layers
2. **Error-driven**: Only errors propagate up
3. **Attention as precision**: Precision (inverse variance) gates error signals — high-precision errors get more attention
4. **Hierarchical**: Each level predicts the level below

### Biological Implementation

- **Feedforward connections**: Carry prediction errors from lower to higher areas
- **Feedback connections**: Carry predictions from higher to lower areas
- **Prediction neurons**: Generate predictions at each level
- **Error neurons**: Compute difference between actual and predicted

### Limitations

- Tested on static images (V1/V2)
- Required extensive training (backprop-like)
- Did not handle temporal sequences
- No online learning rule specified

---

## 2. Hierarchical Error Signals

### Friston's Free Energy Principle (2005, 2010)

Extends Rao & Ballard to a unified theory of brain function.

**Free energy bound**: The brain minimizes variational free energy (surprise about sensory input):
$$F = \mathbb{E}_{q(s)}[\ln q(s) - \ln p(o,s)]$$

Where:
- $q(s)$ = approximate posterior over hidden states
- $p(o,s)$ = generative model
- $o$ = observations

**Minimizing free energy** = minimizing prediction error + model complexity

### Predictive Coding as Gradient Descent

Each layer's activity performs gradient descent on free energy:
$$\frac{da_i^{(l)}}{dt} = -\frac{\partial F}{\partial a_i^{(l)}} = -\epsilon_i^{(l)} + \sum_j \epsilon_j^{(l+1)} \frac{\partial \hat{a}_j^{(l+1)}}{\partial a_i^{(l)}}$$

This naturally implements **prediction error minimization**.

### Hierarchical Structure

**Primary sensory cortex (V1)**:
- Receives raw sensory input
- Computes error between input and prediction from V2
- Sends error to V2

**Secondary areas (V2, V4, IT)**:
- Generate predictions for lower areas
- Receive errors from below
- Generate predictions for areas below

**Prefrontal cortex**:
- Highest level predictions
- Abstract, long-term predictions
- Receives errors from all lower areas

### Error Propagation Dynamics

1. **Bottom-up**: Raw sensory input → V1 error → V2 error → ... → PFC
2. **Top-down**: PFC predictions → ... → V2 predictions → V1 predictions
3. **Convergence**: Errors minimized at all levels simultaneously
4. **Latency**: ~100-200ms for full convergence (matches ERP timing)

---

## 3. Local Learning Rules

### Classical Hebbian Learning

**Original Hebb (1949)**: "Neurons that fire together wire together"
$$\Delta W_{ij} = \eta \cdot x_i \cdot y_j$$

Where $x_i$ = pre-synaptic activity, $y_j$ = post-synaptic activity.

**Problems**: Unstable, weights grow without bound.

### Oja's Rule (1982)

Normalized Hebbian:
$$\Delta W_{ij} = \eta \cdot y_j \cdot (x_i - y_j W_{ij})$$

Converges to first principal component. Stable.

### BCM Rule (1982)

Bienenstock, Cooper, Munro:
$$\Delta W_{ij} = \eta \cdot y_j \cdot (y_j - \theta) \cdot x_i$$

Where $\theta$ is a sliding threshold based on output history. Adaptive learning rate.

### Predictive Coding Learning Rule

**Weight update** (Rao & Ballard):
$$\Delta W_{ij}^{(l)} = \eta \cdot \epsilon_i^{(l)} \cdot a_j^{(l+1)}$$

This is **local**: all terms available at the synapse.
- $\epsilon_i^{(l)}$ = prediction error at neuron $i$ in layer $l$
- $a_j^{(l+1)}$ = activity of neuron $j$ in layer $l+1$ (prediction source)

**Key insight**: Learning strengthens connections that produce accurate predictions.

### L^p Hebbian (Krotov & Hopfield, 2019)

$$\Delta W_{ij} = \eta \cdot y_j \cdot \text{sign}(x_i - \sum_k W_{ik} y_k) \cdot |x_i - \sum_k W_{ik} y_k|^p$$

Competitive, stable associative memory. For $p=1$, equivalent to Hopfield.

### Online Discriminative Learning (Our Novel Rule)

$$\Delta W_{ij} = \eta \cdot e_{ij}(t) \cdot \delta_j(t) \cdot g(t)$$

Where:
- $e_{ij}(t)$ = eligibility trace (credit assignment through time)
- $\delta_j(t)$ = prediction error (discriminative signal)
- $g(t)$ = neuromodulator (global context: surprise, attention)

**Why this is novel**: Combines three signals that are usually separate.

---

## 4. Biological Plausibility

### Synaptic Mechanisms

**NMDA receptors**:
- Require both pre- and post-synaptic activity
- Implement AND-like gating (Hebbian coincidence detection)
- Act as coincidence detectors for prediction error

**Dopamine**:
- Global neuromodulator signal
- Encodes reward prediction error
- Gates learning (our $g(t)$ term)

**Acetylcholine**:
- Encodes expected uncertainty
- Modulates learning rate
- Attention-like precision weighting

### Cortical Microcircuits

**Layer 2/3 pyramidal neurons**:
- Generate predictions (receive top-down input)
- Compare with actual input
- Output prediction error

**Layer 5 pyramidal neurons**:
- Output predictions to lower areas
- Receive feedback from higher areas

**Layer 6 neurons**:
- Encode precision (gain control)
- Modulate prediction error signals

### Spike-Timing-Dependent Plasticity (STDP)

Biological learning rule:
- Pre fires before post → LTP (strengthen)
- Post fires before pre → LTD (weaken)
- Window: ~20-50ms

**Connection to predictive coding**:
- If pre predicts post: pre fires before post → LTP
- If pre doesn't predict post: random timing → no change or LTD
- Natural implementation of "strengthen connections that predict"

### Dendritic Compartments

Modern biology shows:
- **Apical dendrites**: Receive top-down predictions
- **Basal dendrites**: Receive bottom-up input
- **Soma**: Computes prediction error (difference)
- **Spike output**: Error signal

This is a biological implementation of Rao & Ballard's model.

---

## 5. Computational Properties

### Efficiency

**Sparse computation**: If prediction is correct, error is small → few neurons active.
- Estimated 2-5% of neurons active at any time in cortex
- Massive energy savings vs. dense computation

**Incremental update**: Only update when prediction fails.
- No need for full forward pass
- Natural real-time operation

### Robustness

**Graceful degradation**: Damage to some neurons → predictions still mostly correct.
**Noise tolerance**: Small input noise → small prediction error → correctable.

### Learning Speed

**Single-exposure**: Strong prediction error → large weight change.
**Gradual refinement**: Repeated exposures → refined predictions.

### Capacity

**Hierarchical composition**: Simple features combine into complex representations.
**Reusable components**: Same lower features used in many higher representations.

---

## 6. Comparison with Backpropagation

| Property | Backprop | Predictive Coding |
|----------|----------|-------------------|
| Learning rule | Global gradient | Local prediction error |
| Credit assignment | Backpropagated gradient | Forward-propagated error |
| Biological plausibility | ❌ (weight transport) | ✅ (local) |
| Online learning | ❌ (needs batches) | ✅ (per sample) |
| Temporal credit | ❌ (BPTT required) | ✅ (eligibility traces) |
| Forgetting | Catastrophic | Graceful (decay) |

---

## 7. Key References

1. Rao, R.P.N. & Ballard, D.H. (1999). "Predictive coding in the visual cortex"
2. Friston, K. (2005). "A theory of cortical responses"
3. Krotov, D. & Hopfield, J.J. (2019). "Dense associative memory for pattern recognition"
4. Kleyko, D. et al. (2021). "Integer sparse distributed representations"
5. Maass, W. (2000). "On the computational power of winner-take-all"
6. Oja, E. (1982). "A simplified neuron model as a principal component analyzer"
7. Bienenstock, E.L., Cooper, L.N., & Munro, P.W. (1982). "Theory for the development of neuron selectivity"
8. Spruston, N. (2008). "Pyramidal neurons: dendritic structure and synaptic integration"
9. Shipp, S. (2016). "The importance of being agranular"
