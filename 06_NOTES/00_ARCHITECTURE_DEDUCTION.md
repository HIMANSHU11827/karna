# Phase-Coupled Predictive Coding (PCP)

## Architecture Deduction from First Principles

### Core Insight
Intelligence requires minimizing prediction error across multiple timescales. The brain achieves this through:
1. **Sparse, event-driven computation** (neurons fire sparingly)
2. **Temporal binding through synchrony** (features bound by firing together)
3. **Hierarchical prediction** (higher layers predict lower ones)
4. **Local learning** (synapses change based on local activity)

### What Makes This Different

| Existing Work | PCP (Ours) |
|---------------|------------|
| Static feedforward units | Phase-coupled oscillators with dynamics |
| Dense representations | Sparse, event-driven activation |
| Backpropagation | Local energy-minimizing plasticity |
| Separate memory system | Memory as attractor states within the network |
| Single timescale | Multi-timescale temporal hierarchy |

### Mathematical Formulation

**State variables per neuron $i$ in layer $l$:**
- $x_i^l$ — activation state
- $\phi_i^l$ — phase (for temporal binding)
- $m_i^l$ — memory trace (synaptic importance)

**Energy function (the objective the network minimizes):**

$$E = \underbrace{\sum_{l,i} (x_i^l - \hat{x}_i^l)^2}_{\text{prediction error}} + \underbrace{\lambda \sum_{l,i} (x_i^l)^2}_{\text{sparsity}} + \underbrace{\mu \sum_{l,i,j} w_{ij}^l (1 - \cos(\phi_i^l - \phi_j^l))}_{\text{phase coherence}}$$

**Network dynamics (gradient descent on energy):**

$$\tau \frac{dx_i^l}{dt} = -(x_i^l - \hat{x}_i^l) - \lambda x_i^l + \sum_j w_{ij}^{l,l-1} e_j^{l-1}$$

$$\frac{d\phi_i^l}{dt} = \omega^l + \mu \sum_j w_{ij}^l \sin(\phi_j^l - \phi_i^l)$$

**Prediction (top-down):**
$$\hat{x}_i^l = f\left(\sum_j u_{ji}^{l+1,l} \cdot x_j^{l+1} \cdot \cos(\phi_j^{l+1} - \phi_i^l)\right)$$

The $\cos(\Delta\phi)$ term means prediction is strongest when phases are aligned — temporal binding modulates prediction.

**Learning rules (all local):**

Bottom-up (perceptual):
$$\Delta w_{ij}^{l,l-1} = \eta \cdot \underbrace{e_i^l}_{\text{post-synaptic error}} \cdot \underbrace{x_j^{l-1}}_{\text{pre-synaptic activity}}$$

Top-down (predictive):
$$\Delta u_{ij}^{l+1,l} = \eta \cdot \underbrace{(x_i^l - \hat{x}_i^l)}_{\text{prediction error}} \cdot \underbrace{x_j^{l+1}}_{\text{top-down signal}}$$

Phase binding:
$$\Delta c_{ij}^l = \mu \cdot \sin(\phi_j^l - \phi_i^l) \cdot x_i^l \cdot x_j^l$$

Continual learning (importance weighting):
$$\Omega_{ij}(t+1) = \Omega_{ij}(t) + \alpha (\Delta w_{ij})^2$$
$$\eta_{ij} = \frac{\eta_0}{1 + \beta \cdot \Omega_{ij}}$$

**Key properties:**
- No backpropagation — all learning is local
- Sparse activation through phase gating
- Memory via importance-weighted consolidation
- Temporal binding via phase synchrony
- Continual learning via synaptic importance

---

*Bot Army — Orchestrator*
