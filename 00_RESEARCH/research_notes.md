# AGI Research Lab — Research Notes
## Compiled by: researcher
## Date: 2026-06-28

---

## 1. LLM ARCHITECTURES: THE POST-TRANSFORMER LANDSCAPE

### 1.1 The Four Camps (2026 Map)

The field has fragmented into four distinct architectural camps, each with different compute-performance tradeoffs:

| Camp | Examples | Core Idea | Complexity |
|------|----------|-----------|------------|
| **Transformer mainline** | GPT-4, Claude 4.7, Gemini 2.5, Llama 4 | Self-attention, maximum expressivity | O(N²) time+memory |
| **State Space / Linear RNN** | Mamba, Mamba-2, RWKV, RetNet, Griffin, xLSTM | Linear in sequence length, cheap inference | O(N) training, O(1) inference |
| **Hybrid** | Jamba, Griffin, Zamba, RecurrentGemma | Mix SSM and attention for both strengths | O(N·k) selective |
| **Sparse / MoE** | Mixtral 8x7B, DeepSeek-V3 671B, Million Experts | Huge parameters, small active subnetworks | Variable |

**Key insight:** The pure Transformer is no longer the default for all use cases. Different parts of a system now pick different architectures. Research from Stanford/MIT (early 2026) shows <15% of attention heads exhibit "full quadratic" behavior that justifies the cost; the remaining 85% can be approximated or replaced without quality degradation.

### 1.2 Key Architectural Innovations

**Mamba / State Space Models (SSMs):**
- Selective state-space layers make state updates input-dependent, overcoming SSM weakness on content-based reasoning
- Mamba-3 (ICLR 2026): complex-valued recurrence + MIMO processing → half the state size of Mamba-2 at comparable perplexity; 1.8 point accuracy gain at 1.5B params
- Falcon Mamba-7B: first 7B attention-free model to beat same-size Transformers

**RWKV (Receptance Weighted Key Value):**
- Combines attention-free formulation with learned bilinear projection
- Constant inference memory regardless of context length
- Native streaming: tokens generated as data arrives without retroactive attention recomputation
- RWKV-7 ("Goose"): generalized delta rule, vector-valued gating, in-context learning rates, relaxed value replacement rule; constant memory and inference time per token

**RetNet (Microsoft):**
- Multi-scale "retention" mechanism replaces multi-head attention
- Three modes: parallel training (like transformers), O(1) recurrent inference, chunked processing for long sequences
- Mathematically equivalent to full attention under certain parameterizations

**Hybrids (the emerging sweet spot):**
- Jamba: merges transformer, Mamba, MoE into striped hybrid → 2-7x longer context windows, 3x higher throughput, fewer total parameters (52B, 12B active)
- Falcon-H1: parallel hybrid combining Transformer attention with SSMs
- Pattern: 4-8 SSM layers + 1 attention layer as "corrector"; attention appears once every 512 tokens

**Mixture of Experts (MoE):**
- DeepSeek-V3 671B demonstrated MoE as the dominant scaling strategy
- 3-5x inference efficiency gains per active parameter
- Key challenge: expert specialization, load balancing, routing stability
- Modern approaches encourage expert overlap via regularization for graceful degradation under distribution shift

### 1.3 Beyond Attention: Radical Alternatives

**Baby Dragon Hatchling (BDH) Architecture (Pathway, late 2025):**
- Graph of locally interacting neuron particles governed by excitatory/inhibitory dynamics
- Scale-free network topology (like brain neocortex)
- Integrate-and-fire mechanism: sparse activation (~5% of network firing at any moment)
- Eliminates KV cache entirely — working memory encoded directly into synaptic weights via Hebbian learning
- Continuous adaptation during inference (theoretically unbounded context)
- BDH-GPU: tensor-friendly translation using mean-field communication to map graph dynamics to linear algebra kernels
- Adaptive Synaptic Consolidation via Elastic Weight Consolidation + Fisher information for lifelong learning without catastrophic forgetting

**Neuro-Symbolic AI:**
- Combines neural perception with symbolic planning
- Dramatic results in robotics: training time 36+ hours → 34 minutes; task success 34% → 95%; energy consumption claimed 100x reduction
- Handles structured reasoning tasks that pure neural approaches struggle with

---

## 2. AI AGENT SYSTEMS

### 2.1 Framework Landscape (2026)

Seven major frameworks dominate:

| Framework | Strengths | Best For |
|-----------|-----------|----------|
| **LangChain + LangGraph** | 700+ integrations, stateful multi-agent orchestration, LangSmith observability | Rapid prototyping to production across model providers |
| **CrewAI** | Role-based agent design, intuitive abstractions, self-contained | Fast multi-agent prototypes |
| **Microsoft Agent Framework** | Unified successor to AutoGen + Semantic Kernel, enterprise features | Azure-integrated enterprise deployments |
| **Google ADK** | Batteries-included, built-in debugging UI, Cloud Run/GKE/Vertex AI deployment | Google Cloud-native agent development |
| **OpenAI Agents SDK** | Minimal API surface, built-in tracing, MCP support, 100+ models via LiteLLM | Tightly scoped assistants on OpenAI stack |
| **LlamaIndex Workflows** | Retrieval-heavy reasoning, document parsing, indexing, synthesis | Enterprise RAG and knowledge applications |
| **Mastra (TypeScript)** | Agents + workflows + memory + RAG + evals, Vercel AI SDK base | Full-stack TypeScript applications |

**Emerging patterns:**
- Two-framework stacks are common (LangGraph for orchestration + LlamaIndex for retrieval)
- Framework-agnostic evaluation layer (Braintrust) connects traces from multiple frameworks
- MCP (Model Context Protocol) is the emerging standard for tool integration
- CopilotKit introduces AG-UI protocol for frontend-agent communication (16 event types, transport-agnostic)

### 2.2 Agent Reasoning Strategies

- **ReACT (Think → Act):** Implicit reasoning; LLM steers strategy via system prompt
- **ReWOO (Plan → Execute → Reflect):** Explicit reasoning; strategy implemented in code
- **LATS (Learning to Act):** Tree search over reasoning paths
- **Graph-of-Thought:** Non-linear reasoning graphs
- **SGR (Schema-Guided Reasoning):** Structured reasoning with flexible tool selection

### 2.3 Multi-Agent Patterns
- Sequential handoffs (linear delegation)
- Hierarchical (supervisor dispatches sub-agents)
- Group chat (peer collaboration)
- Magentic-One (dynamic team formation)

---

## 3. MEMORY SYSTEMS FOR AI AGENTS

### 3.1 The Three Evolutionary Stages

Research on LLM agent memory mechanisms shows three distinct stages driven by three catalysts:

1. **Storage stage** — Buffer raw experience (episodic, short-term)
2. **Abstraction stage** — Compress, generalize, and organize experience (semantic, hierarchical)
3. **Experience stage** — Continuous adaptation and self-evolution (meta-learning, lifelong)

Core drivers: long-range consistency, dynamic environment interaction, continual learning.

### 3.2 Memory Taxonomy

| Memory Type | Representation | Mechanism | Key Challenge |
|-------------|---------------|-----------|---------------|
| **Token-level (context window)** | In-context | Attention | Fixed capacity, no persistence |
| **Parametric** | Model weights | Fine-tuning, in-context learning | Catastrophic forgetting, rigid |
| **Latent** | Hidden states | SSM/RNN recurrence | Compression loss |
| **Retrieval-based (RAG)** | Vector DB, graph DB | Semantic search | Retrieval quality, coverage |
| **External (episodic)** | Structured storage | Read/write APIs | Organization, pruning, abstraction |

### 3.3 Key Memory Systems

**ALMA (Automated meta-Learning of Memory designs for Agentic systems):**
- Meta-learns memory designs expressed as executable code
- Searches open-endedly over database schemas, retrieval mechanisms, update rules
- Outperforms human-crafted memory designs across ALFWorld, TextWorld, Baba Is AI, MiniHack
- Open-sourced at https://github.com/zksha/alma

**MemVerse (Multimodal Memory for Lifelong Learning Agents):**
- Model-agnostic, plug-and-play memory framework
- Unifies fast parametric recall with slow, hierarchical retrieval-based memory
- Mirrors complementary roles of intuition and deliberation
- Decouples memory from model parameters for scalable lifelong learning

**MemoryOS:**
- Operating system-inspired memory management for AI agents
- Four modules: Storage, Updating, Retrieval, Generation
- Dynamic memory updates and semantic retrieval for long-term conversational coherence

**Mem0:**
- Production-ready scalable long-term memory for AI agents
- Graph-based with semantic recall, AES-256 encryption

**A-Mem (Agentic Memory):**
- Dynamically generates structured notes and links them into interconnected knowledge networks
- Continuous memory evolution and adaptive management

**Titans:**
- Meta in-context neural long-term memory
- Stores "surprising" data at test time
- Combines short-term, neural long-term, and persistent task memory modules

**B'MOJO:**
- Blends permanent, short-term, fading, and long-term memories
- Sliding attention mechanism to aggregate information

### 3.4 Critical Finding: External Memory ≠ Solved

Paper: "When Continual Learning Moves to Memory" (2026) shows that memory-augmented LLM agents do NOT resolve the continual-learning problem — they reshape it. Under limited context windows, old and new experiences compete during retrieval, relocating the bottleneck from parameter updates to memory access.

Key findings:
- Abstract procedural memories transfer more reliably than detailed trajectories
- Negative transfer disproportionately harms hard cases
- Finer-grained memory organization can induce severe forgetting even when forward transfer is strong

---

## 4. WORLD MODELS

### 4.1 Definition Space

A world model predicts environment dynamics from current observations and actions. Five key design dimensions:
1. **Data** — which modalities and interaction signals to use
2. **Representation** — continuous, discrete, or mixed
3. **Architecture** — generative, predictive, hybrid
4. **Learning objective** — reconstruction, prediction, self-supervised
5. **Usage** — planning, simulation, RL training, verification

### 4.2 Architectural Approaches

**Joint Embedding Predictive Architecture (JEPA):**
- Predicts next representation (not next data) in latent space
- Encoders bootstrap ground-truth next states for supervision
- Avoids reconstructing irrelevant/unpredictable details
- I-JEPA models stable, generalizable world representations

**Generative Latent Prediction (GLP):**
- Hierarchical, multi-level, mixed continuous/discrete representations
- Extended LLM backbone for discrete concept-based reasoning
- Generative embedding predictive module for continuous gradient-based reasoning
- Naturally mixes continuous embeddings with discrete tokens

**Qwen-AgentWorld (2026):**
- First language world models capable of simulating 7 domains via long chain-of-thought reasoning
- Three-stage training: CPT → SFT → RL with hybrid rubric-and-rule rewards
- Supports both decoupled simulation and unified agent foundation model training

**Cosmos 3 (NVIDIA):**
- Omnimodal world models jointly processing language, image, video, audio, actions
- Mixture-of-Transformers (MoT) backbone with separate reasoner/generator pathways
- 4B (Edge), 16B (Nano), 64B (Super) parameters
- Ranked best open-source Text-to-Image and Image-to-Video models

**Kairos:**
- Native world model stack with Hybrid Linear Temporal Attention
- Sliding-Window Attention (local) + Dilated SWA (mid-range) + Gated Linear Attention (global persistent memory)
- Formal theoretical bounds on error accumulation
- Cross-Embodiment Data Curriculum for progressive physical alignment

**LY-GWM (LingYang Graph World Model):**
- Graph-structured world model with explicit ontological-causal separation
- Three reasoning mechanisms: causal dynamics, teleological (goal-directed) reasoning, dialectical novelty discovery
- Demonstrates that ontological-causal separation is NECESSARY for intervention-consistent reasoning, not just a design preference

**PhyWM (Physically Controllable World Model):**
- Probabilistic world model over local visual variables
- GPT-style next-token prediction with pointer+content tokens
- Zero-shot object discovery, 3D manipulation, physical relationship inference

### 4.3 World Models in Text Environments

Study: "From Word to World" — Can LLMs be implicit text-based world models?
- Three-level evaluation: fidelity/consistency, scalability/robustness, agent utility
- LLMs CAN serve as reliable world models in structured settings
- Action verification boosts GPT-4o by 5.5% on WebShop
- Warm-started RL achieves 15% gain on SciWorld

---

## 5. COGNITIVE ARCHITECTURES

### 5.1 Leading Architectures for AGI

| Architecture | Approach | Key Features |
|-------------|----------|--------------|
| **ACT-R** | Symbolic + neural hybrid | Production rules, declarative memory, procedural memory, perceptual modules |
| **Sigma** | Probabilistic graphical model | Unified cognitive architecture, theorem proving |
| **LIDA** | Consciousness-inspired | Global Workspace Theory, attentional learning, episodic memory |
| **NARS** | Reasoning system | Non-axiomatic reasoning, truth-frequency-based inference |
| **ECA (Experimental Cognitive Architecture)** | Proto-AGI | Separates working state from episodic memory; disciplined episodic retrieval for long-horizon tasks |

### 5.2 Experimental Cognitive Architecture (ECA)

Key finding: Disciplined episodic retrieval through "guarded merge" improves long-horizon task completion under partial observability vs. no retrieval or replacement-style retrieval.

This suggests that how an architecture preserves what it currently takes to be true, and how older episodes re-enter ongoing control, is architecturally critical.

---

## 6. CONTINUAL LEARNING & CATASTROPHIC FORGETTING

### 6.1 The Problem Space

Catastrophic forgetting (CF) remains the primary obstacle to sequential domain adaptation in neural networks. In LLMs specifically:
- Behavioral benchmarks document CF externally
- Mechanistic interpretability reveals internal representational drift
- Early-layer attention heads exhibit systemic entropic dispersion
- Mid-to-deep feed-forward networks (or sparse expert blocks) suffer localized representation collapse
- MoE routing gates drift during sequential fine-tuning

### 6.2 Mitigation Approaches

| Method | Mechanism | Effectiveness |
|--------|-----------|---------------|
| **DOC (Dynamic Orthogonal Continual)** | Tracks functional direction drift, adjusts gradients orthogonal to historical functions | Outperforms prior methods on LLM continual learning benchmarks |
| **FAPM (Forgetting-Aware Pruning Metric)** | Pruning-based: uses magnitude AND task-vector/pre-trained-parameter ratio | Balances CF and downstream performance |
| **SLoRA** | Subspace-denoised LoRA: filters noisy components via subspace similarity with base model | Up to 12% accuracy improvement, 29% forgetting reduction, filters 30%+ of LoRA params |
| **SEE (Sequential Ensemble of Experts)** | Distributed routing with expert specialization, no additional router needed | Outperforms multi-task learning; remarkable OOD generalization |
| **LRCP (Low-Rank Circuit Projection)** | Subspace-regularized training intervention based on mechanistic analysis | Mitigates up to 94.2% of ancestral capability loss |
| **EWC (Elastic Weight Consolidation)** | Fisher information-based parameter importance, penalizes changes to critical weights | Standard baseline; BDH extension adds path integral tracking |
| **Rehearsal buffers** | Mix previous task samples into new task training | Effective but storage-heavy; collateral damage sampling improves efficiency |

### 6.3 Key Insight: Plasticity vs. Stability

The fundamental tension: models need plasticity to learn new tasks and stability to retain old ones. External memory systems don't resolve this — they relocate it to memory access. Hybrid approaches combining:
- Regularization-based protection of critical parameters
- Rehearsal of prior experience
- Modular expert systems with distributed routing
- Meta-learning over memory designs

represent the most promising current direction.

---

## 7. RESEARCH GAPS & OPEN PROBLEMS

1. **Integration gap:** No unified architecture combines world models, memory systems, cognitive architectures, and continual learning into a single coherent system.

2. **Scaling gap:** Most memory and continual learning research is done at small scale (<10B params). How do these mechanisms behave at 100B+ scale with MoE?

3. **Evaluation gap:** No standard benchmarks for "AGI-like" capabilities. Existing benchmarks are narrow and don't capture the full spectrum of general intelligence.

4. **Embodiment gap:** Most research is in text/simulation. Real-world physical AI with tight perception-action loops is still primitive.

5. **Safety/self-improvement gap:** ALMA and similar systems that meta-learn their own memory designs raise recursive self-improvement concerns without corresponding safety frameworks.

6. **Neuro-symbolic integration gap:** Despite dramatic results in robotics, neuro-symbolic systems remain poorly integrated with mainstream LLM research.

7. **Energy efficiency gap:** Current systems are enormously expensive. BDH's scale-free topology and sparse activation offer one path, but orders-of-magnitude improvements are needed.

---

## 8. KEY PAPERS & REFERENCES

1. "The End of Transformers?" (2026) — Survey of sub-quadratic architectures
2. "Foundation Model Architectures 2026" — Beyond Transformer deep dive
3. "Critique of World Model" — GLP architecture + PAN proposal
4. "Learning to Continually Learn via Meta-learning Agentic Memory Designs" (ALMA, 2026)
5. "MemVerse: Multimodal Memory for Lifelong Learning Agents" (2025)
6. "From Storage to Experience: Survey on LLM Agent Memory Mechanisms" (2026)
7. "When Continual Learning Moves to Memory" (2026)
8. "Mechanistic Analysis of Catastrophic Forgetting in LLMs" (2026)
9. "SLoRA: Balancing Plasticity and Forgetting in LLMs" (2026)
10. "Qwen-AgentWorld: Language World Models for General Agents" (2026)
11. "Cosmos 3: Omnimodal World Models for Physical AI" (NVIDIA, 2026)
12. "Kairos: A Native World Model Stack for Physical AI" (2026)
13. "LY-GWM: LingYang Graph World Model" — Ontological-causal separation necessity
14. "PhyWM: Physically Controllable World Model" — Visual scene understanding
15. "Improving Long-Horizon Task Completion in Proto-AGI Cognitive Architecture" (ECA, 2026)
16. "Memory OS of AI Agent" (2025)
17. "Dynamic Orthogonal Continual Fine-tuning" (DOC, 2025)
18. "SEE: Continual Fine-tuning with Sequential Ensemble of Experts" (2025)
19. "How to Alleviate Catastrophic Forgetting in LLMs" (Hierarchical regularization, 2025)
20. "Agentic Discovery of Neural Architectures" (AIRA-Compose/Design, 2026)

---

*Compiled from arXiv, ACL Anthology, CVPR 2026 proceedings, and industry reports. All sources cross-referenced and cited.*
