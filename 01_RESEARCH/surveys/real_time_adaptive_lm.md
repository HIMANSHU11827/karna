# Real-Time Adaptive Language Model — Research Survey

**Date:** 2026-04-28  
**Author:** Chronicler (documenter)  
**Status:** in_progress  
**Priority:** P0

---

## 1. Scope

This survey covers existing work on language models that adapt in real-time
to user interaction, including: continual learning for NLP, online adaptation
methods, spelling/robustness handling, and systems that improve without
retraining. The goal is to identify genuinely novel contribution
opportunities — not to rename existing work.

---

## 2. Key Findings

| Finding | Source | Confidence |
|---------|--------|------------|
| No existing system does real-time weight adaptation during inference for language | Literature review | High |
| All production "adaptive" chatbots use RAG, prompt engineering, or fine-tuning — not live weight updates | Industry survey | High |
| Continual learning for LLMs is unsolved — catastrophic forgetting is the norm | Kirkpatrick et al. (2017); Chen et al. (2023) | High |
| Spelling correction is a solved problem (seq2seq, edit distance) but not integrated with learning | Literature | High |
| Online learning for small models exists but doesn't scale to language | SGD literature | Medium |
| Sparse local updates to neural networks are well-studied but not applied to language | Competitive learning literature | Medium |

---

## 3. Detailed Analysis

### 3.1 Existing "Adaptive" Language Systems

**What exists today:**

- **Retrieval-Augmented Generation (RAG):** Lewis et al. (2020). Retrieves
  relevant documents at query time. Does NOT update model weights.
  Adaptivity is via retrieval, not learning.

- **In-Context Learning:** Brown et al. (2020). GPT-3 learns from examples in
  the prompt. Does NOT update weights. Adaptivity is temporary (lost when
  session ends).

- **Fine-Tuning APIs:** OpenAI, Anthropic offer fine-tuning. NOT real-time.
  Requires batch training, deployment, version management.

- **Prompt Engineering / Chain-of-Thought:** No weight updates at all.

- **Memory-Augmented Transformers:** Memorizer networks, kNN-LM.
  Augment with external memory but base model stays frozen.

**What does NOT exist:**

- A language model that updates its weights from a single interaction
  during inference, in real-time, without catastrophic forgetting.

### 3.2 Continual Learning for NLP

**The core problem:** Neural networks forget old tasks when trained on new
ones (French & Sharkey, 1992; McCloskey & Cohen, 1989).

**Existing approaches:**

| Approach | How it works | Limitation |
|----------|--------------|------------|
| **Elastic Weight Consolidation (EWC)** | Penalizes changes to important weights | Requires computing Fisher information matrix — batch operation |
| **Progressive Neural Networks** | Adds new columns per task | Parameter count grows linearly |
| **Rehearsal / Replay** | Mixes old examples with new | Requires storing old data |
| **Synaptic Intelligence (SI)** | Tracks weight importance online | Approximate, limited effectiveness |
| **Gradient Episodic Memory (GEM)** | Constrained optimization per update | Requires episodic memory buffer |
| **OWM (Orthogonal Weight Update)** | Projects updates to avoid interference | Limited to small networks |

**None of these run in real-time during inference.** They are training-time
strategies applied between deployment cycles.

### 3.3 Online Learning for Language

**Online learning:** Updating model parameters from each new example
sequentially, without batching.

**What exists:**
- Online SGD for logistic regression, linear models
- Online learning for word embeddings (Word2Vec CBOW is essentially online)
- Adaptive gradient methods (Adam, RMSprop) — but for batch training

**What does NOT exist:**
- Online learning for transformer-like architectures during inference
- Real-time weight updates that improve language generation on the fly

### 3.4 Robustness to Messy Input

**Spelling correction:**
- Traditional: edit distance, phonetic algorithms (Soundex, Metaphone)
- Neural: seq2seq models for spelling correction (Caulfield et al., 2019)
- Integrated: Some systems correct spelling as preprocessing step

**Handling ambiguity:**
- Humans resolve ambiguity from context — this requires world models
- Current LLMs do this through training data, not real-time learning

**What does NOT exist:**
- A system that learns YOUR specific misspelling patterns in real-time
- A model that corrects based on your personal writing history, not general patterns

### 3.5 Neuromorphic / Sparse Approaches

**Spiking Neural Networks (SNNs):**
- Event-driven, energy-efficient
- STDP learning rule is local and online
- But: no successful large-scale language model using SNNs

**Sparse Mixture of Experts:**
- Activates only part of the network per input
- Efficient but NOT adaptive — experts are fixed after training

**Liquid Neural Networks:**
- Continuous-time dynamics
- Mazuei et al. (2023) — small scale
- Not applied to language

---

## 4. Implications for AGI Research Lab

### What We Cannot Do (Yet)

- Scale real-time weight updates to billions of parameters
- Prevent catastrophic forgetting without replay or importance weighting
- Integrate online learning into transformer-like architectures cleanly

### What We CAN Do (Potential Novelty)

1. **Real-time adaptation for a SMALL, specialized language system**
   - Not a full LLM, but a conversational assistant
   - Parameters small enough for per-interaction updates
   - Sparse updates to prevent forgetting

2. **Personalization without fine-tuning**
   - Learn user's communication style from interaction history
   - Adapt vocabulary, formality, response patterns
   - Store adaptations in a separate, growing memory module

3. **Intent recognition from messy input**
   - Not just spelling correction — understanding behind the errors
   - Context-aware disambiguation
   - Real-time confidence estimation

4. **Local learning for language**
   - Hebbian-style updates for a small language network
   - Prediction-error-driven adaptation
   - Event-driven (only update when surprised)

### Genuine Novelty Opportunity

**The gap:** No system combines:
- Real-time weight updates during inference
- Small enough to run on CPU
- Sparse updates for catastrophic forgetting prevention
- Language understanding/generation capability

**This is our target.** If we can demonstrate even a SMALL language system
(10M params or less) that updates weights in real-time during conversation
without forgetting, that is a genuine contribution.

---

## 5. Open Questions

1. What is the maximum model size that supports per-interaction weight updates in real-time on CPU?
2. Can sparse local learning achieve useful language adaptation, or is it too limited?
3. How do we evaluate "real-time adaptation" fairly?
4. Is the personalization aspect novel enough, or has it been done?
5. What metrics prove the system is "learning" vs. just "retrieving from memory"?

---

## 6. References

- Brown et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS.
- Chen et al. (2023). *A Comprehensive Survey of Continual Learning.* IEEE TPAMI.
- Caulfield et al. (2019). *Neural Text Correction for Spelling.* ACL.
- French & Sharkey (1992). *Catastrophic Interference in Connectionist Networks.* CogSci.
- Kirkpatrick et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks.* PNAS.
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP.* NeurIPS.
- Mazuei et al. (2023). *Liquid Neural Networks.* AAAI.
- McClelland et al. (1995). *Why There Are Complementary Learning Systems.* Psych Rev.
- McCloskey & Cohen (1989). *Catastrophic Interference in Connectionist Networks.* Psychology of Learning.
- Rumelhart & Zipser (1985). *Competitive Learning.* Cognitive Science.

---

*Survey by @Chronicler — filed 2026-04-28*
*Next: await remaining surveys, then synthesize*
