# Benchmarks Survey — AGI Research Lab

**Date:** 2026-09-23  
**Researcher:** verifier (Phase 1: Study)  
**Scope:** How to benchmark our from-scratch neural architecture

---

## 1. MNIST Benchmarking

### Standard Protocol
- **Dataset:** 70,000 handwritten digits (0–9), 28×28 grayscale
- **Split:** 60,000 train / 10,000 test
- **Input:** Flattened to 784-dimensional vectors, normalized to [0,1]
- **Output:** 10-class softmax
- **Metric:** Test accuracy (%)
- **Baselines:**
  - Random: 10%
  - K-NN: ~97%
  - SVM: ~98%
  - MLP (2-layer): ~98.5%
  - CNN (LeNet): ~99.5%
  - Modern SOTA (vision transformers): ~99.8%

### Why We Use It
- Small, fast to train → good for rapid prototyping
- Well-understood → easy to spot implementation bugs
- Label errors are low (0.15%) → reliable ground truth

### What We Measure
1. **Final accuracy** — after training converges
2. **Convergence speed** — epochs to reach 90%, 95%, 99%
3. **Samples to threshold** — how many samples needed for X% accuracy (data efficiency)
4. **Wall-clock time** — training and inference latency

---

## 2. CIFAR-10 Benchmarking

### Standard Protocol
- **Dataset:** 60,000 color images (32×32×3), 10 classes
- **Split:** 50,000 train / 10,000 test
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Preprocessing:** Pixel normalization (/255), optional: per-channel mean/std subtraction, data augmentation (random crop, horizontal flip)
- **Metric:** Test accuracy (%)
- **Baselines:**
  - Random: 10%
  - MLP: ~50%
  - CNN (VGG): ~93%
  - ResNet-18: ~95%
  - Modern SOTA: ~99%

### Why We Use It
- Color + spatial structure → harder than MNIST
- Real images (not synthetic) → tests real perception
- Standard comparison → easy to see where we stand

### What We Measure
1. **Final accuracy** — without data augmentation (our baseline)
2. **Per-class accuracy** — which classes are hard/easy
3. **Training convergence** — epochs to converge
4. **Inference latency** — ms per image (real-time requirement)

---

## 3. Continual Learning — Split-MNIST

### Standard Protocol
- **Dataset:** MNIST split into 5 tasks of 2 digits each
  - T1: {0,1}, T2: {2,3}, T3: {4,5}, T4: {6,7}, T5: {8,9}
- **Training:** Sequential, one task at a time. No access to previous task data (unless using replay).
- **Testing:** After each task, evaluate on ALL digits 0–9.
- **Scenarios:**
  - **Multi-headed:** Task ID given at test time. Model predicts only within the 2 classes of current task. (Easier.)
  - **Single-headed:** Task ID NOT given. Model must predict over all 10 classes. (Harder, more realistic.)
  - **Our project:** Single-headed is the right test. We don't get task IDs in the real world.

### Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Average Accuracy (ACC)** | Mean test accuracy across all tasks after learning all tasks | Overall performance. Higher = better. |
| **Backward Transfer (BWT)** | $R_{i,T} - R_{i,i}$ — change in task i's accuracy after learning all later tasks. Negative = forgetting. | Measures catastrophic forgetting. Positive = learning new things helped old things. |
| **Forward Transfer (FWT)** | $R_{i,i} - \text{random init accuracy on task i}$ — how much prior learning helps new tasks. | Measures transfer learning. |
| **Forgetting** | $-BWT$ | Direct measure of how much was forgotten. |

Where $R_{i,j}$ = accuracy on task i after training on task j.

### Why We Use It
- Tests **catastrophic forgetting** — the #1 problem for real-time learning
- Measures **knowledge transfer** — does learning A help with B?
- Standard comparison → we can compare against EWC, SI, GEM, Replay, etc.

### What We Measure
1. **ACC** — final average accuracy (our headline number)
2. **BWT** — must be > 0 for a successful system (no forgetting)
3. **Per-task accuracy matrix** — which tasks are forgotten, which are retained
4. **Convergence per task** — how fast each task is learned
5. **Sample efficiency** — how many samples per task

---

## 4. Real-Time System Benchmarking

### Latency Metrics
- **Inference latency** — ms per input (text token, image frame, audio chunk)
- **Throughput** — inputs/second or tokens/second
- **Real-time factor (RTF)** — processing time / input duration. RTF < 1 means faster than real-time.
  - Example: processing a 1s audio clip in 200ms → RTF = 0.2 (real-time capable)

### Real-Time System Benchmarks
| Benchmark | What it tests | Key metric |
|-----------|---------------|------------|
| **MLPerf Inference** | Inference speed across models | latency (ms), throughput (samples/s) |
| **DeepBench** | Basic ops (matmul, conv) | GFLOPS, latency |
| **TinyML** | Microcontroller inference | latency, energy (mJ), model size (KB) |
| **AIBO** | Online learning speed | time per weight update |
| **Online Learning Benchmarks** | Streaming learning | regret, cumulative error, adaptation time |

### What We Measure
1. **Per-sample inference latency** — target: < 10ms for text, < 50ms for images
2. **Learning update latency** — target: < 1ms per sample (so we can learn on the fly without slowing inference)
3. **Throughput** — how many concurrent streams can we handle?
4. **Memory usage** — peak RAM, model size on disk
5. **Energy** — CPU time per operation (we're local CPU only)

---

## 5. Baselines to Compare Against

### For MNIST / CIFAR-10
| Baseline | Type | MNIST Acc | CIFAR-10 Acc |
|----------|------|-----------|--------------|
| Random | — | 10% | 10% |
| K-NN (k=5) | Non-parametric | ~97% | ~85% |
| Linear classifier (SVM) | Linear | ~98% | ~90% |
| MLP (2-layer, ReLU) | Neural (backprop) | ~98.5% | ~55% |
| CNN (LeNet-style) | Neural (backprop) | ~99.3% | ~75% |
| ResNet-18 | Neural (backprop) | ~99.5% | ~95% |
| EfficientNet-B0 | Neural (backprop) | ~99.7% | ~98% |

**Our target:** Beat MLP with our from-scratch architecture. Match CNN if possible. Don't need to beat ResNet — we're prioritizing real-time learning, not just accuracy.

### For Continual Learning (Split-MNIST)
| Method | Type | ACC (single-headed) | Memory |
|--------|------|---------------------|--------|
| Naive SGD | Baseline | ~20% (catastrophic forgetting) | None |
| EWC | Regularization | ~20% | None |
| SI | Regularization | ~20% | None |
| MAS | Regularization | ~20% | None |
| LwF | Regularization | ~24% | None |
| Replay (naive) | Replay | ~91% | Buffer of past samples |
| GEM | Replay | ~92% | Buffer + optimization |
| DGR | Replay | ~91% | Generative model |
| Offline (upper bound) | Oracle | ~98% | All data available |

**Our target:** ACC > 90%, BWT > 0 (no forgetting), without storing raw samples (we don't want a replay buffer — we want to learn continuously).

### For Real-Time Systems
| System | Latency | Throughput | Online learning? |
|--------|---------|------------|------------------|
| GPT-4 API | 500ms-5s | ~50 tok/s | No |
| Claude API | 200ms-2s | ~80 tok/s | No |
| Llama-3-8B (local) | 50ms | ~30 tok/s | No |
| Ours (target) | <10ms | >100 samples/s | **Yes** |

**Our target:** Sub-10ms inference with learning on every sample. This is the key differentiator.

---

## 6. Our Evaluation Plan

### Phase 1: Static Accuracy (Does it work at all?)
- MNIST → target > 95% accuracy
- CIFAR-10 → target > 70% accuracy (without backprop, this is ambitious)
- Compare against MLP and CNN baselines

### Phase 2: Continual Learning (Does it forget?)
- Split-MNIST (single-headed) → target ACC > 85%, BWT > 0
- Permuted MNIST → target ACC > 80%
- Compare against EWC, Replay baselines

### Phase 3: Real-Time (Is it fast enough?)
- Inference latency < 10ms
- Learning update < 1ms per sample
- Throughput > 100 samples/sec
- Real-time factor < 1.0 for all modalities

### Phase 4: Real-Time Adaptive (Does it improve on the fly?)
- Same user interacts repeatedly → measure accuracy improvement over time
- Intent inference: test with misspoken/misspelled input → target > 90% intent recovery
- No re-prompting: system must self-correct without asking for clarification

### Phase 5: Multimodal (Can it handle all modalities?)
- Text: language understanding + generation
- Image: classification + description
- Audio: speech-to-text + emotion detection
- Video: activity recognition
- Cross-modal: text→image, image→text, audio→text

---

## 7. Critical Notes

1. **MNIST/CIFAR-10 test sets have label errors** — MNIST: 0.15%, CIFAR-10: 0.54%. This means the "true" accuracy ceiling is slightly below 100%. Don't chase the last 0.5%.

2. **Multi-headed vs single-headed Split-MNIST** — multi-headed inflates accuracy by 50-70 percentage points. Only single-headed is a real test. We use single-headed.

3. **Permuted MNIST is too easy for catastrophic forgetting** — the pixel permutation destroys spatial structure, so tasks don't interfere much. Split-MNIST is harder and more realistic.

4. **Real-time learning has no standard benchmark** — we have to define our own metrics. Key ones: samples to adaptation, accuracy delta before/after adaptation, latency overhead of learning.

5. **"Better at everything through real-time interaction"** is not a standard ML metric. We need to define:
   - Same user, repeated interactions → accuracy delta
   - Intent inference under noise → recovery rate
   - Adaptation time → samples to converge to new pattern

---

*Sources: arXiv:2106.01065 (Pervasive Label Errors), arXiv:1810.12488 (Continual Learning Evaluation), Farquhar & Gal (2018), Lopez-Paz & Ranzato (2017), Lomonaco & Maltoni (CORe50).*
