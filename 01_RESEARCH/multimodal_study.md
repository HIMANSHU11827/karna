# Multimodal Systems Study

**Date:** 2026-09-23  
**Author:** verifier  
**Scope:** CLIP, Whisper, fusion strategies, real-time challenges

---

## 1. CLIP (Contrastive Language-Image Pre-training)

### Architecture
- **Dual encoders**: Image encoder + Text encoder
- **Image encoder**: ResNet-50 (modified) or Vision Transformer (ViT)
- **Text encoder**: 12-layer Transformer, 512-wide, 8 attention heads
- **Shared embedding space**: Linear projection from both encoders
- **Contrastive loss**: Maximize cosine similarity of correct (image, text) pairs, minimize incorrect pairs within a batch of N samples (N×N possible pairings)

### Key Design Choices
- Trained from scratch (no pre-trained weights)
- Linear projection only (no non-linear projection head)
- Temperature parameter τ learned during training
- Random square crop as only data augmentation
- Vocab: 49,152 BPE tokens for text

### Training Data
- 400 million (image, text) pairs from internet
- Largest model: 18 days on 592 V100 GPUs (RN50x64) or 12 days on 256 V100 (ViT-L/14)

### Zero-Shot Transfer
- Synthesize linear classifier by embedding class names/descriptions
- No fine-tuning needed — just provide text descriptions of classes
- Matches ResNet-50 on ImageNet without using any labeled training examples

### Limitations for Real-Time Use
- Batch-level contrastive loss (not streaming)
- Fixed encoders after training (no online learning)
- High compute cost (GPU cluster required)
- No temporal/video understanding

---

## 2. Whisper (Speech Recognition)

### Architecture
- **Encoder-decoder Transformer**
- **Input processing**: Audio → 16kHz mono → 80-channel log-mel spectrogram (25ms windows, 10ms stride) → 30-second chunks
- **Encoder**: 2 conv layers (stride-2 downsampling) + sinusoidal position embeddings + Transformer encoder blocks
- **Decoder**: Autoregressive Transformer with cross-attention to encoder, learned positional embeddings, tied input/output embeddings
- **Tokenizer**: BPE (GPT-2 vocabulary for English, extended to 51,865 for multilingual)

### Multitask Token Framework
Special control tokens prepended to decoder input:
- Language tag: `<|en|>`, `<|fr|>`, etc. (99 languages)
- Task token: `<|transcribe|>` or `<|translate|>`
- Timestamp control: `<|notimestamps|>` or timestamp tokens at 20ms intervals
- Voice activity: `<|nospeech|>`

### Model Sizes
| Model | Parameters | Encoder Layers | Decoder Layers | Width | VRAM (FP16) | Relative Speed |
|-------|-----------|----------------|----------------|-------|-------------|----------------|
| tiny | 39M | 4 | 4 | 384 | ~1 GB | ~10x |
| base | 74M | 6 | 6 | 512 | ~1 GB | ~7x |
| small | 244M | 12 | 12 | 768 | ~2 GB | ~4x |
| medium | 769M | 24 | 24 | 1024 | ~5 GB | ~2x |
| large | 1,550M | 32 | 32 | 1280 | ~10 GB | 1x |
| large-v3-turbo | 809M | 32 | 4 | 1280 | ~6 GB | ~8x |

### Training Data
- 680,000 hours of multilingual, multitask supervised audio
- 65% English, 35% other languages (99 total)
- Weak supervision (audio + transcript pairs from web)

### Real-Time Deployment
- **whisper.cpp**: C/C++ port, no Python/PyTorch, quantization (4-8 bit), runs on CPU/Raspberry Pi/iPhone
- **faster-whisper**: CTranslate2-based, 4x faster than original, INT8 quantization
- Real-time factor (RTF) < 1 means faster than real-time

### Limitations for Our Project
- Encoder-decoder Transformer (banned architecture)
- Fixed weights after training (no online learning)
- 30-second fixed window (not truly streaming)
- High compute for large models

---

## 3. Multimodal Fusion Strategies

### Fusion Architectures

| Strategy | Description | Latency | Accuracy | Use Case |
|----------|-------------|---------|----------|----------|
| **Early Fusion** | Concatenate raw inputs, single network | Lowest | Lowest | Edge devices, low compute |
| **Intermediate Fusion** | Fuse at feature extraction layers | Medium | Medium | Balanced systems |
| **Late Fusion** | Separate encoders, fuse at decision level | Highest | Highest | Cloud, high accuracy |
| **Cross-Attention** | One modality attends to another | High | High | Vision-language tasks |

### Fusion Mechanisms
1. **Concatenation**: Simple feature vector joining
2. **Cross-attention**: Query from one modality, Key/Value from another
3. **Tensor fusion**: Outer product of modality features (higher-order interactions)
4. **Mixture of Experts**: Route to modality-specific experts, combine outputs
5. **Bilinear fusion**: Compact bilinear pooling for feature interaction

### Real-Time Fusion Challenges
- **Time alignment**: Different modalities have different sampling rates (text: ~5 tokens/sec typing, audio: 16kHz, video: 30fps)
- **Bursty event rates**: Activity-dependent compute (silent audio vs. speech, static vs. moving video)
- **Modality reliability**: One modality may be noisy/missing (bad lighting, background noise)
- **Clock synchronization**: Sensor clocks differ across modalities
- **Memory movement**: Data transfer between modality-specific processors dominates cost

### Latency-Accuracy Trade-off (from literature)
| Fusion | Accuracy (%) | Latency (ms) |
|--------|-------------|--------------|
| Late Fusion | 84.25 | 21.6 |
| Intermediate Fusion | 72.40 | 13.5 |
| Early Fusion | 67.89 | 11.4 |

---

## 4. Real-Time Multimodal Challenges

### Latency Requirements
| Modality | Target Latency | Throughput |
|----------|---------------|------------|
| Text | <100ms | 1000 tokens/sec |
| Image | <50ms | 20 images/sec |
| Audio | <100ms | RTF < 1.0 |
| Video | <33ms | 30 fps |

### Key Challenges

1. **Streaming Processing**
   - Must process partial input (can't wait for complete)
   - Chunk-based processing with state carryover
   - Commit-and-revise: output best guess, update as more data arrives

2. **Time Synchronization**
   - Audio and video must be lip-synced (±80ms tolerance)
   - Multi-rate fusion: buffer and interpolate to align
   - Event-driven processing: process when data arrives, not on fixed schedule

3. **Compute Budget**
   - Activity-dependent compute: more processing for complex scenes
   - Graceful degradation: reduce quality under load, don't fail
   - Early exit: produce answer when confident enough

4. **Memory Management**
   - Sliding window for recent context
   - Compressed summary for older context
   - Importance-based retention (keep important, forget trivial)

5. **Modality Dropout**
   - System must work when one modality is missing
   - Robustness to noise in individual modalities
   - Cross-modal completion: infer missing modality from others

---

## 5. Implications for Our Architecture

### What We Can Learn (Not Copy)
1. **Unified representation space**: CLIP's shared embedding space is elegant — but we need a from-scratch version
2. **Multitask tokens**: Whisper's special token framework is clever — but we need our own from-scratch mechanism
3. **Streaming chunk processing**: Both CLIP and Whisper process fixed windows — we need true streaming with state
4. **Contrastive learning**: CLIP's contrastive objective is powerful — but we need a local, online version

### What We Must Avoid
1. **Transformer architecture**: Both CLIP and Whisper use Transformers — banned
2. **Backpropagation**: Both trained with backprop — banned
3. **Fixed weights after training**: Both frozen at deployment — we need continuous learning
4. **Batch-level training**: Both require large batches — we need per-sample updates
5. **Massive compute**: Both need GPU clusters — we target local CPU

### Our Approach
- **From-scratch fusion**: Design our own cross-modal binding mechanism (not cross-attention)
- **Local learning rules**: All modalities learn via local plasticity (no backprop)
- **Streaming by design**: Process chunks as they arrive, maintain state across time
- **Sparse representations**: All modalities encoded as sparse distributed representations
- **Real-time guarantees**: Bounded latency, anytime processing, graceful degradation

---

*Sources: Radford et al. (2021) CLIP; Radford et al. (2022) Whisper; arXiv:2511.21889 (Fusion Strategies); arXiv:2607.18171 (FlashRT); IOPScience (Event-based fusion).*
