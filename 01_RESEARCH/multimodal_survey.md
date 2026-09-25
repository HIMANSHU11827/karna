# Multimodal AI Systems — Technical Survey

**Researcher:** coder-1
**Date:** 2026-01-15
**Status:** Phase 1 Study Complete

## 1. CLIP: Vision-Language Understanding

### Architecture

CLIP uses a **dual-tower architecture** with separate encoders for vision and text, both projecting to a shared embedding space.[1][2]

**Vision Encoder:**
- Base: ResNet-50 with modifications (3 stem convolutions instead of 1, anti-aliasing strided convolutions, QKV attention pooling instead of average pooling)[2]
- Alternative: Vision Transformer (ViT) — divides image into patches, processes as token sequence
- Input image → convolutional stem → transformer blocks → 512-dim embedding

**Text Encoder:**
- Compact autoregressive transformer (~1/3 the parameters of vision encoder)[2]
- Tokenizes text via byte-level BPE (same as GPT-2)[2]
- Text → token embedding + positional encoding → transformer blocks → 512-dim embedding

**Training:**
- **Contrastive learning** on 400M (image, text) pairs
- Compute cosine similarity between all image-text pairs in batch
- Maximize similarity for matching pairs, minimize for non-matching pairs
- Uses symmetric cross-entropy loss
- Trained on internet-collected data (WebImageText)[1]

**Key Insight:**
- Jointly trained vision-language models can **zero-shot transfer** to downstream tasks
- No task-specific fine-tuning needed — just prompt engineering with natural language
- "a photo of a {label}" template for classification

**Performance:**
- Zero-shot ImageNet: 76.2% (matches ResNet-50 supervised)[1]
- Robust to distribution shift — outperforms supervised models on many benchmarks[1]
- Batch size: 32,768 (critical for contrastive learning)[1]

**Limitations:**
- Weak at fine-grained tasks (counting, spatial relations)[1]
- Struggles with abstract reasoning and complex visual concepts[1]
- Text encoder is small — limits linguistic understanding[2]
- Data hungry — needs 400M+ pairs for good performance[1]

---

## 2. Whisper: Audio Understanding

### Architecture

Whisper is an **encoder-decoder Transformer** trained on 680,000 hours of multilingual supervised audio data.[3][4]

**Audio Encoder:**
1. Input audio resampled to 16,000 Hz
2. Compute 80-channel log-magnitude Mel spectrogram on 25ms windows, 10ms stride
3. Global normalization to [-1, 1] with zero mean
4. Two convolutional layers (kernel=3, GELU activation, stride=2 on second)[3]
5. Sinusoidal positional embeddings added
6. Transformer blocks (same width as decoder)[3]
7. Final layer normalization[3]

**Text Decoder:**
- Learned positional embeddings (not sinusoidal)[3]
- Token embeddings tied to output projection (shared weights)[3]
- Causal attention mask (can't look ahead)[3]
- Uses byte-level BPE tokenizer (GPT-2 tokenizer for English, refit for multilingual)[3]

**Multitask Format:**
- Special tokens control behavior: `<|startoftranscript|>`, `<|en|>`, `<|transcribe|>`, `<|notimestamps|>`, `<|endoftranscript|>`[3]
- Can perform: transcription, translation, language ID, voice activity detection, timestamp prediction[3]
- All tasks represented as token sequences to predict[3]

**Training:**
- Weak supervision — audio + text pairs from internet (noisy)[3]
- 680k hours total (some English, some multilingual)[3]
- Multilingual: 125k hours non-English[3]
- Speech translation: audio in language X → text in English[3]

**Key Results:**
- Zero-shot across 50+ languages[3]
- 50% fewer errors than previous SOTA on diverse benchmarks[3]
- Robust to accents, background noise, technical language[3]
- LibriSpeech: doesn't beat specialized models, but much better zero-shot generalization[3]

**Limitations:**
- Trained on 30-second chunks — long audio requires chunking algorithm[3]
- Hallucination risk — may generate text not in audio (combines audio prediction with language modeling)[3]
- Repetitive text generation (mitigated by beam search)[3]
- Slow inference — ~1x real-time on GPU for large model[3]

---

## 3. Video Understanding

### Approaches

**3D CNNs (C3D, I3D, ResNet3D, R(2+1)D):**
- Extend 2D convolutions to 3D (spatio-temporal kernels)[6]
- Good at capturing local temporal patterns (motion)[6]
- **Limitation:** Struggle with long-range dependencies — temporal receptive field is limited by kernel size[6]
- Example: C3D uses 3×3×3 kernels, effective temporal receptive field ~10-20 frames[6]

**Video Transformers (TimeSformer, ViViT, UniFormerV2):**
- Divide video into spatio-temporal tokens
- Self-attention across all tokens → captures long-range dependencies[6][9]
- **Limitation:** Quadratic complexity O(n²) where n = T × H × W patches[6]
- TimeSformer uses divided space-time attention to reduce cost[9]

**State Space Models (VideoMamba):**
- Adapt Mamba (selective SSM) to video[5]
- Linear complexity O(n) vs O(n²) for transformers[5]
- Bidirectional processing: forward and backward passes on video sequence[5]
- Spatio-temporal scanning: process spatial tokens first, then temporal[5]
- Self-distillation technique allows training without large pretraining datasets[5]

**Hybrid Approaches (ActNetFormer, UniFormerV2):**
- Combine 3D CNNs (local) with transformers (global)[6][9]
- ActNetFormer: 3D-ResNet50 + Video Transformer Small, cross-architecture pseudo-labeling[6]
- UniFormerV2: 3D convolutions in early layers, spatio-temporal attention in later layers[9]

**Current State-of-the-Art:**
- VideoMamba: Linear complexity, competitive accuracy on UCF101, HMDB51, Kinetics[5]
- UniFormerV2: Uses CLIP-pretrained ViT weights, strong performance[9]
- ActNetFormer: Semi-supervised with 3D CNN + Transformer ensemble[6]

---

## 4. Real-Time Streaming

### WebRTC Protocol

**Architecture:**
- Peer-to-peer communication protocol for real-time audio/video/data[7]
- Uses ICE (Interactive Connectivity Establishment) for NAT traversal[7]
- SRTP (Secure Real-time Transport Protocol) for encrypted media[7]
- SCTP (Stream Control Transmission Protocol) for data channels[7]

**Latency Components:**
- Capture: 1-10ms (audio), 16-33ms (video at 30-60fps)
- Encoding: 5-20ms (audio codec), 10-50ms (video codec)
- Network: 10-100ms (depends on distance, congestion)
- Decoding: 5-20ms
- **Total typical:** 50-200ms one-way[7]

**Key Features:**
- Adaptive bitrate streaming (adjusts quality to network conditions)[7]
- Forward Error Correction (FEC) for packet loss resilience[7]
- Jitter buffer for smooth playback[7]

### Real-Time Multimodal Systems

**Aero Realtime (2026):**
- 4B parameter streaming multimodal model[8]
- **Duplex architecture:** input and output advance on same temporal grid[8]
- 80ms audio slots — each slot predicts token or silence[8]
- KV-cache reuse across continuous updates[8]
- **Results:** 84ms median lag, 173ms P95 lag over 20 minutes of continuous video[8]
- Training: 1,600+ tokens/GPU/second on A100-40G[8]

**StreamWise (2026):**
- Real-time multi-modal generation serving system[7]
- Coordinates LLMs, TTS, video generation models[7]
- Adaptive quality: lower resolution for early scenes, higher for later[7]
- Multi-level parallelism: stage, sequence, tensor parallelism[7]
- **Results:** Sub-second latency on 8×H100 GPUs, $40 per 10-minute video[7]

---

## 5. Bottlenecks for Real-Time Multimodal

### 1. **Compute Heterogeneity**
- Different modalities need different compute profiles[7][10]
- Text: lightweight (milliseconds)[10]
- Images: moderate (under 1 second)[10]
- Video: heavy (seconds to minutes)[10]
- **Problem:** Scheduling heterogeneous workloads on shared resources causes head-of-line blocking[10]

### 2. **Memory Pressure**
- Video requests use 10-100x more memory than text[10]
- KV-cache for long video sequences can exhaust GPU memory[10]
- Under memory pressure, SLO violations surge to 70-90%[10]

### 3. **Head-of-Line Blocking**
- Large video requests monopolize GPU during prefill[10]
- Small text requests wait tens of seconds (unacceptable for interactive use)[10]
- FCFS scheduling fails under multimodal workloads[10]

### 4. **Modality Encoding Bottleneck**
- Preprocessing (image resize, audio spectrogram, video decoding) adds latency[10]
- On-device encoding can be slower than inference[10]
- CPU-GPU data transfer is a bottleneck[10]

### 5. **Duplex vs Turn-Based**
- Most systems are turn-based: prefill → decode → wait for next input[8]
- True real-time requires duplex: input and output simultaneously[8]
- Existing systems use micro-turn polling (adds latency)[8]

### 6. **Model Size vs Latency Tradeoff**
- Larger models = better quality but higher latency[8]
- 4B model: 84ms median lag[8]
- 7B+ models: likely 200ms+ lag[8]
- Quantization and distillation help but reduce quality[8]

### 7. **Network Bandwidth**
- High-resolution video requires significant bandwidth[7]
- 1080p video: ~5-10 Mbps[7]
- 4K video: ~20-40 Mbps[7]
- Mobile networks may not sustain these rates[7]

---

## 6. Key Takeaways for Our System

### What Works:
1. **Contrastive learning** for vision-language alignment (CLIP)[1]
2. **Encoder-decoder Transformers** for audio (Whisper)[3]
3. **State space models** for efficient video understanding (VideoMamba)[5]
4. **Duplex architecture** for real-time streaming (Aero Realtime)[8]
5. **Modular serving** with adaptive quality (StreamWise)[7]

### What Doesn't Work:
1. **Pure FCFS scheduling** for multimodal workloads[10]
2. **Turn-based architectures** for real-time interaction[8]
3. **3D CNNs alone** for long-range video understanding[6]
4. **Quadratic attention** for long videos[6]

### Our Architecture Should:
1. Use **duplex streaming** (input + output simultaneously)[8]
2. Use **modular pipeline** with adaptive quality[7]
3. Use **state space models** for video (linear complexity)[5]
4. Use **contrastive pretraining** for vision-language[1]
5. Use **encoder-decoder** for audio[3]
6. Use **priority scheduling** (not FCFS) for mixed workloads[10]
7. Use **KV-cache reuse** for continuous inference[8]

---

## Sources

[1] https://arxiv.org/abs/2103.00020 — CLIP Original Paper
[2] https://github.com/openai/CLIP — CLIP Code
[3] https://arxiv.org/abs/2212.04356 — Whisper Original Paper
[4] https://github.com/openai/whisper — Whisper Code
[5] https://arxiv.org/abs/2403.06977 — VideoMamba Paper
[6] https://arxiv.org/abs/2404.06243 — ActNetFormer Paper
[7] https://arxiv.org/abs/2603.05800 — StreamWise Paper
[8] https://arxiv.org/abs/2608.08469 — Aero Realtime Paper
[9] https://en.papernotes.org/ECCV2024/video_understanding/videomamba_state_space_model_for_efficient_video_understanding — VideoMamba Notes
[10] https://ascpt.onlinelibrary.wiley.com/doi/10.1049/aie2.70020 — Edge MLLM Inference
