"""
Multimodal Processing — Cross-Modal Attention and Unified Representations
==========================================================================

A multimodal processing system that fuses information from multiple sensory
modalities into coherent, unified representations.

Architecture:
- Modality-specific encoders (vision, audio, text)
- Cross-modal attention mechanism
- Shared representation space
- Modality fusion with gating

"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(Enum):
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    TACTILE = "tactile"
    PROPRIOCEPTION = "proprioception"


@dataclass
class ModalityEmbedding:
    """Embedding from a single modality."""
    modality: Modality
    embedding: list[float]
    attention_weights: list[float] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "modality": self.modality.value,
            "embedding_dim": len(self.embedding),
            "embedding_sample": self.embedding[:5],
            "attention_weight_mean": sum(self.attention_weights) / len(self.attention_weights) if self.attention_weights else 0,
            "confidence": self.confidence,
            "features": self.features,
        }


@dataclass
class FusedRepresentation:
    """Unified representation from multiple modalities."""
    embeddings: list[ModalityEmbedding]
    fused_embedding: list[float]
    fusion_weights: dict[str, float]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "num_modalities": len(self.embeddings),
            "modalities": [e.modality.value for e in self.embeddings],
            "fused_dim": len(self.fused_embedding),
            "fused_sample": self.fused_embedding[:5],
            "fusion_weights": self.fusion_weights,
        }


@dataclass
class CrossModalAttention:
    """Attention weights between modalities."""
    source_modality: Modality
    target_modality: Modality
    attention_matrix: list[list[float]]
    attention_score: float


# ============================================================
# Modality-Specific Encoders
# ============================================================

class ModalityEncoder:
    """Base class for modality-specific encoders."""

    def __init__(self, modality: Modality, output_dim: int = 128):
        self.modality = modality
        self.output_dim = output_dim

    def encode(self, data: Any) -> ModalityEmbedding:
        """Encode raw data into modality embedding."""
        raise NotImplementedError


class SimpleVisionEncoder(ModalityEncoder):
    """Simplified vision encoder for multimodal fusion."""

    def __init__(self, output_dim: int = 128):
        super().__init__(Modality.VISION, output_dim)

    def encode(self, data: Any) -> ModalityEmbedding:
        """Encode visual data."""
        if isinstance(data, dict):
            features = {
                "edges": data.get("edges", []),
                "colors": data.get("colors", []),
                "shapes": data.get("shapes", []),
                "textures": data.get("textures", []),
            }
            raw_embedding = [0.5] * self.output_dim  # Placeholder
        else:
            raw_embedding = [0.0] * self.output_dim
            features = {}

        # Generate deterministic embedding
        seed = hash(str(data)) % 10000
        embedding = [
            math.sin(seed + i * 0.1) * 0.5 + 0.5
            for i in range(self.output_dim)
        ]

        return ModalityEmbedding(
            modality=Modality.VISION,
            embedding=embedding,
            attention_weights=[1.0] * self.output_dim,
            confidence=0.90,
            features=features,
        )


class SimpleAudioEncoder(ModalityEncoder):
    """Simplified audio encoder for multimodal fusion."""

    def __init__(self, output_dim: int = 128):
        super().__init__(Modality.AUDIO, output_dim)

    def encode(self, data: Any) -> ModalityEmbedding:
        """Encode audio data."""
        if isinstance(data, dict):
            features = {
                "spectrogram": data.get("spectrogram", []),
                "pitch": data.get("pitch", 0.0),
                "rhythm": data.get("rhythm", []),
                "timbre": data.get("timbre", ""),
            }
        else:
            features = {}

        seed = hash(str(data)) % 10000
        embedding = [
            math.cos(seed + i * 0.15) * 0.5 + 0.5
            for i in range(self.output_dim)
        ]

        return ModalityEmbedding(
            modality=Modality.AUDIO,
            embedding=embedding,
            attention_weights=[1.0] * self.output_dim,
            confidence=0.85,
            features=features,
        )


class SimpleTextEncoder(ModalityEncoder):
    """Simplified text encoder for multimodal fusion."""

    def __init__(self, output_dim: int = 128):
        super().__init__(Modality.TEXT, output_dim)

    def encode(self, data: Any) -> ModalityEmbedding:
        """Encode text data."""
        if isinstance(data, str):
            features = {
                "tokens": data.lower().split(),
                "length": len(data),
                "language": "en",
            }
        else:
            features = {}
            data = str(data)

        seed = hash(str(data)) % 10000
        embedding = [
            math.sin(seed + i * 0.08) * 0.5 + 0.5
            for i in range(self.output_dim)
        ]

        return ModalityEmbedding(
            modality=Modality.TEXT,
            embedding=embedding,
            attention_weights=[1.0] * self.output_dim,
            confidence=0.95,
            features=features,
        )


# ============================================================
# Cross-Modal Attention
# ============================================================

class CrossModalAttentionModule:
    """Computes attention between different modality embeddings.

    Uses scaled dot-product attention adapted for cross-modal:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    In production: multi-head cross-attention (like in Flamingo, BLIP-2).
    Prototype: simplified single-head attention.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def compute_attention(
        self, source: ModalityEmbedding, target: ModalityEmbedding
    ) -> CrossModalAttention:
        """Compute cross-modal attention from source to target."""
        # Compute attention scores between each dimension
        attention_matrix = []
        for i in range(min(len(source.embedding), self.embedding_dim)):
            row = []
            for j in range(min(len(target.embedding), self.embedding_dim)):
                # Dot product attention
                score = source.embedding[i] * target.embedding[j]
                row.append(score)
            attention_matrix.append(row)

        # Softmax normalize each row
        attention_matrix = self._softmax_rows(attention_matrix)

        # Overall attention score (mean of max values per row)
        max_values = [max(row) for row in attention_matrix if row]
        attention_score = sum(max_values) / len(max_values) if max_values else 0.0

        return CrossModalAttention(
            source_modality=source.modality,
            target_modality=target.modality,
            attention_matrix=attention_matrix,
            attention_score=attention_score,
        )

    def _softmax_rows(self, matrix: list[list[float]]) -> list[list[float]]:
        """Apply softmax normalization to each row."""
        result = []
        for row in matrix:
            if not row:
                result.append([])
                continue
            max_val = max(row)
            exp_row = [math.exp(val - max_val) for val in row]
            sum_exp = sum(exp_row)
            if sum_exp == 0:
                result.append([0.0] * len(row))
            else:
                result.append([val / sum_exp for val in exp_row])
        return result


# ============================================================
# Multimodal Fusion
# ============================================================

class MultimodalFusion:
    """Fuses multiple modality embeddings into a unified representation.

    Uses attention-weighted fusion with learned gating mechanisms.
    In production: transformer-based fusion (like in LLaVA, Gemini).
    Prototype: confidence-weighted fusion with cross-modal attention.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.attention_module = CrossModalAttentionModule(embedding_dim)
        self._fusion_history: list[FusedRepresentation] = []

    def fuse(
        self, embeddings: list[ModalityEmbedding]
    ) -> FusedRepresentation:
        """Fuse multiple modality embeddings into unified representation."""
        if not embeddings:
            raise ValueError("No embeddings to fuse")

        if len(embeddings) == 1:
            return FusedRepresentation(
                embeddings=embeddings,
                fused_embedding=embeddings[0].embedding,
                fusion_weights={embeddings[0].modality.value: 1.0},
            )

        # Compute cross-modal attention
        attention_scores = {}
        for i, source in enumerate(embeddings):
            for j, target in enumerate(embeddings):
                if i != j:
                    attention = self.attention_module.compute_attention(source, target)
                    key = f"{source.modality.value}_to_{target.modality.value}"
                    attention_scores[key] = attention.attention_score

        # Compute fusion weights based on confidence and attention
        fusion_weights = self._compute_fusion_weights(embeddings, attention_scores)

        # Weighted fusion of embeddings
        fused_embedding = [0.0] * self.embedding_dim
        total_weight = sum(fusion_weights.values())

        if total_weight == 0:
            total_weight = 1.0

        for emb in embeddings:
            weight = fusion_weights.get(emb.modality.value, 0.0) / total_weight
            for i in range(min(len(emb.embedding), self.embedding_dim)):
                fused_embedding[i] += weight * emb.embedding[i]

        # Normalize fused embedding
        norm = math.sqrt(sum(x * x for x in fused_embedding))
        if norm > 0:
            fused_embedding = [x / norm for x in fused_embedding]

        result = FusedRepresentation(
            embeddings=embeddings,
            fused_embedding=fused_embedding,
            fusion_weights=fusion_weights,
        )
        self._fusion_history.append(result)
        return result

    def _compute_fusion_weights(
        self,
        embeddings: list[ModalityEmbedding],
        attention_scores: dict[str, float],
    ) -> dict[str, float]:
        """Compute fusion weights for each modality."""
        weights = {}
        for emb in embeddings:
            # Base weight from confidence
            base_weight = emb.confidence

            # Attention bonus: how much other modalities attend to this one
            attention_bonus = 0.0
            for key, score in attention_scores.items():
                if key.endswith(f"to_{emb.modality.value}"):
                    attention_bonus += score

            weights[emb.modality.value] = base_weight + 0.1 * attention_bonus

        return weights

    def get_fusion_stats(self) -> dict:
        """Get statistics on fusion operations."""
        if not self._fusion_history:
            return {"count": 0, "avg_modalities": 0.0}

        total_modalities = sum(len(f.embeddings) for f in self._fusion_history)
        return {
            "count": len(self._fusion_history),
            "avg_modalities": total_modalities / len(self._fusion_history),
            "total_fusions": len(self._fusion_history),
        }


# ============================================================
# Multimodal Processor — Main Interface
# ============================================================

class MultimodalProcessor:
    """High-level multimodal processing interface.

    Coordinates encoding, attention, and fusion across modalities.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.encoders: dict[Modality, ModalityEncoder] = {}
        self.fusion = MultimodalFusion(embedding_dim)
        self._processed_count = 0

        # Register default encoders
        self.register_encoder(SimpleVisionEncoder(embedding_dim))
        self.register_encoder(SimpleAudioEncoder(embedding_dim))
        self.register_encoder(SimpleTextEncoder(embedding_dim))

    def register_encoder(self, encoder: ModalityEncoder) -> None:
        """Register an encoder for a modality."""
        self.encoders[encoder.modality] = encoder

    def process(
        self, inputs: dict[Modality, Any]
    ) -> FusedRepresentation:
        """Process multiple modality inputs and fuse them."""
        embeddings = []
        for modality, data in inputs.items():
            encoder = self.encoders.get(modality)
            if encoder is None:
                continue
            emb = encoder.encode(data)
            embeddings.append(emb)

        if not embeddings:
            raise ValueError("No valid modality inputs")

        self._processed_count += 1
        return self.fusion.fuse(embeddings)

    def process_single(self, modality: Modality, data: Any) -> ModalityEmbedding:
        """Process a single modality input."""
        encoder = self.encoders.get(modality)
        if encoder is None:
            raise ValueError(f"No encoder for modality: {modality}")
        self._processed_count += 1
        return encoder.encode(data)

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "total_processed": self._processed_count,
            "registered_modalities": [m.value for m in self.encoders.keys()],
            "fusion_stats": self.fusion.get_fusion_stats(),
        }


# ============================================================
# Demo
# ============================================================

def main():
    """Run multimodal processing demo."""
    print("=" * 60)
    print("  Multimodal Processing Demo")
    print("=" * 60)
    print()

    processor = MultimodalProcessor(embedding_dim=128)

    # Demo 1: Single modality
    print("[1] Processing single modality (text)...")
    text_emb = processor.process_single(
        Modality.TEXT, "The cat sat on the mat."
    )
    print(f"    Modality: {text_emb.modality.value}")
    print(f"    Embedding dim: {len(text_emb.embedding)}")
    print(f"    Confidence: {text_emb.confidence:.2f}")
    print(f"    Features: {list(text_emb.features.keys())}")
    print()

    # Demo 2: Multimodal fusion
    print("[2] Fusing multimodal inputs...")
    inputs = {
        Modality.VISION: {
            "edges": [0.1, 0.2, 0.3],
            "colors": ["red", "blue"],
            "shapes": ["circle", "square"],
        },
        Modality.AUDIO: {
            "spectrogram": [[0.1, 0.2], [0.3, 0.4]],
            "pitch": 440.0,
            "rhythm": [1.0, 0.5, 1.0],
        },
        Modality.TEXT: "A red circle with a blue square.",
    }
    fused = processor.process(inputs)
    print(f"    Modalities fused: {[e.modality.value for e in fused.embeddings]}")
    print(f"    Fused embedding dim: {len(fused.fused_embedding)}")
    print(f"    Fusion weights: {fused.fusion_weights}")
    print()

    # Demo 3: Cross-modal attention
    print("[3] Computing cross-modal attention...")
    vision_emb = processor.process_single(
        Modality.VISION, {"edges": [0.1, 0.2], "colors": ["red"]}
    )
    text_emb2 = processor.process_single(
        Modality.TEXT, "A red object in the scene."
    )
    attention = processor.fusion.attention_module.compute_attention(
        vision_emb, text_emb2
    )
    print(f"    {attention.source_modality.value} -> {attention.target_modality.value}")
    print(f"    Attention score: {attention.attention_score:.4f}")
    print(f"    Attention matrix shape: {len(attention.attention_matrix)}x{len(attention.attention_matrix[0]) if attention.attention_matrix else 0}")
    print()

    # Demo 4: Stats
    print("[4] Processing stats:")
    stats = processor.get_stats()
    print(f"    Total processed: {stats['total_processed']}")
    print(f"    Registered modalities: {stats['registered_modalities']}")
    print(f"    Fusion stats: {stats['fusion_stats']}")
    print()

    print("=" * 60)
    print("  Multimodal Processing Demo — Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
