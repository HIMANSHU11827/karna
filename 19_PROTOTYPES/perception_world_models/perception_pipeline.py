"""
Perception Pipeline — Modality-Agnostic Sensory Processing
=============================================================

A unified perception system that processes multiple sensory modalities
(vision, audio, text, tactile, etc.) into structured representations.

Architecture:
- Sensory encoders convert raw inputs to latent representations
- Modality-specific preprocessing
- Cross-modal alignment
- Unified representation space

"""

from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# Core Data Structures
# ============================================================

class Modality(Enum):
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    TACTILE = "tactile"
    PROPRIOCEPTION = "proprioception"
    OLFACTION = "olfaction"
    GUSTATION = "gustation"


@dataclass
class SensoryInput:
    """Raw sensory input from any modality."""
    modality: Modality
    data: Any
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    source_id: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "modality": self.modality.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "source_id": self.source_id,
            "data_type": type(self.data).__name__,
        }


@dataclass
class PerceptualRepresentation:
    """Processed representation of sensory input."""
    modality: Modality
    embedding: list[float]
    features: dict[str, Any]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    source_input: SensoryInput | None = None

    def to_dict(self) -> dict:
        return {
            "modality": self.modality.value,
            "embedding_dim": len(self.embedding),
            "embedding_sample": self.embedding[:5],
            "features": self.features,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class CrossModalAlignment:
    """Alignment between representations from different modalities."""
    source_modality: Modality
    target_modality: Modality
    alignment_score: float
    mapping: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Sensory Encoders
# ============================================================

class SensoryEncoder(ABC):
    """Base class for modality-specific sensory encoders."""

    def __init__(self, modality: Modality, embedding_dim: int = 128):
        self.modality = modality
        self.embedding_dim = embedding_dim
        self._is_trained = False

    @abstractmethod
    def encode(self, data: Any) -> PerceptualRepresentation:
        """Encode raw sensory data into a perceptual representation."""
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess raw data before encoding."""
        pass

    @property
    def is_trained(self) -> bool:
        return self._is_trained


class VisionEncoder(SensoryEncoder):
    """Encoder for visual sensory input.

    In production, this would use a vision transformer or CNN.
    For the prototype, we simulate with feature extraction.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__(Modality.VISION, embedding_dim)
        self.feature_extractors = [
            "edges", "colors", "shapes", "textures", "objects", "depth"
        ]

    def preprocess(self, data: Any) -> dict:
        """Preprocess visual data (normalize, resize, etc.)."""
        if isinstance(data, dict):
            return {
                "normalized": True,
                "resolution": data.get("resolution", (224, 224)),
                "channels": data.get("channels", 3),
                "pixel_values": data.get("pixels", []),
            }
        return {"normalized": True, "raw": str(data)[:200]}

    def encode(self, data: Any) -> PerceptualRepresentation:
        """Encode visual input into perceptual representation."""
        preprocessed = self.preprocess(data)

        # Simulate feature extraction with structured embeddings
        # In production: ViT, CNN, or multimodal transformer
        embedding = self._generate_embedding(preprocessed)

        features = {
            "detected_objects": self._detect_objects(preprocessed),
            "color_histogram": self._extract_colors(preprocessed),
            "spatial_layout": self._analyze_spatial(preprocessed),
            "motion_vectors": self._estimate_motion(preprocessed),
        }

        return PerceptualRepresentation(
            modality=Modality.VISION,
            embedding=embedding,
            features=features,
            confidence=0.92,
            source_input=data if isinstance(data, SensoryInput) else None,
        )

    def _generate_embedding(self, preprocessed: dict) -> list[float]:
        """Generate embedding vector from preprocessed data."""
        # Prototype: deterministic pseudo-embedding based on input
        seed = hash(str(preprocessed)) % 10000
        return [math.sin(seed + i * 0.1) * 0.5 + 0.5 for i in range(self.embedding_dim)]

    def _detect_objects(self, preprocessed: dict) -> list[dict]:
        """Detect objects in visual input."""
        return [
            {"label": "prototype_object", "confidence": 0.85, "bbox": [0.1, 0.2, 0.5, 0.8]},
        ]

    def _extract_colors(self, preprocessed: dict) -> dict:
        """Extract dominant colors."""
        return {"dominant": ["#3498db", "#2ecc71", "#e74c3c"], "palette_size": 3}

    def _analyze_spatial(self, preprocessed: dict) -> dict:
        """Analyze spatial layout."""
        return {"depth_layers": 3, "salient_regions": 5, "composition": "centered"}

    def _estimate_motion(self, preprocessed: dict) -> list[float]:
        """Estimate motion vectors."""
        return [0.0, 0.0, 0.0]  # [dx, dy, magnitude]


class AudioEncoder(SensoryEncoder):
    """Encoder for auditory sensory input."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__(Modality.AUDIO, embedding_dim)
        self.feature_extractors = [
            "spectrogram", "mfcc", "pitch", "rhythm", "timbre", "phonemes"
        ]

    def preprocess(self, data: Any) -> dict:
        """Preprocess audio data (resample, normalize, segment)."""
        if isinstance(data, dict):
            return {
                "sample_rate": data.get("sample_rate", 16000),
                "channels": data.get("channels", 1),
                "duration": data.get("duration", 1.0),
                "waveform": data.get("waveform", []),
            }
        return {"sample_rate": 16000, "duration": 1.0, "raw": str(data)[:200]}

    def encode(self, data: Any) -> PerceptualRepresentation:
        """Encode audio input into perceptual representation."""
        preprocessed = self.preprocess(data)
        embedding = self._generate_embedding(preprocessed)

        features = {
            "transcription": self._transcribe(preprocessed),
            "speaker_id": self._identify_speaker(preprocessed),
            "emotion": self._detect_emotion(preprocessed),
            "sound_events": self._detect_events(preprocessed),
        }

        return PerceptualRepresentation(
            modality=Modality.AUDIO,
            embedding=embedding,
            features=features,
            confidence=0.88,
            source_input=data if isinstance(data, SensoryInput) else None,
        )

    def _generate_embedding(self, preprocessed: dict) -> list[float]:
        """Generate embedding from preprocessed audio."""
        seed = hash(str(preprocessed)) % 10000
        return [math.cos(seed + i * 0.15) * 0.5 + 0.5 for i in range(self.embedding_dim)]

    def _transcribe(self, preprocessed: dict) -> str:
        """Transcribe speech (placeholder)."""
        return "[speech transcription placeholder]"

    def _identify_speaker(self, preprocessed: dict) -> dict:
        """Identify speaker."""
        return {"speaker_id": "unknown", "confidence": 0.0}

    def _detect_emotion(self, preprocessed: dict) -> dict:
        """Detect emotional content."""
        return {"emotion": "neutral", "valence": 0.0, "arousal": 0.5}

    def _detect_events(self, preprocessed: dict) -> list[str]:
        """Detect sound events."""
        return []


class TextEncoder(SensoryEncoder):
    """Encoder for textual/language input."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__(Modality.TEXT, embedding_dim)
        self.feature_extractors = [
            "tokens", "entities", "sentiment", "intent", "semantics", "syntax"
        ]

    def preprocess(self, data: Any) -> dict:
        """Preprocess text data."""
        if isinstance(data, str):
            return {
                "raw_text": data,
                "tokens": data.lower().split(),
                "length": len(data),
                "language": "en",
            }
        return {"raw_text": str(data)[:200], "tokens": [], "length": 0}

    def encode(self, data: Any) -> PerceptualRepresentation:
        """Encode text input into perceptual representation."""
        preprocessed = self.preprocess(data)
        embedding = self._generate_embedding(preprocessed)

        features = {
            "entities": self._extract_entities(preprocessed),
            "sentiment": self._analyze_sentiment(preprocessed),
            "intent": self._detect_intent(preprocessed),
            "key_phrases": self._extract_key_phrases(preprocessed),
        }

        return PerceptualRepresentation(
            modality=Modality.TEXT,
            embedding=embedding,
            features=features,
            confidence=0.95,
            source_input=data if isinstance(data, SensoryInput) else None,
        )

    def _generate_embedding(self, preprocessed: dict) -> list[float]:
        """Generate embedding from preprocessed text."""
        seed = hash(str(preprocessed)) % 10000
        return [math.sin(seed + i * 0.08) * 0.5 + 0.5 for i in range(self.embedding_dim)]

    def _extract_entities(self, preprocessed: dict) -> list[dict]:
        """Extract named entities."""
        return []

    def _analyze_sentiment(self, preprocessed: dict) -> dict:
        """Analyze sentiment."""
        return {"polarity": 0.0, "subjectivity": 0.5, "label": "neutral"}

    def _detect_intent(self, preprocessed: dict) -> str:
        """Detect user intent."""
        return "unknown"

    def _extract_key_phrases(self, preprocessed: dict) -> list[str]:
        """Extract key phrases."""
        tokens = preprocessed.get("tokens", [])
        return tokens[:5] if tokens else []


# ============================================================
# Perception Pipeline
# ============================================================

class PerceptionPipeline:
    """Unified perception pipeline that processes multiple modalities.

    Routes sensory inputs to appropriate encoders, collects representations,
    and maintains a coherent multi-modal world state.

    Usage:
        pipeline = PerceptionPipeline()
        pipeline.register_encoder(VisionEncoder())
        pipeline.register_encoder(AudioEncoder())

        # Process single input
        result = pipeline.process(SensoryInput(
            modality=Modality.VISION,
            data={"pixels": [...], "resolution": (224, 224)}
        ))

        # Process batch
        results = pipeline.process_batch([
            SensoryInput(Modality.VISION, vision_data),
            SensoryInput(Modality.AUDIO, audio_data),
            SensoryInput(Modality.TEXT, "hello world"),
        ])
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.encoders: dict[Modality, SensoryEncoder] = {}
        self.representation_history: list[PerceptualRepresentation] = []
        self.cross_modal_alignments: list[CrossModalAlignment] = []
        self._processed_count = 0

        # Register default encoders
        self.register_encoder(VisionEncoder(embedding_dim))
        self.register_encoder(AudioEncoder(embedding_dim))
        self.register_encoder(TextEncoder(embedding_dim))

    def register_encoder(self, encoder: SensoryEncoder) -> None:
        """Register a sensory encoder for a modality."""
        self.encoders[encoder.modality] = encoder

    def process(self, input_data: SensoryInput | Any) -> PerceptualRepresentation:
        """Process a single sensory input."""
        if not isinstance(input_data, SensoryInput):
            input_data = SensoryInput(
                modality=self._infer_modality(input_data),
                data=input_data,
            )

        encoder = self.encoders.get(input_data.modality)
        if encoder is None:
            raise ValueError(f"No encoder registered for modality: {input_data.modality}")

        representation = encoder.encode(input_data)
        self.representation_history.append(representation)
        self._processed_count += 1

        return representation

    def process_batch(self, inputs: list[SensoryInput]) -> list[PerceptualRepresentation]:
        """Process multiple sensory inputs as a batch."""
        results = []
        for inp in inputs:
            try:
                result = self.process(inp)
                results.append(result)
            except ValueError as e:
                print(f"Warning: Skipping input — {e}")
        return results

    def cross_modal_fusion(
        self, representations: list[PerceptualRepresentation]
    ) -> PerceptualRepresentation:
        """Fuse representations from multiple modalities into a unified representation.

        Uses attention-weighted fusion based on confidence scores.
        """
        if not representations:
            raise ValueError("No representations to fuse")

        if len(representations) == 1:
            return representations[0]

        # Weighted average of embeddings based on confidence
        total_confidence = sum(r.confidence for r in representations)
        if total_confidence == 0:
            total_confidence = 1.0

        fused_embedding = [0.0] * self.embedding_dim
        for r in representations:
            weight = r.confidence / total_confidence
            for i in range(min(len(r.embedding), self.embedding_dim)):
                fused_embedding[i] += weight * r.embedding[i]

        # Collect all features
        all_features = {}
        for r in representations:
            all_features[r.modality.value] = r.features

        avg_confidence = total_confidence / len(representations)

        return PerceptualRepresentation(
            modality=Modality.PROPRIOCEPTION,  # Fused = integrated perception
            embedding=fused_embedding,
            features={"fused_modalities": [r.modality.value for r in representations],
                       "modal_features": all_features},
            confidence=avg_confidence,
        )

    def get_world_state(self) -> dict:
        """Get the current aggregated world state from all processed perceptions."""
        if not self.representation_history:
            return {"status": "empty", "modalities_seen": []}

        # Group by modality
        by_modality: dict[str, list[dict]] = {}
        for r in self.representation_history:
            mod = r.modality.value
            if mod not in by_modality:
                by_modality[mod] = []
            by_modality[mod].append(r.to_dict())

        return {
            "total_perceptions": self._processed_count,
            "modalities_seen": list(by_modality.keys()),
            "modalities_count": len(by_modality),
            "per_modality": {
                mod: {"count": len(reps), "latest": reps[-1] if reps else None}
                for mod, reps in by_modality.items()
            },
        }

    def compute_alignment(
        self, source: PerceptualRepresentation, target: PerceptualRepresentation
    ) -> CrossModalAlignment:
        """Compute alignment between representations from different modalities."""
        # Cosine similarity between embeddings
        similarity = self._cosine_similarity(source.embedding, target.embedding)

        alignment = CrossModalAlignment(
            source_modality=source.modality,
            target_modality=target.modality,
            alignment_score=similarity,
            mapping={"method": "cosine_similarity", "dimensions": self.embedding_dim},
        )
        self.cross_modal_alignments.append(alignment)
        return alignment

    def _infer_modality(self, data: Any) -> Modality:
        """Infer modality from data type."""
        if isinstance(data, dict):
            if "pixels" in data or "resolution" in data:
                return Modality.VISION
            if "waveform" in data or "sample_rate" in data:
                return Modality.AUDIO
        if isinstance(data, str):
            return Modality.TEXT
        return Modality.PROPRIENCEPTION

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ============================================================
# Demo
# ============================================================

def main():
    """Run perception pipeline demo."""
    print("=" * 60)
    print("  Perception Pipeline Demo")
    print("=" * 60)
    print()

    pipeline = PerceptionPipeline(embedding_dim=128)

    # Demo 1: Vision
    print("[1] Processing vision input...")
    vision_data = {
        "pixels": [[[0.5] * 3] * 224] * 224,
        "resolution": (224, 224),
        "channels": 3,
    }
    vision_input = SensoryInput(
        modality=Modality.VISION,
        data=vision_data,
        metadata={"source": "camera_0", "fps": 30},
    )
    vision_result = pipeline.process(vision_input)
    print(f"    Modality: {vision_result.modality.value}")
    print(f"    Embedding dim: {len(vision_result.embedding)}")
    print(f"    Confidence: {vision_result.confidence:.2f}")
    print(f"    Features: {list(vision_result.features.keys())}")
    print()

    # Demo 2: Audio
    print("[2] Processing audio input...")
    audio_data = {
        "waveform": [0.1, 0.3, -0.2, 0.5] * 100,
        "sample_rate": 16000,
        "duration": 1.0,
    }
    audio_result = pipeline.process(audio_data)
    print(f"    Modality: {audio_result.modality.value}")
    print(f"    Embedding dim: {len(audio_result.embedding)}")
    print(f"    Confidence: {audio_result.confidence:.2f}")
    print(f"    Features: {list(audio_result.features.keys())}")
    print()

    # Demo 3: Text
    print("[3] Processing text input...")
    text_result = pipeline.process("Hello, this is a test of the perception system.")
    print(f"    Modality: {text_result.modality.value}")
    print(f"    Embedding dim: {len(text_result.embedding)}")
    print(f"    Confidence: {text_result.confidence:.2f}")
    print(f"    Features: {list(text_result.features.keys())}")
    print()

    # Demo 4: Cross-modal fusion
    print("[4] Fusing multimodal representations...")
    fused = pipeline.cross_modal_fusion([vision_result, audio_result, text_result])
    print(f"    Fused embedding dim: {len(fused.embedding)}")
    print(f"    Fused confidence: {fused.confidence:.2f}")
    print(f"    Modalities fused: {fused.features.get('fused_modalities', [])}")
    print()

    # Demo 5: Cross-modal alignment
    print("[5] Computing cross-modal alignment...")
    alignment = pipeline.compute_alignment(vision_result, text_result)
    print(f"    {alignment.source_modality.value} <-> {alignment.target_modality.value}")
    print(f"    Alignment score: {alignment.alignment_score:.4f}")
    print()

    # Demo 6: World state
    print("[6] Current world state:")
    state = pipeline.get_world_state()
    print(f"    Total perceptions: {state['total_perceptions']}")
    print(f"    Modalities seen: {state['modalities_seen']}")
    for mod, info in state.get("per_modality", {}).items():
        print(f"      {mod}: {info['count']} perceptions")
    print()

    print("=" * 60)
    print("  Perception Pipeline Demo — Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
