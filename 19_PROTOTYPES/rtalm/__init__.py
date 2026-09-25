"""
RT-ALM — Real-Time Adaptive Language Model
=============================================

A unified architecture for real-time adaptive language processing.
"""

from .rtalm import (
    SDREncoder,
    kwta,
    AttractorMemory,
    EpisodicMemory,
    OnlineLearningRule,
    ResponseRetriever,
    MultimodalEncoder,
    RTALM,
    compute_overlap,
    compute_similarity,
    decode_sdr,
)

__all__ = [
    'SDREncoder',
    'kwta',
    'AttractorMemory',
    'EpisodicMemory',
    'OnlineLearningRule',
    'ResponseRetriever',
    'MultimodalEncoder',
    'RTALM',
    'compute_overlap',
    'compute_similarity',
    'decode_sdr',
]
