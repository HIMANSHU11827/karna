"""
RT-ALM: Response Generator
============================

Retrieval + perturbation approach (NOT autoregressive).

Based on predictive coding: the system retrieves a prototype response
from memory and perturbs it to fit the current context.

This is fundamentally different from GPT-style autoregressive generation.
"""
import numpy as np
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PrototypeResponse:
    """A stored response prototype with its context SDR."""
    context_sdr: np.ndarray
    response_text: str
    response_sdr: np.ndarray
    count: int = 1


class ResponseGenerator:
    """
    Generates responses by retrieving similar prototypes and perturbing them.
    
    Algorithm:
    1. Encode input to SDR
    2. Find top-k similar prototypes from episodic memory
    3. Retrieve best matching prototype
    4. Perturb response based on current context
    5. Apply learned style/transformation rules
    """
    
    def __init__(self, dim: int = 10000, max_prototypes: int = 10000):
        self.dim = dim
        self.max_prototypes = max_prototypes
        self.prototypes: List[PrototypeResponse] = []
        self.transformation_rules: Dict[str, str] = {}  # pattern → replacement
        self.style_vocabulary: Dict[str, List[str]] = {}  # style → word choices
    
    def store_prototype(self, context_sdr: np.ndarray, response_text: str,
                        response_sdr: np.ndarray) -> None:
        """Store a response prototype."""
        # Check for existing similar prototype
        for proto in self.prototypes:
            overlap = np.sum(context_sdr & proto.context_sdr)
            if overlap > 0.8 * np.sum(context_sdr):
                # Average the prototype
                proto.count += 1
                proto.context_sdr = np.maximum(proto.context_sdr, context_sdr)
                return
        
        if len(self.prototypes) >= self.max_prototypes:
            # Remove least used prototype
            min_count = min(p.count for p in self.prototypes)
            for i, p in enumerate(self.prototypes):
                if p.count == min_count:
                    self.prototypes.pop(i)
                    break
        
        self.prototypes.append(PrototypeResponse(
            context_sdr=context_sdr.copy(),
            response_text=response_text,
            response_sdr=response_sdr.copy(),
            count=1
        ))
    
    def find_best_prototype(self, context_sdr: np.ndarray,
                           top_k: int = 5) -> List[Tuple[float, PrototypeResponse]]:
        """Find top-k similar prototypes."""
        if not self.prototypes:
            return []
        
        similarities = []
        for proto in self.prototypes:
            overlap = np.sum(context_sdr & proto.context_sdr)
            k = np.sum(context_sdr)
            if k > 0:
                similarity = overlap / k
            else:
                similarity = 0
            similarities.append((similarity, proto))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
    
    def perturb_response(self, prototype: PrototypeResponse,
                        context_sdr: np.ndarray,
                        perturbation_strength: float = 0.1) -> str:
        """
        Perturb a prototype response to fit current context.
        
        The perturbation is guided by:
        1. Context overlap (how similar is current context to prototype)
        2. Learned transformation rules
        3. Style vocabulary
        """
        response = prototype.response_text
        
        # Apply transformation rules
        for pattern, replacement in self.transformation_rules.items():
            if pattern in response:
                # Check if context suggests this replacement
                rule_sdr = self._hash_string(replacement)
                overlap = np.sum(context_sdr & rule_sdr)
                k = np.sum(context_sdr)
                if k > 0 and overlap / k > 0.5:
                    response = response.replace(pattern, replacement)
        
        return response
    
    def _hash_string(self, text: str) -> np.ndarray:
        """Hash string to SDR."""
        sdr = np.zeros(self.dim, dtype=np.int8)
        padded = f"#{text.lower()}#"
        for n in [2, 3, 4]:
            for i in range(len(padded) - n + 1):
                ngram = padded[i:i+n]
                h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
                idx = h % self.dim
                sdr[idx] = 1
        return sdr
    
    def generate(self, context_sdr: np.ndarray,
                response_sdr: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Generate response using retrieval + perturbation.
        
        Args:
            context_sdr: Input context as SDR
            response_sdr: Optional target response SDR (for guided generation)
            
        Returns:
            Generated response text or None
        """
        best_protos = self.find_best_prototype(context_sdr, top_k=5)
        
        if not best_protos:
            return None
        
        best_similarity, best_proto = best_protos[0]
        
        # If we have a high-confidence match, use it directly
        if best_similarity > 0.9:
            return best_proto.response_text
        
        # Otherwise, perturb based on context
        response = self.perturb_response(best_proto, context_sdr)
        
        return response
    
    def learn_transformation(self, original_pattern: str,
                           replacement: str,
                           context_sdr: np.ndarray) -> None:
        """Learn a transformation rule from context."""
        self.transformation_rules[original_pattern] = replacement
    
    def learn_style(self, style: str, words: List[str]) -> None:
        """Learn style vocabulary for response variation."""
        if style not in self.style_vocabulary:
            self.style_vocabulary[style] = []
        self.style_vocabulary[style].extend(words)


class TemplateResponseGenerator(ResponseGenerator):
    """
    Response generator with template-based perturbation.
    
    Uses response templates with slots that can be filled based on context.
    """
    
    def __init__(self, dim: int = 10000, max_prototypes: int = 10000):
        super().__init__(dim, max_prototypes)
        self.templates: Dict[str, str] = {}  # intent → template
        self.slot_values: Dict[str, List[str]] = {}  # slot → possible values
    
    def register_template(self, intent: str, template: str) -> None:
        """Register a response template for an intent."""
        self.templates[intent] = template
    
    def register_slot_value(self, slot: str, values: List[str]) -> None:
        """Register possible values for a slot."""
        self.slot_values[slot] = values
    
    def generate(self, context_sdr: np.ndarray,
                intent: Optional[str] = None,
                response_sdr: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Generate response using template filling.
        
        If intent is known, use template. Otherwise, use retrieval.
        """
        if intent and intent in self.templates:
            template = self.templates[intent]
            # Fill slots based on context
            response = self._fill_slots(template, context_sdr)
            return response
        
        # Fall back to retrieval + perturbation
        return super().generate(context_sdr, response_sdr)
    
    def _fill_slots(self, template: str, context_sdr: np.ndarray) -> str:
        """Fill template slots based on context SDR."""
        response = template
        
        # Simple slot filling: replace {slot} with best matching value
        for slot, values in self.slot_values.items():
            placeholder = f"{{{slot}}}"
            if placeholder in response:
                # Find best value for this slot given context
                best_value = self._select_best_value(values, context_sdr)
                response = response.replace(placeholder, best_value)
        
        return response
    
    def _select_best_value(self, values: List[str],
                          context_sdr: np.ndarray) -> str:
        """Select best value for a slot given context."""
        best_value = values[0]
        best_overlap = 0
        
        for value in values:
            value_sdr = self._hash_string(value)
            overlap = np.sum(context_sdr & value_sdr)
            if overlap > best_overlap:
                best_overlap = overlap
                best_value = value
        
        return best_value
