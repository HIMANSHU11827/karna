"""
AGI Research Lab - Memory System Prototype
==========================================

A modular memory system for AI agents with four memory types:
- WorkingMemory: Short-term context buffer with attention-based retrieval
- EpisodicMemory: Long-term experience storage with temporal indexing
- SemanticMemory: Conceptual knowledge with graph-based relationships
- ProceduralMemory: How-to knowledge with skill composition

Architecture inspired by human memory systems (Tulving, Squire) and
modern AI memory approaches (MemGPT, Generative Agents).
"""

import time
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict
from pathlib import Path
import heapq


# =============================================================================
# Base Classes
# =============================================================================

@dataclass
class MemoryTrace:
    """Base unit of memory storage."""
    content: Any
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    trace_id: str = ""
    tags: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = hashlib.md5(
                f"{self.content}{self.timestamp}".encode()
            ).hexdigest()[:12]

    def access(self):
        """Record an access event."""
        self.access_count += 1
        self.last_accessed = time.time()

    def decay(self, current_time: Optional[float] = None) -> float:
        """
        Calculate memory strength with Ebbinghaus-inspired decay.
        Returns value between 0 and 1.
        """
        if current_time is None:
            current_time = time.time()
        age_hours = (current_time - self.timestamp) / 3600
        # forgetting curve: R = e^(-t/S) where S is stability
        stability = 1 + self.importance * 10 + self.access_count * 0.5
        strength = math.exp(-age_hours / stability)
        return strength

    def relevance_score(self, current_time: Optional[float] = None) -> float:
        """
        Combined score for retrieval priority.
        Balances recency, importance, and frequency of access.
        """
        if current_time is None:
            current_time = time.time()
        decay_score = self.decay(current_time)
        # Frequency bonus (log scale to prevent runaway)
        freq_bonus = math.log1p(self.access_count) * 0.1
        return decay_score * (self.importance + freq_bonus)


# =============================================================================
# Working Memory
# =============================================================================

class WorkingMemory:
    """
    Short-term working memory with limited capacity.
    Implements attention-based slot management.
    Capacity follows Miller's law: 7±2 items.
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.slots: list[MemoryTrace] = []

    def add(self, content: Any, importance: float = 0.5, tags: list = None) -> MemoryTrace:
        """Add item to working memory, evicting lowest priority if full."""
        trace = MemoryTrace(
            content=content,
            importance=importance,
            tags=tags or []
        )
        self.slots.append(trace)
        # Evict if over capacity
        if len(self.slots) > self.capacity:
            self._evict()
        return trace

    def _evict(self):
        """Remove lowest priority item."""
        if not self.slots:
            return
        min_idx = min(range(len(self.slots)), key=lambda i: self.slots[i].relevance_score())
        self.slots.pop(min_idx)

    def retrieve(self, query: str = None, k: int = 5) -> list[MemoryTrace]:
        """Retrieve top-k items from working memory."""
        for trace in self.slots:
            trace.access()
        # Return by relevance
        sorted_slots = sorted(self.slots, key=lambda t: t.relevance_score(), reverse=True)
        return sorted_slots[:k]

    def clear(self):
        """Flush working memory."""
        self.slots.clear()

    @property
    def size(self) -> int:
        return len(self.slots)

    @property
    def is_full(self) -> bool:
        return len(self.slots) >= self.capacity

    def to_list(self) -> list[dict]:
        """Serialize to list of dicts."""
        return [
            {
                "id": t.trace_id,
                "content": str(t.content)[:200],
                "importance": t.importance,
                "access_count": t.access_count
            }
            for t in self.slots
        ]


# =============================================================================
# Episodic Memory
# =============================================================================

class EpisodicMemory:
    """
    Long-term episodic memory for storing experiences and events.
    Supports temporal queries and context-based retrieval.
    Uses vector similarity for content matching (simplified with TF-IDF-like scoring).
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.traces: list[MemoryTrace] = []
        self.storage_path = storage_path
        self._index_by_time: list[tuple[float, int]] = []  # (timestamp, trace_idx)
        self._index_by_tag: dict[str, list[int]] = defaultdict(list)

    def store(self, content: Any, importance: float = 0.5,
              tags: list = None, metadata: dict = None) -> MemoryTrace:
        """Store a new episodic memory."""
        trace = MemoryTrace(
            content=content,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {}
        )
        idx = len(self.traces)
        self.traces.append(trace)
        # Update indexes
        self._index_by_time.append((trace.timestamp, idx))
        for tag in trace.tags:
            self._index_by_tag[tag].append(idx)
        return trace

    def retrieve(self, k: int = 5, query: str = None,
                 tags: list = None, time_range: tuple = None) -> list[MemoryTrace]:
        """
        Retrieve relevant episodic memories.
        Supports filtering by tags and time range.
        """
        current_time = time.time()
        candidates = list(range(len(self.traces)))

        # Filter by tags
        if tags:
            tag_sets = [set(self._index_by_tag.get(t, [])) for t in tags]
            if tag_sets:
                candidates = [i for i in candidates if any(i in s for s in tag_sets)]

        # Filter by time range
        if time_range:
            start, end = time_range
            candidates = [i for i in candidates
                         if start <= self.traces[i].timestamp <= end]

        # Score and rank
        scored = []
        for idx in candidates:
            trace = self.traces[idx]
            score = trace.relevance_score(current_time)
            # Boost for query match (simple substring matching for prototype)
            if query and query.lower() in str(trace.content).lower():
                score *= 2.0
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, idx in scored[:k]:
            self.traces[idx].access()
            results.append(self.traces[idx])
        return results

    def get_timeline(self, k: int = 20) -> list[dict]:
        """Return chronological timeline of memories."""
        sorted_traces = sorted(self.traces, key=lambda t: t.timestamp, reverse=True)
        return [
            {
                "id": t.trace_id,
                "timestamp": t.timestamp,
                "content": str(t.content)[:100],
                "importance": t.importance
            }
            for t in sorted_traces[:k]
        ]

    @property
    def size(self) -> int:
        return len(self.traces)

    def save_to_disk(self, path: Optional[str] = None):
        """Persist episodic memory to disk."""
        path = path or self.storage_path
        if not path:
            return
        data = [
            {
                "trace_id": t.trace_id,
                "content": t.content,
                "timestamp": t.timestamp,
                "importance": t.importance,
                "access_count": t.access_count,
                "last_accessed": t.last_accessed,
                "tags": t.tags,
                "metadata": t.metadata
            }
            for t in self.traces
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_disk(self, path: Optional[str] = None):
        """Load episodic memory from disk."""
        path = path or self.storage_path
        if not path or not Path(path).exists():
            return
        with open(path) as f:
            data = json.load(f)
        self.traces = []
        for item in data:
            trace = MemoryTrace(
                content=item["content"],
                timestamp=item["timestamp"],
                importance=item["importance"],
                trace_id=item["trace_id"],
                tags=item.get("tags", []),
                metadata=item.get("metadata", {})
            )
            trace.access_count = item.get("access_count", 0)
            trace.last_accessed = item.get("last_accessed", item["timestamp"])
            self.traces.append(trace)
        # Rebuild indexes
        self._index_by_time = [(t.timestamp, i) for i, t in enumerate(self.traces)]
        self._index_by_tag.clear()
        for i, t in enumerate(self.traces):
            for tag in t.tags:
                self._index_by_tag[tag].append(i)


# =============================================================================
# Semantic Memory
# =============================================================================

class SemanticMemory:
    """
    Semantic memory for storing concepts, facts, and their relationships.
    Uses a graph structure where nodes are concepts and edges are relationships.
    """

    def __init__(self):
        self.concepts: dict[str, dict] = {}  # concept_id -> concept data
        self.relationships: list[tuple[str, str, str, float]] = []  # (source, target, type, weight)

    def add_concept(self, name: str, description: str = "",
                    properties: dict = None, importance: float = 0.5) -> str:
        """Add a concept to semantic memory."""
        concept_id = hashlib.md5(name.encode()).hexdigest()[:12]
        self.concepts[concept_id] = {
            "id": concept_id,
            "name": name,
            "description": description,
            "properties": properties or {},
            "importance": importance,
            "created_at": time.time(),
            "access_count": 0
        }
        return concept_id

    def add_relationship(self, source_id: str, target_id: str,
                        rel_type: str, weight: float = 1.0):
        """Add a relationship between two concepts."""
        if source_id in self.concepts and target_id in self.concepts:
            self.relationships.append((source_id, target_id, rel_type, weight))

    def get_concept(self, concept_id: str) -> Optional[dict]:
        """Retrieve a concept by ID."""
        concept = self.concepts.get(concept_id)
        if concept:
            concept["access_count"] += 1
        return concept

    def find_by_name(self, name: str) -> list[dict]:
        """Find concepts by name (fuzzy matching for prototype)."""
        results = []
        name_lower = name.lower()
        for cid, concept in self.concepts.items():
            if name_lower in concept["name"].lower():
                results.append(concept)
        return results

    def get_related(self, concept_id: str, k: int = 5) -> list[dict]:
        """Get related concepts via graph traversal."""
        related = []
        for src, tgt, rel_type, weight in self.relationships:
            if src == concept_id:
                target = self.concepts.get(tgt)
                if target:
                    related.append({
                        "concept": target,
                        "relationship": rel_type,
                        "weight": weight
                    })
            elif tgt == concept_id:
                source = self.concepts.get(src)
                if source:
                    related.append({
                        "concept": source,
                        "relationship": f"inverse_{rel_type}",
                        "weight": weight
                    })
        related.sort(key=lambda x: x["weight"], reverse=True)
        return related[:k]

    def query(self, concept_name: str) -> dict:
        """Comprehensive query: concept + related concepts."""
        concepts = self.find_by_name(concept_name)
        if not concepts:
            return {"found": False, "concept": None, "related": []}
        concept = concepts[0]
        related = self.get_related(concept["id"])
        return {"found": True, "concept": concept, "related": related}

    @property
    def size(self) -> int:
        return len(self.concepts)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "concepts": self.concepts,
            "relationships": [
                {"source": s, "target": t, "type": r, "weight": w}
                for s, t, r, w in self.relationships
            ]
        }


# =============================================================================
# Procedural Memory
# =============================================================================

class ProceduralMemory:
    """
    Procedural memory for storing skills and procedures.
    Implements skill composition: complex skills built from simpler ones.
    """

    def __init__(self):
        self.skills: dict[str, dict] = {}

    def add_skill(self, name: str, steps: list[str],
                  preconditions: list[str] = None,
                  postconditions: list[str] = None,
                  subskills: list[str] = None,
                  difficulty: float = 0.5) -> str:
        """Add a skill to procedural memory."""
        skill_id = hashlib.md5(name.encode()).hexdigest()[:12]
        self.skills[skill_id] = {
            "id": skill_id,
            "name": name,
            "steps": steps,
            "preconditions": preconditions or [],
            "postconditions": postconditions or [],
            "subskills": subskills or [],
            "difficulty": difficulty,
            "success_count": 0,
            "fail_count": 0,
            "created_at": time.time(),
            "last_practiced": None
        }
        return skill_id

    def execute(self, skill_id: str) -> dict:
        """Simulate skill execution."""
        skill = self.skills.get(skill_id)
        if not skill:
            return {"success": False, "reason": "Skill not found"}

        # Check preconditions (simplified)
        for pre in skill["preconditions"]:
            if not self._check_precondition(pre):
                skill["fail_count"] += 1
                return {"success": False, "reason": f"Precondition failed: {pre}"}

        # Execute steps
        skill["success_count"] += 1
        skill["last_practiced"] = time.time()
        return {
            "success": True,
            "steps_executed": len(skill["steps"]),
            "steps": skill["steps"],
            "skill_name": skill["name"]
        }

    def _check_precondition(self, precondition: str) -> bool:
        """Check if a precondition is met (simplified)."""
        # In a real system, this would check environment state
        return True

    def get_skill(self, skill_id: str) -> Optional[dict]:
        """Retrieve a skill by ID."""
        return self.skills.get(skill_id)

    def find_skills(self, query: str) -> list[dict]:
        """Find skills by name."""
        results = []
        query_lower = query.lower()
        for sid, skill in self.skills.items():
            if query_lower in skill["name"].lower():
                results.append(skill)
        return results

    def get_mastery(self, skill_id: str) -> float:
        """Calculate mastery level (0 to 1) for a skill."""
        skill = self.skills.get(skill_id)
        if not skill:
            return 0.0
        total = skill["success_count"] + skill["fail_count"]
        if total == 0:
            return 0.0
        # Weighted by practice count (law of practice)
        success_rate = skill["success_count"] / total
        practice_factor = 1 - math.exp(-total / 10)  # asymptotic approach to 1
        return success_rate * practice_factor

    @property
    def size(self) -> int:
        return len(self.skills)


# =============================================================================
# Integrated Memory System
# =============================================================================

class MemorySystem:
    """
    Integrated memory system combining all memory types.
    Provides unified interface for storing and retrieving across memory types.
    Implements consolidation (working -> episodic -> semantic).
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.working = WorkingMemory(capacity=7)
        self.episodic = EpisodicMemory(storage_path=storage_path)
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.consolidation_threshold = 0.7  # importance threshold for consolidation

    def remember(self, content: Any, memory_type: str = "episodic",
                 importance: float = 0.5, tags: list = None,
                 metadata: dict = None) -> MemoryTrace:
        """
        Store a memory in the specified memory type.
        """
        if memory_type == "working":
            return self.working.add(content, importance, tags)
        elif memory_type == "episodic":
            return self.episodic.store(content, importance, tags, metadata)
        elif memory_type == "semantic":
            if isinstance(content, dict):
                cid = self.semantic.add_concept(
                    name=content.get("name", str(content)),
                    description=content.get("description", ""),
                    properties=content.get("properties", {}),
                    importance=importance
                )
                # Return a trace-like object for consistency
                trace = MemoryTrace(content=content, importance=importance, tags=tags)
                return trace
            else:
                cid = self.semantic.add_concept(name=str(content), importance=importance)
                trace = MemoryTrace(content=content, importance=importance, tags=tags)
                return trace
        elif memory_type == "procedural":
            if isinstance(content, dict):
                sid = self.procedural.add_skill(
                    name=content.get("name", str(content)),
                    steps=content.get("steps", []),
                    preconditions=content.get("preconditions", []),
                    postconditions=content.get("postconditions", []),
                    difficulty=content.get("difficulty", 0.5)
                )
                trace = MemoryTrace(content=content, importance=importance, tags=tags)
                return trace
            else:
                sid = self.procedural.add_skill(name=str(content), steps=[])
                trace = MemoryTrace(content=content, importance=importance, tags=tags)
                return trace
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    def recall(self, query: str = None, memory_type: str = "all",
               k: int = 5, tags: list = None) -> dict:
        """
        Recall memories across all or specified memory types.
        """
        results = {}
        if memory_type in ("working", "all"):
            results["working"] = self.working.retrieve(query=query, k=k)
        if memory_type in ("episodic", "all"):
            results["episodic"] = self.episodic.retrieve(query=query, k=k, tags=tags)
        if memory_type in ("semantic", "all"):
            if query:
                sem_results = self.semantic.find_by_name(query)
                results["semantic"] = sem_results[:k]
            else:
                results["semantic"] = []
        if memory_type in ("procedural", "all"):
            if query:
                proc_results = self.procedural.find_skills(query)
                results["procedural"] = proc_results[:k]
            else:
                results["procedural"] = []
        return results

    def consolidate(self):
        """
        Consolidate memories:
        - Move important working memories to episodic
        - Extract semantic knowledge from frequent episodic patterns
        """
        consolidated = {"working_to_episodic": 0}
        # Move important working memories to episodic
        for trace in self.working.slots:
            if trace.importance >= self.consolidation_threshold:
                self.episodic.store(
                    content=trace.content,
                    importance=trace.importance,
                    tags=trace.tags
                )
                consolidated["working_to_episodic"] += 1
        # Clear working memory after consolidation
        self.working.clear()
        return consolidated

    def get_stats(self) -> dict:
        """Get statistics about the memory system."""
        return {
            "working_memory": {
                "size": self.working.size,
                "capacity": self.working.capacity,
                "utilization": self.working.size / self.working.capacity
            },
            "episodic_memory": {
                "size": self.episodic.size,
                "total_accesses": sum(t.access_count for t in self.episodic.traces)
            },
            "semantic_memory": {
                "size": self.semantic.size,
                "relationships": len(self.semantic.relationships)
            },
            "procedural_memory": {
                "size": self.procedural.size,
                "avg_mastery": sum(
                    self.procedural.get_mastery(sid)
                    for sid in self.procedural.skills
                ) / max(len(self.procedural.skills), 1)
            }
        }


# =============================================================================
# Demo & Tests
# =============================================================================

def run_demo():
    """Demonstrate the memory system capabilities."""
    print("=" * 60)
    print("AGI Research Lab - Memory System Prototype Demo")
    print("=" * 60)

    # Initialize memory system
    memory = MemorySystem(storage_path="/tmp/agi_memory_test.json")

    # 1. Working Memory
    print("\n--- Working Memory ---")
    memory.remember("Current task: debug authentication bug", "working", importance=0.8)
    memory.remember("User reported 403 error on login", "working", importance=0.9)
    memory.remember("API rate limit is 100 req/min", "working", importance=0.6)
    memory.remember("Database connection pool exhausted", "working", importance=0.7)
    memory.remember("Need to check OAuth token expiry", "working", importance=0.85)
    print(f"Working memory size: {memory.working.size}/{memory.working.capacity}")
    print(f"Working memory full: {memory.working.is_full}")

    # Add more to trigger eviction
    memory.remember("Cache hit ratio is low", "working", importance=0.5)
    memory.remember("SSL cert expires in 30 days", "working", importance=0.4)
    memory.remember("Deploy pending for v2.1", "working", importance=0.6)
    print(f"After overflow - size: {memory.working.size} (should be <= {memory.working.capacity})")

    # Retrieve from working memory
    working_results = memory.recall(query="debug", memory_type="working", k=3)
    print(f"Retrieved {len(working_results['working'])} working memories for 'debug'")

    # 2. Episodic Memory
    print("\n--- Episodic Memory ---")
    memory.remember(
        "Successfully deployed microservice architecture on 2024-01-15",
        "episodic",
        importance=0.8,
        tags=["deployment", "milestone"],
        metadata={"team": "platform", "duration_hours": 6}
    )
    memory.remember(
        "Database migration failed due to constraint violation",
        "episodic",
        importance=0.9,
        tags=["database", "failure"],
        metadata={"rollback": True}
    )
    memory.remember(
        "Implemented new caching layer - 3x performance improvement",
        "episodic",
        importance=0.7,
        tags=["performance", "caching"],
        metadata={"improvement_factor": 3}
    )
    memory.remember(
        "Security audit found 2 critical vulnerabilities",
        "episodic",
        importance=0.95,
        tags=["security", "audit"],
        metadata={"critical_count": 2, "patched": True}
    )

    # Retrieve episodic memories
    episodic_results = memory.recall(query="database", memory_type="episodic", k=3)
    print(f"Episodic memories for 'database': {len(episodic_results['episodic'])}")
    for trace in episodic_results['episodic']:
        print(f"  - {str(trace.content)[:60]}... (importance: {trace.importance})")

    # 3. Semantic Memory
    print("\n--- Semantic Memory ---")
    memory.remember(
        {"name": "Neural Network", "description": "Computational model inspired by biological neural networks",
         "properties": {"type": "architecture", " trainable": True}},
        "semantic", importance=0.9
    )
    memory.remember(
        {"name": "Transformer", "description": "Attention-based neural architecture for sequence processing",
         "properties": {"type": "architecture", "attention_based": True}},
        "semantic", importance=0.95
    )
    memory.remember(
        {"name": "Reinforcement Learning", "description": "Learning through trial and error with rewards",
         "properties": {"type": "paradigm", "online": True}},
        "semantic", importance=0.85
    )
    memory.remember(
        {"name": "AGI", "description": "Artificial General Intelligence - human-level cognitive abilities",
         "properties": {"type": "goal", "achieved": False}},
        "semantic", importance=1.0
    )

    # Add relationships
    concepts = list(memory.semantic.concepts.keys())
    if len(concepts) >= 3:
        # Find concepts by name for relationships
        transformer_id = None
        nn_id = None
        agi_id = None
        for cid, c in memory.semantic.concepts.items():
            if c["name"] == "Transformer":
                transformer_id = cid
            elif c["name"] == "Neural Network":
                nn_id = cid
            elif c["name"] == "AGI":
                agi_id = cid

        if transformer_id and nn_id:
            memory.semantic.add_relationship(transformer_id, nn_id, "is_a", weight=1.0)
        if transformer_id and agi_id:
            memory.semantic.add_relationship(transformer_id, agi_id, "contributes_to", weight=0.8)

    # Query semantic memory
    query_result = memory.semantic.query("Transformer")
    print(f"Semantic query 'Transformer': found={query_result['found']}")
    if query_result['found']:
        print(f"  Related concepts: {len(query_result['related'])}")

    # 4. Procedural Memory
    print("\n--- Procedural Memory ---")
    memory.remember(
        {
            "name": "Run Full System Test Suite",
            "steps": [
                "Set up test environment",
                "Run unit tests",
                "Run integration tests",
                "Run end-to-end tests",
                "Generate coverage report",
                "Analyze results"
            ],
            "preconditions": ["Code compiled", "Test database available"],
            "postconditions": ["All tests passing", "Coverage report generated"],
            "difficulty": 0.6
        },
        "procedural", importance=0.8
    )
    memory.remember(
        {
            "name": "Deploy to Production",
            "steps": [
                "Run pre-deploy checks",
                "Backup current state",
                "Deploy new version",
                "Run smoke tests",
                "Update monitoring"
            ],
            "preconditions": ["Tests passing", "Approval obtained"],
            "postconditions": ["New version live", "Monitoring active"],
            "difficulty": 0.7
        },
        "procedural", importance=0.85
    )

    # Execute a skill
    deploy_skills = memory.procedural.find_skills("Deploy")
    if deploy_skills:
        result = memory.procedural.execute(deploy_skills[0]["id"])
        print(f"Executed 'Deploy to Production': {result['success']}")
        print(f"  Steps: {result['steps_executed']}")

    # 5. Consolidation
    print("\n--- Consolidation ---")
    consolidation_result = memory.consolidate()
    print(f"Consolidated: {consolidation_result['working_to_episodic']} working->episodic")
    print(f"Working memory after consolidation: {memory.working.size}")

    # 6. Stats
    print("\n--- Memory System Stats ---")
    stats = memory.get_stats()
    for memory_type, data in stats.items():
        print(f"  {memory_type}: {data}")

    # 7. Cross-type retrieval
    print("\n--- Cross-type Retrieval ---")
    all_results = memory.recall(query="test", k=3)
    total = sum(len(v) for v in all_results.values())
    print(f"Total memories matching 'test': {total}")
    for mem_type, traces in all_results.items():
        print(f"  {mem_type}: {len(traces)}")

    # Save and load
    memory.episodic.save_to_disk()
    print(f"\nEpisodic memory saved to disk ({memory.episodic.size} traces)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    return memory


if __name__ == "__main__":
    run_demo()
