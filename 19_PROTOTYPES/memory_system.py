"""
AGI Research Lab — Prototype: Memory System
Multi-tier memory with consolidation and retrieval.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import json
import heapq
from pathlib import Path


@dataclass
class MemoryItem:
    """A single memory entry."""
    content: Any
    importance: float = 0.5  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def relevance_score(self, current_time: Optional[float] = None) -> float:
        """Compute relevance score for retrieval."""
        if current_time is None:
            current_time = time.time()

        # Recency decay
        age_hours = (current_time - self.timestamp) / 3600
        recency = 1.0 / (1.0 + age_hours)

        # Access frequency
        frequency = min(self.access_count / 10.0, 1.0)

        # Importance
        importance = self.importance

        # Combined score
        return 0.4 * recency + 0.3 * frequency + 0.3 * importance


@dataclass
class Episode:
    """An episodic memory — a recorded experience."""
    title: str
    events: List[Dict[str, Any]]
    outcome: str = ""
    timestamp: float = field(default_factory=time.time)
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "events": self.events,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "emotional_valence": self.emotional_valence,
            "tags": self.tags,
        }


class MemorySystem:
    """
    Multi-tier memory system:
    - Working memory: short-term, limited capacity
    - Episodic memory: experiences and events
    - Semantic memory: facts and knowledge
    - Procedural memory: skills and procedures
    """

    def __init__(
        self,
        working_memory_capacity: int = 7,
        consolidation_threshold: float = 0.6,
        data_dir: str = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/06_MEMORY",
    ):
        self.working_memory: List[MemoryItem] = []
        self.episodic_memory: List[Episode] = []
        self.semantic_memory: Dict[str, MemoryItem] = {}
        self.procedural_memory: Dict[str, Callable] = {}

        self.working_memory_capacity = working_memory_capacity
        self.consolidation_threshold = consolidation_threshold
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.consolidation_log: List[Dict[str, Any]] = []

    def perceive(self, content: Any, importance: float = 0.5, tags: Optional[List[str]] = None):
        """Add a new item to working memory (perception)."""
        item = MemoryItem(
            content=content,
            importance=importance,
            tags=tags or [],
        )
        self.working_memory.append(item)

        # Enforce capacity limit — consolidate oldest if full
        if len(self.working_memory) > self.working_memory_capacity:
            oldest = self.working_memory.pop(0)
            self._consolidate(oldest)

    def recall(self, query: Optional[str] = None, n: int = 5, tags: Optional[List[str]] = None) -> List[MemoryItem]:
        """Recall relevant memories."""
        # Gather candidates from all memory tiers
        candidates: List[MemoryItem] = []

        # From working memory
        candidates.extend(self.working_memory)

        # From semantic memory
        for item in self.semantic_memory.values():
            candidates.append(item)

        # Score and rank
        now = time.time()
        scored: List[Tuple[float, MemoryItem]] = []
        for item in candidates:
            if tags and not any(t in item.tags for t in tags):
                continue

            score = item.relevance_score(now)

            # Simple keyword boost
            if query and isinstance(item.content, str):
                if query.lower() in item.content.lower():
                    score += 0.5

            scored.append((score, item))

        # Return top n
        top = heapq.nlargest(n, scored, key=lambda x: x[0])
        for _, item in top:
            item.access_count += 1
            item.last_accessed = now

        return [item for _, item in top]

    def remember_fact(self, key: str, fact: str, importance: float = 0.7, tags: Optional[List[str]] = None):
        """Store a fact in semantic memory."""
        self.semantic_memory[key] = MemoryItem(
            content=fact,
            importance=importance,
            tags=tags or ["fact"],
        )

    def recall_fact(self, key: str) -> Optional[str]:
        """Retrieve a fact from semantic memory."""
        item = self.semantic_memory.get(key)
        if item:
            item.access_count += 1
            item.last_accessed = time.time()
            return str(item.content)
        return None

    def record_episode(self, episode: Episode):
        """Record an experience in episodic memory."""
        self.episodic_memory.append(episode)
        # High emotional valence -> higher importance
        if abs(episode.emotional_valence) > 0.5:
            self.perceive(
                f"Episode: {episode.title} — {episode.outcome}",
                importance=0.8,
                tags=["episode"] + episode.tags,
            )

    def register_procedure(self, name: str, description: str, fn: Callable):
        """Register a skill/procedure in procedural memory."""
        self.procedural_memory[name] = fn
        self.remember_fact(f"procedure_{name}", description, tags=["procedure", "skill"])

    def execute_procedure(self, name: str, *args, **kwargs) -> Any:
        """Execute a stored procedure."""
        if name not in self.procedural_memory:
            return {"error": f"Procedure '{name}' not found"}
        try:
            return self.procedural_memory[name](*args, **kwargs)
        except Exception as e:
            return {"error": str(e)}

    def _consolidate(self, item: MemoryItem):
        """Consolidate an item from working to long-term memory."""
        if item.importance >= self.consolidation_threshold:
            # Promote to semantic memory
            if isinstance(item.content, str):
                key = f"consolidated_{item.timestamp}"
                self.semantic_memory[key] = item
                self.consolidation_log.append({
                    "action": "consolidated_to_semantic",
                    "content_preview": item.content[:100],
                    "importance": item.importance,
                    "timestamp": time.time(),
                })

    def search_episodes(self, query: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Episode]:
        """Search episodic memory."""
        results = self.episodic_memory
        if tags:
            results = [e for e in results if any(t in e.tags for t in tags)]
        if query:
            results = [
                e for e in results
                if query.lower() in e.title.lower()
                or query.lower() in e.outcome.lower()
            ]
        return results

    def stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        return {
            "working_memory": len(self.working_memory),
            "semantic_memory": len(self.semantic_memory),
            "episodic_memory": len(self.episodic_memory),
            "procedural_memory": len(self.procedural_memory),
        }

    def save(self):
        """Persist memory to disk."""
        data = {
            "semantic_memory": {
                k: {
                    "content": v.content,
                    "importance": v.importance,
                    "access_count": v.access_count,
                    "tags": v.tags,
                }
                for k, v in self.semantic_memory.items()
            },
            "episodic_memory": [e.to_dict() for e in self.episodic_memory[-100:]],
            "stats": self.stats(),
        }
        path = self.data_dir / "memory_state.json"
        path.write_text(json.dumps(data, indent=2, default=str))

    def reflect(self) -> str:
        """Generate a summary of current memory state."""
        stats = self.stats()
        recent_episodes = self.episodic_memory[-5:] if self.episodic_memory else []
        facts = list(self.semantic_memory.keys())[:10]

        summary = f"""Memory Summary:
- Working memory: {stats['working_memory']} items
- Semantic memory: {stats['semantic_memory']} facts
- Episodic memory: {stats['episodic_memory']} episodes
- Procedural memory: {stats['procedural_memory']} skills

Recent episodes:
{chr(10).join(f"  • {e.title}: {e.outcome}" for e in recent_episodes)}

Key facts:
{chr(10).join(f"  • {f}" for f in facts)}
"""
        return summary


if __name__ == "__main__":
    print("=" * 60)
    print("Memory System — Prototype")
    print("=" * 60)

    mem = MemorySystem()

    # Register a procedure
    mem.register_procedure(
        "greet",
        "Generate a greeting for a given name",
        lambda target: f"Hello, target={target}",
    )

    # Feed some perceptions
    print("\nPerceiving...")
    mem.perceive("The sky is blue today", importance=0.3)
    mem.perceive("User prefers concise answers", importance=0.8, tags=["preference"])
    mem.perceive("Meeting scheduled for 3pm", importance=0.6, tags=["schedule"])
    mem.perceive("Found a bug in the parser", importance=0.9, tags=["bug", "urgent"])
    mem.remember_fact("user_language", "English", tags=["fact", "user"])
    mem.remember_fact("project_name", "AGI Research Lab", tags=["fact", "project"])

    # Record an episode
    print("Recording episode...")
    mem.record_episode(Episode(
        title="First experiment run",
        events=[
            {"action": "start_experiment", "time": time.time()},
            {"action": "observe_result", "time": time.time() + 1},
        ],
        outcome="Successfully ran 10 experiments with positive results",
        emotional_valence=0.7,
        tags=["experiment", "success"],
    ))

    # Recall
    print("\nRecalling facts about 'bug'...")
    results = mem.recall(query="bug", n=3)
    for item in results:
        print(f"  • {item.content} (importance: {item.importance:.1f})")

    # Reflect
    print("\n" + mem.reflect())

    # Execute procedure
    print("\nExecuting 'greet' procedure:")
    result = mem.execute_procedure("greet", "World")
    print(f"  Result: {result}")

    # Save
    mem.save()
    print("\nMemory state saved.")
