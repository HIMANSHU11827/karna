# Interface Definitions

## Internal Service Contracts

All components communicate via these interface contracts. Implementations must be interface-compliant for system integration.

---

### MemoryStore Interface

```python
class MemoryStore:
    """Primary interface for all memory operations."""
    
    def store(self, key: str, value: Any, metadata: Dict) -> bool:
        """Store a value with metadata. Returns success confirmation."""
        
    def retrieve(self, query: str, context: Dict, limit: int = 10) -> List[MemoryItem]:
        """Retrieve memories matching query, ranked by relevance."""
        
    def consolidate(self, memory_type: str) -> ConsolidationReport:
        """Consolidate memories (e.g., move short-term to long-term)."""
        
    def forget(self, criteria: Dict) -> int:
        """Forget memories matching criteria. Returns count forgotten."""
        
    def get_stats(self) -> Dict:
        """Return memory system statistics."""
```

### Learner Interface

```python
class Learner:
    """Primary interface for all learning operations."""
    
    def train(self, experience: Experience, objective: Objective) -> ModelUpdate:
        """Train on experience to optimize objective."""
        
    def evaluate(self, test_suite: TestSuite) -> Metrics:
        """Evaluate model on test suite."""
        
    def adapt(self, feedback: Feedback) -> UpdatedStrategy:
        """Adapt based on feedback signal."""
        
    def transfer(self, source_domain: str, target_domain: str) -> AdaptedModel:
        """Transfer knowledge from source to target domain."""
        
    def get_skill_library(self) -> List[Skill]:
        """Return all acquired skills."""
```

### PerceptionModule Interface

```python
class PerceptionModule:
    """Primary interface for perception and world modeling."""
    
    def encode(self, modality: str, input_data: Any) -> Representation:
        """Encode input from specific modality into representation."""
        
    def fuse(self, inputs: List[Representation]) -> UnifiedRepresentation:
        """Fuse multimodal representations into unified form."""
        
    def predict(self, state: Representation, action: Any) -> Distribution:
        """Predict next state distribution given current state and action."""
        
    def generate(self, representation: Representation, modality: str) -> Any:
        """Generate output in target modality from representation."""
```

### Reasoner Interface

```python
class Reasoner:
    """Primary interface for reasoning and planning."""
    
    def plan(self, goal: Goal, constraints: Constraints) -> ActionSequence:
        """Generate action sequence to achieve goal under constraints."""
        
    def deduce(self, premises: List, rules: Rules) -> Conclusions:
        """Apply logical deduction from premises and rules."""
        
    def abstract(self, examples: List) -> Concepts:
        """Form abstract concepts from examples."""
        
    def intervene(self, model: CausalModel, variable: str) -> Counterfactual:
        """Perform causal intervention and return counterfactual outcome."""
```

### Experiment Interface

```python
class Experiment:
    """Primary interface for experiment management."""
    
    def __init__(self, config: Dict, name: str):
        """Initialize with config and name."""
        
    def run(self) -> ExperimentResult:
        """Run the experiment and return results."""
        
    def save(self) -> str:
        """Save experiment state. Returns path."""
        
    def load(self, path: str) -> 'Experiment':
        """Load experiment from path."""
        
    def compare(self, other: 'Experiment') -> ComparisonReport:
        """Compare with another experiment."""
```

---

## Data Types

### MemoryItem
```python
@dataclass
class MemoryItem:
    key: str
    value: Any
    metadata: Dict
    created: datetime
    accessed: datetime
    importance: float  # 0.0 to 1.0
```

### Experience
```python
@dataclass
class Experience:
    input_data: Any
    output_data: Any
    context: Dict
    reward: float
    timestamp: datetime
```

### ModelUpdate
```python
@dataclass
class ModelUpdate:
    changed_parameters: int
    loss_before: float
    loss_after: float
    update_type: str  # "gradient", "rule", "memory"
```

### Representation
```python
@dataclass
class Representation:
    vector: np.ndarray
    modality: str
    metadata: Dict
    confidence: float
```

### Goal
```python
@dataclass
class Goal:
    description: str
    success_criteria: Dict
    priority: int
    deadline: Optional[datetime]
```

### BenchmarkSuite
```python
@dataclass
class BenchmarkSuite:
    name: str
    tasks: List[Task]
    metrics: List[str]
    difficulty_range: Tuple[float, float]
```

---

## Communication Patterns

### Synchronous Request-Response
For immediate operations (retrieval, encoding, deduction).

### Asynchronous Task Queue
For long-running operations (training, planning, consolidation).

### Event-Driven Notifications
For system events (memory consolidation, anomaly detection, completion).

---

*All interfaces are versioned. Breaking changes require major version bump.*
