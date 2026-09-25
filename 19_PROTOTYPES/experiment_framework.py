"""
AGI Research Lab — Experiment Framework
Core system for defining, running, tracking, and reproducing experiments.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import json
import time
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agi-lab")


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    tags: List[str] = field(default_factory=list)
    version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "seed": self.seed,
            "tags": self.tags,
            "version": self.version,
        }

    def fingerprint(self) -> str:
        """Unique hash for this experiment config (for deduplication)."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class ExperimentResult:
    """Results from running an experiment."""
    config: ExperimentConfig
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "fingerprint": self.config.fingerprint(),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "logs": self.logs,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "error": self.error,
        }


class ExperimentFramework:
    """Define, run, track, and reproduce experiments."""

    def __init__(self, root_dir: str = "/home/himanshu/Desktop/AGI_RESEARCH_LAB"):
        self.root = Path(root_dir)
        self.experiments_dir = self.root / "03_EXPERIMENTS"
        self.artifacts_dir = self.root / "20_OUTPUT"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, ExperimentResult] = {}
        logger.info(f"Experiment framework initialized at {self.root}")

    def create_experiment(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        tags: Optional[List[str]] = None,
    ) -> ExperimentConfig:
        """Create a new experiment configuration."""
        config = ExperimentConfig(
            name=name,
            description=description,
            parameters=parameters or {},
            seed=seed,
            tags=tags or [],
        )
        logger.info(f"Created experiment: {config.name} [{config.fingerprint()}]")
        return config

    def run(
        self,
        config: ExperimentConfig,
        fn: Callable[[ExperimentConfig, Path], Dict[str, float]],
    ) -> ExperimentResult:
        """Run an experiment function and track results."""
        result = ExperimentResult(config=config, status="running")
        exp_dir = self.experiments_dir / config.fingerprint()
        exp_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = self.artifacts_dir / config.fingerprint()
        artifact_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running experiment: {config.name} [{config.fingerprint()}]")
        start = time.time()

        try:
            metrics = fn(config, artifact_dir)
            result.metrics = metrics
            result.status = "completed"
            result.artifacts = [str(p) for p in artifact_dir.iterdir()]
            logger.info(f"Completed: {config.name} — metrics: {metrics}")
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            logger.error(f"Failed: {config.name} — {e}")

        result.duration_seconds = time.time() - start
        result.logs.append(f"Experiment {config.name} finished in {result.duration_seconds:.2f}s")

        # Save results
        result_path = exp_dir / "result.json"
        result_path.write_text(json.dumps(result.to_dict(), indent=2))
        self._registry[config.fingerprint()] = result

        return result

    def get_experiment(self, fingerprint: str) -> Optional[ExperimentResult]:
        """Retrieve a previous experiment by fingerprint."""
        return self._registry.get(fingerprint)

    def list_experiments(self, tag: Optional[str] = None) -> List[ExperimentResult]:
        """List all experiments, optionally filtered by tag."""
        results = list(self._registry.values())
        if tag:
            results = [r for r in results if tag in r.config.tags]
        return results

    def compare(self, *fingerprints: str) -> Dict[str, Dict[str, float]]:
        """Compare metrics across experiments."""
        comparison = {}
        for fp in fingerprints:
            result = self._registry.get(fp)
            if result:
                comparison[result.config.name] = result.metrics
        return comparison
