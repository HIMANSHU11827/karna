"""
AGI Research Lab — Prototype: Neural Architecture Search (NAS)
Simplified implementation for exploring optimal network architectures.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import json
import time
from pathlib import Path


@dataclass
class LayerConfig:
    """Configuration for a single neural network layer."""
    layer_type: str  # "linear", "conv2d", "attention", "lstm", "gru"
    units: int = 64
    activation: str = "relu"
    dropout: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_type": self.layer_type,
            "units": self.units,
            "activation": self.activation,
            "dropout": self.dropout,
            "extra": self.extra,
        }


@dataclass
class Architecture:
    """A candidate neural architecture."""
    layers: List[LayerConfig] = field(default_factory=list)
    fitness: float = 0.0
    id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fitness": self.fitness,
            "layers": [l.to_dict() for l in self.layers],
        }


class NeuralArchitectureSearch:
    """
    Evolutionary Neural Architecture Search.
    Searches for optimal network structures through mutation and selection.
    """

    def __init__(
        self,
        seed_architectures: Optional[List[Architecture]] = None,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        max_layers: int = 8,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_layers = max_layers
        self.rng = random.Random(seed)
        self.generation = 0
        self.population: List[Architecture] = seed_architectures or []
        self.history: List[Dict[str, Any]] = []

        self.available_layers = ["linear", "conv2d", "attention", "lstm", "gru"]
        self.available_activations = ["relu", "gelu", "silu", "tanh", "sigmoid"]
        self.unit_options = [32, 64, 128, 256, 512, 768, 1024]

    def _random_layer(self) -> LayerConfig:
        """Generate a random layer configuration."""
        return LayerConfig(
            layer_type=self.rng.choice(self.available_layers),
            units=self.rng.choice(self.unit_options),
            activation=self.rng.choice(self.available_activations),
            dropout=self.rng.choice([0.0, 0.1, 0.2, 0.3, 0.5]),
        )

    def _random_architecture(self) -> Architecture:
        """Generate a random architecture."""
        n_layers = self.rng.randint(1, self.max_layers)
        arch = Architecture(
            layers=[self._random_layer() for _ in range(n_layers)],
            id=f"arch_{self.generation}_{self.rng.randint(1000, 9999)}",
        )
        return arch

    def initialize_population(self):
        """Create initial random population."""
        self.population = [self._random_architecture() for _ in range(self.population_size)]
        self.generation = 0
        print(f"Initialized population of {self.population_size} architectures")

    def mutate(self, arch: Architecture) -> Architecture:
        """Mutate an architecture."""
        new_layers = [LayerConfig(**l.to_dict()) for l in arch.layers]

        for i in range(len(new_layers)):
            if self.rng.random() < self.mutation_rate:
                mutation_type = self.rng.choice(["units", "activation", "dropout", "type"])
                if mutation_type == "units":
                    new_layers[i].units = self.rng.choice(self.unit_options)
                elif mutation_type == "activation":
                    new_layers[i].activation = self.rng.choice(self.available_activations)
                elif mutation_type == "dropout":
                    new_layers[i].dropout = self.rng.choice([0.0, 0.1, 0.2, 0.3, 0.5])
                elif mutation_type == "type":
                    new_layers[i].layer_type = self.rng.choice(self.available_layers)

        # Add or remove layers
        if self.rng.random() < self.mutation_rate / 2:
            if len(new_layers) < self.max_layers and self.rng.random() < 0.5:
                new_layers.append(self._random_layer())
            elif len(new_layers) > 1:
                new_layers.pop(self.rng.randint(0, len(new_layers) - 1))

        return Architecture(
            layers=new_layers,
            id=f"arch_{self.generation}_{self.rng.randint(1000, 9999)}",
        )

    def crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Single-point crossover between two architectures."""
        point = self.rng.randint(
            0, min(len(parent1.layers), len(parent2.layers))
        )
        child_layers = parent1.layers[:point] + parent2.layers[point:]
        return Architecture(
            layers=[LayerConfig(**l.to_dict()) for l in child_layers],
            id=f"arch_{self.generation}_{self.rng.randint(1000, 9999)}",
        )

    def evaluate_fitness(
        self, eval_fn: Callable[[Architecture], float]
    ):
        """Evaluate all architectures in the population."""
        for arch in self.population:
            arch.fitness = eval_fn(arch)

    def select_parents(self, tournament_size: int = 3) -> Tuple[Architecture, Architecture]:
        """Tournament selection."""
        def tournament() -> Architecture:
            contestants = self.rng.sample(self.population, min(tournament_size, len(self.population)))
            return max(contestants, key=lambda a: a.fitness)
        return tournament(), tournament()

    def evolve(self, eval_fn: Callable[[Architecture], float], generations: int = 10):
        """Run the evolutionary search."""
        if not self.population:
            self.initialize_population()

        for gen in range(generations):
            self.generation = gen

            # Evaluate fitness
            self.evaluate_fitness(eval_fn)

            # Sort by fitness
            self.population.sort(key=lambda a: a.fitness, reverse=True)

            # Record history
            self.history.append({
                "generation": gen,
                "best_fitness": self.population[0].fitness,
                "mean_fitness": sum(a.fitness for a in self.population) / len(self.population),
                "best_arch": self.population[0].to_dict(),
            })

            print(f"Gen {gen}: best={self.population[0].fitness:.4f}, "
                  f"mean={sum(a.fitness for a in self.population) / len(self.population):.4f}")

            # Elitism: keep top 2
            new_population = self.population[:2]

            # Fill rest with offspring
            while len(new_population) < self.population_size:
                if self.rng.random() < self.crossover_rate:
                    p1, p2 = self.select_parents()
                    child = self.crossover(p1, p2)
                else:
                    parent = self.select_parents()[0]
                    child = parent

                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population

        return self.population[0]

    def get_best(self) -> Architecture:
        """Return the best architecture found."""
        return max(self.population, key=lambda a: a.fitness)

    def save(self, path: str):
        """Save search state."""
        state = {
            "generation": self.generation,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "history": self.history,
            "best": self.get_best().to_dict(),
        }
        Path(path).write_text(json.dumps(state, indent=2))


# === Demo: Dummy fitness function ===
def dummy_fitness(arch: Architecture) -> float:
    """Pretend fitness: prefer deeper networks with dropout."""
    score = 0.0
    score += len(arch.layers) * 0.1
    for layer in arch.layers:
        if layer.dropout > 0:
            score += 0.05
        if layer.units >= 256:
            score += 0.02
    return score + random.random() * 0.1


if __name__ == "__main__":
    print("=" * 60)
    print("Neural Architecture Search — Prototype")
    print("=" * 60)

    nas = NeuralArchitectureSearch(population_size=20, seed=42)
    nas.initialize_population()

    print("\nRunning evolution for 10 generations...")
    best = nas.evolve(eval_fn=dummy_fitness, generations=10)

    print(f"\nBest architecture (fitness={best.fitness:.4f}):")
    for i, layer in enumerate(best.layers):
        print(f"  Layer {i+1}: {layer.layer_type} — units={layer.units}, "
              f"activation={layer.activation}, dropout={layer.dropout}")

    nas.save("/home/himanshu/Desktop/AGI_RESEARCH_LAB/20_OUTPUT/nas_results.json")
    print("\nResults saved to 20_OUTPUT/nas_results.json")
