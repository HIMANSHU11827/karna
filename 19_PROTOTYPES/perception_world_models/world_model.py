"""
World Model — Predictive State Representations and Mental Simulation
=====================================================================

A world model that maintains an internal representation of the environment,
predicts future states, and enables mental simulation for planning.

Architecture:
- State encoder: compresses perceptual representations into latent states
- Transition model: predicts next state given current state and action
- Reward model: estimates expected reward for state-action pairs
- Simulation engine: rolls out trajectories for planning

"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# Core Data Structures
# ============================================================

@dataclass
class WorldState:
    """Latent state representation of the world."""
    state_id: str
    embedding: list[float]
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "state_id": self.state_id,
            "embedding_dim": len(self.embedding),
            "embedding_sample": self.embedding[:5],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Action:
    """An action that can be taken in the world."""
    action_id: str
    action_type: str
    parameters: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
        }


@dataclass
class Transition:
    """A predicted transition from one state to another."""
    current_state: WorldState
    action: Action
    next_state: WorldState
    predicted_reward: float
    probability: float = 1.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "current_state_id": self.current_state.state_id,
            "action_id": self.action.action_id,
            "next_state_id": self.next_state.state_id,
            "predicted_reward": self.predicted_reward,
            "probability": self.probability,
            "confidence": self.confidence,
        }


@dataclass
class SimulationResult:
    """Result of a mental simulation rollout."""
    initial_state: WorldState
    actions: list[Action]
    states_visited: list[WorldState]
    total_reward: float
    trajectory_length: int
    success_probability: float = 0.0

    def to_dict(self) -> dict:
        return {
            "initial_state_id": self.initial_state.state_id,
            "trajectory_length": self.trajectory_length,
            "actions_taken": [a.action_type for a in self.actions],
            "total_reward": self.total_reward,
            "success_probability": self.success_probability,
            "states_visited": len(self.states_visited),
        }


# ============================================================
# State Encoder
# ============================================================

class StateEncoder:
    """Encodes perceptual representations into compact latent states.

    In production: variational autoencoder, world transformer, or RSSM
    (Recurrent State-Space Model from DreamerV3).

    Prototype: deterministic hashing-based encoding with structure.
    """

    def __init__(self, state_dim: int = 64, seed: int = 42):
        self.state_dim = state_dim
        self.seed = seed
        self._state_counter = 0

    def encode(self, perceptual_embedding: list[float], context: dict = None) -> WorldState:
        """Encode a perceptual embedding into a world state."""
        self._state_counter += 1
        state_id = f"state_{self._state_counter:06d}"

        # Combine perceptual embedding with context
        combined = list(perceptual_embedding)
        if context:
            context_hash = hash(str(context)) % 10000
            combined.append(context_hash / 10000.0)

        # Project to state dimension
        state_embedding = self._project(combined, self.state_dim)

        return WorldState(
            state_id=state_id,
            embedding=state_embedding,
            metadata={
                "source": "perceptual_encoder",
                "context_keys": list(context.keys()) if context else [],
                "original_dim": len(perceptual_embedding),
            },
        )

    def _project(self, vector: list[float], target_dim: int) -> list[float]:
        """Project a vector to target dimension using sinusoidal features."""
        result = []
        for i in range(target_dim):
            val = 0.0
            for j, v in enumerate(vector):
                freq = (j + 1) * 0.1
                val += v * math.sin(freq * (i + 1) + self.seed)
            # Normalize to [-1, 1]
            val = math.tanh(val)
            result.append(val)
        return result


# ============================================================
# Transition Model
# ============================================================

class TransitionModel:
    """Predicts next state given current state and action.

    In production: dynamics model (e.g., RSSM, transformer-based world model).
    Prototype: learned transformation with stochastic sampling.
    """

    def __init__(self, state_dim: int = 64, action_dim: int = 32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._transition_count = 0
        # Learned transition weights (simplified)
        self._weights = self._initialize_weights()

    def _initialize_weights(self) -> list[list[float]]:
        """Initialize transition weights."""
        random.seed(42)
        return [
            [random.gauss(0, 0.1) for _ in range(self.action_dim)]
            for _ in range(self.state_dim)
        ]

    def predict(
        self,
        current_state: WorldState,
        action: Action,
        stochastic: bool = True,
    ) -> Transition:
        """Predict next state given current state and action."""
        self._transition_count += 1

        # Encode action
        action_embedding = self._encode_action(action)

        # Apply transition: next_state = f(current_state, action)
        next_embedding = self._transition(current_state.embedding, action_embedding)

        # Add stochasticity
        if stochastic:
            next_embedding = self._add_noise(next_embedding, sigma=0.05)

        # Predict reward (simplified)
        predicted_reward = self._predict_reward(current_state, action)

        next_state = WorldState(
            state_id=f"state_pred_{self._transition_count:06d}",
            embedding=next_embedding,
            metadata={
                "predicted_from": current_state.state_id,
                "action_type": action.action_type,
                "stochastic": stochastic,
            },
        )

        # Confidence based on prediction uncertainty
        confidence = 0.85 if not stochastic else 0.75

        return Transition(
            current_state=current_state,
            action=action,
            next_state=next_state,
            predicted_reward=predicted_reward,
            confidence=confidence,
        )

    def _encode_action(self, action: Action) -> list[float]:
        """Encode action into a fixed-size vector."""
        action_str = f"{action.action_type}:{action.action_id}"
        seed = hash(action_str) % 10000
        return [math.sin(seed + i * 0.12) * 0.5 + 0.5 for i in range(self.action_dim)]

    def _transition(
        self, state_embedding: list[float], action_embedding: list[float]
    ) -> list[float]:
        """Apply transition function to get next state embedding."""
        # Simplified: element-wise combination with tanh
        result = []
        for i in range(self.state_dim):
            state_val = state_embedding[i] if i < len(state_embedding) else 0
            action_val = (
                action_embedding[i % len(action_embedding)] if action_embedding else 0
            )
            # Weighted combination
            combined = 0.8 * state_val + 0.2 * action_val
            result.append(math.tanh(combined))
        return result

    def _predict_reward(self, state: WorldState, action: Action) -> float:
        """Predict reward for state-action pair."""
        # Prototype: reward based on state-action compatibility
        state_norm = math.sqrt(sum(x * x for x in state.embedding))
        action_hash = hash(action.action_id) % 1000
        return math.sin(state_norm + action_hash) * 0.5

    def _add_noise(self, embedding: list[float], sigma: float = 0.05) -> list[float]:
        """Add Gaussian noise for stochastic predictions."""
        random.seed(int(time.time() * 1000) % 10000)
        return [
            val + random.gauss(0, sigma) for val in embedding
        ]


# ============================================================
# Reward Model
# ============================================================

class RewardModel:
    """Estimates expected rewards for state-action pairs.

    In production: learned reward function from human feedback or
    intrinsic motivation signals.
    """

    def __init__(self, state_dim: int = 64):
        self.state_dim = state_dim
        self._reward_history: list[dict] = []

    def estimate(self, state: WorldState, action: Action) -> float:
        """Estimate reward for taking action in state."""
        # Prototype: heuristic reward based on state features
        state_energy = math.sqrt(sum(x * x for x in state.embedding))
        action_type_bonus = self._action_type_bonus(action.action_type)
        reward = state_energy * 0.1 + action_type_bonus

        self._reward_history.append({
            "state_id": state.state_id,
            "action_type": action.action_type,
            "estimated_reward": reward,
        })

        return reward

    def _action_type_bonus(self, action_type: str) -> float:
        """Bonus based on action type."""
        bonuses = {
            "explore": 0.1,
            "observe": 0.05,
            "interact": 0.2,
            "communicate": 0.15,
            "move": 0.0,
            "analyze": 0.1,
        }
        return bonuses.get(action_type, 0.0)

    def get_reward_stats(self) -> dict:
        """Get statistics on reward predictions."""
        if not self._reward_history:
            return {"count": 0, "mean_reward": 0.0, "max_reward": 0.0}
        rewards = [r["estimated_reward"] for r in self._reward_history]
        return {
            "count": len(rewards),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
        }


# ============================================================
# Simulation Engine
# ============================================================

class SimulationEngine:
    """Mental simulation engine for planning via model-based RL.

    Uses the world model to simulate trajectories and evaluate action
    sequences before executing them in the real world.
    """

    def __init__(
        self,
        state_encoder: StateEncoder,
        transition_model: TransitionModel,
        reward_model: RewardModel,
    ):
        self.state_encoder = state_encoder
        self.transition_model = transition_model
        self.reward_model = reward_model
        self._simulation_count = 0

    def simulate(
        self,
        initial_state: WorldState,
        planned_actions: list[Action],
        num_rollouts: int = 10,
        discount_factor: float = 0.95,
    ) -> list[SimulationResult]:
        """Simulate multiple rollouts of planned actions.

        Args:
            initial_state: Starting world state
            planned_actions: Sequence of actions to simulate
            num_rollouts: Number of stochastic rollouts
            discount_factor: Reward discount factor (gamma)

        Returns:
            List of simulation results, one per rollout
        """
        results = []

        for rollout_idx in range(num_rollouts):
            self._simulation_count += 1
            current_state = initial_state
            states_visited = [current_state]
            total_reward = 0.0
            discounted_reward = 0.0
            success_count = 0

            for step_idx, action in enumerate(planned_actions):
                # Predict next state
                transition = self.transition_model.predict(
                    current_state, action, stochastic=True
                )

                # Estimate reward
                reward = self.reward_model.estimate(transition.next_state, action)
                discounted_reward += (discount_factor ** step_idx) * reward

                # Check success (simplified)
                if transition.confidence > 0.7:
                    success_count += 1

                current_state = transition.next_state
                states_visited.append(current_state)

            success_prob = success_count / len(planned_actions) if planned_actions else 0

            result = SimulationResult(
                initial_state=initial_state,
                actions=planned_actions,
                states_visited=states_visited,
                total_reward=discounted_reward,
                trajectory_length=len(planned_actions),
                success_probability=success_prob,
            )
            results.append(result)

        return results

    def plan(
        self,
        initial_state: WorldState,
        candidate_actions: list[Action],
        planning_horizon: int = 5,
        num_rollouts: int = 10,
    ) -> dict:
        """Plan optimal action sequence via simulation.

        Uses greedy rollout-based planning: at each step, evaluate all
        candidate actions by simulating future trajectories.
        """
        planned_actions = []
        current_state = initial_state
        planning_trace = []

        for step in range(planning_horizon):
            best_action = None
            best_value = -float("inf")
            best_result = None

            # Evaluate each candidate action
            for action in candidate_actions:
                # Simulate this action
                transitions = self.transition_model.predict(
                    current_state, action, stochastic=False
                )
                reward = self.reward_model.estimate(transitions.next_state, action)

                # Look ahead: simulate future with random actions
                future_value = 0.0
                for _ in range(3):  # 3-step lookahead
                    random_action = random.choice(candidate_actions)
                    future_transitions = self.transition_model.predict(
                        transitions.next_state, random_action, stochastic=False
                    )
                    future_value += self.reward_model.estimate(
                        future_transitions.next_state, random_action
                    )

                total_value = reward + 0.9 * future_value

                if total_value > best_value:
                    best_value = total_value
                    best_action = action
                    best_result = transitions

            if best_action is None:
                break

            planned_actions.append(best_action)
            current_state = best_result.next_state if best_result else current_state

            planning_trace.append({
                "step": step,
                "action": best_action.action_type,
                "value": best_value,
                "state_id": current_state.state_id,
            })

        # Final simulation with planned actions
        final_results = self.simulate(
            initial_state, planned_actions, num_rollouts=num_rollouts
        )
        avg_reward = sum(r.total_reward for r in final_results) / len(final_results)
        avg_success = sum(r.success_probability for r in final_results) / len(final_results)

        return {
            "planned_actions": [a.action_type for a in planned_actions],
            "planning_trace": planning_trace,
            "average_reward": avg_reward,
            "average_success_probability": avg_success,
            "num_simulations": self._simulation_count,
            "horizon": planning_horizon,
        }

    def get_simulation_stats(self) -> dict:
        """Get simulation statistics."""
        return {
            "total_simulations": self._simulation_count,
            "transition_model_calls": self.transition_model._transition_count,
            "reward_model_calls": len(self.reward_model._reward_history),
        }


# ============================================================
# World Model — Main Interface
# ============================================================

class WorldModel:
    """Unified world model combining state encoding, transition, and simulation.

    This is the high-level interface that the agent uses to maintain
    its understanding of the world and plan actions.
    """

    def __init__(self, state_dim: int = 64):
        self.state_dim = state_dim
        self.state_encoder = StateEncoder(state_dim)
        self.transition_model = TransitionModel(state_dim)
        self.reward_model = RewardModel(state_dim)
        self.simulation_engine = SimulationEngine(
            self.state_encoder, self.transition_model, self.reward_model
        )
        self._current_state: WorldState | None = None
        self._state_history: list[WorldState] = []

    def observe(self, perceptual_embedding: list[float], context: dict = None) -> WorldState:
        """Process a new observation and update the world state."""
        state = self.state_encoder.encode(perceptual_embedding, context)
        self._current_state = state
        self._state_history.append(state)
        return state

    def predict(self, action: Action) -> Transition:
        """Predict the outcome of an action from the current state."""
        if self._current_state is None:
            raise RuntimeError("No current state. Call observe() first.")
        return self.transition_model.predict(self._current_state, action)

    def simulate(
        self, actions: list[Action], num_rollouts: int = 10
    ) -> list[SimulationResult]:
        """Simulate a sequence of actions from the current state."""
        if self._current_state is None:
            raise RuntimeError("No current state. Call observe() first.")
        return self.simulation_engine.simulate(
            self._current_state, actions, num_rollouts
        )

    def plan(
        self,
        candidate_actions: list[Action],
        horizon: int = 5,
        num_rollouts: int = 10,
    ) -> dict:
        """Plan optimal action sequence from the current state."""
        if self._current_state is None:
            raise RuntimeError("No current state. Call observe() first.")
        return self.simulation_engine.plan(
            self._current_state,
            candidate_actions,
            planning_horizon=horizon,
            num_rollouts=num_rollouts,
        )

    def get_world_state_summary(self) -> dict:
        """Get a summary of the world model's current state."""
        return {
            "current_state": self._current_state.to_dict() if self._current_state else None,
            "state_history_length": len(self._state_history),
            "total_observations": self.state_encoder._state_counter,
            "total_predictions": self.transition_model._transition_count,
            "reward_stats": self.reward_model.get_reward_stats(),
            "simulation_stats": self.simulation_engine.get_simulation_stats(),
        }


# ============================================================
# Demo
# ============================================================

def main():
    """Run world model demo."""
    print("=" * 60)
    print("  World Model Demo")
    print("=" * 60)
    print()

    wm = WorldModel(state_dim=64)

    # Demo 1: Observation
    print("[1] Observing environment...")
    perceptual_embedding = [math.sin(i * 0.1) * 0.5 + 0.5 for i in range(128)]
    state = wm.observe(perceptual_embedding, context={"modality": "vision", "source": "camera"})
    print(f"    State ID: {state.state_id}")
    print(f"    Embedding dim: {len(state.embedding)}")
    print(f"    Metadata: {state.metadata}")
    print()

    # Demo 2: Action prediction
    print("[2] Predicting action outcomes...")
    action = Action(action_id="act_001", action_type="explore", parameters={"direction": "forward"})
    transition = wm.predict(action)
    print(f"    Current state: {transition.current_state.state_id}")
    print(f"    Action: {transition.action.action_type}")
    print(f"    Predicted next state: {transition.next_state.state_id}")
    print(f"    Predicted reward: {transition.predicted_reward:.4f}")
    print(f"    Confidence: {transition.confidence:.2f}")
    print()

    # Demo 3: Mental simulation
    print("[3] Running mental simulation...")
    actions = [
        Action(action_id=f"act_{i:03d}", action_type=random.choice(["explore", "observe", "interact"]))
        for i in range(5)
    ]
    results = wm.simulate(actions, num_rollouts=5)
    print(f"    Simulated {len(results)} rollouts")
    print(f"    Trajectory length: {results[0].trajectory_length}")
    print(f"    Avg total reward: {sum(r.total_reward for r in results) / len(results):.4f}")
    print(f"    Avg success prob: {sum(r.success_probability for r in results) / len(results):.2f}")
    print()

    # Demo 4: Planning
    print("[4] Planning optimal action sequence...")
    candidate_actions = [
        Action(action_id="explore_1", action_type="explore"),
        Action(action_id="observe_1", action_type="observe"),
        Action(action_id="interact_1", action_type="interact"),
        Action(action_id="analyze_1", action_type="analyze"),
    ]
    plan_result = wm.plan(candidate_actions, horizon=3, num_rollouts=5)
    print(f"    Planned actions: {plan_result['planned_actions']}")
    print(f"    Average reward: {plan_result['average_reward']:.4f}")
    print(f"    Success probability: {plan_result['average_success_probability']:.2f}")
    print(f"    Total simulations: {plan_result['num_simulations']}")
    print()

    # Demo 5: World state summary
    print("[5] World model summary:")
    summary = wm.get_world_state_summary()
    print(f"    Total observations: {summary['total_observations']}")
    print(f"    Total predictions: {summary['total_predictions']}")
    print(f"    State history length: {summary['state_history_length']}")
    print(f"    Reward stats: {summary['reward_stats']}")
    print(f"    Simulation stats: {summary['simulation_stats']}")
    print()

    print("=" * 60)
    print("  World Model Demo — Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
