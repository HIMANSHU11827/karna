"""
AGI Research Lab — Prototype: World Model
Simple predictive world model for environment simulation and planning.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random
import json
import math
from pathlib import Path


@dataclass
class State:
    """A state in the environment."""
    features: Dict[str, float]
    step: int = 0
    terminal: bool = False

    def to_vector(self) -> List[float]:
        return list(self.features.values())

    def distance(self, other: "State") -> float:
        """Euclidean distance between states."""
        keys = set(self.features.keys()) & set(other.features.keys())
        if not keys:
            return float("inf")
        return sum((self.features[k] - other.features[k]) ** 2 for k in keys) ** 0.5


@dataclass
class Action:
    """An action in the environment."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0


@dataclass
class Transition:
    """A state-action-next_state transition."""
    state: State
    action: Action
    next_state: State
    reward: float
    done: bool = False


class SimpleEnvironment:
    """
    A simple grid-like environment for testing world models.
    The agent must navigate to a goal while avoiding obstacles.
    """

    def __init__(self, size: int = 10, seed: int = 42):
        self.size = size
        self.rng = random.Random(seed)
        self.goal = {"x": size - 1, "y": size - 1}
        self.obstacles: List[Tuple[int, int]] = []
        self.agent_pos = {"x": 0, "y": 0}
        self.step_count = 0
        self.max_steps = size * 4

        # Generate random obstacles
        n_obstacles = size
        for _ in range(n_obstacles):
            x = self.rng.randint(0, size - 1)
            y = self.rng.randint(0, size - 1)
            if (x, y) != (0, 0) and (x, y) != (self.goal["x"], self.goal["y"]):
                self.obstacles.append((x, y))

    def get_state(self) -> State:
        """Get current state."""
        return State(
            features={
                "agent_x": float(self.agent_pos["x"]),
                "agent_y": float(self.agent_pos["y"]),
                "goal_x": float(self.goal["x"]),
                "goal_y": float(self.goal["y"]),
                "dist_to_goal": self._distance_to_goal(),
                "step_ratio": self.step_count / self.max_steps,
            },
            step=self.step_count,
            terminal=self._is_terminal(),
        )

    def _distance_to_goal(self) -> float:
        dx = self.agent_pos["x"] - self.goal["x"]
        dy = self.agent_pos["y"] - self.goal["y"]
        return (dx ** 2 + dy ** 2) ** 0.5

    def _is_terminal(self) -> bool:
        return (
            self.agent_pos["x"] == self.goal["x"]
            and self.agent_pos["y"] == self.goal["y"]
        ) or self.step_count >= self.max_steps

    def step(self, action: Action) -> Tuple[State, float, bool]:
        """Take an action, return (next_state, reward, done)."""
        old_dist = self._distance_to_goal()

        # Apply action
        dx, dy = 0, 0
        if action.name == "up":
            dy = -1
        elif action.name == "down":
            dy = 1
        elif action.name == "left":
            dx = -1
        elif action.name == "right":
            dx = 1

        new_x = max(0, min(self.size - 1, self.agent_pos["x"] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos["y"] + dy))

        # Check obstacle collision
        if (new_x, new_y) not in self.obstacles:
            self.agent_pos["x"] = new_x
            self.agent_pos["y"] = new_y

        self.step_count += 1

        # Compute reward
        new_dist = self._distance_to_goal()
        reward = -0.1  # small penalty for each step
        if new_dist < old_dist:
            reward += 0.2  # reward for getting closer
        if self._is_terminal():
            if self.agent_pos["x"] == self.goal["x"] and self.agent_pos["y"] == self.goal["y"]:
                reward += 10.0  # big reward for reaching goal
            else:
                reward -= 1.0  # penalty for timeout

        return self.get_state(), reward, self._is_terminal()

    def get_available_actions(self) -> List[Action]:
        """Get all possible actions."""
        return [
            Action(name="up"),
            Action(name="down"),
            Action(name="left"),
            Action(name="right"),
        ]

    def reset(self) -> State:
        """Reset environment."""
        self.agent_pos = {"x": 0, "y": 0}
        self.step_count = 0
        return self.get_state()


class WorldModel:
    """
    Learns a predictive model of the environment from observed transitions.
    Uses a simple table-based approach for discrete environments.
    """

    def __init__(self):
        self.transitions: List[Transition] = []
        self.state_action_counts: Dict[str, int] = {}
        self.state_action_rewards: Dict[str, List[float]] = {}
        # O(1) lookup: key -> (next_state, running_reward_sum, count)
        self._index: Dict[str, Tuple[State, float, int]] = {}

    def observe(self, transition: Transition):
        """Record an observed transition."""
        key = self._key(transition.state, transition.action)
        self.state_action_counts[key] = self.state_action_counts.get(key, 0) + 1
        if key not in self.state_action_rewards:
            self.state_action_rewards[key] = []
        self.state_action_rewards[key].append(transition.reward)

        # Update O(1) index
        if key in self._index:
            old_next, old_sum, old_count = self._index[key]
            self._index[key] = (transition.next_state, old_sum + transition.reward, old_count + 1)
        else:
            self._index[key] = (transition.next_state, transition.reward, 1)

    def predict(self, state: State, action: Action) -> Tuple[State, float]:
        """Predict next state and reward — O(1) lookup."""
        key = self._key(state, action)
        
        # O(1) exact match lookup
        if key in self._index:
            next_state, reward_sum, count = self._index[key]
            return next_state, reward_sum / count

        # No match — return state unchanged, zero reward
        # (approximate nearest-neighbor deferred to production)
        return state, 0.0

    def _key(self, state: State, action: Action) -> str:
        """Create a lookup key for state-action pair."""
        state_str = "_".join(f"{k}={v:.1f}" for k, v in sorted(state.features.items()))
        return f"{state_str}|{action.name}"

    def plan(
        self, initial_state: State, actions: List[Action], depth: int = 5
    ) -> List[Action]:
        """Simple greedy planning using the learned model."""
        plan = []
        current_state = initial_state

        for _ in range(depth):
            best_action = None
            best_reward = float("-inf")

            for action in actions:
                _, predicted_reward = self.predict(current_state, action)
                if predicted_reward > best_reward:
                    best_reward = predicted_reward
                    best_action = action

            if best_action:
                plan.append(best_action)
                next_state, _ = self.predict(current_state, best_action)
                current_state = next_state

        return plan

    def save(self, path: str):
        """Save world model."""
        data = {
            "n_transitions": sum(self.state_action_counts.values()),
            "unique_state_action_pairs": len(self._index),
            "state_action_counts": self.state_action_counts,
            "mean_rewards": {
                k: sum(v) / len(v) for k, v in self.state_action_rewards.items()
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    print("=" * 60)
    print("World Model — Prototype")
    print("=" * 60)

    env = SimpleEnvironment(size=8, seed=42)
    model = WorldModel()

    # Collect random experience
    print("\nCollecting experience...")
    for episode in range(50):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            actions = env.get_available_actions()
            action = random.choice(actions)
            next_state, reward, done = env.step(action)
            model.observe(Transition(state, action, next_state, reward, done))
            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            print(f"  Episode {episode}: reward={total_reward:.2f}, steps={env.step_count}")

    print(f"\nCollected {len(model.transitions)} transitions")

    # Test planning
    print("\nTesting greedy plan...")
    state = env.reset()
    actions = env.get_available_actions()
    plan = model.plan(state, actions, depth=10)
    print(f"Planned {len(plan)} actions: {[a.name for a in plan]}")

    # Execute plan
    print("\nExecuting plan...")
    state = env.reset()
    total_reward = 0
    done = False
    for action in plan:
        if done:
            break
        state, reward, done = env.step(action)
        total_reward += reward
        print(f"  {action.name} -> pos=({state.features['agent_x']:.0f}, {state.features['agent_y']:.0f}), reward={reward:.2f}")

    print(f"\nTotal reward: {total_reward:.2f}")
    model.save("/home/himanshu/Desktop/AGI_RESEARCH_LAB/20_OUTPUT/world_model.json")
    print("World model saved.")
