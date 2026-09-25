"""
AGI Research Lab — Prototype: Baseline LLM Agent
Tests core agent loop with memory and tool use.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ToolCall:
    """A tool invocation by the agent."""
    name: str
    arguments: Dict[str, Any]
    result: Any = None
    timestamp: float = field(default_factory=time.time)


class AgentMemory:
    """Sliding-window memory with importance weighting."""

    def __init__(self, max_messages: int = 100, max_tokens: int = 4000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: List[Message] = []
        self.important_facts: List[str] = []

    def add(self, message: Message):
        """Add a message to memory."""
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def recall(self, n: int = 10) -> List[Message]:
        """Recall the last n messages."""
        return self.messages[-n:]

    def remember_fact(self, fact: str):
        """Store an important fact permanently."""
        if fact not in self.important_facts:
            self.important_facts.append(fact)

    def get_context(self, system_prompt: str = "") -> List[Dict[str, str]]:
        """Build the full context for an LLM call."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if self.important_facts:
            facts_str = "\n".join(f"- {f}" for f in self.important_facts[-20:])
            messages.append({
                "role": "system",
                "content": f"Important facts remembered:\n{facts_str}"
            })
        for msg in self.messages[-self.max_messages:]:
            messages.append({"role": msg.role, "content": msg.content})
        return messages


class ToolRegistry:
    """Registry of available tools for the agent."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(self, name: str, description: str, parameters: Dict[str, Any], fn: Callable):
        """Register a tool."""
        self._tools[name] = fn
        self._schemas.append({
            "name": name,
            "description": description,
            "parameters": parameters,
        })

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by name."""
        if name not in self._tools:
            return {"error": f"Tool '{name}' not found"}
        try:
            return self._tools[name](**arguments)
        except Exception as e:
            return {"error": str(e)}

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas."""
        return self._schemas


class BaselineAgent:
    """
    Baseline AGI research agent.
    - Maintains memory across interactions
    - Uses tools for external actions
    - Supports multi-step reasoning
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_fn: Callable[[List[Dict[str, str]]], str],
        tools: Optional[ToolRegistry] = None,
        memory: Optional[AgentMemory] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm_fn = llm_fn
        self.tools = tools or ToolRegistry()
        self.memory = memory or AgentMemory()
        self.tool_calls: List[ToolCall] = []

    def step(self, user_input: str) -> str:
        """Single interaction step."""
        # Add user message to memory
        self.memory.add(Message(role="user", content=user_input))

        # Build context
        messages = self.memory.get_context(self.system_prompt)

        # Call LLM
        response = self.llm_fn(messages)

        # Add assistant response to memory
        self.memory.add(Message(role="assistant", content=response))

        return response

    def run_loop(self, prompt: str, max_steps: int = 10) -> str:
        """Multi-step reasoning loop with tool use."""
        current_input = prompt
        for i in range(max_steps):
            response = self.step(current_input)
            # In a real implementation, parse tool calls from response
            # For now, just return the response
            return response
        return "Max steps reached"

    def remember(self, fact: str):
        """Store an important fact."""
        self.memory.remember_fact(fact)

    def save(self, path: str):
        """Save agent state to disk."""
        state = {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "tool_calls": [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "timestamp": tc.timestamp,
                }
                for tc in self.tool_calls
            ],
            "memory": {
                "messages": [m.to_dict() for m in self.memory.messages],
                "important_facts": self.memory.important_facts,
            },
        }
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: str, llm_fn: Callable, tools: Optional[ToolRegistry] = None) -> "BaselineAgent":
        """Load agent state from disk."""
        state = json.loads(Path(path).read_text())
        agent = cls(
            name=state["name"],
            system_prompt=state["system_prompt"],
            llm_fn=llm_fn,
            tools=tools,
        )
        agent.memory.important_facts = state["memory"]["important_facts"]
        return agent


# === Demo: Mock LLM function for testing ===
def mock_llm(messages: List[Dict[str, str]]) -> str:
    """A simple mock LLM for testing without API keys."""
    last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
    return f"I received your message: '{last_user}'. This is a mock response."


if __name__ == "__main__":
    # Demo usage
    agent = BaselineAgent(
        name="agi-researcher-v0",
        system_prompt="You are an AGI research assistant. Think step by step.",
        llm_fn=mock_llm,
    )

    # Register a simple tool
    agent.tools.register(
        name="search",
        description="Search for information",
        parameters={"query": {"type": "string"}},
        fn=lambda query: f"Mock search results for: {query}",
    )

    # Run a test interaction
    response = agent.step("What is the current state of AGI research?")
    print(f"Agent: {response}")

    agent.remember("AGI research is progressing rapidly in 2025.")
    print(f"Facts remembered: {agent.memory.important_facts}")

    # Save state
    agent.save("/home/himanshu/Desktop/AGI_RESEARCH_LAB/20_OUTPUT/agent_state.json")
    print("Agent state saved.")
