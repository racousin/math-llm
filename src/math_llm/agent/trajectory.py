"""
Trajectory management for agent interactions.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

from math_llm.lean.executor import ExecutionFeedback


@dataclass
class Step:
    """A single step in the agent's trajectory."""

    step_num: int
    action: str  # The code/tactic produced by the LLM
    result: ExecutionFeedback  # Result from Lean execution
    thinking: Optional[str] = None  # LLM's reasoning (if available)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_prompt_format(self) -> str:
        """Format step for inclusion in LLM prompt."""
        parts = [
            f"<step num=\"{self.step_num}\">",
            f"<action>\n{self.action}\n</action>",
            self.result.to_prompt(),
            "</step>",
        ]
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "step_num": self.step_num,
            "action": self.action,
            "thinking": self.thinking,
            "result": {
                "success": self.result.success,
                "complete": self.result.complete,
                "message": self.result.message,
                "goals": self.result.goals,
                "errors": self.result.errors,
                "hints": self.result.hints,
            },
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Step":
        """Create from dictionary."""
        # Create a minimal feedback object
        feedback = ExecutionFeedback(
            success=data["result"]["success"],
            complete=data["result"]["complete"],
            message=data["result"]["message"],
            goals=data["result"]["goals"],
            errors=data["result"]["errors"],
            hints=data["result"]["hints"],
            raw_result=None,  # Not preserved in serialization
        )
        return cls(
            step_num=data["step_num"],
            action=data["action"],
            result=feedback,
            thinking=data.get("thinking"),
            timestamp=data.get("timestamp", 0),
        )


@dataclass
class Trajectory:
    """
    Complete trajectory of an agent solving a problem.

    Contains:
    - The problem being solved
    - All steps taken
    - Final outcome
    - Metadata for training
    """

    problem_id: str
    statement: str
    description: Optional[str] = None
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    total_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def final_action(self) -> Optional[str]:
        if self.steps:
            return self.steps[-1].action
        return None

    @property
    def is_complete(self) -> bool:
        """Check if the trajectory led to a complete proof."""
        if self.steps:
            return self.steps[-1].result.complete
        return False

    def add_step(self, action: str, result: ExecutionFeedback, thinking: Optional[str] = None) -> Step:
        """Add a new step to the trajectory."""
        step = Step(
            step_num=len(self.steps) + 1,
            action=action,
            result=result,
            thinking=thinking,
        )
        self.steps.append(step)
        self.success = result.complete
        return step

    def to_prompt_format(self) -> str:
        """Convert entire trajectory to prompt format for LLM."""
        parts = ["<trajectory>"]
        for step in self.steps:
            parts.append(step.to_prompt_format())
        parts.append("</trajectory>")
        return "\n".join(parts)

    def to_training_format(self) -> dict:
        """
        Convert to format suitable for RL training.

        Returns dict with:
        - prompt: Initial prompt with problem
        - trajectory: Full trajectory string
        - actions: List of actions taken
        - rewards: List of step rewards
        - final_reward: Overall success reward
        """
        # Build the initial prompt
        prompt_parts = []
        if self.description:
            prompt_parts.append(f"Problem: {self.description}")
        prompt_parts.append(f"Prove the following in Lean 4:\n\n```lean4\n{self.statement}\n```")
        prompt_parts.append("\nProvide your proof step by step.")

        # Build trajectory string
        trajectory_str = self.to_prompt_format()

        # Extract actions and compute step rewards
        actions = [step.action for step in self.steps]
        rewards = []
        for i, step in enumerate(self.steps):
            if step.result.complete:
                rewards.append(1.0)  # Success
            elif step.result.success:
                rewards.append(0.1)  # Progress
            else:
                rewards.append(-0.1)  # Error

        return {
            "prompt": "\n\n".join(prompt_parts),
            "trajectory": trajectory_str,
            "actions": actions,
            "rewards": rewards,
            "final_reward": 1.0 if self.success else 0.0,
            "num_steps": self.num_steps,
            "success": self.success,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "problem_id": self.problem_id,
            "statement": self.statement,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "success": self.success,
            "total_time": self.total_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        """Create from dictionary."""
        trajectory = cls(
            problem_id=data["problem_id"],
            statement=data["statement"],
            description=data.get("description"),
            success=data.get("success", False),
            total_time=data.get("total_time", 0.0),
            metadata=data.get("metadata", {}),
        )
        trajectory.steps = [Step.from_dict(s) for s in data.get("steps", [])]
        return trajectory

    def save(self, path: str) -> None:
        """Save trajectory to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        """Load trajectory from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class TrajectoryBatch:
    """Batch of trajectories for training."""

    trajectories: list[Trajectory]

    @property
    def success_rate(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(1 for t in self.trajectories if t.success) / len(self.trajectories)

    @property
    def avg_steps(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(t.num_steps for t in self.trajectories) / len(self.trajectories)

    def to_training_data(self) -> list[dict]:
        """Convert batch to training format."""
        return [t.to_training_format() for t in self.trajectories]

    def save(self, path: str) -> None:
        """Save batch to JSON file."""
        data = [t.to_dict() for t in self.trajectories]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrajectoryBatch":
        """Load batch from JSON file."""
        with open(path) as f:
            data = json.load(f)
        trajectories = [Trajectory.from_dict(d) for d in data]
        return cls(trajectories)
