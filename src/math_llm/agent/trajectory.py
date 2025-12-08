"""
Trajectory management for agent interactions.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from math_llm.lean.executor import ExecutionResult


@dataclass
class Step:
    """A single step in the agent's trajectory."""

    step_num: int
    action: str
    result: ExecutionResult
    thinking: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = {
            "step_num": self.step_num,
            "action": self.action,
            "thinking": self.thinking,
            "result": {
                "success": self.result.success,
                "complete": self.result.complete,
                "goals": self.result.goals,
                "errors": self.result.errors,
            },
            "timestamp": self.timestamp,
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Step":
        """Create from dictionary."""
        result = ExecutionResult(
            success=data["result"]["success"],
            complete=data["result"]["complete"],
            goals=data["result"]["goals"],
            errors=data["result"]["errors"],
            raw=None,
        )
        return cls(
            step_num=data["step_num"],
            action=data["action"],
            result=result,
            thinking=data.get("thinking"),
            timestamp=data.get("timestamp", 0),
            metadata=data.get("metadata"),
        )


@dataclass
class Trajectory:
    """Complete trajectory of an agent solving a problem."""

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
        return self.steps[-1].action if self.steps else None

    @property
    def is_complete(self) -> bool:
        return self.steps[-1].result.complete if self.steps else False

    def add_step(self, action: str, result: ExecutionResult, thinking: Optional[str] = None) -> Step:
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

    def to_training_format(self) -> dict:
        """Convert to format suitable for RL training."""
        prompt_parts = []
        if self.description:
            prompt_parts.append(f"Problem: {self.description}")
        prompt_parts.append(f"Prove:\n```lean4\n{self.statement}\n```")

        actions = [step.action for step in self.steps]
        rewards = []
        for step in self.steps:
            if step.result.complete:
                rewards.append(1.0)
            elif step.result.success:
                rewards.append(0.1)
            else:
                rewards.append(-0.1)

        return {
            "prompt": "\n\n".join(prompt_parts),
            "actions": actions,
            "rewards": rewards,
            "final_reward": 1.0 if self.success else 0.0,
            "num_steps": self.num_steps,
            "success": self.success,
        }

    def to_dict(self) -> dict:
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
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Trajectory":
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
        return [t.to_training_format() for t in self.trajectories]

    def save(self, path: str) -> None:
        data = [t.to_dict() for t in self.trajectories]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrajectoryBatch":
        with open(path) as f:
            data = json.load(f)
        return cls(trajectories=[Trajectory.from_dict(d) for d in data])
