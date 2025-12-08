"""
Reward functions for RL training on Lean theorem proving.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from math_llm.agent.trajectory import Trajectory, Step


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    success_reward: float = 1.0
    failure_reward: float = 0.0
    step_penalty: float = -0.01
    iteration_decay: float = 0.95
    partial_credit: bool = True
    error_penalty: float = -0.1


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def compute_step_reward(self, step: Step, trajectory: Trajectory) -> float:
        """Compute reward for a single step."""
        pass

    @abstractmethod
    def compute_trajectory_reward(self, trajectory: Trajectory) -> float:
        """Compute total reward for a trajectory."""
        pass


class ProofReward(RewardFunction):
    """
    Reward function for theorem proving.

    Reward structure:
    - +1.0 for completing the proof
    - +0.1 for progress (code accepted, goals reduced)
    - -0.01 per step (encourages efficiency)
    - -0.1 for errors
    - Exponential decay for longer trajectories
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def compute_step_reward(self, step: Step, trajectory: Trajectory) -> float:
        """Compute reward for a single step."""
        reward = 0.0

        if step.result.complete:
            # Proof completed - big reward with bonus for fewer steps
            base_reward = self.config.success_reward
            efficiency_bonus = max(0, 0.5 - 0.05 * step.step_num)  # Bonus for early completion
            reward = base_reward + efficiency_bonus
        elif step.result.success:
            # Code accepted but incomplete
            if self.config.partial_credit:
                # Give partial credit based on progress
                # More goals = less progress
                num_goals = len(step.result.goals)
                if num_goals == 0:
                    reward = 0.3  # No goals but not complete (rare)
                elif num_goals == 1:
                    reward = 0.2  # One goal remaining
                else:
                    reward = 0.1  # Multiple goals
            else:
                reward = 0.0
        else:
            # Error
            reward = self.config.error_penalty

        # Apply step penalty
        reward += self.config.step_penalty

        return reward

    def compute_trajectory_reward(self, trajectory: Trajectory) -> float:
        """
        Compute total reward for a trajectory.

        Uses discounted sum of step rewards with bonus for success.
        """
        if not trajectory.steps:
            return 0.0

        total_reward = 0.0
        gamma = self.config.iteration_decay

        for i, step in enumerate(trajectory.steps):
            step_reward = self.compute_step_reward(step, trajectory)
            # Apply discount factor
            discounted_reward = step_reward * (gamma ** i)
            total_reward += discounted_reward

        # Final success/failure adjustment
        if trajectory.success:
            # Bonus for success (already included in step reward but ensure positive)
            total_reward = max(total_reward, self.config.success_reward)
        else:
            # Penalty for failure
            total_reward = min(total_reward, self.config.failure_reward)

        return total_reward

    def compute_step_rewards(self, trajectory: Trajectory) -> list[float]:
        """Compute rewards for all steps in trajectory."""
        return [self.compute_step_reward(step, trajectory) for step in trajectory.steps]


class SparseReward(RewardFunction):
    """
    Simple sparse reward function.

    Only gives reward at the end:
    - +1 for success
    - 0 for failure
    """

    def __init__(self, success_reward: float = 1.0, failure_reward: float = 0.0):
        self.success_reward = success_reward
        self.failure_reward = failure_reward

    def compute_step_reward(self, step: Step, trajectory: Trajectory) -> float:
        """Only give reward on final step."""
        if step.step_num == len(trajectory.steps):
            return self.success_reward if trajectory.success else self.failure_reward
        return 0.0

    def compute_trajectory_reward(self, trajectory: Trajectory) -> float:
        return self.success_reward if trajectory.success else self.failure_reward


class EfficiencyReward(RewardFunction):
    """
    Reward function that emphasizes efficiency.

    Higher rewards for solving in fewer steps.
    """

    def __init__(
        self,
        max_steps: int = 10,
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
    ):
        self.max_steps = max_steps
        self.success_reward = success_reward
        self.failure_reward = failure_reward

    def compute_step_reward(self, step: Step, trajectory: Trajectory) -> float:
        if step.result.complete:
            # Reward inversely proportional to number of steps
            efficiency = 1.0 - (step.step_num - 1) / self.max_steps
            return self.success_reward * efficiency
        return 0.0

    def compute_trajectory_reward(self, trajectory: Trajectory) -> float:
        if trajectory.success:
            efficiency = 1.0 - (trajectory.num_steps - 1) / self.max_steps
            return self.success_reward * max(0.1, efficiency)
        return self.failure_reward


def compute_trajectory_reward(
    trajectory: Trajectory,
    reward_fn: Optional[RewardFunction] = None,
) -> float:
    """Convenience function to compute trajectory reward."""
    if reward_fn is None:
        reward_fn = ProofReward()
    return reward_fn.compute_trajectory_reward(trajectory)


def compute_advantages(
    rewards: list[float],
    values: list[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> list[float]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: List of rewards for each step
        values: List of value estimates for each step
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        List of advantage estimates
    """
    advantages = []
    last_advantage = 0.0
    last_value = 0.0

    # Compute advantages in reverse order
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0  # Terminal state
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        advantage = delta + gamma * lam * last_advantage
        advantages.insert(0, advantage)
        last_advantage = advantage
        last_value = values[t]

    return advantages
