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


class DiverseTacticReward(RewardFunction):
    """
    Reward function that encourages diverse tactic usage.

    Better for learning advanced strategies because:
    - Rewards using different tactics (not just simp)
    - Rewards using library lemmas
    - Tracks progress through goal reduction
    - Penalizes repetitive/stuck behavior
    """

    # Tactics categorized by type
    AUTOMATION_TACTICS = {"simp", "omega", "ring", "linarith", "decide", "trivial", "norm_num", "nlinarith"}
    STRUCTURAL_TACTICS = {"intro", "intros", "constructor", "cases", "induction", "rcases", "obtain"}
    PROOF_TACTICS = {"exact", "apply", "rw", "rewrite", "rfl", "have", "let", "show"}
    ADVANCED_TACTICS = {"ext", "funext", "congr", "conv", "calc", "suffices", "by_contra", "push_neg"}

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self._used_tactics: set[str] = set()
        self._used_lemmas: set[str] = set()

    def compute_step_reward(self, step: Step, trajectory: Trajectory) -> float:
        """Compute reward with diversity bonus."""
        import re
        reward = 0.0

        if step.result.complete:
            # Big reward for completion
            base_reward = self.config.success_reward
            efficiency_bonus = max(0.1, 1.0 - 0.08 * step.step_num)

            # Diversity bonus: reward using different tactic types
            diversity_bonus = self._compute_diversity_bonus(trajectory)

            reward = base_reward * efficiency_bonus + diversity_bonus
        elif step.result.success:
            # Partial credit for progress
            reward = 0.1

            # Extract tactics used in this step
            tactics_used = self._extract_tactics(step.action)
            lemmas_used = self._extract_lemmas(step.action)

            # Bonus for using new tactics
            new_tactics = tactics_used - self._used_tactics
            if new_tactics:
                reward += 0.05 * len(new_tactics)
                self._used_tactics.update(new_tactics)

            # Bonus for using lemmas (not just automation)
            if lemmas_used:
                reward += 0.1 * min(len(lemmas_used), 3)  # Cap bonus
                self._used_lemmas.update(lemmas_used)

            # Bonus for structural tactics (building proof)
            structural_used = tactics_used & self.STRUCTURAL_TACTICS
            if structural_used:
                reward += 0.05

            # Penalty for only using simp repeatedly
            if tactics_used == {"simp"} and "simp" in self._used_tactics:
                reward -= 0.02
        else:
            # Error
            reward = self.config.error_penalty

        # Step cost
        reward += self.config.step_penalty

        return reward

    def _extract_tactics(self, action: str) -> set[str]:
        """Extract tactic names from proof code."""
        import re
        tactics = set()

        # Match tactic at start of line or after semicolon
        tactic_pattern = r'(?:^|\n|;)\s*(\w+)'
        for match in re.finditer(tactic_pattern, action):
            tactic = match.group(1).lower()
            all_tactics = (self.AUTOMATION_TACTICS | self.STRUCTURAL_TACTICS |
                          self.PROOF_TACTICS | self.ADVANCED_TACTICS)
            if tactic in all_tactics:
                tactics.add(tactic)

        return tactics

    def _extract_lemmas(self, action: str) -> set[str]:
        """Extract lemma names from proof code."""
        import re
        lemmas = set()

        # Match qualified names like Nat.add_comm, List.map_append
        lemma_pattern = r'(\w+\.\w+(?:\.\w+)*)'
        for match in re.finditer(lemma_pattern, action):
            lemmas.add(match.group(1))

        return lemmas

    def _compute_diversity_bonus(self, trajectory: Trajectory) -> float:
        """Compute bonus based on tactic diversity in trajectory."""
        all_tactics = set()
        all_lemmas = set()

        for step in trajectory.steps:
            all_tactics.update(self._extract_tactics(step.action))
            all_lemmas.update(self._extract_lemmas(step.action))

        # Count tactic categories used
        categories_used = 0
        if all_tactics & self.AUTOMATION_TACTICS:
            categories_used += 1
        if all_tactics & self.STRUCTURAL_TACTICS:
            categories_used += 1
        if all_tactics & self.PROOF_TACTICS:
            categories_used += 1
        if all_tactics & self.ADVANCED_TACTICS:
            categories_used += 1

        # Bonus: 0.1 per category, plus 0.05 per unique lemma (capped)
        diversity_bonus = 0.1 * categories_used + 0.05 * min(len(all_lemmas), 5)

        return diversity_bonus

    def compute_trajectory_reward(self, trajectory: Trajectory) -> float:
        """Compute total trajectory reward with diversity."""
        # Reset tracking for new trajectory
        self._used_tactics = set()
        self._used_lemmas = set()

        if not trajectory.steps:
            return 0.0

        total_reward = 0.0
        gamma = self.config.iteration_decay

        for i, step in enumerate(trajectory.steps):
            step_reward = self.compute_step_reward(step, trajectory)
            total_reward += step_reward * (gamma ** i)

        # Final adjustment
        if trajectory.success:
            total_reward = max(total_reward, self.config.success_reward * 0.5)
        else:
            total_reward = min(total_reward, 0.0)

        return total_reward


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
