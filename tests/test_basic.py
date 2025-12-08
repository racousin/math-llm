"""
Basic tests for Math-LLM.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConfig:
    """Test configuration module."""

    def test_load_config(self):
        from math_llm.config import load_config

        config = load_config("configs/dummy_test.yaml")
        assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.agent.max_steps == 3
        assert config.debug is True

    def test_default_config(self):
        from math_llm.config import Config

        config = Config()
        assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.agent.max_steps == 10
        assert config.training.learning_rate == 1e-5


class TestData:
    """Test data module."""

    def test_lean_problem(self):
        from math_llm.data.datasets import LeanProblem

        problem = LeanProblem(
            id="test/1",
            statement="theorem test : 1 = 1 := by sorry",
            description="Simple equality",
            source="test",
        )

        assert problem.id == "test/1"
        assert "sorry" in problem.statement
        assert problem.source == "test"

    def test_lean_problem_to_prompt(self):
        from math_llm.data.datasets import LeanProblem

        problem = LeanProblem(
            id="test/1",
            statement="theorem test : 1 = 1 := by sorry",
            description="Prove one equals one",
        )

        prompt = problem.to_prompt()
        assert "Prove one equals one" in prompt
        assert "lean4" in prompt
        assert problem.statement in prompt

    def test_lean_dataset(self):
        from math_llm.data.datasets import LeanDataset, LeanProblem

        problems = [
            LeanProblem(id=f"test/{i}", statement=f"theorem t{i} : True := trivial")
            for i in range(5)
        ]

        dataset = LeanDataset(problems)
        assert len(dataset) == 5
        assert dataset[0]["id"] == "test/0"

    def test_dummy_dataset(self):
        from math_llm.data.loaders import create_dummy_dataset

        dataset = create_dummy_dataset(10)
        assert len(dataset) == 10
        assert all(p.source == "dummy" for p in dataset.problems)

    def test_dataset_split(self):
        from math_llm.data.loaders import create_dummy_dataset

        dataset = create_dummy_dataset(100)
        train, val = dataset.split(train_ratio=0.8)

        assert len(train) == 80
        assert len(val) == 20


class TestLean:
    """Test Lean module."""

    def test_lean_result(self):
        from math_llm.lean.server import LeanResult, LeanResultStatus

        result = LeanResult(
            status=LeanResultStatus.SUCCESS,
            output="",
            errors=[],
            warnings=[],
            goals=[],
        )

        assert result.is_success
        assert result.is_complete  # No goals = complete

    def test_lean_result_incomplete(self):
        from math_llm.lean.server import LeanResult, LeanResultStatus

        result = LeanResult(
            status=LeanResultStatus.INCOMPLETE,
            output="",
            errors=[],
            warnings=[],
            goals=["⊢ 1 = 1"],
        )

        assert not result.is_complete

    def test_lean_action_from_code_block(self):
        from math_llm.lean.executor import LeanAction

        output = """Here's my proof:
```lean4
rfl
```
This should work."""

        action = LeanAction.from_llm_output(output)
        assert action.code == "rfl"

    def test_lean_action_from_plain_text(self):
        from math_llm.lean.executor import LeanAction

        output = "simp only [add_comm]"
        action = LeanAction.from_llm_output(output)
        assert "simp" in action.code


class TestTrajectory:
    """Test trajectory module."""

    def test_trajectory_creation(self):
        from math_llm.agent.trajectory import Trajectory

        traj = Trajectory(
            problem_id="test/1",
            statement="theorem test : True := trivial",
        )

        assert traj.problem_id == "test/1"
        assert traj.num_steps == 0
        assert not traj.success

    def test_trajectory_add_step(self):
        from math_llm.agent.trajectory import Trajectory
        from math_llm.lean.executor import ExecutionFeedback
        from math_llm.lean.server import LeanResult, LeanResultStatus

        traj = Trajectory(
            problem_id="test/1",
            statement="theorem test : True := trivial",
        )

        mock_result = LeanResult(
            status=LeanResultStatus.SUCCESS,
            output="",
            goals=[],
        )
        feedback = ExecutionFeedback(
            success=True,
            complete=True,
            message="Done",
            goals=[],
            errors=[],
            hints=[],
            raw_result=mock_result,
        )

        traj.add_step("trivial", feedback)

        assert traj.num_steps == 1
        assert traj.success
        assert traj.is_complete

    def test_trajectory_serialization(self):
        from math_llm.agent.trajectory import Trajectory
        from math_llm.lean.executor import ExecutionFeedback
        from math_llm.lean.server import LeanResult, LeanResultStatus

        traj = Trajectory(
            problem_id="test/1",
            statement="theorem test : True := trivial",
            description="Test problem",
        )

        mock_result = LeanResult(status=LeanResultStatus.SUCCESS, output="", goals=[])
        feedback = ExecutionFeedback(
            success=True, complete=True, message="", goals=[], errors=[], hints=[],
            raw_result=mock_result,
        )
        traj.add_step("trivial", feedback)

        # Serialize and deserialize
        data = traj.to_dict()
        loaded = Trajectory.from_dict(data)

        assert loaded.problem_id == traj.problem_id
        assert loaded.num_steps == traj.num_steps
        assert loaded.success == traj.success


class TestRewards:
    """Test reward functions."""

    def test_proof_reward_success(self):
        from math_llm.training.rewards import ProofReward
        from math_llm.agent.trajectory import Trajectory, Step
        from math_llm.lean.executor import ExecutionFeedback
        from math_llm.lean.server import LeanResult, LeanResultStatus

        reward_fn = ProofReward()

        # Create successful trajectory
        traj = Trajectory(problem_id="test", statement="theorem t : True")
        mock_result = LeanResult(status=LeanResultStatus.SUCCESS, output="", goals=[])
        feedback = ExecutionFeedback(
            success=True, complete=True, message="", goals=[], errors=[], hints=[],
            raw_result=mock_result,
        )
        traj.add_step("trivial", feedback)

        reward = reward_fn.compute_trajectory_reward(traj)
        assert reward >= 1.0  # Success should give positive reward

    def test_proof_reward_failure(self):
        from math_llm.training.rewards import ProofReward
        from math_llm.agent.trajectory import Trajectory
        from math_llm.lean.executor import ExecutionFeedback
        from math_llm.lean.server import LeanResult, LeanResultStatus

        reward_fn = ProofReward()

        # Create failed trajectory
        traj = Trajectory(problem_id="test", statement="theorem t : True")
        mock_result = LeanResult(status=LeanResultStatus.ERROR, output="", goals=[], errors=["Error"])
        feedback = ExecutionFeedback(
            success=False, complete=False, message="", goals=[], errors=["Error"], hints=[],
            raw_result=mock_result,
        )
        traj.add_step("wrong", feedback)

        reward = reward_fn.compute_trajectory_reward(traj)
        assert reward <= 0.0  # Failure should give non-positive reward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
