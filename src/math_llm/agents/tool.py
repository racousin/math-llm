"""
Tool agent: Iterative proof with Lean tool calls.

Uses the LLM to generate tactics step by step, calling the Lean server
to verify each step and get feedback (errors, remaining goals).
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from math_llm.data import Problem
from math_llm.lean_server import LeanServer, LeanResult
from math_llm.agents.simple import AgentResult, extract_proof


# System prompt for tool-based agent
SYSTEM_PROMPT = """You are a Lean 4 theorem prover with access to a Lean REPL tool.

You can call the lean_check tool to verify tactics. After each call, you'll see:
- SUCCESS: Tactic accepted, remaining goals shown
- ERROR: Tactic failed, error message shown
- COMPLETE: Proof finished!

Strategy:
1. Analyze the goal type
2. Try appropriate tactics based on goal structure
3. If a tactic fails, try alternatives
4. Build proof incrementally

Tactics reference:
- Finishing: rfl, norm_num, decide, ring, omega, linarith, positivity
- Rewriting: rw [h], simp, ring_nf, field_simp
- Structure: intro, obtain, rcases, use, constructor, left/right
- Application: exact, apply, have, calc, refine
- Other: ext, induction, cases, simp_all

Output format: When suggesting a tactic, output ONLY the tactic on a single line.
"""


@dataclass
class Step:
    """A single step in the proof trajectory."""
    tactic: str
    result: LeanResult
    step_num: int

    def to_dict(self) -> dict:
        return {
            "step": self.step_num,
            "tactic": self.tactic,
            "success": self.result.success,
            "complete": self.result.complete,
            "errors": self.result.errors,
            "goals": self.result.goals,
        }


@dataclass
class Trajectory:
    """Complete proof attempt trajectory."""
    problem_id: str
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    complete: bool = False

    def add_step(self, tactic: str, result: LeanResult) -> None:
        step = Step(tactic=tactic, result=result, step_num=len(self.steps) + 1)
        self.steps.append(step)
        if result.complete:
            self.success = True
            self.complete = True

    @property
    def final_proof(self) -> str:
        """Get the combined proof from successful steps."""
        successful = [s.tactic for s in self.steps if s.result.success]
        return "\n".join(successful)

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "num_steps": len(self.steps),
            "success": self.success,
            "complete": self.complete,
            "steps": [s.to_dict() for s in self.steps],
            "final_proof": self.final_proof,
        }


class ToolAgent:
    """
    Tool-based iterative proof agent.

    Generates tactics step by step, using Lean server feedback
    to guide the proof search.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        lean_server: Optional[LeanServer] = None,
        max_steps: int = 10,
        max_new_tokens: int = 128,
        temperature: float = 0.3,
    ):
        self.model_name = model_name
        self.lean_server = lean_server
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """Load the LLM model."""
        if self._model is not None:
            return

        print(f"[agent] Loading model {self.model_name}...")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.bfloat16
            elif torch.backends.mps.is_available():
                device_map = "mps"
                torch_dtype = torch.float16
            else:
                device_map = "cpu"
                torch_dtype = torch.float32

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

            print(f"[agent] Model loaded on {device_map}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _build_prompt(
        self,
        problem: Problem,
        trajectory: Trajectory,
    ) -> str:
        """Build prompt with problem and trajectory context."""
        parts = []

        # Problem statement
        if problem.description:
            parts.append(f"Problem: {problem.description}")
        parts.append(f"Theorem:\n```lean4\n{problem.statement}\n```")

        # Trajectory history
        if trajectory.steps:
            parts.append("\n--- Proof Progress ---")
            for step in trajectory.steps[-5:]:  # Last 5 steps
                status = "✓" if step.result.success else "✗"
                parts.append(f"Step {step.step_num}: `{step.tactic}` [{status}]")
                if step.result.errors:
                    parts.append(f"  Error: {step.result.errors[0][:100]}")
                elif step.result.goals:
                    parts.append(f"  Goals: {step.result.goals[0][:100]}")

            # Current state
            last = trajectory.steps[-1]
            if last.result.success and last.result.goals:
                parts.append(f"\nCurrent goal: {last.result.goals[0]}")

        parts.append("\nNext tactic (one line only):")
        return "\n".join(parts)

    def _generate_tactic(self, prompt: str) -> str:
        """Generate next tactic using LLM."""
        self.load_model()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        chat_prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(chat_prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return extract_proof(response)

    def solve(self, problem: Problem) -> AgentResult:
        """
        Solve a problem iteratively with Lean tool feedback.

        Args:
            problem: The problem to solve

        Returns:
            AgentResult with trajectory information
        """
        if self.lean_server is None:
            return AgentResult(
                problem_id=problem.id,
                success=False,
                complete=False,
                proof="",
                error="ToolAgent requires a LeanServer",
            )

        trajectory = Trajectory(problem_id=problem.id)
        accumulated_proof = []

        for step_num in range(self.max_steps):
            # Build prompt with current state
            prompt = self._build_prompt(problem, trajectory)

            # Generate next tactic
            try:
                tactic = self._generate_tactic(prompt)
            except Exception as e:
                return AgentResult(
                    problem_id=problem.id,
                    success=trajectory.success,
                    complete=trajectory.complete,
                    proof=trajectory.final_proof,
                    error=f"Generation failed at step {step_num + 1}: {e}",
                )

            if not tactic.strip():
                continue  # Skip empty tactics

            # Build accumulated proof
            test_proof = "\n".join(accumulated_proof + [tactic])

            # Verify with Lean
            try:
                result = self.lean_server.check_proof(problem.statement, test_proof)
            except Exception as e:
                return AgentResult(
                    problem_id=problem.id,
                    success=trajectory.success,
                    complete=trajectory.complete,
                    proof=trajectory.final_proof,
                    error=f"Lean check failed at step {step_num + 1}: {e}",
                )

            trajectory.add_step(tactic, result)

            # Update accumulated proof if step succeeded
            if result.success:
                accumulated_proof.append(tactic)

            # Check for completion
            if result.complete:
                print(f"[agent] Proof complete in {step_num + 1} steps!")
                break

        return AgentResult(
            problem_id=problem.id,
            success=trajectory.success,
            complete=trajectory.complete,
            proof=trajectory.final_proof,
            lean_result=trajectory.steps[-1].result if trajectory.steps else None,
        )
