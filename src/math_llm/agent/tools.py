"""
Tool system for Lean theorem proving agent.

Provides structured tools that the agent can call:
- run_tactic: Execute tactic, get new state
- suggest_tactics: Get exact?/apply? suggestions
- search_lemmas: Search by name/type
- check_type: Get type of expression
- get_goal: Current goal state
- store_hypothesis: Remember intermediate result
"""

import re
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum


class ToolStatus(Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    status: ToolStatus
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_prompt(self) -> str:
        """Convert result to prompt format."""
        if self.status == ToolStatus.SUCCESS:
            if isinstance(self.output, list):
                return "\n".join(str(x) for x in self.output)
            return str(self.output)
        else:
            return f"Error: {self.error}"

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
        }


@dataclass
class ToolCall:
    """Record of a tool call."""
    tool_name: str
    arguments: dict
    result: ToolResult
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result.to_dict(),
            "timestamp": self.timestamp,
        }


class Tool(ABC):
    """Abstract base class for proof tools."""

    name: str
    description: str
    parameters: dict  # JSON Schema for parameters

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

    def to_schema(self) -> dict:
        """Get tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolMetrics:
    """Metrics for monitoring tool usage."""

    def __init__(self):
        self.calls: list[ToolCall] = []
        self.by_tool: dict[str, list[ToolCall]] = {}
        self.success_count: dict[str, int] = {}
        self.error_count: dict[str, int] = {}
        self.total_time: dict[str, float] = {}

    def record(self, call: ToolCall) -> None:
        """Record a tool call."""
        self.calls.append(call)

        tool_name = call.tool_name
        if tool_name not in self.by_tool:
            self.by_tool[tool_name] = []
            self.success_count[tool_name] = 0
            self.error_count[tool_name] = 0
            self.total_time[tool_name] = 0.0

        self.by_tool[tool_name].append(call)
        self.total_time[tool_name] += call.result.execution_time

        if call.result.status == ToolStatus.SUCCESS:
            self.success_count[tool_name] += 1
        else:
            self.error_count[tool_name] += 1

    def reset(self) -> None:
        """Reset all metrics."""
        self.calls = []
        self.by_tool = {}
        self.success_count = {}
        self.error_count = {}
        self.total_time = {}

    def summary(self) -> dict:
        """Get summary of tool usage."""
        summary = {
            "total_calls": len(self.calls),
            "by_tool": {},
        }

        for tool_name in self.by_tool:
            total = len(self.by_tool[tool_name])
            success = self.success_count[tool_name]
            summary["by_tool"][tool_name] = {
                "calls": total,
                "success": success,
                "errors": self.error_count[tool_name],
                "success_rate": success / total if total > 0 else 0,
                "avg_time": self.total_time[tool_name] / total if total > 0 else 0,
            }

        return summary

    def to_table(self) -> str:
        """Format metrics as table."""
        lines = ["Tool Usage Metrics", "=" * 60]
        lines.append(f"{'Tool':<20} {'Calls':>8} {'Success':>8} {'Rate':>8} {'Avg Time':>10}")
        lines.append("-" * 60)

        for tool_name in sorted(self.by_tool.keys()):
            total = len(self.by_tool[tool_name])
            success = self.success_count[tool_name]
            rate = success / total if total > 0 else 0
            avg_time = self.total_time[tool_name] / total if total > 0 else 0

            lines.append(f"{tool_name:<20} {total:>8} {success:>8} {rate:>7.1%} {avg_time:>9.3f}s")

        lines.append("-" * 60)
        lines.append(f"{'Total':<20} {len(self.calls):>8}")

        return "\n".join(lines)


# =============================================================================
# Concrete Tool Implementations
# =============================================================================

class RunTacticTool(Tool):
    """Execute a tactic and get the new proof state."""

    name = "run_tactic"
    description = "Execute a Lean 4 tactic and get the resulting proof state. Returns success/failure and remaining goals."
    parameters = {
        "type": "object",
        "properties": {
            "tactic": {
                "type": "string",
                "description": "The tactic to execute (e.g., 'simp', 'exact Nat.add_comm a b', 'intro h')"
            }
        },
        "required": ["tactic"]
    }

    def __init__(self, executor: "LeanExecutor", statement: str):
        self.executor = executor
        self.statement = statement
        self.current_proof: list[str] = []

    def execute(self, tactic: str) -> ToolResult:
        start_time = time.time()

        try:
            # Build proof with new tactic
            self.current_proof.append(tactic)
            full_proof = "\n  ".join(self.current_proof)

            # Execute
            result = self.executor.execute_proof_attempt(
                statement=self.statement,
                proof=full_proof,
            )

            if result.complete:
                output = {"status": "complete", "message": "Proof complete!"}
            elif result.success:
                output = {
                    "status": "progress",
                    "goals": result.goals,
                    "message": f"Tactic accepted. {len(result.goals)} goal(s) remaining."
                }
            else:
                # Tactic failed - remove it
                self.current_proof.pop()
                output = {
                    "status": "failed",
                    "errors": result.errors,
                    "message": "Tactic failed."
                }
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output=output,
                    error="; ".join(result.errors[:2]),
                    execution_time=time.time() - start_time,
                )

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            if self.current_proof:
                self.current_proof.pop()
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def reset(self) -> None:
        """Reset proof state for new problem."""
        self.current_proof = []


class SuggestTacticsTool(Tool):
    """Get tactic suggestions using Lean's exact?/apply?."""

    name = "suggest_tactics"
    description = "Use Lean's suggestion tactics (exact?, apply?) to find applicable lemmas and tactics for the current goal."
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["exact", "apply", "simp", "all"],
                "description": "Type of suggestion: 'exact' for exact?, 'apply' for apply?, 'simp' for simp?, 'all' for all suggestions",
                "default": "all"
            }
        },
        "required": []
    }

    def __init__(self, server: "LeanServer", statement: str):
        self.server = server
        self.statement = statement

    def execute(self, mode: str = "all") -> ToolResult:
        start_time = time.time()

        try:
            suggestions = []

            if mode in ["exact", "all"]:
                suggestions.extend(self._try_suggestion("exact?"))

            if mode in ["apply", "all"]:
                suggestions.extend(self._try_suggestion("apply?"))

            if mode in ["simp", "all"]:
                suggestions.extend(self._try_suggestion("simp?"))

            # Deduplicate
            suggestions = list(dict.fromkeys(suggestions))

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=suggestions[:10],  # Limit results
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=[],
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _try_suggestion(self, tactic: str) -> list[str]:
        """Try a suggestion tactic and parse results."""
        if ":= by" in self.statement:
            code = self.statement.replace("sorry", tactic)
        else:
            code = f"{self.statement} := by {tactic}"

        result = self.server.execute(code)

        # Parse "Try this:" suggestions
        suggestions = []
        try_pattern = r"Try this:\s*(.+?)(?:\n|$)"
        for match in re.finditer(try_pattern, result.output):
            suggestions.append(match.group(1).strip())

        return suggestions


class SearchLemmasTool(Tool):
    """Search for lemmas by name or type pattern."""

    name = "search_lemmas"
    description = "Search Mathlib/Lean for lemmas matching a query. Can search by name pattern or type signature."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query: name pattern (e.g., 'add_comm') or type pattern"
            },
            "search_type": {
                "type": "string",
                "enum": ["name", "type"],
                "description": "Search by 'name' or 'type'",
                "default": "name"
            }
        },
        "required": ["query"]
    }

    # Common lemma database (subset for quick lookup)
    COMMON_LEMMAS = {
        # Nat arithmetic
        "add": ["Nat.add_comm", "Nat.add_assoc", "Nat.add_zero", "Nat.zero_add",
                "Nat.add_succ", "Nat.succ_add", "Nat.add_one"],
        "mul": ["Nat.mul_comm", "Nat.mul_assoc", "Nat.mul_one", "Nat.one_mul",
                "Nat.mul_zero", "Nat.zero_mul", "Nat.mul_add", "Nat.add_mul"],
        "sub": ["Nat.sub_self", "Nat.sub_zero", "Nat.add_sub_cancel"],
        "succ": ["Nat.succ_eq_add_one", "Nat.succ_pos", "Nat.succ_ne_zero"],
        # Int arithmetic
        "int": ["Int.add_comm", "Int.mul_comm", "Int.add_assoc", "Int.mul_assoc"],
        # Logic
        "and": ["And.intro", "And.left", "And.right", "And.comm"],
        "or": ["Or.inl", "Or.inr", "Or.comm", "Or.assoc"],
        "not": ["not_not", "not_and", "not_or"],
        "eq": ["Eq.refl", "Eq.symm", "Eq.trans", "Eq.subst"],
        "iff": ["Iff.intro", "Iff.mp", "Iff.mpr"],
        # Functions
        "comp": ["Function.comp", "Function.id"],
        "inj": ["Function.Injective", "Function.Surjective"],
        # Lists
        "list": ["List.nil", "List.cons", "List.append", "List.length", "List.map"],
        # Sets
        "set": ["Set.mem_def", "Set.subset_def", "Set.union_def", "Set.inter_def"],
    }

    def __init__(self, server: Optional["LeanServer"] = None):
        self.server = server

    def execute(self, query: str, search_type: str = "name") -> ToolResult:
        start_time = time.time()

        try:
            results = []
            query_lower = query.lower()

            # Search in common lemmas
            for category, lemmas in self.COMMON_LEMMAS.items():
                if query_lower in category:
                    results.extend(lemmas)
                else:
                    for lemma in lemmas:
                        if query_lower in lemma.lower():
                            results.append(lemma)

            # If server available, try #check
            if self.server and search_type == "type":
                check_result = self.server.execute(f"#check @{query}")
                if check_result.is_success:
                    results.append(f"{query}: {check_result.output.strip()}")

            # Deduplicate and limit
            results = list(dict.fromkeys(results))[:15]

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=results,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=[],
                error=str(e),
                execution_time=time.time() - start_time,
            )


class CheckTypeTool(Tool):
    """Get the type of an expression."""

    name = "check_type"
    description = "Get the type of a Lean expression using #check. Useful for understanding what type a lemma or term has."
    parameters = {
        "type": "object",
        "properties": {
            "expr": {
                "type": "string",
                "description": "The expression to check (e.g., 'Nat.add_comm', '@List.map')"
            }
        },
        "required": ["expr"]
    }

    def __init__(self, server: "LeanServer"):
        self.server = server

    def execute(self, expr: str) -> ToolResult:
        start_time = time.time()

        try:
            # Use #check to get type
            code = f"#check @{expr}" if not expr.startswith("@") else f"#check {expr}"
            result = self.server.execute(code)

            if result.is_success:
                # Parse type from output
                output = result.output.strip()
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=output,
                    execution_time=time.time() - start_time,
                )
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output=None,
                    error="; ".join(result.errors),
                    execution_time=time.time() - start_time,
                )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
            )


class GetGoalTool(Tool):
    """Get the current goal state."""

    name = "get_goal"
    description = "Get the current proof goal(s). Useful when context is long or to refresh understanding of what needs to be proved."
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    def __init__(self, run_tactic_tool: RunTacticTool):
        self.run_tactic = run_tactic_tool

    def execute(self) -> ToolResult:
        start_time = time.time()

        try:
            # Build current proof state
            if not self.run_tactic.current_proof:
                # No tactics yet - return initial goal
                result = self.run_tactic.executor.execute_proof_attempt(
                    statement=self.run_tactic.statement,
                    proof="sorry",
                )
            else:
                full_proof = "\n  ".join(self.run_tactic.current_proof)
                result = self.run_tactic.executor.execute_proof_attempt(
                    statement=self.run_tactic.statement,
                    proof=full_proof + "\n  sorry",  # Add sorry to see remaining goals
                )

            goals = result.goals if result.goals else ["(no goals - proof may be complete)"]

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output={
                    "goals": goals,
                    "tactics_so_far": self.run_tactic.current_proof,
                },
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
            )


class StoreHypothesisTool(Tool):
    """Store an intermediate hypothesis/result."""

    name = "store_hypothesis"
    description = "Remember an intermediate result or hypothesis for later use. Useful for multi-step proofs."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name for the hypothesis"
            },
            "statement": {
                "type": "string",
                "description": "The statement/type of the hypothesis"
            },
            "proof": {
                "type": "string",
                "description": "How to prove this hypothesis (tactic or term)",
                "default": ""
            }
        },
        "required": ["name", "statement"]
    }

    def __init__(self, memory: "ProofMemory"):
        self.memory = memory

    def execute(self, name: str, statement: str, proof: str = "") -> ToolResult:
        start_time = time.time()

        try:
            # Store in memory
            self.memory.hypotheses.append(f"{name} : {statement}")

            # Generate tactic to add this hypothesis
            if proof:
                tactic = f"have {name} : {statement} := {proof}"
            else:
                tactic = f"have {name} : {statement} := by sorry"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output={
                    "stored": f"{name} : {statement}",
                    "tactic": tactic,
                    "message": f"Hypothesis '{name}' stored. Use 'have' tactic to introduce it in the proof."
                },
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
            )


# =============================================================================
# Tool Registry and Manager
# =============================================================================

class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self.metrics = ToolMetrics()

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[dict]:
        """List all available tools."""
        return [tool.to_schema() for tool in self.tools.values()]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool and record metrics."""
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=f"Unknown tool: {tool_name}",
            )

        result = tool.execute(**kwargs)

        # Record call
        call = ToolCall(
            tool_name=tool_name,
            arguments=kwargs,
            result=result,
        )
        self.metrics.record(call)

        return result

    def get_metrics(self) -> ToolMetrics:
        """Get tool usage metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset metrics for new evaluation."""
        self.metrics.reset()

    def to_prompt(self) -> str:
        """Generate tool documentation for prompt."""
        lines = ["## Available Tools\n"]

        for tool in self.tools.values():
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}\n")

            if tool.parameters.get("properties"):
                lines.append("**Parameters:**")
                for param, schema in tool.parameters["properties"].items():
                    required = param in tool.parameters.get("required", [])
                    req_str = " (required)" if required else ""
                    lines.append(f"- `{param}`{req_str}: {schema.get('description', '')}")
                lines.append("")

        return "\n".join(lines)


def create_proof_tools(
    executor: "LeanExecutor",
    server: "LeanServer",
    statement: str,
    memory: "ProofMemory",
) -> ToolRegistry:
    """Create a tool registry with all proof tools for a problem."""
    registry = ToolRegistry()

    # Create tools
    run_tactic = RunTacticTool(executor, statement)
    registry.register(run_tactic)

    registry.register(SuggestTacticsTool(server, statement))
    registry.register(SearchLemmasTool(server))
    registry.register(CheckTypeTool(server))
    registry.register(GetGoalTool(run_tactic))
    registry.register(StoreHypothesisTool(memory))

    return registry
