# Context Engineering Strategy

How we construct prompts and manage context for the Lean theorem proving agent.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LeanAgent                             │
├─────────────────────────────────────────────────────────────┤
│  PromptBuilder          │  LeanExecutor                      │
│  - system prompt        │  - execute tactic                  │
│  - initial prompt       │  - parse LLM output                │
│  - continuation prompt  │  - clean errors                    │
│                         │                                    │
│  ProofState             │  ExecutionResult                   │
│  - problem              │  - success/complete                │
│  - steps history        │  - goals                           │
│  - failed tactics       │  - errors                          │
└─────────────────────────────────────────────────────────────┘
```

## 1. Lean Environment

Every proof runs with:

```lean
import Mathlib
import Aesop
```

**No namespaces opened.** Use qualified names: `Nat.add_comm`, not `add_comm`.

This avoids name conflicts between theorem names and Mathlib lemmas.

## 2. System Prompt

Minimal, direct:

```
You are a Lean 4 theorem prover. Output ONE tactic.

Environment: Mathlib imported.

Tactics: simp, omega, ring, linarith, rfl, exact, apply, rw, intro, cases, induction
```

## 3. Initial Prompt

```
Problem: {description}

Theorem:
```lean4
{statement}
```

Goal: `{extracted_goal}`
```

## 4. Continuation Prompt

```
Problem: {description}

Theorem:
```lean4
{statement}
```

Last tactic: `{tactic}`
Result: {OK | ERROR - message}

[If goals remain:]
Remaining goals:
  ⊢ {goal_1}

[If failed tactics:]
Avoid: {tactic_1}, {tactic_2}
```

## 5. Proof State

```python
@dataclass
class ProofState:
    problem: LeanProblem
    steps: list[dict]        # {tactic, success, errors, goals}
    failed_tactics: list[str]
```

## 6. Execution Flow

```
1. Initial prompt → LLM → tactic
2. Parse tactic
3. Execute in Lean
4. Update state
5. Build continuation prompt
6. Repeat until complete or max_steps
```

## 7. Error Handling

No hints. Model receives raw Lean errors (file paths stripped).

## 8. Files

| File | Purpose |
|------|---------|
| `agent/context.py` | System prompt, prompt builder, proof state |
| `agent/agent.py` | Main agent loop |
| `lean/executor.py` | Tactic parsing, execution |
| `lean/server.py` | Lean process communication |
