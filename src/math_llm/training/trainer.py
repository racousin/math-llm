"""
RL Trainer for Lean theorem proving using TRL.

Implements PPO-based training on proof trajectories.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from rich.console import Console
from rich.progress import Progress

console = Console()


@dataclass
class TrainingStats:
    """Statistics from training."""

    epoch: int
    step: int
    loss: float
    policy_loss: float
    value_loss: float
    kl_divergence: float
    entropy: float
    mean_reward: float
    success_rate: float

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "kl_divergence": self.kl_divergence,
            "entropy": self.entropy,
            "mean_reward": self.mean_reward,
            "success_rate": self.success_rate,
        }


class RLTrainer:
    """
    Reinforcement Learning trainer for theorem proving.

    Uses TRL's PPOTrainer with custom reward function
    based on Lean execution feedback.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: "TrainingConfig",
        lean_executor=None,
        reward_fn=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.lean_executor = lean_executor
        self.reward_fn = reward_fn

        self.ppo_trainer = None
        self.training_stats: List[TrainingStats] = []

    def setup(self) -> None:
        """Setup the trainer with TRL."""
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
        from peft import LoraConfig, get_peft_model, TaskType

        console.print("[blue]Setting up RL trainer...[/blue]")

        # Setup PEFT/LoRA if configured
        if self.config.use_peft:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )

            # Wrap model with value head and PEFT
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model,
                peft_config=lora_config,
            )
        else:
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model)

        # PPO configuration
        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.batch_size // 2,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            ppo_epochs=self.config.ppo_epochs,
            kl_penalty="kl",
            init_kl_coef=self.config.kl_coef,
            target_kl=0.1,
            cliprange=self.config.clip_range,
            cliprange_value=self.config.value_clip_range,
            vf_coef=0.5,
            gamma=self.config.gamma,
            log_with="wandb" if self.config.wandb_project else None,
        )

        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Setup wandb if configured
        if self.config.wandb_project:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__,
            )

        console.print("[green]RL trainer setup complete[/green]")

    def train_on_trajectories(
        self,
        trajectories: List["Trajectory"],
    ) -> TrainingStats:
        """
        Train on a batch of trajectories.

        Args:
            trajectories: List of proof trajectories

        Returns:
            Training statistics
        """
        from math_llm.training.rewards import ProofReward

        if self.ppo_trainer is None:
            self.setup()

        reward_fn = self.reward_fn or ProofReward()

        # Prepare training data
        queries = []
        responses = []
        rewards = []

        for traj in trajectories:
            # Build query from problem
            query = self._build_query(traj)

            # Build response from trajectory
            response = self._build_response(traj)

            # Compute reward
            reward = reward_fn.compute_trajectory_reward(traj)

            queries.append(query)
            responses.append(response)
            rewards.append(reward)

        # Tokenize
        query_tensors = [
            self.tokenizer.encode(q, return_tensors="pt").squeeze()
            for q in queries
        ]
        response_tensors = [
            self.tokenizer.encode(r, return_tensors="pt").squeeze()
            for r in responses
        ]
        reward_tensors = [torch.tensor(r) for r in rewards]

        # PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        # Record stats
        training_stats = TrainingStats(
            epoch=len(self.training_stats) // 100,
            step=len(self.training_stats),
            loss=stats.get("ppo/loss/total", 0.0),
            policy_loss=stats.get("ppo/loss/policy", 0.0),
            value_loss=stats.get("ppo/loss/value", 0.0),
            kl_divergence=stats.get("ppo/mean_kl", 0.0),
            entropy=stats.get("ppo/mean_entropy", 0.0),
            mean_reward=sum(rewards) / len(rewards),
            success_rate=sum(1 for t in trajectories if t.success) / len(trajectories),
        )
        self.training_stats.append(training_stats)

        return training_stats

    def _build_query(self, trajectory: "Trajectory") -> str:
        """Build the query string from a trajectory."""
        parts = []
        if trajectory.description:
            parts.append(f"Problem: {trajectory.description}")
        parts.append(f"Prove in Lean 4:\n```lean4\n{trajectory.statement}\n```")
        return "\n\n".join(parts)

    def _build_response(self, trajectory: "Trajectory") -> str:
        """Build the response string from a trajectory."""
        # Use the full trajectory as the response for training
        return trajectory.to_prompt_format()

    def collect_trajectories(
        self,
        agent: "LeanAgent",
        problems: List["LeanProblem"],
        num_trajectories: int,
    ) -> List["Trajectory"]:
        """
        Collect trajectories by running the agent on problems.

        Args:
            agent: The LeanAgent to use
            problems: List of problems to sample from
            num_trajectories: Number of trajectories to collect

        Returns:
            List of trajectories
        """
        import random

        trajectories = []

        with Progress(console=console) as progress:
            task = progress.add_task("Collecting trajectories...", total=num_trajectories)

            while len(trajectories) < num_trajectories:
                # Sample a problem
                problem = random.choice(problems)

                # Run agent
                trajectory = agent.solve(problem)
                trajectories.append(trajectory)

                progress.update(task, advance=1)

        return trajectories

    def train(
        self,
        agent: "LeanAgent",
        train_problems: List["LeanProblem"],
        val_problems: Optional[List["LeanProblem"]] = None,
        num_epochs: int = 3,
        trajectories_per_epoch: int = 100,
        eval_every: int = 10,
    ) -> None:
        """
        Full training loop.

        Args:
            agent: The LeanAgent (with model to be trained)
            train_problems: Training problems
            val_problems: Optional validation problems
            num_epochs: Number of training epochs
            trajectories_per_epoch: Trajectories to collect per epoch
            eval_every: Evaluate every N batches
        """
        console.print(f"[blue]Starting training for {num_epochs} epochs[/blue]")

        for epoch in range(num_epochs):
            console.print(f"\n[yellow]Epoch {epoch + 1}/{num_epochs}[/yellow]")

            # Collect trajectories
            trajectories = self.collect_trajectories(
                agent,
                train_problems,
                trajectories_per_epoch,
            )

            # Train on trajectories in batches
            batch_size = self.config.batch_size
            for i in range(0, len(trajectories), batch_size):
                batch = trajectories[i:i + batch_size]
                stats = self.train_on_trajectories(batch)

                if (i // batch_size + 1) % self.config.logging_steps == 0:
                    console.print(
                        f"  Step {i // batch_size + 1}: "
                        f"loss={stats.loss:.4f}, "
                        f"reward={stats.mean_reward:.4f}, "
                        f"success={stats.success_rate:.1%}"
                    )

            # Evaluate
            if val_problems:
                val_trajectories = self.collect_trajectories(
                    agent,
                    val_problems,
                    min(50, len(val_problems)),
                )
                val_success = sum(1 for t in val_trajectories if t.success) / len(val_trajectories)
                console.print(f"  Validation success rate: {val_success:.1%}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1)

        console.print("[green]Training complete![/green]")

    def save_checkpoint(self, epoch: int) -> None:
        """Save a training checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = output_dir / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training stats
        stats_file = checkpoint_dir / "training_stats.json"
        with open(stats_file, "w") as f:
            json.dump([s.to_dict() for s in self.training_stats], f, indent=2)

        console.print(f"[green]Checkpoint saved to {checkpoint_dir}[/green]")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a training checkpoint."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print(f"[blue]Loading checkpoint from {checkpoint_path}[/blue]")

        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        # Load training stats if available
        stats_file = Path(checkpoint_path) / "training_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats_data = json.load(f)
            # Reconstruct stats objects would require more work
            console.print(f"[blue]Loaded {len(stats_data)} training stats entries[/blue]")


class TrajectoryDataset(torch.utils.data.Dataset):
    """PyTorch dataset of trajectories for training."""

    def __init__(
        self,
        trajectories: List["Trajectory"],
        tokenizer,
        max_length: int = 2048,
    ):
        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]

        # Build query
        query = ""
        if traj.description:
            query += f"Problem: {traj.description}\n\n"
        query += f"Prove in Lean 4:\n```lean4\n{traj.statement}\n```"

        # Build response (final action)
        response = traj.final_action or ""

        # Tokenize
        query_encoded = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors="pt",
        )

        response_encoded = self.tokenizer(
            response,
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_encoded["input_ids"].squeeze(),
            "query_attention_mask": query_encoded["attention_mask"].squeeze(),
            "response_input_ids": response_encoded["input_ids"].squeeze(),
            "response_attention_mask": response_encoded["attention_mask"].squeeze(),
            "success": torch.tensor(1.0 if traj.success else 0.0),
        }
