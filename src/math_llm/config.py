"""
Configuration management using Hydra and OmegaConf.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_length: int = 2048
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True


class LeanConfig(BaseModel):
    """Lean server configuration."""
    lean_path: str = "lake"
    project_path: Optional[str] = None
    timeout: int = 60
    memory_limit: int = 4096  # MB
    max_retries: int = 3


class AgentConfig(BaseModel):
    """Agent configuration."""
    max_steps: int = 10
    temperature: float = 0.7
    top_p: float = 0.95
    stop_on_error: bool = False
    verbose: bool = True


class DataConfig(BaseModel):
    """Data configuration."""
    sources: list[str] = field(default_factory=lambda: ["formal-conjectures", "mathlib"])
    cache_dir: str = ".cache/datasets"
    train_split: float = 0.9
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42


class TrainingConfig(BaseModel):
    """Training configuration."""
    # Basic training params
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # RL specific
    reward_success: float = 1.0
    reward_failure: float = 0.0
    reward_step_penalty: float = -0.01
    reward_iteration_decay: float = 0.95
    kl_coef: float = 0.1
    gamma: float = 0.99

    # PPO specific
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_clip_range: float = 0.2

    # LoRA/PEFT
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "outputs"
    wandb_project: Optional[str] = "math-llm"


class EvalConfig(BaseModel):
    """Evaluation configuration."""
    num_samples: int = 100
    beam_size: int = 1
    temperature: float = 0.0
    save_trajectories: bool = True
    output_file: str = "eval_results.json"


class Config(BaseModel):
    """Main configuration."""
    model: ModelConfig = ModelConfig()
    lean: LeanConfig = LeanConfig()
    agent: AgentConfig = AgentConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    eval: EvalConfig = EvalConfig()

    seed: int = 42
    debug: bool = False

    class Config:
        arbitrary_types_allowed = True


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    omega_conf = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(omega_conf, resolve=True)
    return Config(**config_dict)


def merge_configs(base: Config, overrides: dict[str, Any]) -> Config:
    """Merge base config with overrides."""
    base_dict = base.model_dump()

    def deep_merge(d1: dict, d2: dict) -> dict:
        result = d1.copy()
        for key, value in d2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(base_dict, overrides)
    return Config(**merged)


# Fix the field default factory issue by using __init__
DataConfig.__init__ = lambda self, **kwargs: BaseModel.__init__(
    self,
    sources=kwargs.get("sources", ["formal-conjectures", "mathlib"]),
    **{k: v for k, v in kwargs.items() if k != "sources"}
)
