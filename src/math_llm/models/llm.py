"""
LLM wrapper for theorem proving.

Provides a unified interface for different model backends:
- HuggingFace Transformers
- vLLM (for fast inference)
- API-based models
"""

from typing import Optional, Union, List
from pathlib import Path

import torch
from rich.console import Console

console = Console()


class LLMWrapper:
    """
    Wrapper for LLM inference.

    Supports:
    - HuggingFace models (local or from hub)
    - PEFT/LoRA adapters
    - Multiple generation backends
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_length: int = 2048,
        trust_remote_code: bool = True,
        adapter_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.adapter_path = adapter_path

        # Parse torch dtype
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)

        # Quantization config
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        console.print(f"[blue]Loading model: {self.model_name}[/blue]")

        # Setup quantization if requested
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Determine device map
        device_map = self.device if self.device != "auto" else "auto"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": device_map,
            "torch_dtype": self.torch_dtype,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Load adapter if specified
        if self.adapter_path:
            self._load_adapter(self.adapter_path)

        self._loaded = True
        console.print(f"[green]Model loaded successfully[/green]")

    def _load_adapter(self, adapter_path: str) -> None:
        """Load a PEFT/LoRA adapter."""
        from peft import PeftModel

        console.print(f"[blue]Loading adapter from: {adapter_path}[/blue]")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        stop_strings: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to sample (False for greedy)
            stop_strings: Optional stop strings

        Returns:
            Generated text
        """
        if not self._loaded:
            self.load()

        # Build messages for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback for models without chat template
            input_text = prompt
            if system_prompt:
                input_text = f"{system_prompt}\n\n{prompt}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens,
        )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response (only new tokens)
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        # Apply stop strings if specified
        if stop_strings:
            for stop in stop_strings:
                if stop in response:
                    response = response.split(stop)[0]

        return response.strip()

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, system_prompt=system_prompt, **kwargs) for p in prompts]

    def get_model_for_training(self):
        """Get the underlying model for training."""
        if not self._loaded:
            self.load()
        return self.model

    def get_tokenizer(self):
        """Get the tokenizer."""
        if not self._loaded:
            self.load()
        return self.tokenizer

    def save_adapter(self, path: str) -> None:
        """Save the current adapter weights."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        from peft import PeftModel

        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
            console.print(f"[green]Adapter saved to: {path}[/green]")
        else:
            console.print("[yellow]Warning: Model doesn't have PEFT adapter[/yellow]")


def load_model(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    **kwargs,
) -> LLMWrapper:
    """
    Convenience function to load a model.

    Recommended models for Lean theorem proving:
    - Qwen/Qwen2.5-0.5B-Instruct: Fast, good for testing
    - Qwen/Qwen2.5-1.5B-Instruct: Better quality
    - Qwen/Qwen2.5-7B-Instruct: High quality
    - deepseek-ai/deepseek-math-7b-instruct: Specialized for math
    - meta-llama/Llama-3.2-3B-Instruct: Good balance
    """
    wrapper = LLMWrapper(model_name=model_name, **kwargs)
    wrapper.load()
    return wrapper


# Pre-configured model factories
def load_qwen_small(**kwargs) -> LLMWrapper:
    """Load Qwen 0.5B for fast testing."""
    return load_model("Qwen/Qwen2.5-0.5B-Instruct", **kwargs)


def load_qwen_medium(**kwargs) -> LLMWrapper:
    """Load Qwen 1.5B for better quality."""
    return load_model("Qwen/Qwen2.5-1.5B-Instruct", **kwargs)


def load_deepseek_math(**kwargs) -> LLMWrapper:
    """Load DeepSeek Math 7B - specialized for mathematics."""
    return load_model("deepseek-ai/deepseek-math-7b-instruct", **kwargs)
