from dataclasses import dataclass
from typing import Literal

from peft import LoraConfig, TaskType


@dataclass(frozen=True)
class LoRASettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | tuple[str, ...] = "all-linear"
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False

    def build(self, task_type: TaskType) -> LoraConfig:
        target_modules = (
            list(self.target_modules)
            if isinstance(self.target_modules, tuple)
            else self.target_modules
        )
        return LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=target_modules,
            bias=self.bias,
            use_rslora=self.use_rslora,
        )
