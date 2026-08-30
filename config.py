from dataclasses import dataclass, field
import os

from Models.lora import LoRASettings


CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_DIR", "checkpoints")


@dataclass(frozen=True)
class RunnerConfig:
    policy_name: str = "Qwen/Qwen3-0.6B"
    reward_name: str = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    judge_name: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M"
    dataset_limit: int = 1000
    static_dataset_length: int = 100
    value_name: str | None = None
    evaluator_name: str = "gpt-5.4-mini"
    reward_model_load: bool = False
    reward_model_checkpoint: str | None = None
    initial_classifier_checkpoints: tuple[tuple[str, str], ...] = ()
    policy_lora_settings: LoRASettings | None = field(
        default_factory=LoRASettings
    )
    ppo_output_dir: str = f"{CHECKPOINT_ROOT}/policy"
    ppo_epochs: float = 1.0
    ppo_batch_size: int = 1
    ppo_gradient_accumulation_steps: int = 1
    ppo_learning_rate: float = 3e-6
    ppo_response_length: int = 128
    ppo_optimization_epochs: int = 4
    ppo_num_mini_batches: int = 1
    ppo_save_steps: int = 100
    ppo_logging_steps: int = 10
    generation_batch_size: int = 2
    evaluator_concurrency: int = 5
    evaluation_seed: int = 42
    evaluation_output_dir: str = f"{CHECKPOINT_ROOT}/evaluations"
    classifier_name: str = "reward_gap_classifier"
    classifier_model_name: str | None = None
    classifier_output_dir: str = f"{CHECKPOINT_ROOT}/classifier"
    classifier_theta: float = 2.0
    classifier_test_size: float = 0.2
    classifier_epochs: float = 3.0
    classifier_batch_size: int = 8
    classifier_learning_rate: float = 2e-5
    classifier_max_length: int = 512
    classifier_lora_settings: LoRASettings | None = field(
        default_factory=LoRASettings
    )


@dataclass(frozen=True)
class ClassifierTrainingConfig:
    output_dir: str
    epochs: float = 3.0
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    lora_settings: LoRASettings | None = None
