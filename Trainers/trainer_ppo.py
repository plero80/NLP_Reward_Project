from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from transformers import PreTrainedModel

from Models.model_policy import PolicyModel
from Models.model_reward import RewardModel
from Models.model_value import ValueModel


@dataclass(frozen=True)
class PPOTrainingConfig:
    output_dir: str
    epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-6
    response_length: int = 128
    num_ppo_epochs: int = 4
    num_mini_batches: int = 1
    save_steps: int = 100
    logging_steps: int = 10


class PolicyPPOTrainer:
    """Adapter around TRL 1.x's experimental dataset-driven PPO trainer."""

    def __init__(
        self,
        policy: PolicyModel,
        reward: RewardModel,
        value: ValueModel,
        train_dataset: Dataset,
        config: PPOTrainingConfig,
        eval_dataset: Dataset | None = None,
    ) -> None:
        self.policy = policy
        self.reward = reward
        self.value = value
        self.config = config
        policy_vocabulary = self.policy.tokenizer.get_vocab()
        for model_name, tokenizer in (
            ("reward", self.reward.tokenizer),
            ("value", self.value.tokenizer),
        ):
            if tokenizer.get_vocab() != policy_vocabulary:
                raise ValueError(
                    f"The policy and {model_name} models must use the same "
                    "tokenizer vocabulary for TRL PPO"
                )
        for entry in self.reward.classifiers:
            if entry["classifier"].tokenizer.get_vocab() != policy_vocabulary:
                raise ValueError(
                    "Every reward classifier must use the policy tokenizer "
                    "vocabulary for TRL PPO"
                )
        if self.reward.model.config.num_labels != 1:
            raise ValueError("TRL PPO requires a reward model with num_labels=1")
        self.train_dataset = train_dataset
        self.eval_dataset = (
            eval_dataset
            if eval_dataset is not None
            else None
        )
        self.trainer: Any | None = None

    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        prompt_column = (
            "prompt"
            if "prompt" in dataset.column_names
            else "prompts"
            if "prompts" in dataset.column_names
            else None
        )
        if prompt_column is None:
            raise ValueError("PPO dataset must contain 'prompt' or 'prompts'")

        tokenizer = self.policy.tokenizer

        def tokenize(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            prompts = batch[prompt_column]
            if not all(isinstance(prompt, str) for prompt in prompts):
                raise TypeError("all PPO prompts must be strings")
            return tokenizer(prompts, truncation=True)

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
        )

    @staticmethod
    def _trl_classes() -> tuple[type[Any], type[Any]]:
        try:
            from trl.experimental.ppo import PPOConfig, PPOTrainer
        except ImportError as exc:
            raise RuntimeError(
                "TRL PPO could not be imported. TRL 1.10 requires a PyTorch "
                "build that provides torch.distributed.fsdp.FSDPModule."
            ) from exc
        return PPOConfig, PPOTrainer

    def train(self, resume_from_checkpoint=None) -> Any:
        PPOConfig, PPOTrainer = self._trl_classes()
        config = self.config

        args = PPOConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            response_length=config.response_length,
            num_ppo_epochs=config.num_ppo_epochs,
            num_mini_batches=config.num_mini_batches,
            stop_token="eos",
            save_strategy="steps",
            save_steps=config.save_steps,
            logging_steps=config.logging_steps,
            report_to="none",
        )

        self.trainer = PPOTrainer(
            args=args,
            processing_class=self.policy.tokenizer,
            model=cast(PreTrainedModel, self.policy.model),
            ref_model=None,
            reward_model=self.reward.for_ppo(),
            train_dataset=self.train_dataset,
            value_model=self.value.model,
            eval_dataset=self.eval_dataset,
        )

        result = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )

        final_directory = Path(config.output_dir) / "final"

        self.trainer.save_model(str(final_directory))
        self.policy.tokenizer.save_pretrained(final_directory)

        return result
