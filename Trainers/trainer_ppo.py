from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from transformers import PreTrainedModel

from Models.model_policy import PolicyModel
from Models.models import PPORewardModelProtocol
from Models.runtime import current_device
from Models.model_value import ValueModel


@dataclass(frozen=True)
class PPOTrainingConfig:
    output_dir: str
    epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    rollout_forward_batch_size: int = 16
    learning_rate: float = 3e-6
    response_length: int = 128
    num_ppo_epochs: int = 4
    num_mini_batches: int = 1
    save_steps: int = 1000
    save_total_limit: int = 1
    logging_steps: int = 10


class PolicyPPOTrainer:
    """Adapter around TRL 1.x's experimental dataset-driven PPO trainer."""

    def __init__(
        self,
        policy: PolicyModel,
        reward: PPORewardModelProtocol,
        value: ValueModel,
        train_dataset: Dataset,
        config: PPOTrainingConfig,
        eval_dataset: Dataset | None = None,
        reference_policy: PolicyModel | None = None,
    ) -> None:
        self.policy = policy
        self.reward = reward
        self.value = value
        self.config = config
        self.reference_policy = reference_policy
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
        if (
            self.reference_policy is not None
            and self.reference_policy.tokenizer.get_vocab() != policy_vocabulary
        ):
            raise ValueError(
                "The reference policy must use the policy tokenizer vocabulary"
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

    def _place_models_for_training(self) -> None:
        """Restore every PPO component after any earlier CPU offload."""
        device = current_device()
        self.policy.model.to(device)
        self.reward.model.to(device)
        self.value.model.to(device)
        if self.reference_policy is not None:
            self.reference_policy.model.to(device)
        for entry in self.reward.classifiers:
            entry["classifier"].model.to(device)

    def train(self) -> Any:
        PPOConfig, PPOTrainer = self._trl_classes()
        config = self.config

        self._place_models_for_training()

        args = PPOConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            local_rollout_forward_batch_size=(
                config.rollout_forward_batch_size
            ),
            response_length=config.response_length,
            num_ppo_epochs=config.num_ppo_epochs,
            num_mini_batches=config.num_mini_batches,
            stop_token="eos",
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            logging_steps=config.logging_steps,
            report_to="none",
        )

        self.trainer = PPOTrainer(
            args=args,
            processing_class=self.policy.tokenizer,
            model=cast(PreTrainedModel, self.policy.model),
            ref_model=(
                cast(PreTrainedModel, self.reference_policy.model)
                if self.reference_policy is not None
                else None
            ),
            reward_model=self.reward.for_ppo(),
            train_dataset=self.train_dataset,
            value_model=self.value.model,
            eval_dataset=self.eval_dataset,
        )

        result = self.trainer.train()

        final_directory = Path(config.output_dir) / "final"

        # TRL trains a PolicyAndValueWrapper. Saving that wrapper directly
        # does not create a checkpoint that PolicyModel.load() can consume.
        # Save the updated policy itself (plus tokenizer and current dataset).
        self.policy.save(final_directory)

        return result
