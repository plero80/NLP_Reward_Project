from dataclasses import dataclass, field
import gc
import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset
from peft import PeftModel
import torch
from transformers.trainer_utils import get_last_checkpoint

from Datasets.dataset_classifier import DatasetClassifier
from Datasets.dataset_request import RequestDataset
from Models.lora import LoRASettings
from Models.model_classifier import Classifier
import Models.model_evaluator as model_evaluator
from Models.model_policy import PolicyModel
from Models.model_reward import RewardModel
from Models.model_value import ValueModel
from Trainers.trainer_classifier import (
    ClassifierTrainer,
    ClassifierTrainingConfig,
)
from Trainers.trainer_ppo import PPOTrainingConfig, PolicyPPOTrainer


logger = logging.getLogger(__name__)


def empty_cuda_cache() -> None:
    """Collect unreachable objects and release unused CUDA cache blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def offload_to_cpu(owner: object | None) -> None:
    """Move a model, or an object's ``model`` attribute, to CPU."""
    if owner is None:
        return

    offload = getattr(owner, "offload", None)
    if callable(offload):
        offload()
        return

    module = (
        owner
        if isinstance(owner, torch.nn.Module)
        else getattr(owner, "model", None)
    )
    if isinstance(module, torch.nn.Module):
        module.to("cpu")


def delete_model(model: object) -> None:
    """Backward-compatible GPU cleanup helper.

    Python functions cannot delete a variable owned by their caller. This
    helper therefore offloads the model tensors and clears unused CUDA cache.
    Use ``del model`` (or assign ``None``) in the caller if the object itself
    should also be released.
    """
    offload_to_cpu(model)
    empty_cuda_cache()


def latest_ppo_checkpoint(output_dir: str | Path) -> str:
    """Return the latest completed ``checkpoint-*`` in a PPO output folder."""
    checkpoint = get_last_checkpoint(str(output_dir))
    if checkpoint is None:
        raise FileNotFoundError(
            f"No PPO checkpoint was found under: {output_dir}"
        )
    return checkpoint


@dataclass
class ConfigEval:
    """Configuration for PPO training followed by policy evaluation."""

    dataset_name: str = "Anthropic/hh-rlhf"
    start: int = 0
    end: int = 1000

    policy_name: str = "Qwen/Qwen3-0.6B"
    reward_model_name: str = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    reward_mode_name: str = "proxy"

    epochs: float = 1.0
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    rollout_forward_batch_size: int = 16
    response_length: int = 128
    policy_batch_size: int = 64
    save_steps: int = 10

    # This is a policy-only warm start, not an exact PPO resume. Optimizer,
    # value-model, RNG, and dataloader state are not restored.
    policy_checkpoint: str | Path | None = None
    output_dir: str | Path | None = None


@dataclass
class ConfigTrainClassifier:
    """Configuration for generating and training a reward-gap classifier."""

    dataset_name: str = "Anthropic/hh-rlhf"
    policy_load_path: str | Path | None = None
    policy_name: str = "Qwen/Qwen3-0.6B"
    start_dataset: int = 0
    end_dataset: int = 1000

    reward_model_name: str = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    reward_mode_name: str = "proxy"
    judge_model_name: str = "Skywork/Skywork-Reward-V2-Qwen3-4B"
    judge_mode_name: str = "judge"

    generation_batch_size: int = 8
    reward_batch_size: int = 1
    judge_batch_size: int = 1
    score_max_length: int = 2_048

    classifier_model_name: str = "Qwen/Qwen3-0.6B"
    classifier_theta: float = 2.0
    classifier_test_size: float = 0.2
    classifier_random_state: int = 42
    classifier_output_root: str | Path = "outputs/classifiers"
    classifier_epochs: float = 3.0
    classifier_batch_size: int = 8
    classifier_learning_rate: float = 2e-5
    classifier_max_length: int = 512
    lora_settings: LoRASettings | None = field(default_factory=LoRASettings)


def _validate_classifier_config(config: ConfigTrainClassifier) -> None:
    if config.start_dataset < 0:
        raise ValueError("start_dataset must be non-negative")
    if config.end_dataset <= config.start_dataset:
        raise ValueError("end_dataset must be greater than start_dataset")
    if config.reward_mode_name == config.judge_mode_name:
        raise ValueError("reward_mode_name and judge_mode_name must be different")

    positive_sizes = {
        "generation_batch_size": config.generation_batch_size,
        "reward_batch_size": config.reward_batch_size,
        "judge_batch_size": config.judge_batch_size,
        "score_max_length": config.score_max_length,
        "classifier_batch_size": config.classifier_batch_size,
        "classifier_max_length": config.classifier_max_length,
    }
    for name, value in positive_sizes.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    if config.classifier_epochs <= 0:
        raise ValueError("classifier_epochs must be positive")
    if not 0.0 < config.classifier_test_size < 1.0:
        raise ValueError("classifier_test_size must be between 0 and 1")


def create_classifier(
    config: ConfigTrainClassifier | None = None,
) -> Classifier:
    """Create reward/judge labels, train a LoRA classifier, and return it."""
    config = config or ConfigTrainClassifier()
    _validate_classifier_config(config)

    policy: PolicyModel | None = None
    classifier_dataset: DatasetClassifier

    try:
        policy = (
            PolicyModel.load(config.policy_load_path)
            if config.policy_load_path is not None
            else PolicyModel(config.policy_name)
        )

        raw_dataset = load_dataset(config.dataset_name)
        request_dataset = RequestDataset.from_raw(
            raw_dataset,
            policy.model_name,
        )
        request_dataset.truncate(
            config.start_dataset,
            config.end_dataset,
        )
        if len(request_dataset) == 0:
            raise ValueError("The selected classifier dataset range is empty")

        policy.generate_new_dataset(
            request_dataset,
            batch_size=config.generation_batch_size,
        )

        # Keep the generated rows, but make room for one scorer at a time.
        policy.offload()
        empty_cuda_cache()

        scoring_stages = (
            (
                config.reward_model_name,
                config.reward_mode_name,
                config.reward_batch_size,
            ),
            (
                config.judge_model_name,
                config.judge_mode_name,
                config.judge_batch_size,
            ),
        )
        for model_name, mode_name, batch_size in scoring_stages:
            scorer: RewardModel | None = None
            try:
                scorer = RewardModel(model_name, mode_name)
                scorer.init_normalization(policy, batch_size)
                
                if mode_name == "judge":
                    config.classifier_theta = scorer.std
                
                scorer.score_policy(
                    policy,
                    batch_size=batch_size,
                    max_length=config.score_max_length,
                    normalize_score = True
                )
            except BaseException:
                # Notebook tracebacks may retain the failed scorer. Offload it
                # before propagating the original exception.
                offload_to_cpu(scorer)
                raise
            finally:
                scorer = None
                empty_cuda_cache()

        classifier_dataset = DatasetClassifier(
            theta=config.classifier_theta
        )
        classifier_dataset.add(
            prompts=policy.get_dataset_col("prompts"),
            answers=policy.get_dataset_col("answers"),
            reward_scores=policy.get_dataset_col(config.reward_mode_name),
            judge_scores=policy.get_dataset_col(config.judge_mode_name),
        )
    except BaseException:
        offload_to_cpu(policy)
        raise
    finally:
        policy = None
        empty_cuda_cache()

    class_counts = classifier_dataset.class_counts()
    if len(class_counts) < 2:
        raise ValueError(
            "Classifier labels contain only one class. Adjust "
            f"classifier_theta; class counts: {class_counts}"
        )

    train_dataset, test_dataset = classifier_dataset.split(
        test_size=config.classifier_test_size,
        random_state=config.classifier_random_state,
    )

    classifier = Classifier(config.classifier_model_name)
    training_config = ClassifierTrainingConfig(
        output_dir=str(
            Path(config.classifier_output_root) / f"id={classifier.id}"
        ),
        epochs=config.classifier_epochs,
        batch_size=config.classifier_batch_size,
        learning_rate=config.classifier_learning_rate,
        max_length=config.classifier_max_length,
        lora_settings=config.lora_settings,
    )
    classifier_trainer = ClassifierTrainer(classifier, training_config)
    hf_trainer = None

    try:
        hf_trainer = classifier_trainer.train(train_dataset)
        metrics = classifier_trainer.evaluate(hf_trainer, test_dataset)
        logger.info("Classifier test metrics: %s", metrics)
        classifier.model.eval()
        return classifier
    except BaseException:
        offload_to_cpu(classifier)
        raise
    finally:
        hf_trainer = None
        classifier_trainer = None
        empty_cuda_cache()


def _ppo_output_directory(
    config: ConfigEval,
) -> str:
    default_output_dir = Path(
        "outputs/ppo_policy/"
        f"{config.reward_mode_name}_{config.start}_{config.end}"
    )
    if config.output_dir is None:
        if config.policy_checkpoint is None:
            return str(default_output_dir)
        return str(
            default_output_dir.parent
            / (
                f"{default_output_dir.name}_from_"
                f"{Path(config.policy_checkpoint).name}"
            )
        )

    selected_output = Path(config.output_dir).expanduser()
    if config.policy_checkpoint is not None:
        checkpoint = Path(config.policy_checkpoint).expanduser().resolve()
        resolved_output = selected_output.resolve()
        if resolved_output in {checkpoint, checkpoint.parent}:
            raise ValueError(
                "A PPO warm start must use a new output directory so the "
                "source checkpoint cannot be overwritten"
            )
    return str(selected_output)


def eval_policy_with_reward(
    config: ConfigEval,
) -> float:
    """Train PPO, generate answers, and evaluate the resulting policy.

    ``config.policy_checkpoint`` performs a policy-weight warm start. It is
    not an exact PPO resume because optimizer, value-model, RNG, and dataloader
    state are not restored.
    """
    if config.start < 0 or config.end <= config.start:
        raise ValueError("end must be greater than a non-negative start")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    for name, value in (
        ("batch_size", config.batch_size),
        (
            "gradient_accumulation_steps",
            config.gradient_accumulation_steps,
        ),
        (
            "rollout_forward_batch_size",
            config.rollout_forward_batch_size,
        ),
        ("policy_batch_size", config.policy_batch_size),
        ("response_length", config.response_length),
        ("save_steps", config.save_steps),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    training_output_dir = _ppo_output_directory(config)

    policy: PolicyModel | None = None
    reference_policy: PolicyModel | None = None
    value_model: ValueModel | None = None
    reward_model: RewardModel | None = None
    trainer: PolicyPPOTrainer | None = None
    evaluator: Any | None = None

    try:
        policy = (
            PolicyModel.load(
                config.policy_checkpoint,
                is_trainable=True,
            )
            if config.policy_checkpoint is not None
            else PolicyModel(config.policy_name)
        )

        raw_dataset = load_dataset(config.dataset_name)
        dataset = RequestDataset.from_raw(raw_dataset, policy.model_name)
        dataset.truncate(config.start, config.end)
        if len(dataset) == 0:
            raise ValueError("The selected PPO dataset range is empty")

        # Full-model warm starts need the original policy for KL reference.
        # For PEFT, TRL can use the base model with the adapter disabled.
        reference_policy = (
            PolicyModel(config.policy_name)
            if config.policy_checkpoint is not None
            and not isinstance(policy.model, PeftModel)
            else None
        )
        value_model = ValueModel(config.policy_name)
        reward_model = RewardModel(
            config.reward_model_name,
            config.reward_mode_name,
        )

        ppo_config = PPOTrainingConfig(
            output_dir=training_output_dir,
            epochs=config.epochs,
            batch_size=config.batch_size,
            gradient_accumulation_steps=(
                config.gradient_accumulation_steps
            ),
            rollout_forward_batch_size=(
                config.rollout_forward_batch_size
            ),
            response_length=config.response_length,
            num_ppo_epochs=4,
            num_mini_batches=1,
            learning_rate=3e-6,
            save_steps=config.save_steps,
            save_total_limit=1,
        )
        trainer = PolicyPPOTrainer(
            policy,
            reward_model,
            value_model,
            dataset,
            ppo_config,
            reference_policy=reference_policy,
        )

        try:
            trainer.train()
        except BaseException:
            for owner in (
                policy,
                reference_policy,
                value_model,
                reward_model,
            ):
                offload_to_cpu(owner)
            raise
        finally:
            trainer = None
            reference_policy = None
            value_model = None
            reward_model = None
            empty_cuda_cache()

        try:
            policy.generate_new_dataset(
                dataset,
                batch_size=config.policy_batch_size,
            )
        except BaseException:
            offload_to_cpu(policy)
            raise

        # Prometheus is large, so it must not overlap the policy on CUDA.
        policy.offload()
        empty_cuda_cache()

        evaluator = model_evaluator.PrometheusEvaluator()
        evaluation_score = float(evaluator.evaluate(policy))
        return evaluation_score
    except BaseException:
        offload_to_cpu(evaluator)
        offload_to_cpu(policy)
        raise
    finally:
        evaluator = None
        policy = None
        reference_policy = None
        value_model = None
        reward_model = None
        trainer = None
        empty_cuda_cache()
