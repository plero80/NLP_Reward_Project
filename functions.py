from dataclasses import dataclass, field
import gc
import json
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
import numpy as np

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
    calibration_quantile: float = 0.95
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
    if not 0.0 < config.calibration_quantile < 1.0:
        raise ValueError("calibration_quantile must be between 0 and 1")
    if not 0.0 < config.classifier_test_size < 1.0:
        raise ValueError("classifier_test_size must be between 0 and 1")


@dataclass(frozen=True)
class GapCalibration:
    proxy_mean: float
    proxy_std: float
    judge_mean: float
    judge_std: float
    theta: float

    def save(self, path: str | Path) -> Path:
        """Persist this frozen calibration as a versioned JSON file."""
        _validate_gap_calibration(self)
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "proxy_mean": self.proxy_mean,
            "proxy_std": self.proxy_std,
            "judge_mean": self.judge_mean,
            "judge_std": self.judge_std,
            "theta": self.theta,
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "GapCalibration":
        """Load and validate a calibration previously written by ``save``."""
        source = Path(path).expanduser()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Calibration file is not valid JSON: {source}"
            ) from error

        if not isinstance(payload, dict):
            raise ValueError("Calibration JSON must contain an object")
        if payload.get("format_version") != 1:
            raise ValueError(
                "Unsupported calibration format_version: "
                f"{payload.get('format_version')!r}"
            )

        field_names = (
            "proxy_mean",
            "proxy_std",
            "judge_mean",
            "judge_std",
            "theta",
        )
        missing = [name for name in field_names if name not in payload]
        if missing:
            raise ValueError(
                f"Calibration file is missing fields: {missing}"
            )
        try:
            calibration = cls(
                **{name: float(payload[name]) for name in field_names}
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Calibration fields must contain numeric values"
            ) from error
        _validate_gap_calibration(calibration)
        return calibration


_NORMALIZATION_EPSILON = 1e-8
PolicyRows = tuple[list[str], list[str]]


def _validate_gap_calibration(calibration: GapCalibration) -> None:
    values = {
        "proxy_mean": calibration.proxy_mean,
        "proxy_std": calibration.proxy_std,
        "judge_mean": calibration.judge_mean,
        "judge_std": calibration.judge_std,
        "theta": calibration.theta,
    }
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"Calibration {name} must be finite")
    for name in ("proxy_std", "judge_std"):
        if values[name] <= _NORMALIZATION_EPSILON:
            raise ValueError(f"Calibration {name} must be positive")


def _generate_policy_rows(
    config: ConfigTrainClassifier,
    request_dataset: RequestDataset,
    *,
    reference: bool,
    supplied_policy: PolicyModel | None = None,
) -> PolicyRows:
    policy: PolicyModel | None = None
    try:
        if supplied_policy is not None:
            policy = supplied_policy
        elif reference or config.policy_load_path is None:
            policy = PolicyModel(config.policy_name)
        else:
            policy = PolicyModel.load(config.policy_load_path)
        policy.generate_new_dataset(
            request_dataset,
            batch_size=config.generation_batch_size,
        )
        prompts = list(policy.get_dataset_col("prompts"))
        answers = list(policy.get_dataset_col("answers"))
        if len(prompts) != len(answers):
            raise RuntimeError(
                "Policy generation returned different prompt and answer counts"
            )
        if not all(isinstance(value, str) for value in prompts + answers):
            raise TypeError("Generated prompts and answers must all be strings")
        return prompts, answers
    except BaseException:
        offload_to_cpu(policy)
        raise
    finally:
        if supplied_policy is not None:
            # The caller retains this policy. Move it off CUDA before reward
            # and judge models are loaded, but keep the object reusable.
            supplied_policy.offload()
        policy = None
        empty_cuda_cache()


def _finite_score_array(
    scores: list[float],
    *,
    expected_size: int,
    label: str,
) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if array.shape != (expected_size,):
        raise ValueError(
            f"{label} returned shape {array.shape}; expected "
            f"({expected_size},)"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} returned a non-finite score")
    return array


def _score_reference_and_target(
    *,
    model_name: str,
    mode_name: str,
    batch_size: int,
    max_length: int,
    reference_rows: PolicyRows | None,
    target_rows: PolicyRows,
) -> tuple[np.ndarray | None, np.ndarray]:
    scorer: RewardModel | None = None
    try:
        scorer = RewardModel(model_name, mode_name)
        reference_scores = (
            _finite_score_array(
                scorer.score(
                    reference_rows[0],
                    reference_rows[1],
                    batch_size=batch_size,
                    max_length=max_length,
                ),
                expected_size=len(reference_rows[0]),
                label=f"reference {mode_name}",
            )
            if reference_rows is not None
            else None
        )
        target_scores = _finite_score_array(
            scorer.score(
                target_rows[0],
                target_rows[1],
                batch_size=batch_size,
                max_length=max_length,
            ),
            expected_size=len(target_rows[0]),
            label=f"target {mode_name}",
        )
        return reference_scores, target_scores
    except BaseException:
        offload_to_cpu(scorer)
        raise
    finally:
        scorer = None
        empty_cuda_cache()


def _build_gap_calibration(
    proxy_scores: np.ndarray,
    judge_scores: np.ndarray,
    quantile: float,
) -> GapCalibration:
    proxy_mean = float(proxy_scores.mean())
    proxy_std = float(proxy_scores.std(ddof=0))
    judge_mean = float(judge_scores.mean())
    judge_std = float(judge_scores.std(ddof=0))

    calibration_without_theta = GapCalibration(
        proxy_mean=proxy_mean,
        proxy_std=proxy_std,
        judge_mean=judge_mean,
        judge_std=judge_std,
        theta=0.0,
    )
    _validate_gap_calibration(calibration_without_theta)

    proxy_z = (proxy_scores - proxy_mean) / proxy_std
    judge_z = (judge_scores - judge_mean) / judge_std
    theta = float(np.quantile(proxy_z - judge_z, quantile))
    calibration = GapCalibration(
        proxy_mean=proxy_mean,
        proxy_std=proxy_std,
        judge_mean=judge_mean,
        judge_std=judge_std,
        theta=theta,
    )
    _validate_gap_calibration(calibration)
    return calibration


def create_classifier(
    config: ConfigTrainClassifier | None = None,
    *,
    policy: PolicyModel | None = None,
    calibration: GapCalibration | None = None,
) -> tuple[Classifier, GapCalibration]:
    """Train a classifier using a frozen reference reward-gap calibration."""
    config = config or ConfigTrainClassifier()
    _validate_classifier_config(config)
    if policy is not None and config.policy_load_path is not None:
        raise ValueError(
            "Pass either policy or config.policy_load_path, not both"
        )

    raw_dataset = load_dataset(config.dataset_name)
    request_dataset = RequestDataset.from_raw(
        raw_dataset,
        config.policy_name,
    )
    request_dataset.truncate(
        config.start_dataset,
        config.end_dataset,
    )
    if len(request_dataset) == 0:
        raise ValueError("The selected classifier dataset range is empty")

    reference_rows = (
        _generate_policy_rows(
            config,
            request_dataset,
            reference=True,
        )
        if calibration is None
        else None
    )
    target_rows = _generate_policy_rows(
        config,
        request_dataset,
        reference=False,
        supplied_policy=policy,
    )

    reference_proxy, target_proxy = _score_reference_and_target(
        model_name=config.reward_model_name,
        mode_name=config.reward_mode_name,
        batch_size=config.reward_batch_size,
        max_length=config.score_max_length,
        reference_rows=reference_rows,
        target_rows=target_rows,
    )
    reference_judge, target_judge = _score_reference_and_target(
        model_name=config.judge_model_name,
        mode_name=config.judge_mode_name,
        batch_size=config.judge_batch_size,
        max_length=config.score_max_length,
        reference_rows=reference_rows,
        target_rows=target_rows,
    )

    if calibration is None:
        if reference_proxy is None or reference_judge is None:
            raise RuntimeError("Reference scores are required for calibration")
        calibration = _build_gap_calibration(
            reference_proxy,
            reference_judge,
            config.calibration_quantile,
        )
    else:
        _validate_gap_calibration(calibration)

    target_proxy_z = (
        target_proxy - calibration.proxy_mean
    ) / calibration.proxy_std
    target_judge_z = (
        target_judge - calibration.judge_mean
    ) / calibration.judge_std

    classifier_dataset = DatasetClassifier(theta=calibration.theta)
    classifier_dataset.add(
        prompts=target_rows[0],
        answers=target_rows[1],
        reward_scores=target_proxy_z.tolist(),
        judge_scores=target_judge_z.tolist(),
    )

    class_counts = classifier_dataset.class_counts()
    if len(class_counts) < 2:
        raise ValueError(
            "Classifier labels contain only one class under the frozen "
            f"calibration threshold; class counts: {class_counts}"
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
        return classifier, calibration
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
) -> tuple[PolicyModel, float]:
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
        return policy, evaluation_score
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
