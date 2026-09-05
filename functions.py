from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import gc
import json
import logging
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from datasets import load_dataset
from peft import PeftModel
import torch
from transformers.trainer_utils import get_last_checkpoint

from Datasets.dataset_classifier import DatasetClassifier
from Datasets.dataset_gap_finder import DatasetGapFinder
from Datasets.dataset_request import RequestDataset
from Models.lora import LoRASettings
from Models.model_classifier import Classifier
from Models.model_gap_finder import GapFinder
from Models.reward_adjustment import RewardAdjustment
from Models.models import PPORewardModelProtocol
from Factory import *
import Models.model_evaluator as model_evaluator
from Models.model_policy import PolicyModel
from Models.model_reward import RewardModel
from Models.model_value import ValueModel
from Models.model_evaluator import PrometheusEvaluator
from Trainers.trainer_classifier import (
    ClassifierTrainer,
    ClassifierTrainingConfig,
)
from Trainers.trainer_gap_finder import GapFinderTrainer, GapFinderTrainingConfig
from Trainers.trainer_ppo import PPOTrainingConfig, PolicyPPOTrainer
import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def empty_cuda_cache() -> None:
    """Collect unreachable objects and release unused CUDA cache blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def offload_to_cpu(owner: object | None) -> None:
    """Move an owner model and any attached classifier models to CPU."""
    if owner is None:
        return

    offload = getattr(owner, "offload", None)
    if callable(offload):
        offload()
    else:
        module = (
            owner
            if isinstance(owner, torch.nn.Module)
            else getattr(owner, "model", None)
        )
        if isinstance(module, torch.nn.Module):
            module.to("cpu")

    for entry in getattr(owner, "classifiers", ()):
        classifier = entry.get("classifier") if isinstance(entry, dict) else entry
        offload_to_cpu(classifier)
    for adjustment in getattr(owner, "adjustments", ()):
        offload_to_cpu(adjustment)


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


@dataclass(frozen=True)
class PolicySpec:
    class_name: str = "PolicyModel"
    model_name: str = "Qwen/Qwen3-0.6B"
    lora_config: LoRASettings | None = field(default_factory=LoRASettings)
    checkpoint: str | Path | None = None


@dataclass(frozen=True)
class RewardSpec:
    class_name: str = "RewardModel"
    model_name: str = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    mode_name: str = "proxy"
    mean: float | None = None
    std: float | None = None





@dataclass
class ConfigTrainClassifier:
    """Configuration for generating and training a reward-gap classifier."""

    dataset_name: str = "Anthropic/hh-rlhf"
    policy: PolicySpec = field(default_factory=PolicySpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    judge: RewardSpec = field(
        default_factory=lambda: RewardSpec(
            model_name="Skywork/Skywork-Reward-V2-Qwen3-4B",
            mode_name="judge",
        )
    )
    start_dataset: int = 0
    end_dataset: int = 1000

    generation_batch_size: int = 64
    reward_batch_size: int = 64
    judge_batch_size: int = 32
    score_max_length: int = 2_048

    classifier_model_name: str = "Qwen/Qwen3-0.6B"
    calibration_quantile: float = 0.95
    classifier_test_size: float = 0.2
    classifier_random_state: int = 42
    classifier_output_root: str | Path = "outputs/classifiers"
    classifier_epochs: float = 3.0
    classifier_batch_size: int = 64
    classifier_learning_rate: float = 2e-5
    classifier_max_length: int = 512
    lora_settings: LoRASettings | None = field(default_factory=LoRASettings)

    gap_finder_model_name: str = "Qwen/Qwen3-0.6B"
    gap_finder_output_root: str | Path = "outputs/gap_finders"
    gap_finder_epochs: float = 3.0
    gap_finder_batch_size: int = 32
    gap_finder_learning_rate: float = 2e-5
    gap_finder_max_length: int = 512
    gap_finder_lora_settings: LoRASettings | None = field(default_factory=LoRASettings)


def _validate_classifier_config(config: ConfigTrainClassifier) -> None:
    if config.start_dataset < 0:
        raise ValueError("start_dataset must be non-negative")
    if config.end_dataset <= config.start_dataset:
        raise ValueError("end_dataset must be greater than start_dataset")
    if config.reward.mode_name == config.judge.mode_name:
        raise ValueError(
            "reward.mode_name and judge.mode_name must be different"
        )
    for name, spec in (("reward", config.reward), ("judge", config.judge)):
        if (spec.mean is None) != (spec.std is None):
            raise ValueError(
                f"{name}.mean and {name}.std must be provided together"
            )
        if spec.mean is not None and spec.std is not None:
            if not np.isfinite(spec.mean) or not np.isfinite(spec.std):
                raise ValueError(f"{name} normalization values must be finite")
            if spec.std <= _NORMALIZATION_EPSILON:
                raise ValueError(f"{name}.std must be greater than zero")

    positive_sizes = {
        "generation_batch_size": config.generation_batch_size,
        "reward_batch_size": config.reward_batch_size,
        "judge_batch_size": config.judge_batch_size,
        "score_max_length": config.score_max_length,
        "classifier_batch_size": config.classifier_batch_size,
        "classifier_max_length": config.classifier_max_length,
        "gap_finder_batch_size": config.gap_finder_batch_size,
        "gap_finder_max_length": config.gap_finder_max_length,
    }
    for name, value in positive_sizes.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    if config.classifier_epochs <= 0:
        raise ValueError("classifier_epochs must be positive")
    if config.gap_finder_epochs <= 0:
        raise ValueError("gap_finder_epochs must be positive")
    if config.gap_finder_learning_rate <= 0:
        raise ValueError("gap_finder_learning_rate must be positive")
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
        else:
            checkpoint = None if reference else config.policy.checkpoint
            policy = PolicyModelFactory.create_model(
                config.policy.class_name,
                config.policy.model_name,
                config.policy.lora_config,
                checkpoint=checkpoint,
                is_trainable=False,
            )
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


def _score_policy_rows(
    *,
    spec: RewardSpec,
    batch_size: int,
    max_length: int,
    row_groups: dict[str, PolicyRows],
) -> dict[str, np.ndarray]:
    scorer: object | None = None
    try:
        scorer = RewardModelFactory.create_model(
            spec.class_name,
            spec.model_name,
            spec.mode_name,
            mean=spec.mean,
            std=spec.std,
        )
        score = getattr(scorer, "score", None)
        if not callable(score):
            raise TypeError(
                f"Reward class {spec.class_name!r} cannot score prompt/answer "
                "batches for classifier creation"
            )
        score_options = {
            "batch_size": batch_size,
            "max_length": max_length,
        }
        # Classifier labels use their own frozen GapCalibration below, so
        # collect raw scores even when the reward carries PPO normalization.
        return {
            label: _finite_score_array(
                score(
                    rows[0],
                    rows[1],
                    **score_options,
                ),
                expected_size=len(rows[0]),
                label=f"{label} {spec.mode_name}",
            )
            for label, rows in row_groups.items()
        }
    except BaseException:
        offload_to_cpu(scorer)
        raise
    finally:
        scorer = None
        empty_cuda_cache()


def _score_reference_and_target(
    *,
    spec: RewardSpec,
    batch_size: int,
    max_length: int,
    reference_rows: PolicyRows | None,
    target_rows: PolicyRows,
) -> tuple[np.ndarray | None, np.ndarray]:
    row_groups = {"target": target_rows}
    if reference_rows is not None:
        row_groups = {"reference": reference_rows, **row_groups}
    scores = _score_policy_rows(
        spec=spec,
        batch_size=batch_size,
        max_length=max_length,
        row_groups=row_groups,
    )
    return scores.get("reference"), scores["target"]


def _build_gap_calibration(
    proxy_scores: list[float] | np.ndarray,
    judge_scores: list[float] | np.ndarray,
    quantile: float,
) -> GapCalibration:
    proxy_scores = np.asarray(proxy_scores, dtype=np.float64)
    judge_scores = np.asarray(judge_scores, dtype=np.float64)

    if proxy_scores.ndim != 1 or judge_scores.ndim != 1:
        raise ValueError("Proxy and judge scores must be one-dimensional")
    if proxy_scores.size == 0 or judge_scores.size == 0:
        raise ValueError("Proxy and judge scores cannot be empty")
    if proxy_scores.shape != judge_scores.shape:
        raise ValueError(
            "Proxy and judge score shapes differ: "
            f"{proxy_scores.shape} != {judge_scores.shape}"
        )
    if not np.isfinite(proxy_scores).all():
        raise ValueError("Proxy scores must all be finite")
    if not np.isfinite(judge_scores).all():
        raise ValueError("Judge scores must all be finite")

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


def calculate_gap_calibration(
    config: ConfigTrainClassifier | None = None,
) -> GapCalibration:
    """Calculate frozen proxy/judge normalization from the reference policy."""
    config = config or ConfigTrainClassifier()
    _validate_classifier_config(config)

    raw_dataset = load_dataset(config.dataset_name)
    request_dataset = RequestDataset.from_raw(
        raw_dataset,
        config.policy.model_name,
    )
    request_dataset.truncate(config.start_dataset, config.end_dataset)
    if len(request_dataset) == 0:
        raise ValueError("The selected calibration dataset range is empty")

    reference_rows = _generate_policy_rows(
        config,
        request_dataset,
        reference=True,
    )
    proxy_scores = _score_policy_rows(
        spec=config.reward,
        batch_size=config.reward_batch_size,
        max_length=config.score_max_length,
        row_groups={"reference": reference_rows},
    )["reference"]
    judge_scores = _score_policy_rows(
        spec=config.judge,
        batch_size=config.judge_batch_size,
        max_length=config.score_max_length,
        row_groups={"reference": reference_rows},
    )["reference"]
    return _build_gap_calibration(
        proxy_scores,
        judge_scores,
        config.calibration_quantile,
    )


def _policy_source_label(
    config: ConfigTrainClassifier,
    policy: PolicyModel | None,
) -> str:
    checkpoint = (
        getattr(policy, "checkpoint_path", None)
        if policy is not None
        else config.policy.checkpoint
    )
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        source = (
            checkpoint_path.parent.name
            if checkpoint_path.name.casefold() == "final"
            else checkpoint_path.name
        )
    else:
        source = Path(config.policy.model_name).name

    safe_source = re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip("._-")
    return safe_source or "unknown"


def create_classifier(
    config: ConfigTrainClassifier | None = None,
    *,
    policy: PolicyModel | None = None,
    calibration: GapCalibration | None = None,
) -> tuple[Classifier, GapCalibration]:
    """Train a classifier using a frozen reference reward-gap calibration."""
    config = config or ConfigTrainClassifier()
    _validate_classifier_config(config)
    if policy is not None and config.policy.checkpoint is not None:
        raise ValueError(
            "Pass either policy or config.policy.checkpoint, not both"
        )

    raw_dataset = load_dataset(config.dataset_name)
    request_dataset = RequestDataset.from_raw(
        raw_dataset,
        config.policy.model_name,
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
        spec=config.reward,
        batch_size=config.reward_batch_size,
        max_length=config.score_max_length,
        reference_rows=reference_rows,
        target_rows=target_rows,
    )
    reference_judge, target_judge = _score_reference_and_target(
        spec=config.judge,
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

    artifact_id = str(uuid4())
    classifier_dataset = DatasetClassifier(
        theta=calibration.theta,
        id=artifact_id,
    )
    classifier_dataset.add(
        prompts=target_rows[0],
        answers=target_rows[1],
        reward_scores=target_proxy_z.tolist(),
        judge_scores=target_judge_z.tolist(),
    )

    class_counts = classifier_dataset.class_counts()
    total_count = len(classifier_dataset)
    hack_count = class_counts.get(1, 0)
    non_hack_count = class_counts.get(0, 0)
    hack_percentage = 100.0 * hack_count / total_count
    label_summary = (
        f"Reward-hack labels: {hack_count} of {total_count} prompts "
        f"({hack_percentage:.2f}%) are hacks; {non_hack_count} are "
        f"non-hacks; theta={calibration.theta:.6f}; "
        f"dataset_id={artifact_id}"
    )
    print(f"\n{label_summary}\n", flush=True)
    logger.info("%s", label_summary)
    if len(class_counts) < 2:
        raise ValueError(
            "Classifier labels contain only one class under the frozen "
            f"calibration threshold; class counts: {class_counts}"
        )

    policy_source = _policy_source_label(config, policy)
    run_directory = (
        Path(config.classifier_output_root)
        / f"policy={policy_source}"
        / f"id={artifact_id}"
    )
    dataset_path = classifier_dataset.save(run_directory / "dataset.json")
    logger.info("Saved classifier dataset to %s", dataset_path)

    train_dataset, test_dataset = classifier_dataset.split(
        test_size=config.classifier_test_size,
        random_state=config.classifier_random_state,
    )
    split_summary = (
        "Classifier label split: "
        f"train={train_dataset.class_counts()}; "
        f"test={test_dataset.class_counts()}"
    )
    print(split_summary, flush=True)
    logger.info("%s", split_summary)

    classifier: Classifier | None = None
    classifier_trainer: ClassifierTrainer | None = None
    hf_trainer = None

    try:
        classifier = Classifier(
            config.classifier_model_name,
            classifier_id=artifact_id,
            source_policy=policy_source,
        )
        training_config = ClassifierTrainingConfig(
            output_dir=str(run_directory),
            epochs=config.classifier_epochs,
            batch_size=config.classifier_batch_size,
            learning_rate=config.classifier_learning_rate,
            max_length=config.classifier_max_length,
            lora_settings=config.lora_settings,
        )
        classifier_trainer = ClassifierTrainer(classifier, training_config)
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


def create_gap_finder(
    config: ConfigTrainClassifier | None = None,
    *,
    policy: PolicyModel | None = None,
    calibration: GapCalibration | None = None,
) -> tuple[GapFinder, GapCalibration]:
    """Train a regressor for the continuous normalized proxy-minus-judge gap."""
    config = config or ConfigTrainClassifier()
    _validate_classifier_config(config)
    if policy is not None and config.policy.checkpoint is not None:
        raise ValueError("Pass either policy or config.policy.checkpoint, not both")

    raw_dataset = load_dataset(config.dataset_name)
    request_dataset = RequestDataset.from_raw(
        raw_dataset,
        config.policy.model_name,
    )
    request_dataset.truncate(config.start_dataset, config.end_dataset)
    if len(request_dataset) == 0:
        raise ValueError("The selected GapFinder dataset range is empty")

    reference_rows = (
        _generate_policy_rows(config, request_dataset, reference=True)
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
        spec=config.reward,
        batch_size=config.reward_batch_size,
        max_length=config.score_max_length,
        reference_rows=reference_rows,
        target_rows=target_rows,
    )
    reference_judge, target_judge = _score_reference_and_target(
        spec=config.judge,
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

    target_proxy_z = (target_proxy - calibration.proxy_mean) / calibration.proxy_std
    target_judge_z = (target_judge - calibration.judge_mean) / calibration.judge_std
    artifact_id = str(uuid4())
    dataset = DatasetGapFinder(id=artifact_id)
    dataset.add(
        target_rows[0],
        target_rows[1],
        target_proxy_z.tolist(),
        target_judge_z.tolist(),
    )

    policy_source = _policy_source_label(config, policy)
    run_directory = (
        Path(config.gap_finder_output_root)
        / f"policy={policy_source}"
        / f"id={artifact_id}"
    )
    dataset.save(run_directory / "dataset.json")
    calibration.save(run_directory / "calibration.json")
    train_dataset, test_dataset = dataset.split(
        test_size=config.classifier_test_size,
        random_state=config.classifier_random_state,
    )
    gaps = np.asarray([row["labels"] for row in dataset.dataset])
    print(
        "GapFinder targets: "
        f"count={len(gaps)}, mean={gaps.mean():.6f}, "
        f"std={gaps.std():.6f}, theta={calibration.theta:.6f}",
        flush=True,
    )

    gap_finder: GapFinder | None = None
    trainer: GapFinderTrainer | None = None
    hf_trainer = None
    try:
        gap_finder = GapFinder(
            config.gap_finder_model_name,
            gap_finder_id=artifact_id,
            source_policy=policy_source,
        )
        training_config = GapFinderTrainingConfig(
            output_dir=str(run_directory),
            epochs=config.gap_finder_epochs,
            batch_size=config.gap_finder_batch_size,
            learning_rate=config.gap_finder_learning_rate,
            max_length=config.gap_finder_max_length,
            lora_settings=config.gap_finder_lora_settings,
        )
        trainer = GapFinderTrainer(gap_finder, training_config)
        hf_trainer = trainer.train(train_dataset)
        metrics = trainer.evaluate(hf_trainer, test_dataset)
        logger.info("GapFinder test metrics: %s", metrics)
        gap_finder.model.eval()
        return gap_finder, calibration
    except BaseException:
        offload_to_cpu(gap_finder)
        raise
    finally:
        hf_trainer = None
        trainer = None
        empty_cuda_cache()


def evaluate_gap_finder(
    prompts: Sequence[str],
    *,
    policy: PolicyModel,
    gap_finder: GapFinder,
    config: ConfigTrainClassifier,
    calibration: GapCalibration,
    generation_batch_size: int = 8,
) -> list[dict[str, str | float]]:
    """Inspect raw scores, normalized gaps, and GapFinder corrections.

    GapFinder predicts ``proxy_z - judge_z``.  The final corrected proxy is
    therefore ``proxy_z - predicted_gap`` (or equivalently, in raw proxy
    units, ``proxy_score - predicted_gap * proxy_std``).
    """
    _validate_classifier_config(config)
    _validate_gap_calibration(calibration)
    if generation_batch_size < 1:
        raise ValueError("generation_batch_size must be at least 1")
    prompt_list = list(prompts)
    if not prompt_list:
        raise ValueError("prompts cannot be empty")
    if not all(isinstance(prompt, str) for prompt in prompt_list):
        raise TypeError("prompts must contain only strings")

    answers: list[str] = []
    try:
        for start in range(0, len(prompt_list), generation_batch_size):
            generated = policy.generate_batch(
                prompt_list[start : start + generation_batch_size]
            )
            answers.extend(generated)
    finally:
        policy.offload()
        empty_cuda_cache()
    if len(answers) != len(prompt_list):
        raise RuntimeError("Policy returned a different number of answers than prompts")

    rows = (prompt_list, answers)
    proxy_scores = _score_policy_rows(
        spec=config.reward,
        batch_size=config.reward_batch_size,
        max_length=config.score_max_length,
        row_groups={"evaluation": rows},
    )["evaluation"]
    judge_scores = _score_policy_rows(
        spec=config.judge,
        batch_size=config.judge_batch_size,
        max_length=config.score_max_length,
        row_groups={"evaluation": rows},
    )["evaluation"]

    gap_finder.move_to_current_device()
    try:
        predicted_gaps = np.asarray(
            gap_finder.predict_gap(
                prompt_list,
                answers,
                batch_size=config.gap_finder_batch_size,
                max_length=config.gap_finder_max_length,
            ),
            dtype=np.float64,
        )
    finally:
        gap_finder.offload()
        empty_cuda_cache()
    if predicted_gaps.shape != proxy_scores.shape:
        raise ValueError(
            f"GapFinder returned shape {predicted_gaps.shape}; expected "
            f"{proxy_scores.shape}"
        )
    if not np.isfinite(predicted_gaps).all():
        raise ValueError("GapFinder predictions must be finite")

    proxy_z = (proxy_scores - calibration.proxy_mean) / calibration.proxy_std
    judge_z = (judge_scores - calibration.judge_mean) / calibration.judge_std
    actual_gaps = proxy_z - judge_z
    corrected_proxy_z = proxy_z - predicted_gaps
    corrected_proxy_scores = (
        proxy_scores - predicted_gaps * calibration.proxy_std
    )
    residual_gaps = corrected_proxy_z - judge_z

    results: list[dict[str, str | float]] = []
    for index, (prompt, answer) in enumerate(zip(prompt_list, answers)):
        results.append(
            {
                "prompt": prompt,
                "answer": answer,
                "proxy_score": float(proxy_scores[index]),
                "judge_score": float(judge_scores[index]),
                "proxy_z": float(proxy_z[index]),
                "judge_z": float(judge_z[index]),
                "actual_gap": float(actual_gaps[index]),
                "predicted_gap": float(predicted_gaps[index]),
                "final_proxy_score": float(corrected_proxy_scores[index]),
                "final_proxy_z": float(corrected_proxy_z[index]),
                "final_gap_vs_judge": float(residual_gaps[index]),
            }
        )
    return results


def _ppo_output_directory(
    config: ConfigEval  | TrainingPPOConfig,
) -> str:
    if isinstance(config, TrainingPPOConfig):
        reward_mode_name = config.reward.mode_name
        start_dataset = config.dataset.start
        end_dataset = config.dataset.end
        policy_checkpoint = config.policy.checkpoint
        output_dir = config.output_dir
    else:
        reward_mode_name = config.reward_mode_name
        start_dataset = config.start_dataset
        end_dataset = config.end_dataset
        policy_checkpoint = config.policy_checkpoint
        output_dir = config.output_dir

    default_output_dir = Path(
        "outputs/ppo_policy/"
        f"{reward_mode_name}_{start_dataset}_{end_dataset}"
    )
    if output_dir is None:
        if policy_checkpoint is None:
            return str(default_output_dir)
        return str(
            default_output_dir.parent
            / (
                f"{default_output_dir.name}_from_"
                f"{Path(policy_checkpoint).name}"
            )
        )

    selected_output = Path(output_dir).expanduser()
    if policy_checkpoint is not None:
        checkpoint = Path(policy_checkpoint).expanduser().resolve()
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
    """Train a policy with PPO, generate answers, and return it on CPU."""
    
    if config.start_dataset < 0 or config.end_dataset <= config.start_dataset:
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
        dataset.truncate(config.start_dataset, config.end_dataset)
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
            policy.save_dataset(Path(training_output_dir) / "final")
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



@dataclass
class ConfigEval:
    """Configuration for PPO training followed by policy evaluation."""

    dataset_name: str = "Anthropic/hh-rlhf"
    start_dataset: int = 0
    end_dataset: int = 1000

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


@dataclass(frozen=True)
class DatasetSpec:
    class_name: str = "RequestDataset"
    dataset_name: str = "Anthropic/hh-rlhf"
    start: int = 0
    end: int = 100
    
    
@dataclass(frozen=True)
class EvaluatorSpec:
    class_name: str = "PrometheusEvaluator"
    model_name: str = "prometheus-eval/prometheus-7b-v2.0"
    


@dataclass(frozen=True)
class TrainingPPOConfig:
    policy: PolicySpec = field(default_factory=PolicySpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    classifier_load: tuple[str | Path, ...] = ()
    output_dir: str | Path | None = None
    epochs: float = 1.0
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    rollout_forward_batch_size: int = 16
    response_length: int = 128
    generation_batch_size: int = 64
    num_ppo_epochs: int = 4
    num_mini_batches: int = 1
    learning_rate: float = 3e-6
    save_steps: int = 10
    save_total_limit: int = 1
    logging_steps: int = 10


def _validate_training_ppo_config(config: TrainingPPOConfig) -> None:
    if config.dataset.start < 0 or config.dataset.end <= config.dataset.start:
        raise ValueError("dataset end must be greater than a non-negative start")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    for name, value in (
        ("batch_size", config.batch_size),
        ("gradient_accumulation_steps", config.gradient_accumulation_steps),
        ("rollout_forward_batch_size", config.rollout_forward_batch_size),
        ("generation_batch_size", config.generation_batch_size),
        ("response_length", config.response_length),
        ("num_ppo_epochs", config.num_ppo_epochs),
        ("num_mini_batches", config.num_mini_batches),
        ("save_steps", config.save_steps),
        ("save_total_limit", config.save_total_limit),
        ("logging_steps", config.logging_steps),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")


def temp_ppo_train_policy(
    config: TrainingPPOConfig,
    *,
    classifiers: Sequence[Classifier] = (),
    adjustments: Sequence[RewardAdjustment] = (),
) -> PolicyModel:
    """Build configured components, train with PPO, and return policy on CPU."""
    _validate_training_ppo_config(config)
    training_output_dir = _ppo_output_directory(config)

    raw_dataset = load_dataset(config.dataset.dataset_name)
    dataset = DatasetFactory.create_from_raw(
        config.dataset.class_name,
        raw_dataset,
        config.policy.model_name,
        config.dataset.start,
        config.dataset.end,
    )
    if not isinstance(dataset, RequestDataset):
        raise TypeError("PPO training requires a RequestDataset")
    if len(dataset) == 0:
        raise ValueError("The selected PPO dataset range is empty")

    policy = PolicyModelFactory.create_model(
        config.policy.class_name,
        config.policy.model_name,
        config.policy.lora_config,
        checkpoint=config.policy.checkpoint,
        is_trainable=True,
    )
    
    
    reward_classifiers = list(classifiers)
    for classifier_path in config.classifier_load:
        reward_classifiers.append(Classifier.load(classifier_path))
    
    
    reward_options = {
        "mean": config.reward.mean,
        "std": config.reward.std,
    }
    if adjustments:
        reward_options["adjustments"] = adjustments
    reward = RewardModelFactory.create_model(
        config.reward.class_name,
        config.reward.model_name,
        config.reward.mode_name,
        reward_classifiers,
        **reward_options,
    )
    
    return _train_policy(
        policy,
        reward,
        dataset,
        config,
        training_output_dir,
    )


def _train_policy(
    policy: PolicyModel,
    reward: PPORewardModelProtocol,
    dataset: RequestDataset,
    config: TrainingPPOConfig,
    training_output_dir: str,
) -> PolicyModel:
    reference_policy: PolicyModel | None = None
    value_model: ValueModel | None = None
    trainer: PolicyPPOTrainer | None = None
    try:
        if config.policy.checkpoint is not None and not isinstance(
            policy.model,
            PeftModel,
        ):
            reference_policy = PolicyModelFactory.create_model(
                config.policy.class_name,
                config.policy.model_name,
                None,
            )

        value_model = ValueModel(config.policy.model_name)
        ppo_config = PPOTrainingConfig(
            output_dir=training_output_dir,
            epochs=config.epochs,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            rollout_forward_batch_size=config.rollout_forward_batch_size,
            response_length=config.response_length,
            num_ppo_epochs=config.num_ppo_epochs,
            num_mini_batches=config.num_mini_batches,
            learning_rate=config.learning_rate,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            logging_steps=config.logging_steps,
        )
        trainer = PolicyPPOTrainer(
            policy,
            reward,
            value_model,
            dataset,
            ppo_config,
            reference_policy=reference_policy,
        )
        trainer.train()

        policy.generate_new_dataset(
            dataset,
            batch_size=config.generation_batch_size,
        )
        policy.save_dataset(Path(training_output_dir) / "final")
        policy.offload()
        return policy
    except BaseException:
        offload_to_cpu(policy)
        raise
    finally:
        offload_to_cpu(reference_policy)
        offload_to_cpu(value_model)
        offload_to_cpu(reward)
        trainer = None
        reference_policy = None
        value_model = None
        empty_cuda_cache()
    
    
    


def ppo_train_policy(config) -> PolicyModel:
    
    """Train PPO, generate answers, and evaluate the resulting policy.

    ``config.policy_checkpoint`` performs a policy-weight warm start. It is
    not an exact PPO resume because optimizer, value-model, RNG, and dataloader
    state are not restored.
    """
    if config.start_dataset < 0 or config.end_dataset <= config.start_dataset:
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
        dataset.truncate(config.start_dataset, config.end_dataset)
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
            policy.save_dataset(Path(training_output_dir) / "final")
        except BaseException:
            offload_to_cpu(policy)
            raise

        # Prometheus is large, so it must not overlap the policy on CUDA.
        policy.offload()
        empty_cuda_cache()
      
        return policy

    except BaseException:
            offload_to_cpu(policy)
            raise
    finally:
        policy = None
        reference_policy = None
        value_model = None
        reward_model = None
        trainer = None
        empty_cuda_cache()


def get_gap_calibration(
    dataset,
    policy: PolicyModel,
    config: ConfigTrainClassifier,
    quantile: float = 0.95,
    policy_batch_size: int = 64,
    proxy_batch_size: int = 64,
    judge_batch_size: int = 16,
) -> GapCalibration:
    
    if not 0.0 < quantile < 1.0:

        raise ValueError("quantile must be between 0 and 1")

    policy.generate_new_dataset(dataset, policy_batch_size)
    offload_to_cpu(policy)
    empty_cuda_cache()


    proxy = RewardModel(config.reward_model_name, config.reward_mode_name)
    
    try:
        proxy.score_policy(policy, proxy_batch_size)
    finally:
        offload_to_cpu(proxy)
        empty_cuda_cache()

    judge = RewardModel(config.judge_model_name, config.judge_mode_name)
    try:
        judge.score_policy(policy, judge_batch_size)
    finally:
        offload_to_cpu(judge)
        empty_cuda_cache()

    return _build_gap_calibration(
        policy.get_dataset_col("proxy"),
        policy.get_dataset_col("judge"),
        quantile,
    )



def reward_similarity(dataset, policy: PolicyModel, proxy: RewardModel, judge: RewardModel, policy_batch_size=64, proxy_batch_size=64, judge_batch_size=16):


    policy.generate_new_dataset(dataset, policy_batch_size)

    try:
        proxy.score_policy(policy, proxy_batch_size)
    finally:
        offload_to_cpu(proxy)
        proxy = None
        empty_cuda_cache()

    try:
        judge.score_policy(policy, judge_batch_size)
    finally:
        offload_to_cpu(judge)
        judge = None
        empty_cuda_cache()


    gap = _build_gap_calibration(policy.get_dataset_col("proxy"), policy.get_dataset_col("judge"), 0.95)
    policy.normalize_score_col("proxy", gap.proxy_mean, gap.proxy_std)
    policy.normalize_score_col("judge", gap.judge_mean, gap.judge_std)

    r_small = policy.get_dataset_col("proxy")
    r_large = policy.get_dataset_col("judge")

    print("Pearson :", pearsonr(r_small, r_large))
    print("Spearman:", spearmanr(r_small, r_large))

    r_small = np.asarray(r_small)
    r_large = np.asarray(r_large)

    print("small mean/std:", r_small.mean(), r_small.std())
    print("large mean/std:", r_large.mean(), r_large.std())



@dataclass(frozen=True)
class EvaluateConfig:
    policy: PolicySpec
    evaluator: EvaluatorSpec = field(
        default_factory=lambda: EvaluatorSpec(
            class_name="PrometheusEvaluator",
            model_name="prometheus-eval/prometheus-7b-v2.0",
        )
    )
    evaluator_batch_size: int = 1
    evaluator_reset: bool = False



def evaluate_policy(config: EvaluateConfig) -> float:
    if config.policy.checkpoint is None:
        raise ValueError("A policy checkpoint is required for evaluation")

    policy = PolicyModelFactory.create_model(
        config.policy.class_name,
        config.policy.model_name,
        config.policy.lora_config,
        checkpoint=config.policy.checkpoint,
        is_trainable=False,
    )

    if policy.dataset is None:
        raise ValueError("The loaded policy checkpoint contains no dataset")

    # Saved answers are already present, so the policy does not need CUDA
    # while the much larger evaluator is loaded.
    policy.offload()
    empty_cuda_cache()
    evaluator: PrometheusEvaluator | None = None
    try:
        created_evaluator = EvaluatorModelFactory.create_model(
            config.evaluator.class_name,
            config.evaluator.model_name,
        )
        if not isinstance(created_evaluator, PrometheusEvaluator):
            raise TypeError(
                "This evaluation function currently requires PrometheusEvaluator"
            )
        evaluator = created_evaluator
        return float(
            evaluator.evaluate(
                policy,
                reset=config.evaluator_reset,
                batch_size=config.evaluator_batch_size,
            )
        )
    finally:
        offload_to_cpu(policy)
        evaluator = None
        empty_cuda_cache()
