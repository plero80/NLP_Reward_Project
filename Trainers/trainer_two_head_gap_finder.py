from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
import numpy as np
from peft import PeftModel, TaskType, get_peft_model
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, EvalPrediction, Trainer, TrainingArguments

from Datasets.dataset_gap_finder import DatasetGapFinder
from Models.lora import LoRASettings
from Models.model_two_head_gap_finder import TwoHeadGapFinder
from Trainers.trainer_gap_finder import compute_gap_finder_report


def compute_two_head_report(
    actual_gaps,
    predicted_gaps,
    detection_probabilities,
    theta: float,
) -> dict[str, float | int]:
    """Report regression quality and dedicated tail-head detection quality."""
    labels = np.asarray(actual_gaps, dtype=np.float64).reshape(-1)
    gaps = np.asarray(predicted_gaps, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(
        detection_probabilities,
        dtype=np.float64,
    ).reshape(-1)
    if labels.shape != gaps.shape or labels.shape != probabilities.shape:
        raise ValueError("actual gaps, predicted gaps, and probabilities must align")
    if labels.size == 0:
        raise ValueError("two-head evaluation inputs cannot be empty")
    if not all(np.isfinite(values).all() for values in (labels, gaps, probabilities)):
        raise ValueError("two-head evaluation inputs must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("detection probabilities must be between zero and one")

    regression_report = compute_gap_finder_report(labels, gaps, theta)
    actual_tail = labels > float(theta)
    predicted_tail = probabilities >= 0.5
    report: dict[str, float | int] = {
        "mse": regression_report["mse"],
        "mae": regression_report["mae"],
        "rmse": regression_report["rmse"],
        "r2": regression_report["r2"],
        "pearson": regression_report["pearson"],
        "spearman": regression_report["spearman"],
        "mae_d_gt_theta": regression_report["mae_d_gt_theta"],
        "detector_precision": float(
            precision_score(actual_tail, predicted_tail, zero_division=0)
        ),
        "detector_recall": float(
            recall_score(actual_tail, predicted_tail, zero_division=0)
        ),
        "detector_f1": float(
            f1_score(actual_tail, predicted_tail, zero_division=0)
        ),
        "regression_detector_precision": regression_report[
            "precision_d_gt_theta"
        ],
        "regression_detector_recall": regression_report["recall_d_gt_theta"],
        "regression_detector_f1": regression_report["f1_d_gt_theta"],
        "test_examples": int(labels.size),
        "actual_d_gt_theta": int(actual_tail.sum()),
        "predicted_d_gt_theta": int(predicted_tail.sum()),
    }
    if np.unique(actual_tail).size == 2:
        report["detector_pr_auc"] = float(
            average_precision_score(actual_tail, probabilities)
        )
        report["detector_roc_auc"] = float(
            roc_auc_score(actual_tail, probabilities)
        )
    else:
        report["detector_pr_auc"] = float("nan")
        report["detector_roc_auc"] = float("nan")
    return report


def _metric_function(theta: float):
    def compute(evaluation: EvalPrediction) -> dict[str, float | int]:
        if isinstance(evaluation.predictions, tuple):
            logits = np.asarray(evaluation.predictions[0])
        else:
            logits = np.asarray(evaluation.predictions)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected [examples, 2] logits, got {logits.shape}")
        return compute_two_head_report(
            evaluation.label_ids,
            logits[:, 0],
            expit(logits[:, 1]),
            theta,
        )

    return compute


@dataclass(frozen=True)
class TwoHeadGapFinderTrainingConfig:
    output_dir: str
    theta: float
    epochs: float = 3.0
    batch_size: int = 16
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-5
    max_length: int = 512
    high_gap_weight: float = 5.0
    detector_loss_weight: float = 1.0
    detector_positive_weight: float | None = None
    lora_settings: LoRASettings | None = None


class _TwoHeadTrainer(Trainer):
    theta: float
    high_gap_weight: float
    detector_loss_weight: float
    detector_positive_weight: float

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        model_inputs = dict(inputs)
        gaps = model_inputs.pop("labels").reshape(-1).float()
        outputs = model(**model_inputs)
        logits = outputs.logits.float()
        predicted_gaps = logits[:, 0]
        detector_logits = logits[:, 1]
        tail_labels = (gaps > self.theta).float()

        regression_weights = torch.where(
            tail_labels.bool(),
            torch.full_like(gaps, self.high_gap_weight),
            torch.ones_like(gaps),
        )
        regression_loss = torch.mean(
            regression_weights * (predicted_gaps - gaps).square()
        )
        detector_loss = F.binary_cross_entropy_with_logits(
            detector_logits,
            tail_labels,
            pos_weight=torch.as_tensor(
                self.detector_positive_weight,
                dtype=detector_logits.dtype,
                device=detector_logits.device,
            ),
        )
        loss = regression_loss + self.detector_loss_weight * detector_loss
        return (loss, outputs) if return_outputs else loss


class TwoHeadGapFinderTrainer:
    def __init__(
        self,
        gap_finder: TwoHeadGapFinder,
        config: TwoHeadGapFinderTrainingConfig,
    ) -> None:
        self.gap_finder = gap_finder
        self.config = config
        if abs(gap_finder.theta - config.theta) > 1e-9:
            raise ValueError("model theta and training theta must match")
        if config.high_gap_weight <= 0 or config.detector_loss_weight < 0:
            raise ValueError("loss weights are invalid")
        if config.lora_settings is not None:
            if isinstance(gap_finder.model, PeftModel):
                raise ValueError("The two-head GapFinder already has a PEFT adapter")
            gap_finder.model = get_peft_model(
                gap_finder.model,
                config.lora_settings.build(TaskType.SEQ_CLS),
            )
            gap_finder.model.print_trainable_parameters()

    def _prepare_dataset(self, dataset: DatasetGapFinder) -> HFDataset:
        if len(dataset) == 0:
            raise ValueError("Cannot train on an empty GapFinder dataset")
        hf_dataset = HFDataset.from_list(dataset.dataset)

        def tokenize(batch):
            return self.gap_finder.tokenizer(
                batch["prompt"],
                batch["answer"],
                truncation=True,
                max_length=self.config.max_length,
            )

        return hf_dataset.map(
            tokenize,
            batched=True,
            remove_columns=[
                column for column in hf_dataset.column_names if column != "labels"
            ],
        )

    def _positive_weight(self, dataset: DatasetGapFinder) -> float:
        if self.config.detector_positive_weight is not None:
            weight = float(self.config.detector_positive_weight)
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError("detector_positive_weight must be positive")
            return weight
        positives = sum(row["labels"] > self.config.theta for row in dataset.dataset)
        negatives = len(dataset) - positives
        if positives == 0:
            raise ValueError("Training split has no examples above theta")
        return float(negatives / positives)

    def train(
        self,
        train_dataset: DatasetGapFinder,
        validation_dataset: DatasetGapFinder,
    ) -> Trainer:
        prepared_train = self._prepare_dataset(train_dataset)
        prepared_validation = self._prepare_dataset(validation_dataset)
        arguments = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="detector_pr_auc",
            greater_is_better=True,
            report_to="none",
        )
        trainer = _TwoHeadTrainer(
            model=self.gap_finder.model,
            args=arguments,
            train_dataset=prepared_train,
            eval_dataset=prepared_validation,
            processing_class=self.gap_finder.tokenizer,
            data_collator=DataCollatorWithPadding(
                tokenizer=self.gap_finder.tokenizer
            ),
            compute_metrics=_metric_function(self.config.theta),
        )
        trainer.theta = self.config.theta
        trainer.high_gap_weight = self.config.high_gap_weight
        trainer.detector_loss_weight = self.config.detector_loss_weight
        trainer.detector_positive_weight = self._positive_weight(train_dataset)
        trainer.train()
        self.gap_finder.save(Path(self.config.output_dir) / "final")
        return trainer

    def evaluate(
        self,
        trainer: Trainer,
        dataset: DatasetGapFinder,
        prefix: str = "test",
    ) -> dict[str, Any]:
        return trainer.evaluate(
            eval_dataset=self._prepare_dataset(dataset),
            metric_key_prefix=prefix,
        )
