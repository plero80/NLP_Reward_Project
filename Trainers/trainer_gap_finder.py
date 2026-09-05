from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
import numpy as np
from peft import PeftModel, TaskType, get_peft_model
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score
from transformers import DataCollatorWithPadding, EvalPrediction, Trainer, TrainingArguments

from Datasets.dataset_gap_finder import DatasetGapFinder
from Models.lora import LoRASettings
from Models.model_gap_finder import GapFinder


def compute_gap_metrics(evaluation: EvalPrediction) -> dict[str, float]:
    predictions = np.asarray(evaluation.predictions).reshape(-1)
    labels = np.asarray(evaluation.label_ids).reshape(-1)
    errors = predictions - labels
    metrics = {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": float(r2_score(labels, predictions)) if len(labels) >= 2 else float("nan"),
    }
    if len(labels) >= 2 and np.std(labels) > 0 and np.std(predictions) > 0:
        metrics["pearson"] = float(pearsonr(labels, predictions).statistic)
        metrics["spearman"] = float(spearmanr(labels, predictions).statistic)
    else:
        metrics["pearson"] = float("nan")
        metrics["spearman"] = float("nan")
    return metrics


def compute_gap_finder_report(
    actual_gaps,
    predicted_gaps,
    theta: float,
) -> dict[str, float | int]:
    """Regression quality plus ``gap > theta`` detector performance."""
    labels = np.asarray(actual_gaps, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predicted_gaps, dtype=np.float64).reshape(-1)
    if labels.shape != predictions.shape or labels.size == 0:
        raise ValueError("actual and predicted gaps must be non-empty and have equal shape")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("actual and predicted gaps must be finite")
    theta = float(theta)
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")

    metrics = compute_gap_metrics(
        EvalPrediction(predictions=predictions, label_ids=labels)
    )
    actual_detector = labels > theta
    predicted_detector = predictions > theta
    high_gap_errors = np.abs(predictions[actual_detector] - labels[actual_detector])
    return {
        **metrics,
        "mae_d_gt_theta": (
            float(high_gap_errors.mean()) if high_gap_errors.size else float("nan")
        ),
        "precision_d_gt_theta": float(
            precision_score(actual_detector, predicted_detector, zero_division=0)
        ),
        "recall_d_gt_theta": float(
            recall_score(actual_detector, predicted_detector, zero_division=0)
        ),
        "f1_d_gt_theta": float(
            f1_score(actual_detector, predicted_detector, zero_division=0)
        ),
        "test_examples": int(labels.size),
        "actual_d_gt_theta": int(actual_detector.sum()),
        "predicted_d_gt_theta": int(predicted_detector.sum()),
    }


@dataclass(frozen=True)
class GapFinderTrainingConfig:
    output_dir: str
    epochs: float = 3.0
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 2e-5
    max_length: int = 512
    lora_settings: LoRASettings | None = None


class _Float32RegressionTrainer(Trainer):
    """Compute MSE in float32 while the model runs in BF16 or FP16."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        outputs = model(**model_inputs)
        predictions = outputs.logits.reshape(-1).float()
        loss = F.mse_loss(predictions, labels.reshape(-1).float())
        return (loss, outputs) if return_outputs else loss


class GapFinderTrainer:
    def __init__(self, gap_finder: GapFinder, config: GapFinderTrainingConfig) -> None:
        self.gap_finder = gap_finder
        self.config = config
        if config.lora_settings is not None:
            if isinstance(gap_finder.model, PeftModel):
                raise ValueError("The GapFinder already has a PEFT adapter")
            gap_finder.model = get_peft_model(
                gap_finder.model,
                config.lora_settings.build(TaskType.SEQ_CLS),
            )
            gap_finder.model.print_trainable_parameters()

    def _prepare_dataset(self, dataset: DatasetGapFinder) -> HFDataset:
        if len(dataset) == 0:
            raise ValueError("Cannot train GapFinder with an empty dataset")
        hf_dataset = HFDataset.from_list(dataset.dataset)
        tokenizer = self.gap_finder.tokenizer

        def tokenize(batch):
            return tokenizer(
                batch["prompt"],
                batch["answer"],
                truncation=True,
                max_length=self.config.max_length,
            )

        return hf_dataset.map(
            tokenize,
            batched=True,
            remove_columns=[c for c in hf_dataset.column_names if c != "labels"],
        )

    def train(
        self,
        train_dataset: DatasetGapFinder,
        eval_dataset: DatasetGapFinder | None = None,
    ) -> Trainer:
        train = self._prepare_dataset(train_dataset)
        evaluation = self._prepare_dataset(eval_dataset) if eval_dataset else None
        has_evaluation = evaluation is not None
        arguments = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            eval_strategy="epoch" if has_evaluation else "no",
            save_strategy="epoch",
            load_best_model_at_end=has_evaluation,
            metric_for_best_model="mse" if has_evaluation else None,
            greater_is_better=False if has_evaluation else None,
            report_to="none",
        )
        trainer = _Float32RegressionTrainer(
            model=self.gap_finder.model,
            args=arguments,
            train_dataset=train,
            eval_dataset=evaluation,
            processing_class=self.gap_finder.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.gap_finder.tokenizer),
            compute_metrics=compute_gap_metrics,
        )
        trainer.train()
        self.gap_finder.save(Path(self.config.output_dir) / "final")
        return trainer

    def evaluate(self, trainer: Trainer, dataset: DatasetGapFinder) -> dict[str, Any]:
        return trainer.evaluate(
            eval_dataset=self._prepare_dataset(dataset),
            metric_key_prefix="test",
        )
