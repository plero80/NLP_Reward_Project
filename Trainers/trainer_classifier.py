from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

from datasets import Dataset as HFDataset
import numpy as np
from peft import PeftModel, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from Datasets.dataset_classifier import DatasetClassifier
from Models.lora import LoRASettings
from Models.model_classifier import Classifier


logger = logging.getLogger(__name__)


def compute_classifier_metrics(
    evaluation: EvalPrediction,
) -> dict[str, float]:
    """Compute binary classification metrics from Trainer predictions."""
    logits = evaluation.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    labels = np.asarray(evaluation.label_ids).astype(int)

    if logits.ndim == 1 or logits.shape[-1] == 1:
        flat_logits = logits.reshape(-1)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(flat_logits, -80, 80)))
    elif logits.shape[-1] == 2:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials[:, 1] / exponentials.sum(axis=-1)
    else:
        raise ValueError(
            f"Expected one or two classifier logits, got {logits.shape[-1]}"
        )

    predictions = (probabilities >= 0.5).astype(int)
    unique_labels = np.unique(labels)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
            if len(unique_labels) == 2
            else accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
    }
    if len(unique_labels) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    return metrics


@dataclass(frozen=True)
class ClassifierTrainingConfig:
    output_dir: str
    epochs: float = 3.0
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    lora_settings: LoRASettings | None = None


class ClassifierTrainer:
    def __init__(
        self,
        classifier: Classifier,
        config: ClassifierTrainingConfig,
    ) -> None:
        self.classifier = classifier
        self.config = config

        if config.lora_settings is not None:
            if isinstance(self.classifier.model, PeftModel):
                raise ValueError("The classifier already has a PEFT adapter")

            self.classifier.model = get_peft_model(
                self.classifier.model,
                config.lora_settings.build(TaskType.SEQ_CLS),
            )
            self.classifier.model.print_trainable_parameters()

    def _prepare_dataset(
        self,
        dataset: DatasetClassifier,
    ) -> HFDataset:
        if len(dataset) == 0:
            raise ValueError("Cannot train a classifier with an empty dataset")

        hf_dataset = HFDataset.from_list(dataset.dataset)
        tokenizer = self.classifier.tokenizer
        max_length = self.config.max_length

        def tokenize(batch):
            return tokenizer(
                batch["prompt"],
                batch["answer"],
                truncation=True,
                max_length=max_length,
            )

        columns_to_remove = [
            column
            for column in hf_dataset.column_names
            if column != "labels"
        ]
        return hf_dataset.map(
            tokenize,
            batched=True,
            remove_columns=columns_to_remove,
        )

    def train(
        self,
        train_dataset: DatasetClassifier,
        eval_dataset: DatasetClassifier | None = None,
    ) -> Trainer:
        prepared_train_dataset = self._prepare_dataset(train_dataset)
        prepared_eval_dataset = (
            self._prepare_dataset(eval_dataset)
            if eval_dataset is not None and len(eval_dataset) > 0
            else None
        )

        has_evaluation = prepared_eval_dataset is not None
        arguments = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            eval_strategy="epoch" if has_evaluation else "no",
            save_strategy="epoch",
            load_best_model_at_end=has_evaluation,
            metric_for_best_model="f1" if has_evaluation else None,
            greater_is_better=True if has_evaluation else None,
            report_to="none",
        )

        trainer = Trainer(
            model=self.classifier.model,
            args=arguments,
            train_dataset=prepared_train_dataset,
            eval_dataset=prepared_eval_dataset,
            processing_class=self.classifier.tokenizer,
            data_collator=DataCollatorWithPadding(
                tokenizer=self.classifier.tokenizer
            ),
            compute_metrics=compute_classifier_metrics,
        )

        trainer.train()

        final_directory = Path(self.config.output_dir) / "final"
        trainer.save_model(str(final_directory))
        self.classifier.tokenizer.save_pretrained(final_directory)

        return trainer

    def evaluate(
        self,
        trainer: Trainer,
        test_dataset: DatasetClassifier,
    ) -> dict[str, Any]:
        """Evaluate the trained classifier on a held-out dataset."""
        prepared_test_dataset = self._prepare_dataset(test_dataset)
        metrics = trainer.evaluate(
            eval_dataset=prepared_test_dataset,
            metric_key_prefix="test",
        )
        logger.info("Held-out classifier metrics: %s", metrics)
        return metrics
