from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset as HFDataset
from peft import PeftModel, TaskType, get_peft_model
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from Datasets.dataset_classifier import DatasetClassifier
from Models.lora import LoRASettings
from Models.model_classifier import Classifier


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
        )

        trainer.train()

        final_directory = Path(self.config.output_dir) / "final"
        trainer.save_model(str(final_directory))
        self.classifier.tokenizer.save_pretrained(final_directory)

        return trainer
