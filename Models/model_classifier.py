from Models.models import BinaryClassifier
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
import torch
from peft import PeftMixedModel, PeftModel
from peft import PeftConfig

from pathlib import Path
import json

from collections.abc import Sequence
import logging

from Models.runtime import best_dtype, current_device
from uuid import uuid4


logger = logging.getLogger(__name__)

class Classifier(BinaryClassifier):

    METADATA_FILE_NAME = "classifier_metadata.json"
    
    def __init__(
        self,
        model_name: str,
        classifier_id: str | None = None,
        source_policy: str | None = None,
    ) -> None:
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel | PeftModel | PeftMixedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                dtype=best_dtype(),
            )
        )
        self.model.to(current_device())
        self.id = classifier_id or str(uuid4())
        self.source_policy = source_policy
        self.checkpoint_path: Path | None = None

    def save(self, path: str | Path) -> Path:
        """Save a loadable classifier, tokenizer, and stable classifier ID."""
        destination = Path(path).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(destination))
        self.tokenizer.save_pretrained(destination)
        metadata_path = destination / self.METADATA_FILE_NAME
        metadata_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "classifier_id": self.id,
                    "model_name": self.model_name,
                    "source_policy": self.source_policy,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.checkpoint_path = destination
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> "Classifier":
        """Restore a full-model or PEFT classifier checkpoint."""
        checkpoint = Path(path).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {checkpoint}"
            )
        if not checkpoint.is_dir():
            raise NotADirectoryError(
                f"Classifier checkpoint must be a directory: {checkpoint}"
            )

        checkpoint_source = str(checkpoint)
        adapter_checkpoint = (checkpoint / "adapter_config.json").is_file()
        if adapter_checkpoint:
            peft_config = PeftConfig.from_pretrained(checkpoint_source)
            base_model_name = peft_config.base_model_name_or_path
            if not base_model_name:
                raise ValueError(
                    "Classifier PEFT checkpoint does not identify its base "
                    f"model: {checkpoint}"
                )
            tokenizer_source = (
                checkpoint_source
                if (checkpoint / "tokenizer_config.json").is_file()
                else base_model_name
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                dtype=best_dtype(),
            )
            model = PeftModel.from_pretrained(
                base_model,
                checkpoint_source,
                is_trainable=False,
            )
            model_name = base_model_name
        else:
            tokenizer_source = checkpoint_source
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_source,
                dtype=best_dtype(),
            )
            model_name = checkpoint_source

        classifier_id = checkpoint.name
        source_policy = None
        metadata_path = checkpoint / cls.METADATA_FILE_NAME
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Classifier metadata is not valid JSON: {metadata_path}"
                ) from error
            if not isinstance(metadata, dict) or metadata.get(
                "format_version"
            ) != 1:
                raise ValueError(
                    f"Unsupported classifier metadata: {metadata_path}"
                )
            stored_id = metadata.get("classifier_id")
            if not isinstance(stored_id, str) or not stored_id.strip():
                raise ValueError("Classifier metadata has an invalid ID")
            classifier_id = stored_id
            stored_model_name = metadata.get("model_name")
            if isinstance(stored_model_name, str) and stored_model_name:
                model_name = stored_model_name
            stored_source_policy = metadata.get("source_policy")
            if stored_source_policy is not None and not isinstance(
                stored_source_policy,
                str,
            ):
                raise ValueError(
                    "Classifier metadata has an invalid source policy"
                )
            source_policy = stored_source_policy

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        loaded = cls.__new__(cls)
        loaded.model_name = model_name
        loaded.tokenizer = tokenizer
        loaded.model = model
        loaded.model.to(device if device is not None else current_device())
        loaded.model.eval()
        loaded.id = classifier_id
        loaded.source_policy = source_policy
        loaded.checkpoint_path = checkpoint
        return loaded
        
        
    def predict(
            self,
            prompts: Sequence[str],
            answers: Sequence[str],
            threshold: float = 0.5
        ) -> list[int]:

            probabilities = self.predict_proba(prompts, answers)
            logger.info("Classifier:%s predicting the labels", self.id)
            
            labels = [
                int(label >= threshold) for label in probabilities
            ]
            
            logger.debug("Classifier:%s labels: %s", self.id, labels)
            return labels
    
    
    def predict_proba(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
    ) -> list[float]:

        if len(prompts) != len(answers):
            raise ValueError("prompts and answers must have the same length")

        if not prompts:
            return []

        
        logger.info("Classifier:%s predicting the inputs", self.id)
        
        
        inputs = self.tokenizer(
            list(prompts),
            list(answers),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        self.model.eval()

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits

        if logits.shape[-1] == 2:
            # Two outputs: probability for class 0 and class 1.
            probabilities = torch.softmax(logits, dim=-1)[:, 1]

        elif logits.shape[-1] == 1:
            # One output: convert the single logit to a probability.
            probabilities = torch.sigmoid(logits.squeeze(-1))

        else:
            raise ValueError(
                f"Expected 1 or 2 output logits, got {logits.shape[-1]}"
            )

        outputs = probabilities.float().cpu().tolist()
        
        logger.debug("Classifier:%s probabilities: %s", self.id, outputs)
        return outputs
