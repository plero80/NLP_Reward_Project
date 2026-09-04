from __future__ import annotations

from collections.abc import Sequence
import json
import math
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Models.runtime import best_dtype, current_device


class GapFinder:
    """Regression model predicting ``proxy_z - judge_z`` from text."""

    METADATA_FILE_NAME = "gap_finder_metadata.json"

    def __init__(
        self,
        model_name: str,
        gap_finder_id: str,
        source_policy: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
            dtype=best_dtype(),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(current_device())
        self.id = gap_finder_id
        self.source_policy = source_policy
        self.checkpoint_path: Path | None = None

    def predict_gap(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
        *,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> list[float]:
        if len(prompts) != len(answers):
            raise ValueError("prompts and answers must have the same length")
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive")
        device = next(self.model.parameters()).device
        predictions: list[float] = []
        self.model.eval()
        for start in range(0, len(prompts), batch_size):
            inputs = self.tokenizer(
                list(prompts[start : start + batch_size]),
                list(answers[start : start + batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.inference_mode():
                logits = self.model(**inputs).logits.reshape(-1)
            values = logits.float().cpu().tolist()
            if not all(math.isfinite(value) for value in values):
                raise ValueError("GapFinder returned a non-finite prediction")
            predictions.extend(values)
        return predictions

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(destination))
        self.tokenizer.save_pretrained(destination)
        (destination / self.METADATA_FILE_NAME).write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "gap_finder_id": self.id,
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
    ) -> GapFinder:
        checkpoint = Path(path).expanduser()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"GapFinder checkpoint not found: {checkpoint}")
        metadata_path = checkpoint / cls.METADATA_FILE_NAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError(f"Unsupported GapFinder metadata: {metadata_path}")
        adapter = (checkpoint / "adapter_config.json").is_file()
        if adapter:
            peft_config = PeftConfig.from_pretrained(str(checkpoint))
            base_name = peft_config.base_model_name_or_path
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_name,
                num_labels=1,
                problem_type="regression",
                dtype=best_dtype(),
            )
            model = PeftModel.from_pretrained(base_model, str(checkpoint), is_trainable=False)
            tokenizer_source = str(checkpoint) if (checkpoint / "tokenizer_config.json").is_file() else base_name
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                str(checkpoint), dtype=best_dtype()
            )
            tokenizer_source = str(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        loaded = cls.__new__(cls)
        loaded.model_name = metadata["model_name"]
        loaded.tokenizer = tokenizer
        loaded.model = model
        loaded.model.to(device if device is not None else current_device())
        loaded.model.eval()
        loaded.id = metadata["gap_finder_id"]
        loaded.source_policy = metadata.get("source_policy")
        loaded.checkpoint_path = checkpoint
        return loaded
