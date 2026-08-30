from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Models.runtime import best_dtype, current_device


class ValueModel:
    """Single-score critic model used by TRL's PPO trainer."""

    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            dtype=best_dtype(),
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(current_device())
