from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
import torch

from Models.models import ScoreModel
import logging



logger = logging.getLogger(__name__)
NAME = "Reward Model"

class RewardModel(ScoreModel):
    
    
    def __init__(self, model_name) -> None:
        self.base_model_prefix = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
    def score(self, prompt: str, answer: str) -> float:
        logger.info("%s: Starting to calculate the score", NAME)

        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]

        formatted = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
        )

        if (
            self.tokenizer.bos_token is not None
            and formatted.startswith(self.tokenizer.bos_token)
        ):
            formatted = formatted[len(self.tokenizer.bos_token):]

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=16_384,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        reward = outputs.logits[0, 0].item()
        logger.debug("%s: Reward score: %s", NAME, reward)

        return reward