from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
import torch

from Models.models import ScoreModel

from collections.abc import Sequence
import logging



logger = logging.getLogger(__name__)
NAME = " Model"

class RewardModel(ScoreModel):
    
    
    def __init__(self, model_name) -> None:
        self.base_model_prefix = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        

    def score(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
    ) -> list[float]:

        if len(prompts) != len(answers):
            logger.error("%s: prompts and answers must have the same length")
            raise ValueError("prompts and answers must have the same length")

        
        logger.info("%s: Starting to calculate the score", NAME)
        
        conversations = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            for prompt, answer in zip(prompts, answers)
        ]

        formatted_texts = [
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
            )
            for conversation in conversations
        ]

        # Remove a duplicated BOS token when present.
        if self.tokenizer.bos_token is not None:
            formatted_texts = [
                text[len(self.tokenizer.bos_token):]
                if text.startswith(self.tokenizer.bos_token)
                else text
                for text in formatted_texts
            ]

        inputs = self.tokenizer(
            formatted_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16_384,
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            
        
        scores = outputs.logits.squeeze(-1).float().cpu().tolist()
        logger.debug("%s: Reward scores: %s", NAME, scores)

        return scores