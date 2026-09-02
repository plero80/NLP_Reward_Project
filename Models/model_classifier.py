from Models.models import BinaryClassifier
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
import torch
from peft import PeftMixedModel, PeftModel

from pathlib import Path

from collections.abc import Sequence
import logging

from Models.runtime import best_dtype, current_device
from uuid import uuid4


logger = logging.getLogger(__name__)

class Classifier(BinaryClassifier):
    
    def __init__(self, model_name: str) -> None:
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel | PeftModel | PeftMixedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                dtype=best_dtype(),
            )
        )
        self.model.to(current_device())
        self.id = str(uuid4())
        
        
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
