from typing import Protocol
from transformers import PreTrainedModel
from collections.abc import Sequence
import torch

class GenerateModel(Protocol):
    
    
    
    def generate(self, prompt: str) -> str:
        """Return an answer in text"""
        ...
    
    def generate_with_question(self, prompt: str) -> tuple:
        ...
        
    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Return a batch of answers in text"""
        ...
        
        

class BinaryClassifier(Protocol):
    def predict(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
    ) -> list[int]:
        """Return one label (0 or 1) for each prompt-answer pair."""
        ...

    def predict_proba(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
    ) -> list[float]:
        """Return the probability of label 1 for each pair."""
        ...  
    

class ScoreModel(Protocol):
    
    base_model_prefix: str
    model: PreTrainedModel
       
    def score(self, prompts: Sequence[str], answers: Sequence[str]) -> list[float]:
        """Return a batch of scores from prompts and answers"""
        ...