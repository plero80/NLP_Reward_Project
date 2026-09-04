from typing import Protocol
from transformers import PreTrainedModel
from collections.abc import Sequence
import torch

class GenerateModel(Protocol):
    
    
    
    def generate(self, prompt: str) -> str:
        """Return an answer in text"""
        ...
    
        
    def generate_batch(self, prompts: Sequence[str]) -> Sequence[str]:
        """Return a batch of answers in text"""
        ...


class ResidualClassifier(Protocol):
    def predict_gap(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
    ) -> list[float]:
        """Return normalized ``proxy_z - judge_z`` predictions."""
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
        
        
        
from collections.abc import Sequence
from typing import Any, Protocol

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class PPORewardModelProtocol(Protocol):
    """Reward wrapper accepted by PolicyPPOTrainer."""

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        ...

    @property
    def model(self) -> PreTrainedModel:
        ...

    @property
    def classifiers(self) -> Sequence[dict[str, Any]]:
        ...

    def for_ppo(self) -> torch.nn.Module:
        """Return the actual reward module that TRL will execute."""
        ...

    def add_classifier(
        self,
        name: str,
        classifier: BinaryClassifier,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Attach a classifier adjustment to PPO rewards."""
        ...

    def add_adjustment(self, adjustment: Any) -> None:
        """Attach a generic callable reward adjustment."""
        ...
