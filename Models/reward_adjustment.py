from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Protocol

import torch
from transformers import PreTrainedTokenizerBase

from Models.model_classifier import Classifier
from Models.model_gap_finder import GapFinder


class PPOAdjustmentHead(torch.nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class RewardAdjustment(Protocol):
    """Callable policy that returns an additive change to a raw reward."""

    id: str
    tokenizer: PreTrainedTokenizerBase
    model: torch.nn.Module

    def __call__(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
    ) -> list[float]: ...

    def ppo_head(self) -> PPOAdjustmentHead: ...


class ProbabilityPenaltyHead(PPOAdjustmentHead):
    def __init__(self, classifier_model: torch.nn.Module) -> None:
        super().__init__()
        self.classifier_model = classifier_model

    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        logits = self.classifier_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits
        if logits.shape[-1] == 2:
            probability = torch.softmax(logits.float(), dim=-1)[:, 1]
        elif logits.shape[-1] == 1:
            probability = torch.sigmoid(logits.float().squeeze(-1))
        else:
            raise ValueError("Classifier adjustment expects one or two logits")
        return torch.log1p(-probability.clamp(0.0, 1.0 - 1e-12))


class ClassifierProbabilityPenalty:
    """Owns the old log(1-p) policy outside of RewardModel."""

    def __init__(self, classifier: Classifier) -> None:
        self.classifier = classifier
        self.id = getattr(classifier, "id", "classifier")
        self.tokenizer = getattr(classifier, "tokenizer", None)
        self.model = getattr(classifier, "model", None)

    def __call__(self, prompts, answers) -> list[float]:
        adjustments: list[float] = []
        for probability in self.classifier.predict_proba(prompts, answers):
            probability = float(probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"Classifier {self.id!r} returned invalid probability {probability}"
                )
            adjustments.append(math.log1p(-min(probability, 1.0 - 1e-12)))
        return adjustments

    def ppo_head(self) -> PPOAdjustmentHead:
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("Classifier adjustment needs a torch model for PPO")
        return ProbabilityPenaltyHead(self.model)


class GapCorrectionHead(PPOAdjustmentHead):
    def __init__(self, gap_model: torch.nn.Module, reward_std: float) -> None:
        super().__init__()
        self.gap_model = gap_model
        self.reward_std = float(reward_std)

    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        predicted_gap = self.gap_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits.reshape(-1).float()
        return -predicted_gap * self.reward_std


class GapFinderCorrection:
    """Turn predicted ``proxy_z - judge_z`` into a raw reward correction."""

    def __init__(self, gap_finder: GapFinder, reward_std: float) -> None:
        reward_std = float(reward_std)
        if not math.isfinite(reward_std) or reward_std <= 1e-8:
            raise ValueError("reward_std must be finite and greater than zero")
        self.gap_finder = gap_finder
        self.reward_std = reward_std
        self.id = gap_finder.id
        self.tokenizer = gap_finder.tokenizer
        self.model = gap_finder.model

    def __call__(self, prompts, answers) -> list[float]:
        return [
            -float(gap) * self.reward_std
            for gap in self.gap_finder.predict_gap(prompts, answers)
        ]

    def ppo_head(self) -> PPOAdjustmentHead:
        return GapCorrectionHead(self.model, self.reward_std)
