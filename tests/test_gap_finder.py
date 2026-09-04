from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import EvalPrediction

from Datasets.dataset_gap_finder import DatasetGapFinder
from Models.model_reward import CompositeRewardModel
from Models.reward_adjustment import GapFinderCorrection
from Trainers.trainer_gap_finder import compute_gap_metrics


def test_gap_dataset_uses_continuous_normalized_difference(tmp_path: Path) -> None:
    dataset = DatasetGapFinder(id="gap-data")
    dataset.add(["p"], ["a"], [1.5], [-0.25])

    assert dataset[0]["labels"] == pytest.approx(1.75)
    loaded = DatasetGapFinder.load(dataset.save(tmp_path / "dataset.json"))
    assert loaded.id == "gap-data"
    assert loaded.dataset == dataset.dataset


def test_gap_metrics_are_regression_metrics() -> None:
    metrics = compute_gap_metrics(
        EvalPrediction(
            predictions=np.array([[1.0], [3.0]]),
            label_ids=np.array([2.0, 3.0]),
        )
    )
    assert metrics == pytest.approx({"mae": 0.5, "mse": 0.5, "rmse": 0.5 ** 0.5})


class _GapModel(torch.nn.Module):
    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(logits=input_ids[:, :1].float())


class _FakeGapFinder:
    id = "gap-finder"
    tokenizer = None

    def __init__(self) -> None:
        self.model = _GapModel()

    def predict_gap(self, prompts, answers):
        return [2.0] * len(prompts)


def test_gap_correction_scales_and_reverses_proxy_minus_judge_gap() -> None:
    correction = GapFinderCorrection(_FakeGapFinder(), reward_std=0.5)
    assert correction(["p"], ["a"]) == [-1.0]

    head = correction.ppo_head()
    assert head(torch.tensor([[2, 0]])).tolist() == [-1.0]


def test_composite_adds_raw_gap_correction_before_normalizing() -> None:
    class Backbone(torch.nn.Module):
        def forward(self, input_ids, **_kwargs):
            hidden = torch.full((*input_ids.shape, 1), 3.0)
            return SimpleNamespace(hidden_states=[hidden])

    class Reward(torch.nn.Module):
        base_model_prefix = "core"

        def __init__(self):
            super().__init__()
            self.core = Backbone()
            self.config = SimpleNamespace()

        def score(self, hidden_states):
            return hidden_states

    correction = GapFinderCorrection(_FakeGapFinder(), reward_std=0.5)
    model = CompositeRewardModel(
        Reward(),
        [correction.ppo_head()],
        mean=1.0,
        std=0.5,
    )
    output = model.backbone(input_ids=torch.tensor([[2, 0]]))
    scores = model.score(output.hidden_states[-1])

    # raw: 3 - (2 * .5) = 2; normalized: (2 - 1) / .5 = 2
    assert torch.allclose(scores, torch.full_like(scores, 2.0))
