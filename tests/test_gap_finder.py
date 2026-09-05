from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import EvalPrediction

from Datasets.dataset_gap_finder import DatasetGapFinder
from Models.model_reward import CompositeRewardModel
from Models.reward_adjustment import GapFinderCorrection, TwoHeadGapFinderCorrection
from Trainers.trainer_gap_finder import compute_gap_metrics
from Trainers.trainer_gap_finder import compute_gap_finder_report
from Trainers.trainer_gap_finder import _Float32RegressionTrainer
import functions


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
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["mse"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(0.5 ** 0.5)
    assert metrics["r2"] == pytest.approx(-1.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["spearman"] == pytest.approx(1.0)


def test_three_way_split_is_70_15_15_and_disjoint() -> None:
    dataset = DatasetGapFinder(id="split")
    dataset.add(
        [f"p{i}" for i in range(20)],
        [f"a{i}" for i in range(20)],
        list(range(20)),
        [0.0] * 20,
    )
    train, validation, test = dataset.split_three_way(random_state=7)

    assert (len(train), len(validation), len(test)) == (14, 3, 3)
    prompt_sets = [
        {row["prompt"] for row in split.dataset}
        for split in (train, validation, test)
    ]
    assert prompt_sets[0].isdisjoint(prompt_sets[1])
    assert prompt_sets[0].isdisjoint(prompt_sets[2])
    assert prompt_sets[1].isdisjoint(prompt_sets[2])


def test_gap_report_includes_high_gap_detector_metrics() -> None:
    report = compute_gap_finder_report(
        actual_gaps=[0.0, 1.5, 2.0, 3.0],
        predicted_gaps=[0.1, 1.2, 2.5, 0.0],
        theta=1.0,
    )

    assert report["mae_d_gt_theta"] == pytest.approx((0.3 + 0.5 + 3.0) / 3)
    assert report["precision_d_gt_theta"] == pytest.approx(1.0)
    assert report["recall_d_gt_theta"] == pytest.approx(2 / 3)
    assert report["f1_d_gt_theta"] == pytest.approx(0.8)
    assert report["actual_d_gt_theta"] == 3
    assert report["predicted_d_gt_theta"] == 2


def test_regression_loss_supports_bfloat16_logits_and_float32_labels() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.bfloat16))

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.value.expand(input_ids.shape[0], 1))

    model = Model()
    trainer = object.__new__(_Float32RegressionTrainer)
    loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.ones((2, 1), dtype=torch.long),
            "labels": torch.tensor([0.0, 2.0], dtype=torch.float32),
        },
    )

    assert loss.dtype == torch.float32
    loss.backward()
    assert model.value.grad is not None


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


class _TwoHeadModel(torch.nn.Module):
    def forward(self, input_ids, **_kwargs):
        gap = input_ids[:, 0].float()
        detector_logit = input_ids[:, 1].float()
        return SimpleNamespace(logits=torch.stack((gap, detector_logit), dim=1))


class _FakeTwoHeadGapFinder:
    id = "two-head-gap-finder"
    tokenizer = None

    def __init__(self) -> None:
        self.model = _TwoHeadModel()

    def predict(self, prompts, answers):
        return [-2.0, 2.0, 8.0], [0.9, 0.4, 0.9]


def test_two_head_correction_is_gated_non_positive_and_capped() -> None:
    correction = TwoHeadGapFinderCorrection(
        _FakeTwoHeadGapFinder(),
        reward_std=0.5,
        detector_threshold=0.7,
        max_gap=3.0,
    )
    assert correction(["a", "b", "c"], ["x", "y", "z"]) == [0.0, 0.0, -1.5]

    head = correction.ppo_head()
    values = head(torch.tensor([[-2, 3], [2, 0], [8, 3]]))
    assert values.tolist() == [0.0, 0.0, -1.5]


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


def test_evaluate_gap_finder_reports_scores_and_corrected_outcome(monkeypatch) -> None:
    class Policy:
        def __init__(self) -> None:
            self.offloaded = False

        def generate_batch(self, prompts):
            return [f"answer:{prompt}" for prompt in prompts]

        def offload(self):
            self.offloaded = True

    class GapFinder:
        def __init__(self) -> None:
            self.moved = False
            self.offloaded = False

        def move_to_current_device(self):
            self.moved = True

        def predict_gap(self, prompts, answers, **_kwargs):
            return [0.5] * len(prompts)

        def offload(self):
            self.offloaded = True

    def score_rows(*, spec, row_groups, **_kwargs):
        values = [5.0] if spec.mode_name == "proxy" else [2.0]
        return {"evaluation": np.asarray(values)}

    monkeypatch.setattr(functions, "_score_policy_rows", score_rows)
    monkeypatch.setattr(functions, "empty_cuda_cache", lambda: None)
    policy = Policy()
    gap_finder = GapFinder()
    calibration = functions.GapCalibration(
        proxy_mean=1.0,
        proxy_std=2.0,
        judge_mean=0.0,
        judge_std=1.0,
        theta=1.0,
    )

    results = functions.evaluate_gap_finder(
        ["prompt"],
        policy=policy,
        gap_finder=gap_finder,
        config=functions.ConfigTrainClassifier(),
        calibration=calibration,
    )

    numeric_result = {
        key: value
        for key, value in results[0].items()
        if key not in {"prompt", "answer"}
    }
    assert numeric_result == pytest.approx(
        {
            "proxy_score": 5.0,
            "judge_score": 2.0,
            "proxy_z": 2.0,
            "judge_z": 2.0,
            "actual_gap": 0.0,
            "predicted_gap": 0.5,
            "final_proxy_score": 4.0,
            "final_proxy_z": 1.5,
            "final_gap_vs_judge": -0.5,
        }
    )
    assert results[0]["prompt"] == "prompt"
    assert results[0]["answer"] == "answer:prompt"
    assert policy.offloaded
    assert gap_finder.moved and gap_finder.offloaded
