from types import SimpleNamespace

import pytest
import torch

from Trainers.trainer_two_head_gap_finder import (
    _TwoHeadTrainer,
    compute_two_head_report,
)


def test_two_head_report_uses_dedicated_detector_probabilities() -> None:
    report = compute_two_head_report(
        actual_gaps=[0.0, 2.0, 3.0],
        predicted_gaps=[0.0, 1.5, 2.5],
        detection_probabilities=[0.1, 0.9, 0.4],
        theta=1.0,
    )

    assert report["mae"] == pytest.approx(1 / 3)
    assert report["mae_d_gt_theta"] == pytest.approx(0.5)
    assert report["detector_precision"] == pytest.approx(1.0)
    assert report["detector_recall"] == pytest.approx(0.5)
    assert report["detector_f1"] == pytest.approx(2 / 3)
    assert report["detector_pr_auc"] == pytest.approx(1.0)


def test_two_head_report_accepts_tuned_detector_threshold() -> None:
    report = compute_two_head_report(
        actual_gaps=[0.0, 2.0, 3.0],
        predicted_gaps=[0.0, 1.5, 2.5],
        detection_probabilities=[0.1, 0.9, 0.4],
        theta=1.0,
        detector_threshold=0.3,
    )

    assert report["detector_threshold"] == pytest.approx(0.3)
    assert report["detector_precision"] == pytest.approx(1.0)
    assert report["detector_recall"] == pytest.approx(1.0)
    assert report["detector_f1"] == pytest.approx(1.0)


def test_two_head_loss_weights_tail_regression_and_positive_detection() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(
                torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.bfloat16)
            )

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.logits[: input_ids.shape[0]])

    trainer = object.__new__(_TwoHeadTrainer)
    trainer.theta = 1.0
    trainer.high_gap_weight = 5.0
    trainer.detector_loss_weight = 0.5
    trainer.detector_positive_weight = 1.0
    model = Model()
    loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.ones((2, 1), dtype=torch.long),
            "labels": torch.tensor([0.0, 2.0]),
        },
    )

    expected = 2.5 + 0.5 * torch.log(torch.tensor(2.0)).item()
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert model.logits.grad is not None
