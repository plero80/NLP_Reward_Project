import math

import pytest

from Models.model_reward import RewardModel


class _FakeClassifier:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def predict_proba(
        self,
        prompts: list[str],
        answers: list[str],
    ) -> list[float]:
        self.calls.append((list(prompts), list(answers)))
        return [0.25] * len(prompts)


def _reward_without_loading_model() -> RewardModel:
    reward = object.__new__(RewardModel)
    reward.model_mode = "judge"
    reward.mean = None
    reward.std = None
    return reward


def test_score_uses_ordered_micro_batches() -> None:
    reward = _reward_without_loading_model()
    classifier = _FakeClassifier()
    reward.classifiers = [
        {"name": "undesirable", "classifier": classifier}
    ]
    reward_calls: list[tuple[list[str], list[str], int]] = []

    def fake_reward_score(
        prompts: list[str],
        answers: list[str],
        max_length: int,
    ) -> list[float]:
        reward_calls.append((list(prompts), list(answers), max_length))
        return [float(prompt) for prompt in prompts]

    reward._score_reward_model = fake_reward_score
    scores = reward.score(
        ["1", "2", "3", "4", "5"],
        ["a", "b", "c", "d", "e"],
        batch_size=2,
        max_length=123,
    )

    penalty = math.log1p(-0.25)
    assert scores == pytest.approx(
        [value + penalty for value in (1, 2, 3, 4, 5)]
    )
    assert [len(prompts) for prompts, _, _ in reward_calls] == [2, 2, 1]
    assert [max_length for _, _, max_length in reward_calls] == [123] * 3
    assert [len(prompts) for prompts, _ in classifier.calls] == [2, 2, 1]


def test_score_applies_frozen_float_normalization() -> None:
    reward = _reward_without_loading_model()
    reward.classifiers = []
    reward.mean = 1.0
    reward.std = 2.0
    reward._score_reward_model = (
        lambda prompts, answers, max_length: [1.0, 3.0]
    )

    scores = reward.score(
        ["p1", "p2"],
        ["a1", "a2"],
        batch_size=2,
        normalize_score=True,
    )

    assert scores == pytest.approx([0.0, 1.0])


@pytest.mark.parametrize(
    ("batch_size", "max_length", "message"),
    [
        (0, 2_048, "batch_size"),
        (1, 0, "max_length"),
    ],
)
def test_score_rejects_invalid_memory_limits(
    batch_size: int,
    max_length: int,
    message: str,
) -> None:
    reward = _reward_without_loading_model()
    reward.classifiers = []

    with pytest.raises(ValueError, match=message):
        reward.score(
            [],
            [],
            batch_size=batch_size,
            max_length=max_length,
        )
