from collections.abc import Sequence
from typing import Any

from torch.utils.data import Dataset


class DatasetClassifier(Dataset):
    def __init__(self, theta: float = 2.0) -> None:
        self.theta = theta
        self.dataset: list[dict[str, Any]] = []

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def add(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
        reward_scores: Sequence[float],
        judge_scores: Sequence[float],
    ) -> None:
        lengths = {
            len(prompts),
            len(answers),
            len(reward_scores),
            len(judge_scores),
        }
        if len(lengths) != 1:
            raise ValueError(
                "prompts, answers, reward_scores, and judge_scores "
                "must have the same length"
            )

        for prompt, answer, reward_score, judge_score in zip(
            prompts,
            answers,
            reward_scores,
            judge_scores,
        ):
            reward_score = float(reward_score)
            judge_score = float(judge_score)
            score_difference = reward_score - judge_score

            self.dataset.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "reward_score": reward_score,
                    "judge_score": judge_score,
                    "score_difference": score_difference,
                    "labels": int(score_difference > self.theta),
                }
            )
