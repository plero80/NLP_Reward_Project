from collections.abc import Sequence
from collections import Counter
import math
from typing import Any

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class DatasetClassifier(Dataset):
    def __init__(self, theta: float = 2.0) -> None:
        self.theta = theta
        self.dataset: list[dict[str, Any]] = []

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def class_counts(self) -> dict[int, int]:
        return dict(Counter(int(row["labels"]) for row in self.dataset))

    def split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple["DatasetClassifier", "DatasetClassifier"]:
        """Create disjoint train/test datasets, stratifying when possible."""
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if len(self.dataset) < 2:
            raise ValueError("at least two examples are required for a split")

        labels = [int(row["labels"]) for row in self.dataset]
        counts = Counter(labels)
        test_count = math.ceil(len(labels) * test_size)
        train_count = len(labels) - test_count
        can_stratify = (
            len(counts) > 1
            and min(counts.values()) >= 2
            and test_count >= len(counts)
            and train_count >= len(counts)
        )
        stratify = labels if can_stratify else None
        train_rows, test_rows = train_test_split(
            self.dataset,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )

        train_dataset = DatasetClassifier(theta=self.theta)
        train_dataset.dataset = list(train_rows)
        test_dataset = DatasetClassifier(theta=self.theta)
        test_dataset.dataset = list(test_rows)
        return train_dataset, test_dataset

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
