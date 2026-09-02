from collections.abc import Sequence
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class DatasetClassifier(Dataset):
    def __init__(
        self,
        theta: float = 2.0,
        id: str | None = None,
    ) -> None:
        theta = float(theta)
        if not math.isfinite(theta):
            raise ValueError("theta must be finite")
        if id is not None and (not isinstance(id, str) or not id.strip()):
            raise ValueError("id must be a non-empty string")

        self.id = id or str(uuid4())
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

        train_dataset = DatasetClassifier(theta=self.theta, id=self.id)
        train_dataset.dataset = list(train_rows)
        test_dataset = DatasetClassifier(theta=self.theta, id=self.id)
        test_dataset.dataset = list(test_rows)
        return train_dataset, test_dataset

    @staticmethod
    def _normalize_row(row: object, theta: float) -> dict[str, Any]:
        if not isinstance(row, dict):
            raise ValueError("Every classifier dataset row must be an object")

        required_fields = {
            "prompt",
            "answer",
            "reward_score",
            "judge_score",
            "score_difference",
            "labels",
        }
        if set(row) != required_fields:
            raise ValueError(
                "Classifier dataset row fields must be exactly: "
                f"{sorted(required_fields)}"
            )
        prompt = row["prompt"]
        answer = row["answer"]
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ValueError("prompt and answer must be strings")

        try:
            reward_score = float(row["reward_score"])
            judge_score = float(row["judge_score"])
            score_difference = float(row["score_difference"])
        except (TypeError, ValueError) as error:
            raise ValueError("Classifier scores must be numeric") from error
        if not all(
            math.isfinite(value)
            for value in (reward_score, judge_score, score_difference)
        ):
            raise ValueError("Classifier scores must be finite")

        expected_difference = reward_score - judge_score
        if not math.isclose(
            score_difference,
            expected_difference,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "score_difference does not match reward_score - judge_score"
            )

        label = row["labels"]
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError("labels must be the integer 0 or 1")
        expected_label = int(expected_difference > theta)
        if label != expected_label:
            raise ValueError("labels does not match score_difference > theta")

        return {
            "prompt": prompt,
            "answer": answer,
            "reward_score": reward_score,
            "judge_score": judge_score,
            "score_difference": expected_difference,
            "labels": label,
        }

    def save(self, path: str | Path) -> Path:
        """Save the dataset rows and metadata to a versioned JSON file."""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            self._normalize_row(row, self.theta) for row in self.dataset
        ]
        payload = {
            "format_version": 1,
            "id": self.id,
            "theta": self.theta,
            "rows": rows,
        }
        destination.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "DatasetClassifier":
        """Load a dataset created by :meth:`save`."""
        source = Path(path).expanduser()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Classifier dataset is not valid JSON: {source}"
            ) from error

        if not isinstance(payload, dict):
            raise ValueError("Classifier dataset JSON must contain an object")
        required_fields = {"format_version", "id", "theta", "rows"}
        if set(payload) != required_fields:
            raise ValueError(
                "Classifier dataset fields must be exactly: "
                f"{sorted(required_fields)}"
            )
        if payload["format_version"] != 1:
            raise ValueError(
                "Unsupported classifier dataset format_version: "
                f"{payload['format_version']!r}"
            )
        rows = payload["rows"]
        if not isinstance(rows, list):
            raise ValueError("Classifier dataset rows must be a list")

        dataset = cls(theta=payload["theta"], id=payload["id"])
        dataset.dataset = [
            cls._normalize_row(row, dataset.theta) for row in rows
        ]
        return dataset

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
            if not isinstance(prompt, str) or not isinstance(answer, str):
                raise ValueError("prompts and answers must contain strings")
            reward_score = float(reward_score)
            judge_score = float(judge_score)
            if not math.isfinite(reward_score) or not math.isfinite(
                judge_score
            ):
                raise ValueError("reward_scores and judge_scores must be finite")
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
