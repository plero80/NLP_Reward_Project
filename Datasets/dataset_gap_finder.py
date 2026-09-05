from __future__ import annotations

from collections.abc import Sequence
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class DatasetGapFinder(Dataset):
    """Prompt/answer pairs labelled with a continuous normalized reward gap.

    ``labels`` is ``proxy_z - judge_z``.  A positive value therefore means
    that the proxy overestimates the answer relative to the judge.
    """

    def __init__(self, id: str | None = None) -> None:
        if id is not None and (not isinstance(id, str) or not id.strip()):
            raise ValueError("id must be a non-empty string")
        self.id = id or str(uuid4())
        self.dataset: list[dict[str, Any]] = []

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def add(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
        proxy_scores: Sequence[float],
        judge_scores: Sequence[float],
    ) -> None:
        if len({len(prompts), len(answers), len(proxy_scores), len(judge_scores)}) != 1:
            raise ValueError(
                "prompts, answers, proxy_scores, and judge_scores must have "
                "the same length"
            )
        for prompt, answer, proxy_score, judge_score in zip(
            prompts, answers, proxy_scores, judge_scores
        ):
            if not isinstance(prompt, str) or not isinstance(answer, str):
                raise ValueError("prompts and answers must contain strings")
            proxy_score = float(proxy_score)
            judge_score = float(judge_score)
            if not math.isfinite(proxy_score) or not math.isfinite(judge_score):
                raise ValueError("proxy_scores and judge_scores must be finite")
            self.dataset.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "proxy_score": proxy_score,
                    "judge_score": judge_score,
                    "labels": proxy_score - judge_score,
                }
            )

    def split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[DatasetGapFinder, DatasetGapFinder]:
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if len(self.dataset) < 2:
            raise ValueError("at least two examples are required for a split")
        train_rows, test_rows = train_test_split(
            self.dataset,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        train = DatasetGapFinder(id=self.id)
        test = DatasetGapFinder(id=self.id)
        train.dataset = list(train_rows)
        test.dataset = list(test_rows)
        return train, test

    def split_three_way(
        self,
        train_size: float = 0.70,
        validation_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> tuple[DatasetGapFinder, DatasetGapFinder, DatasetGapFinder]:
        """Create disjoint randomized train/validation/test datasets."""
        sizes = (float(train_size), float(validation_size), float(test_size))
        if not all(math.isfinite(size) and size > 0.0 for size in sizes):
            raise ValueError("all split sizes must be finite and positive")
        if not math.isclose(sum(sizes), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("train, validation, and test sizes must sum to 1")
        if len(self.dataset) < 3:
            raise ValueError("at least three examples are required for a three-way split")

        train_rows, remaining_rows = train_test_split(
            self.dataset,
            train_size=train_size,
            random_state=random_state,
            shuffle=True,
        )
        relative_test_size = test_size / (validation_size + test_size)
        validation_rows, test_rows = train_test_split(
            remaining_rows,
            test_size=relative_test_size,
            random_state=random_state,
            shuffle=True,
        )
        train = DatasetGapFinder(id=self.id)
        validation = DatasetGapFinder(id=self.id)
        test = DatasetGapFinder(id=self.id)
        train.dataset = list(train_rows)
        validation.dataset = list(validation_rows)
        test.dataset = list(test_rows)
        return train, validation, test

    @staticmethod
    def _normalize_row(row: object) -> dict[str, Any]:
        if not isinstance(row, dict):
            raise ValueError("Every gap-finder dataset row must be an object")
        required = {"prompt", "answer", "proxy_score", "judge_score", "labels"}
        if set(row) != required:
            raise ValueError(
                f"Gap-finder dataset row fields must be exactly: {sorted(required)}"
            )
        prompt, answer = row["prompt"], row["answer"]
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ValueError("prompt and answer must be strings")
        try:
            proxy_score = float(row["proxy_score"])
            judge_score = float(row["judge_score"])
            label = float(row["labels"])
        except (TypeError, ValueError) as error:
            raise ValueError("Gap-finder scores must be numeric") from error
        if not all(math.isfinite(v) for v in (proxy_score, judge_score, label)):
            raise ValueError("Gap-finder scores must be finite")
        expected = proxy_score - judge_score
        if not math.isclose(label, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("labels does not match proxy_score - judge_score")
        return {
            "prompt": prompt,
            "answer": answer,
            "proxy_score": proxy_score,
            "judge_score": judge_score,
            "labels": expected,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "id": self.id,
            "rows": [self._normalize_row(row) for row in self.dataset],
        }
        destination.write_text(
            json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> DatasetGapFinder:
        source = Path(path).expanduser()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Gap-finder dataset is not valid JSON: {source}") from error
        if not isinstance(payload, dict) or set(payload) != {"format_version", "id", "rows"}:
            raise ValueError("Invalid gap-finder dataset document")
        if payload["format_version"] != 1 or not isinstance(payload["rows"], list):
            raise ValueError("Unsupported gap-finder dataset format")
        dataset = cls(id=payload["id"])
        dataset.dataset = [cls._normalize_row(row) for row in payload["rows"]]
        return dataset
