from dataclasses import replace
import inspect

import pytest
import torch

import functions


def test_eval_policy_with_reward_accepts_only_config() -> None:
    parameters = inspect.signature(
        functions.eval_policy_with_reward
    ).parameters
    assert tuple(parameters) == ("config",)

    invalid_config = replace(
        functions.ConfigEval(),
        start=5,
        end=5,
    )
    with pytest.raises(ValueError, match="end must be greater"):
        functions.eval_policy_with_reward(invalid_config)


def test_classifier_config_rejects_an_empty_range() -> None:
    config = replace(
        functions.ConfigTrainClassifier(),
        start_dataset=10,
        end_dataset=10,
    )

    with pytest.raises(ValueError, match="end_dataset"):
        functions.create_classifier(config)


def test_warm_start_cannot_overwrite_its_source_directory(tmp_path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-10"
    checkpoint.mkdir(parents=True)

    with pytest.raises(ValueError, match="new output directory"):
        functions._ppo_output_directory(
            replace(
                functions.ConfigEval(),
                start=0,
                end=10,
                policy_checkpoint=checkpoint,
                output_dir=checkpoint.parent,
            )
        )


def test_create_classifier_runs_sequential_scorers_and_lora(
    monkeypatch,
) -> None:
    prompts = ["p0", "p1", "p2", "p3"]
    score_calls: list[tuple[str, int, int]] = []
    trainer_configs = []

    class FakeRequestDataset:
        def __init__(self) -> None:
            self.prompts = list(prompts)

        @classmethod
        def from_raw(cls, raw_dataset, tokenizer_name):
            assert raw_dataset == "raw"
            assert tokenizer_name == "policy"
            return cls()

        def truncate(self, start: int, end: int) -> None:
            self.prompts = self.prompts[start:end]

        def __len__(self) -> int:
            return len(self.prompts)

    class FakePolicy:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.model = torch.nn.Linear(1, 1)
            self.rows: dict[str, list] = {}

        def generate_new_dataset(
            self,
            dataset: FakeRequestDataset,
            batch_size: int,
        ) -> None:
            assert batch_size == 2
            self.rows["prompts"] = list(dataset.prompts)
            self.rows["answers"] = [f"answer-{index}" for index in range(4)]

        def get_dataset_col(self, name: str) -> list:
            return self.rows[name]

        def offload(self) -> None:
            self.model.to("cpu")

    class FakeRewardModel:
        def __init__(self, model_name: str, mode_name: str) -> None:
            self.model = torch.nn.Linear(1, 1)
            self.mode_name = mode_name
            self.std = None

        def init_normalization(
            self,
            policy: FakePolicy,
            batch_size: int,
        ) -> None:
            self.std = torch.tensor(0.5)

        def score_policy(
            self,
            policy: FakePolicy,
            batch_size: int,
            max_length: int,
            normalize_score: bool,
        ) -> None:
            assert normalize_score is True
            score_calls.append((self.mode_name, batch_size, max_length))
            policy.rows[self.mode_name] = (
                [1.0, -1.0, 1.0, -1.0]
                if self.mode_name == "proxy"
                else [0.0, 0.0, 0.0, 0.0]
            )

    class FakeClassifier:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.id = "test-id"
            self.model = torch.nn.Linear(1, 2)

    class FakeClassifierTrainer:
        def __init__(self, classifier, config) -> None:
            self.classifier = classifier
            trainer_configs.append(config)

        def train(self, train_dataset):
            assert len(train_dataset) == 3
            return object()

        def evaluate(self, trainer, test_dataset):
            assert len(test_dataset) == 1
            return {"test_f1": 1.0}

    monkeypatch.setattr(functions, "load_dataset", lambda name: "raw")
    monkeypatch.setattr(functions, "RequestDataset", FakeRequestDataset)
    monkeypatch.setattr(functions, "PolicyModel", FakePolicy)
    monkeypatch.setattr(functions, "RewardModel", FakeRewardModel)
    monkeypatch.setattr(functions, "Classifier", FakeClassifier)
    monkeypatch.setattr(functions, "ClassifierTrainer", FakeClassifierTrainer)
    monkeypatch.setattr(functions, "empty_cuda_cache", lambda: None)

    config = replace(
        functions.ConfigTrainClassifier(),
        policy_name="policy",
        start_dataset=0,
        end_dataset=4,
        generation_batch_size=2,
        reward_batch_size=2,
        judge_batch_size=1,
        score_max_length=128,
        classifier_theta=0.0,
        classifier_test_size=0.25,
    )

    classifier = functions.create_classifier(config)

    assert classifier.id == "test-id"
    assert score_calls == [("proxy", 2, 128), ("judge", 1, 128)]
    assert trainer_configs[0].lora_settings == config.lora_settings
    assert trainer_configs[0].output_dir.endswith("id=test-id")
    assert classifier.model.training is False
