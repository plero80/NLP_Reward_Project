from dataclasses import replace
import inspect
import logging

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
        start_dataset=5,
        end_dataset=5,
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


def test_gap_calibration_json_round_trip(tmp_path) -> None:
    calibration = functions.GapCalibration(
        proxy_mean=1.25,
        proxy_std=2.5,
        judge_mean=-0.75,
        judge_std=0.5,
        theta=1.7,
    )
    path = tmp_path / "calibration" / "reward_gap.json"

    saved_path = calibration.save(path)

    assert saved_path == path
    assert functions.GapCalibration.load(path) == calibration


def test_gap_calibration_load_rejects_invalid_std(tmp_path) -> None:
    path = tmp_path / "invalid-calibration.json"
    path.write_text(
        '{"format_version": 1, "proxy_mean": 0, "proxy_std": 0, '
        '"judge_mean": 0, "judge_std": 1, "theta": 1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="proxy_std"):
        functions.GapCalibration.load(path)


def test_warm_start_cannot_overwrite_its_source_directory(tmp_path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-10"
    checkpoint.mkdir(parents=True)

    with pytest.raises(ValueError, match="new output directory"):
        functions._ppo_output_directory(
            replace(
                functions.ConfigEval(),
                start_dataset=0,
                end_dataset=10,
                policy_checkpoint=checkpoint,
                output_dir=checkpoint.parent,
            )
        )


def test_create_classifier_runs_sequential_scorers_and_lora(
    caplog,
    capsys,
    monkeypatch,
    tmp_path,
) -> None:
    caplog.set_level(logging.INFO, logger="functions")
    prompts = ["p0", "p1", "p2", "p3"]
    score_calls: list[tuple[str, str, int, int]] = []
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
        instance_count = 0

        def __init__(self, model_name: str) -> None:
            FakePolicy.instance_count += 1
            self.model_name = model_name
            self.model = torch.nn.Linear(1, 1)
            self.rows: dict[str, list] = {}
            self.offload_count = 0
            self.role = (
                "reference"
                if FakePolicy.instance_count == 1
                else "target"
            )

        def generate_new_dataset(
            self,
            dataset: FakeRequestDataset,
            batch_size: int,
        ) -> None:
            assert batch_size == 2
            self.rows["prompts"] = list(dataset.prompts)
            self.rows["answers"] = [
                f"{self.role}-answer-{index}" for index in range(4)
            ]

        def get_dataset_col(self, name: str) -> list:
            return self.rows[name]

        def offload(self) -> None:
            self.offload_count += 1
            self.model.to("cpu")

    class FakeRewardModel:
        def __init__(self, model_name: str, mode_name: str) -> None:
            self.model = torch.nn.Linear(1, 1)
            self.mode_name = mode_name

        def score(
            self,
            batch_prompts: list[str],
            batch_answers: list[str],
            batch_size: int,
            max_length: int,
        ) -> list[float]:
            role = batch_answers[0].split("-", maxsplit=1)[0]
            score_calls.append(
                (self.mode_name, role, batch_size, max_length)
            )
            if role == "reference":
                return (
                    [-1.0, 1.0, -1.0, 1.0]
                    if self.mode_name == "proxy"
                    else [-1.0, 1.0, 1.0, -1.0]
                )
            return (
                [3.0, 0.0, 3.0, 0.0]
                if self.mode_name == "proxy"
                else [0.0, 0.0, 0.0, 0.0]
            )

    class FakeClassifier:
        def __init__(
            self,
            model_name: str,
            classifier_id: str | None = None,
        ) -> None:
            self.model_name = model_name
            self.id = classifier_id
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
        classifier_test_size=0.25,
        classifier_output_root=tmp_path,
    )

    classifier, calibration = functions.create_classifier(config)

    assert classifier.id is not None
    saved_dataset = functions.DatasetClassifier.load(
        tmp_path / f"id={classifier.id}" / "dataset.json"
    )
    assert saved_dataset.id == classifier.id
    assert saved_dataset.theta == calibration.theta
    assert len(saved_dataset) == 4
    assert (
        "Reward-hack labels: 2 of 4 prompts (50.00%) are hacks"
        in caplog.text
    )
    assert "Classifier label split:" in caplog.text
    notebook_output = capsys.readouterr().out
    assert (
        "Reward-hack labels: 2 of 4 prompts (50.00%) are hacks"
        in notebook_output
    )
    assert "Classifier label split: train=" in notebook_output
    assert score_calls == [
        ("proxy", "reference", 2, 128),
        ("proxy", "target", 2, 128),
        ("judge", "reference", 1, 128),
        ("judge", "target", 1, 128),
    ]
    assert calibration.proxy_mean == pytest.approx(0.0)
    assert calibration.proxy_std == pytest.approx(1.0)
    assert calibration.judge_mean == pytest.approx(0.0)
    assert calibration.judge_std == pytest.approx(1.0)
    assert calibration.theta == pytest.approx(1.7)
    assert trainer_configs[0].lora_settings == config.lora_settings
    assert trainer_configs[0].output_dir.endswith(f"id={classifier.id}")
    assert classifier.model.training is False

    score_calls.clear()
    _, reused_calibration = functions.create_classifier(
        config,
        calibration=calibration,
    )
    assert reused_calibration is calibration
    assert score_calls == [
        ("proxy", "target", 2, 128),
        ("judge", "target", 1, 128),
    ]

    supplied_policy = FakePolicy("trained-policy")
    score_calls.clear()
    _, supplied_calibration = functions.create_classifier(
        config,
        policy=supplied_policy,
        calibration=calibration,
    )
    assert supplied_calibration is calibration
    assert supplied_policy.offload_count == 1
    assert score_calls == [
        ("proxy", "target", 2, 128),
        ("judge", "target", 1, 128),
    ]

    with pytest.raises(ValueError, match="either policy"):
        functions.create_classifier(
            replace(config, policy_load_path="checkpoint"),
            policy=supplied_policy,
            calibration=calibration,
        )
