from types import SimpleNamespace

import pytest

import Datasets.dataset_request as dataset_request
from Datasets.dataset_classifier import DatasetClassifier
from Datasets.dataset_request import RequestDataset
from Factory.factory import DatasetFactory
from Factory.factory import RewardModelFactory


def test_dataset_factory_creates_request_dataset(monkeypatch) -> None:
    tokenizer = SimpleNamespace()
    monkeypatch.setattr(
        dataset_request.AutoTokenizer,
        "from_pretrained",
        lambda model_name: tokenizer,
    )

    dataset = DatasetFactory.create_dataset(
        "RequestDataset",
        requests=["First prompt", "Second prompt"],
        tokenizer_name="test-tokenizer",
    )

    assert isinstance(dataset, RequestDataset)
    assert dataset.tokenizer is tokenizer
    assert dataset["prompts"] == ["First prompt", "Second prompt"]


def test_dataset_factory_creates_classifier_dataset() -> None:
    dataset = DatasetFactory.create_dataset(
        "DatasetClassifier",
        theta=1.5,
        id="classifier-data",
    )

    assert isinstance(dataset, DatasetClassifier)
    assert dataset.theta == 1.5
    assert dataset.id == "classifier-data"


def test_dataset_factory_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="Unknown dataset class 'Missing'"):
        DatasetFactory.create_dataset("Missing")


def test_dataset_factory_explains_invalid_arguments() -> None:
    with pytest.raises(TypeError, match="Invalid arguments.*RequestDataset"):
        DatasetFactory.create_dataset(
            "RequestDataset",
            requests=["prompt"],
        )


def test_request_dataset_from_raw_uses_optional_range(monkeypatch) -> None:
    monkeypatch.setattr(
        dataset_request.AutoTokenizer,
        "from_pretrained",
        lambda _model_name: SimpleNamespace(),
    )
    raw_dataset = {
        "train": {
            "chosen": [
                "Human: First? Assistant: A",
                "Human: Second? Assistant: B",
                "Human: Third? Assistant: C",
            ],
        },
    }

    dataset = RequestDataset.from_raw(
        raw_dataset,
        "test-tokenizer",
        start=1,
        end=3,
    )

    assert dataset["prompts"] == ["Human: Second?", "Human: Third?"]


def test_reward_factory_attaches_loaded_classifiers(monkeypatch) -> None:
    attached: list[tuple] = []
    reward = SimpleNamespace(
        add_classifier=lambda *args: attached.append(args),
    )
    builders = dict(RewardModelFactory._builders)
    builders["RewardModel"] = lambda _model, _mode: reward
    monkeypatch.setattr(RewardModelFactory, "_builders", builders)
    classifier = SimpleNamespace(
        id="classifier-id",
        tokenizer="tokenizer",
    )

    created = RewardModelFactory.create_model(
        "RewardModel",
        "reward/model",
        "proxy",
        [classifier],
    )

    assert created is reward
    assert attached == [("classifier-id", classifier, "tokenizer")]
