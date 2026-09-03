from pathlib import Path

import torch

import Models.model_classifier as model_classifier
from Models.model_classifier import Classifier


class _FakeTokenizer:
    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


def test_classifier_save_and_load_preserves_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "classifier" / "final"
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    classifier = Classifier.__new__(Classifier)
    classifier.model_name = "base/classifier"
    classifier.tokenizer = tokenizer
    classifier.model = model
    classifier.id = "stable-classifier-id"

    saved_path = classifier.save(checkpoint)

    monkeypatch.setattr(
        model_classifier.AutoTokenizer,
        "from_pretrained",
        lambda _source: tokenizer,
    )
    monkeypatch.setattr(
        model_classifier.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda _source, **_kwargs: model,
    )
    loaded = Classifier.load(checkpoint, device="cpu")

    assert saved_path == checkpoint
    assert loaded.id == "stable-classifier-id"
    assert loaded.model_name == "base/classifier"
    assert loaded.model is model
    assert loaded.tokenizer is tokenizer
    assert loaded.model.training is False
