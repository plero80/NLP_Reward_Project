from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import Models.model_policy as model_policy
from Models.model_policy import PolicyModel


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = None
        self.pad_token = None
        self.eos_token = "<eos>"
        self.padding_side = "right"


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def test_load_full_policy_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    tokenizer_sources: list[str] = []
    model_sources: list[str] = []

    def load_tokenizer(source: str) -> _FakeTokenizer:
        tokenizer_sources.append(source)
        return tokenizer

    def load_model(source: str, **_kwargs) -> _FakeModel:
        model_sources.append(source)
        return model

    monkeypatch.setattr(
        model_policy.AutoTokenizer,
        "from_pretrained",
        load_tokenizer,
    )
    monkeypatch.setattr(
        model_policy.AutoModelForCausalLM,
        "from_pretrained",
        load_model,
    )

    loaded = PolicyModel.load(checkpoint, device="cpu")

    assert loaded.model is model
    assert loaded.model_name == str(checkpoint)
    assert loaded.dataset is None
    assert not model.training
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.padding_side == "left"
    assert tokenizer_sources == [str(checkpoint)]
    assert model_sources == [str(checkpoint)]


def test_load_trainable_peft_policy_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    tokenizer = _FakeTokenizer()
    base_model = _FakeModel()
    adapter_model = _FakeModel()
    peft_calls: list[tuple[torch.nn.Module, str, bool]] = []

    monkeypatch.setattr(
        model_policy.PeftConfig,
        "from_pretrained",
        lambda _source: SimpleNamespace(
            base_model_name_or_path="base/policy"
        ),
    )
    monkeypatch.setattr(
        model_policy.AutoTokenizer,
        "from_pretrained",
        lambda source: tokenizer
        if source == "base/policy"
        else pytest.fail(f"Unexpected tokenizer source: {source}"),
    )
    monkeypatch.setattr(
        model_policy.AutoModelForCausalLM,
        "from_pretrained",
        lambda source, **_kwargs: base_model
        if source == "base/policy"
        else pytest.fail(f"Unexpected model source: {source}"),
    )

    def load_adapter(
        model: torch.nn.Module,
        source: str,
        is_trainable: bool,
    ) -> _FakeModel:
        peft_calls.append((model, source, is_trainable))
        return adapter_model

    monkeypatch.setattr(
        model_policy.PeftModel,
        "from_pretrained",
        staticmethod(load_adapter),
    )

    loaded = PolicyModel.load(
        checkpoint,
        is_trainable=True,
        device="cpu",
    )

    assert loaded.model is adapter_model
    assert loaded.model_name == "base/policy"
    assert adapter_model.training
    assert peft_calls == [(base_model, str(checkpoint), True)]


def test_load_policy_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Policy checkpoint not found"):
        PolicyModel.load(tmp_path / "missing")


def test_move_to_current_device_restores_offloaded_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.devices: list[torch.device] = []

        def to(self, device: torch.device) -> None:
            self.devices.append(device)

    expected_device = torch.device("cuda:3")
    policy = PolicyModel.__new__(PolicyModel)
    policy.model = RecordingModel()
    monkeypatch.setattr(model_policy, "current_device", lambda: expected_device)

    selected_device = policy.move_to_current_device()

    assert selected_device == expected_device
    assert policy.model.devices == [expected_device]
