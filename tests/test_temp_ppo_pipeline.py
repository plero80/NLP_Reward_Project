from pathlib import Path

import torch

import functions
from Datasets.dataset_request import RequestDataset


class _FakePolicy:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.generated: tuple[RequestDataset, int] | None = None
        self.dataset_save_path: Path | None = None
        self.offload_count = 0

    def generate_new_dataset(
        self,
        dataset: RequestDataset,
        batch_size: int,
    ) -> None:
        self.generated = (dataset, batch_size)

    def save_dataset(self, checkpoint_directory: str | Path) -> None:
        self.dataset_save_path = Path(checkpoint_directory)

    def offload(self) -> None:
        self.offload_count += 1
        self.model.to("cpu")


class _FakeTrainer:
    instances: list["_FakeTrainer"] = []

    def __init__(
        self,
        policy,
        reward,
        value,
        dataset,
        config,
        reference_policy=None,
    ) -> None:
        self.policy = policy
        self.reward = reward
        self.value = value
        self.dataset = dataset
        self.config = config
        self.reference_policy = reference_policy
        self.trained = False
        self.instances.append(self)

    def train(self) -> None:
        self.trained = True


def test_temp_ppo_pipeline_uses_direct_component_specs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = RequestDataset.__new__(RequestDataset)
    dataset.columns = {"prompts": ["prompt"]}
    policy = _FakePolicy()
    reward = type(
        "FakeReward",
        (),
        {"model": torch.nn.Linear(1, 1)},
    )()
    value = type(
        "FakeValue",
        (),
        {"model": torch.nn.Linear(1, 1)},
    )()
    calls: dict[str, tuple | str] = {}

    monkeypatch.setattr(
        functions,
        "load_dataset",
        lambda name: calls.setdefault("loaded_dataset", name),
    )

    def create_dataset(class_name, raw, tokenizer_name, start, end):
        calls["dataset"] = (class_name, raw, tokenizer_name, start, end)
        return dataset

    def create_policy(
        class_name,
        model_name,
        lora_config,
        **kwargs,
    ):
        calls["policy"] = (
            class_name,
            model_name,
            lora_config,
            kwargs,
        )
        return policy

    def create_reward(class_name, model_name, mode_name):
        calls["reward"] = (class_name, model_name, mode_name)
        return reward

    monkeypatch.setattr(
        functions.DatasetFactory,
        "create_from_raw",
        create_dataset,
    )
    monkeypatch.setattr(
        functions.PolicyModelFactory,
        "create_model",
        create_policy,
    )
    monkeypatch.setattr(
        functions.RewardModelFactory,
        "create_model",
        create_reward,
    )
    monkeypatch.setattr(functions, "ValueModel", lambda _name: value)
    monkeypatch.setattr(functions, "PolicyPPOTrainer", _FakeTrainer)
    monkeypatch.setattr(functions, "empty_cuda_cache", lambda: None)

    config = functions.TrainingPPOConfig(
        policy=functions.PolicySpec(
            class_name="PolicyModel",
            model_name="policy/model",
            lora_config=None,
        ),
        reward=functions.RewardSpec(
            class_name="DeterministicReward",
            model_name="policy/model",
            mode_name="proxy",
        ),
        dataset=functions.DatasetSpec(
            class_name="RequestDataset",
            dataset_name="dataset/name",
            start=2,
            end=7,
        ),
        output_dir=tmp_path / "ppo",
        generation_batch_size=12,
    )

    result = functions.temp_ppo_train_policy(config)

    assert result is policy
    assert calls["loaded_dataset"] == "dataset/name"
    assert calls["dataset"] == (
        "RequestDataset",
        "dataset/name",
        "policy/model",
        2,
        7,
    )
    assert calls["policy"] == (
        "PolicyModel",
        "policy/model",
        None,
        {"checkpoint": None, "is_trainable": True},
    )
    assert calls["reward"] == (
        "DeterministicReward",
        "policy/model",
        "proxy",
    )
    assert policy.generated == (dataset, 12)
    assert policy.dataset_save_path == tmp_path / "ppo" / "final"
    assert policy.offload_count == 1
    trainer = _FakeTrainer.instances[-1]
    assert trainer.trained is True
    assert trainer.reference_policy is None
    assert trainer.config.output_dir == str(tmp_path / "ppo")
