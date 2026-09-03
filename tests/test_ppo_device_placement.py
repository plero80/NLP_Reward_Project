from types import SimpleNamespace

import torch

import Trainers.trainer_ppo as trainer_ppo
from Trainers.trainer_ppo import PolicyPPOTrainer


class _RecordingModel:
    def __init__(self) -> None:
        self.devices: list[torch.device] = []

    def to(self, device: torch.device) -> None:
        self.devices.append(device)


def test_ppo_restores_all_models_after_cpu_offload(monkeypatch) -> None:
    expected_device = torch.device("cuda:2")
    policy_model = _RecordingModel()
    reference_model = _RecordingModel()
    reward_model = _RecordingModel()
    value_model = _RecordingModel()
    classifier_model = _RecordingModel()

    trainer = PolicyPPOTrainer.__new__(PolicyPPOTrainer)
    trainer.policy = SimpleNamespace(model=policy_model)
    trainer.reference_policy = SimpleNamespace(model=reference_model)
    trainer.reward = SimpleNamespace(
        model=reward_model,
        classifiers=[
            {"classifier": SimpleNamespace(model=classifier_model)},
        ],
    )
    trainer.value = SimpleNamespace(model=value_model)
    monkeypatch.setattr(trainer_ppo, "current_device", lambda: expected_device)

    trainer._place_models_for_training()

    for model in (
        policy_model,
        reference_model,
        reward_model,
        value_model,
        classifier_model,
    ):
        assert model.devices == [expected_device]
