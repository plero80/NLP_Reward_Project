from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, overload

from torch.utils.data import Dataset

from Datasets.dataset_classifier import DatasetClassifier
from Datasets.dataset_request import RequestDataset
from Models.lora import LoRASettings
from Models.model_evaluator import EvaluatorOpenAIModel, PrometheusEvaluator
from Models.model_policy import PolicyModel
from Models.model_reward import DeterministicReward, RewardModel
from Models.models import PPORewardModelProtocol


class RewardModelFactory:
    _builders: dict[str, Callable[[str, str], PPORewardModelProtocol]] = {
        "RewardModel": lambda model_name, mode_name: RewardModel(
            model_name,
            mode_name,
        ),
        "DeterministicReward": lambda model_name, _mode_name: (
            DeterministicReward(model_name)
        ),
    }

    @classmethod
    def create_model(
        cls,
        class_name: str,
        model_name: str,
        mode_name: str,
    ) -> PPORewardModelProtocol:
        try:
            builder = cls._builders[class_name]
        except KeyError:
            available = ", ".join(cls._builders)
            raise ValueError(
                f"Unknown reward model class {class_name!r}. "
                f"Available classes: {available}"
            ) from None

        return builder(model_name, mode_name)


class PolicyModelFactory:
    _builders: dict[
        str,
        Callable[[str, LoRASettings | None], PolicyModel],
    ] = {
        "PolicyModel": lambda model_name, lora_config: PolicyModel(
            model_name,
            lora_config,
        ),
    }

    @classmethod
    def create_model(
        cls,
        class_name: str,
        model_name: str,
        lora_config: LoRASettings | None = None,
        checkpoint: str | Path | None = None,
        is_trainable: bool = True,
    ) -> PolicyModel:
        try:
            builder = cls._builders[class_name]
        except KeyError:
            available = ", ".join(cls._builders)
            raise ValueError(
                f"Unknown policy model class {class_name!r}. "
                f"Available classes: {available}"
            ) from None

        if checkpoint is not None:
            if class_name != "PolicyModel":
                raise ValueError(
                    f"Policy class {class_name!r} does not support checkpoints"
                )
            return PolicyModel.load(
                checkpoint,
                is_trainable=is_trainable,
            )

        return builder(model_name, lora_config)


class EvaluatorModelFactory:
    _builders: dict[
        str,
        Callable[[str], EvaluatorOpenAIModel | PrometheusEvaluator],
    ] = {
        "EvaluatorOpenAIModel": EvaluatorOpenAIModel,
        "PrometheusEvaluator": PrometheusEvaluator,
    }

    @overload
    @classmethod
    def create_model(
        cls,
        class_name: Literal["EvaluatorOpenAIModel"],
        model_name: str,
    ) -> EvaluatorOpenAIModel: ...

    @overload
    @classmethod
    def create_model(
        cls,
        class_name: Literal["PrometheusEvaluator"],
        model_name: str,
    ) -> PrometheusEvaluator: ...

    @overload
    @classmethod
    def create_model(
        cls,
        class_name: str,
        model_name: str,
    ) -> EvaluatorOpenAIModel | PrometheusEvaluator: ...

    @classmethod
    def create_model(
        cls,
        class_name: str,
        model_name: str,
    ) -> EvaluatorOpenAIModel | PrometheusEvaluator:
        try:
            builder = cls._builders[class_name]
        except KeyError:
            available = ", ".join(cls._builders)
            raise ValueError(
                f"Unknown evaluator model class {class_name!r}. "
                f"Available classes: {available}"
            ) from None

        return builder(model_name)


class DatasetFactory:
    """Construct project datasets while preserving their native arguments."""

    _builders: dict[str, Callable[..., Dataset]] = {
        "RequestDataset": RequestDataset,
        "DatasetClassifier": DatasetClassifier,
    }
    _raw_builders: dict[
        str,
        Callable[[Any, str, int, int | None], Dataset],
    ] = {
        "RequestDataset": lambda raw, tokenizer_name, start, end: (
            RequestDataset.from_raw(raw, tokenizer_name, start, end)
        ),
    }

    @overload
    @classmethod
    def create_dataset(
        cls,
        class_name: Literal["RequestDataset"],
        *,
        requests: list[str],
        tokenizer_name: str,
        dic: dict[str, list[Any]] | None = None,
        do_dict: bool = False,
    ) -> RequestDataset: ...

    @overload
    @classmethod
    def create_dataset(
        cls,
        class_name: Literal["DatasetClassifier"],
        *,
        theta: float = 2.0,
        id: str | None = None,
    ) -> DatasetClassifier: ...

    @overload
    @classmethod
    def create_dataset(
        cls,
        class_name: str,
        **kwargs: Any,
    ) -> Dataset: ...

    @classmethod
    def create_dataset(
        cls,
        class_name: str,
        **kwargs: Any,
    ) -> Dataset:
        try:
            builder = cls._builders[class_name]
        except KeyError:
            available = ", ".join(cls._builders)
            raise ValueError(
                f"Unknown dataset class {class_name!r}. "
                f"Available classes: {available}"
            ) from None

        try:
            return builder(**kwargs)
        except TypeError as error:
            raise TypeError(
                f"Invalid arguments for dataset class {class_name!r}: {error}"
            ) from error

    @classmethod
    def create_from_raw(
        cls,
        class_name: str,
        raw_dataset: Any,
        tokenizer_name: str,
        start: int = 0,
        end: int | None = None,
    ) -> Dataset:
        try:
            builder = cls._raw_builders[class_name]
        except KeyError:
            available = ", ".join(cls._raw_builders)
            raise ValueError(
                f"Dataset class {class_name!r} cannot be created from raw "
                f"data. Available classes: {available}"
            ) from None

        return builder(raw_dataset, tokenizer_name, start, end)
