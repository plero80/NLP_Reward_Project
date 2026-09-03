from types import SimpleNamespace

import pytest

from Factory import EvaluatorModelFactory


@pytest.mark.parametrize(
    "class_name",
    ["EvaluatorOpenAIModel", "PrometheusEvaluator"],
)
def test_evaluator_factory_creates_registered_model(
    monkeypatch,
    class_name: str,
) -> None:
    calls: list[str] = []
    expected = SimpleNamespace(class_name=class_name)
    builders = dict(EvaluatorModelFactory._builders)
    builders[class_name] = lambda model_name: (
        calls.append(model_name) or expected
    )
    monkeypatch.setattr(EvaluatorModelFactory, "_builders", builders)

    evaluator = EvaluatorModelFactory.create_model(
        class_name,
        "test/evaluator",
    )

    assert evaluator is expected
    assert calls == ["test/evaluator"]


def test_evaluator_factory_rejects_unknown_class() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown evaluator model class 'Missing'",
    ):
        EvaluatorModelFactory.create_model("Missing", "test/evaluator")
