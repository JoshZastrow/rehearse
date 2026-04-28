"""Every registered eval, dataset, and environment satisfies its Protocol."""

from __future__ import annotations

from rehearse.eval.datasets import DATASETS, get_dataset
from rehearse.eval.environments import ENVIRONMENTS, get_environment
from rehearse.eval.evals import EVALS, get_eval
from rehearse.eval.protocols import Dataset, Environment, Eval


def test_every_eval_satisfies_protocol():
    for name in EVALS:
        eval_spec = get_eval(name)
        assert isinstance(eval_spec, Eval), f"{name} is not an Eval"
        assert eval_spec.name == name
        assert isinstance(eval_spec.version, str) and eval_spec.version
        assert eval_spec.preferred_environment in eval_spec.supported_environments
        assert isinstance(eval_spec.dataset, Dataset)


def test_every_dataset_satisfies_protocol():
    for name in DATASETS:
        dataset = get_dataset(name)
        assert isinstance(dataset, Dataset), f"{name} is not a Dataset"
        assert dataset.name == name
        assert isinstance(dataset.version, str) and dataset.version


def test_every_environment_satisfies_protocol():
    for name in ENVIRONMENTS:
        environment = get_environment(name, model_slots={})
        assert isinstance(environment, Environment), f"{name} is not an Environment"
        assert environment.name == name
        assert isinstance(environment.version, str) and environment.version


def test_unknown_eval_raises():
    import pytest

    with pytest.raises(KeyError):
        get_eval("does-not-exist")


def test_unknown_environment_raises():
    import pytest

    with pytest.raises(KeyError):
        get_environment("does-not-exist", {})
