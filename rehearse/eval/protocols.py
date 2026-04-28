"""Plugin contracts for the eval harness.

Four protocols define the public vocabulary:

  Eval        — declares the dataset, compatible environments, and scoring plan
  Dataset     — loads examples
  Environment — runs a rollout against one example
  Scorer      — produces RubricScore rows from a rollout
  Executor    — runs Environments, isolated and in parallel

Concrete implementations live under evals/, datasets/, environments/, scorers/,
and executors/.
The runner imports only this module and the registries.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from rehearse.types import RubricScore


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, use_enum_values=False)


class BenchmarkExample(_Strict):
    """One row from a dataset.

    `payload` is dataset-defined input handed to the Environment. `expected` is
    dataset-defined ground truth handed to the Scorer. Environments never read
    `expected`; Scorers never read raw runtime state.

    The `benchmark` field name is retained for serialized compatibility with
    existing run artifacts. New code should treat it as the eval/dataset name.
    """

    id: str
    benchmark: str
    payload: dict[str, Any]
    expected: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutResult(_Strict):
    """What an Environment produced for one Example.

    `artifacts_dir` is set when the rollout produced a Session bundle on disk
    (full / synthesis environments). `payload` carries inline outputs for environments
    that don't emit a session (raw-llm). Both may be None on error.

    The `target_*` field names are retained for serialized compatibility with
    existing run artifacts. New code should treat them as environment metadata.
    """

    example_id: str
    target_name: str
    target_version: str
    status: Literal["ok", "error", "timeout"]
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    artifacts_dir: Path | None = None
    payload: dict[str, Any] | None = None
    error: str | None = None


@runtime_checkable
class Dataset(Protocol):
    name: str
    version: str

    def load(self) -> Iterable[BenchmarkExample]: ...


@runtime_checkable
class Eval(Protocol):
    name: str
    version: str
    dataset: Dataset
    supported_environments: frozenset[str]
    preferred_environment: str

    def load(self) -> Iterable[BenchmarkExample]: ...

    def scoring_plan(self) -> list[Scorer]: ...

    def rollout_timeout_s(self) -> int: ...


@runtime_checkable
class Environment(Protocol):
    name: str
    version: str

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult: ...


@runtime_checkable
class Scorer(Protocol):
    name: str
    dimension: str

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]: ...


@runtime_checkable
class Executor(Protocol):
    async def submit(
        self,
        target_name: str,
        target_version: str,
        model_slots: dict[str, str],
        example: BenchmarkExample,
        run_dir: Path,
        timeout_s: int,
        rng_seed: int,
    ) -> RolloutResult: ...


# Backwards-compatible protocol names while callers migrate to the new shape.
Benchmark = Eval
Target = Environment
