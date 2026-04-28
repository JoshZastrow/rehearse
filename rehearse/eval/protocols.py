"""Plugin contracts for the eval harness.

Four protocols define the entire vocabulary:

  Benchmark — loads examples + declares scoring plan
  Target    — runs a rollout against one example
  Scorer    — produces RubricScore rows from a rollout
  Executor  — runs Targets, isolated and in parallel

Concrete implementations live under benchmarks/, targets/, scorers/, executors/.
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
    """One row from a benchmark.

    `payload` is benchmark-defined input handed to the Target. `expected` is
    benchmark-defined ground truth handed to the Scorer. Targets never read
    `expected`; Scorers never read raw runtime state.
    """

    id: str
    benchmark: str
    payload: dict[str, Any]
    expected: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutResult(_Strict):
    """What a Target produced for one Example.

    `artifacts_dir` is set when the rollout produced a Session bundle on disk
    (full / synthesis targets). `payload` carries inline outputs for targets
    that don't emit a session (raw-llm). Both may be None on error.
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
class Benchmark(Protocol):
    name: str
    version: str
    supported_targets: frozenset[str]
    preferred_target: str

    def load(self) -> Iterable[BenchmarkExample]: ...

    def scoring_plan(self) -> list[Scorer]: ...

    def rollout_timeout_s(self) -> int: ...


@runtime_checkable
class Target(Protocol):
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
