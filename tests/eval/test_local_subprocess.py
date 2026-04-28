"""LocalSubprocessExecutor: timeout enforcement and crash isolation.

We don't have an environment that hangs by design, so we test the timeout path by
calling submit() with timeout_s=0 (worker has no chance to finish before
asyncio.wait_for raises). Crash isolation is exercised by the runner test
plus the worker's exception handler.
"""

from __future__ import annotations

from pathlib import Path

from rehearse.eval.executors import LocalSubprocessExecutor
from rehearse.eval.protocols import BenchmarkExample


async def test_echo_environment_via_subprocess(tmp_path: Path):
    executor = LocalSubprocessExecutor()
    example = BenchmarkExample(
        id="ex1", benchmark="noop", payload={"echo": "hi"}, expected={}
    )
    result = await executor.submit(
        target_name="echo",
        target_version="v0",
        model_slots={},
        example=example,
        run_dir=tmp_path / "ex1",
        timeout_s=10,
        rng_seed=0,
    )
    assert result.status == "ok"
    assert result.payload == {"echo": "hi"}


async def test_subprocess_timeout_marks_rollout_timeout(tmp_path: Path):
    executor = LocalSubprocessExecutor()
    example = BenchmarkExample(
        id="ex1", benchmark="noop", payload={"echo": "hi"}, expected={}
    )
    result = await executor.submit(
        target_name="echo",
        target_version="v0",
        model_slots={},
        example=example,
        run_dir=tmp_path / "ex1",
        timeout_s=0,
        rng_seed=0,
    )
    assert result.status == "timeout"
    assert "exceeded" in (result.error or "")


async def test_unknown_environment_in_worker_marks_error(tmp_path: Path):
    executor = LocalSubprocessExecutor()
    example = BenchmarkExample(
        id="ex1", benchmark="noop", payload={}, expected={}
    )
    result = await executor.submit(
        target_name="does-not-exist",
        target_version="v0",
        model_slots={},
        example=example,
        run_dir=tmp_path / "ex1",
        timeout_s=10,
        rng_seed=0,
    )
    assert result.status == "error"
