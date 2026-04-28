"""Noop dataset — two synthetic examples for harness smoke tests."""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.protocols import BenchmarkExample


class NoopDataset:
    name = "noop"
    version = "v0"

    def load(self) -> Iterable[BenchmarkExample]:
        return [
            BenchmarkExample(
                id=f"noop-{i:03d}",
                benchmark=self.name,
                payload={"echo": f"hello-{i}"},
                expected={"echo": f"hello-{i}"},
            )
            for i in range(2)
        ]
