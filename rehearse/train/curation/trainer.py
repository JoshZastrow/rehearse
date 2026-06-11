"""Trainer protocol for ablation runs.

Implementations dispatch one budget-matched training run over the curated
corpus minus excluded_ids and return the held-out loss. Tests use FakeTrainer
(tests/curation/conftest.py); the live Modal adapter lives here too.
"""

from __future__ import annotations

from typing import Protocol


class Trainer(Protocol):
    def run(self, *, excluded_ids: list[str], seed: int, max_steps: int, kind: str) -> float:
        """Run training without excluded_ids; return held-out loss."""
        ...
