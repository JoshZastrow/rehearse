"""Smoke dataset for the sandboxed voice-agent environment."""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.protocols import BenchmarkExample


class VoiceAgentSmokeDataset:
    name = "voice-agent-smoke"
    version = "v0"

    def load(self) -> Iterable[BenchmarkExample]:
        yield BenchmarkExample(
            id="voice-agent-smoke-001",
            benchmark=self.name,
            payload={
                "customer_script": [
                    "I need to tell my cofounder I want to revisit equity.",
                    "I am worried they will get defensive.",
                ],
                "max_turns": 4,
            },
            expected={"min_runtime_turns": 2},
        )
