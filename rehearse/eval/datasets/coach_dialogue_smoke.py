"""Coach-dialogue smoke dataset.

One inlined scenario for the LLM-vs-LLM coaching dialogue smoke. No external
manifest in v0; richer datasets land alongside the MME audio seed in RLE3
per `docs/specs/v2026-04-29-mme-seeded-rl-sandbox-eval.md`.
"""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.protocols import BenchmarkExample


class CoachDialogueSmokeDataset:
    name = "coach-dialogue-smoke"
    version = "v0"

    def load(self) -> Iterable[BenchmarkExample]:
        yield BenchmarkExample(
            id="coach-dialogue-cofounder-equity-001",
            benchmark=self.name,
            payload={
                "scenario": {
                    "situation": (
                        "I want to revisit the equity split with my cofounder. "
                        "We agreed on 60/40 a year ago and I've been carrying "
                        "more of the load since then."
                    ),
                    "goal": (
                        "raise this without escalating, get to a 50/50 split, "
                        "preserve the working relationship"
                    ),
                    "counterparty_role": "cofounder",
                    "counterparty_style": "defensive but fair-minded",
                    "stakes": "the relationship and the trajectory of the company",
                    "emotional_state": (
                        "anxious about bringing it up, afraid of being seen "
                        "as transactional, wants to feel heard"
                    ),
                },
                "max_turns": 6,
                "timeout_s": 90,
                "customer_agent": "llm",
                "coach_agent": "llm",
            },
            expected={"min_runtime_turns": 3},
        )
