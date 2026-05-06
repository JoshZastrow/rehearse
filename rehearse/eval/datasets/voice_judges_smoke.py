"""Smoke dataset for the voice-judges-smoke eval (Spec 2 Mini-spec).

One example with a fixture transcript + per-turn audio durations, used by
`AudioFixtureEnvironment` to synthesize silent WAVs that the audio
judges then score (with stub or live judge configurations).
"""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.protocols import BenchmarkExample


class VoiceJudgesSmokeDataset:
    name = "voice-judges-smoke"
    version = "v0"

    def load(self) -> Iterable[BenchmarkExample]:
        return [
            BenchmarkExample(
                id="vjs-001",
                benchmark=self.name,
                payload={
                    "scenario": {
                        "situation": "delivering hard feedback to a peer",
                        "goal": "stay grounded; land the message",
                        "counterparty_role": "peer",
                    },
                    "transcript": [
                        {
                            "speaker": "user",
                            "text": "I keep avoiding this conversation.",
                        },
                        {
                            "speaker": "coach",
                            "text": "What do you imagine happens if you don't have it?",
                        },
                        {
                            "speaker": "user",
                            "text": "Things get worse, slowly.",
                        },
                        {
                            "speaker": "coach",
                            "text": (
                                "Then let's get it ready. "
                                "What's the first thing you'd say?"
                            ),
                        },
                    ],
                    # Per-turn audio durations (seconds) — synthesized as silent WAVs.
                    "user_audio_durations_s": [1.2, 0.9],
                    "coach_audio_durations_s": [1.0, 1.4],
                },
                expected={"opening_emotion": "anxious"},
            ),
        ]
