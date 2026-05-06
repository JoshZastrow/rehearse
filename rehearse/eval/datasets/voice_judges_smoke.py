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
                    # Sized so naturalness sub-metrics land in their ideal bands:
                    #   coach turn 0: 9 words / 3.6s ≈ 150 wpm
                    #   coach turn 1: 12 words / 4.8s = 150 wpm
                    "user_audio_durations_s": [2.5, 1.8],
                    "coach_audio_durations_s": [3.6, 4.8],
                    # Pad between turns so silence_after_user falls in [1.5, 4.0]s.
                    "silence_between_turns_s": 2.0,
                },
                expected={"opening_emotion": "anxious"},
            ),
        ]
