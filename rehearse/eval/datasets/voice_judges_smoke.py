"""Smoke dataset for the voice-judges-smoke eval (Spec 2 Mini-spec).

Five examples spanning the valence × arousal grid so the audio judges
have differentiable signal when run live (and so naturalness
sub-metrics get exercised across multiple per-turn duration bands).
Roadmap item #5: a cheap "do the judges differentiate?" probe before
paying for full calibration.

Coverage:
  1. anxious-avoidant        (negative valence, mid arousal)
  2. flat-withdrawn          (negative valence, low arousal)
  3. escalating-frustrated   (negative valence, high arousal)
  4. breakthrough-positive   (positive valence, high arousal)
  5. confused-seeking        (neutral valence, low arousal)

The `description` field on each transcript turn is the emotion cue
the Hume Octave TTS bridge conditions on when synthesizing audio
(see `rehearse/eval/tts_bridge.py`). With stub judges + silent WAVs
this field is unused; with live TTS + live judges it produces the
acoustic variety the rubric grades.

Per-turn durations are sized so naturalness sub-metrics land in
distinct bands across the five examples — ideal pacing, rushed coach,
long silences, brisk-but-warm, and very tight back-and-forth.
"""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.protocols import BenchmarkExample


class VoiceJudgesSmokeDataset:
    name = "voice-judges-smoke"
    version = "v1"

    def load(self) -> Iterable[BenchmarkExample]:
        return [
            # 1. Anxious / avoidant user, grounded coach. Ideal pacing.
            BenchmarkExample(
                id="vjs-anxious-avoidant",
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
                            "description": "anxious, hesitant, soft-spoken",
                        },
                        {
                            "speaker": "coach",
                            "text": "What do you imagine happens if you don't have it?",
                            "description": "warm, steady, curious",
                        },
                        {
                            "speaker": "user",
                            "text": "Things get worse, slowly.",
                            "description": "resigned, quiet",
                        },
                        {
                            "speaker": "coach",
                            "text": (
                                "Then let's get it ready. "
                                "What's the first thing you'd say?"
                            ),
                            "description": "encouraging, gently directive",
                        },
                    ],
                    # Coach: 9 words / 3.6s ≈ 150 wpm; 12 words / 4.8s = 150 wpm.
                    "user_audio_durations_s": [2.5, 1.8],
                    "coach_audio_durations_s": [3.6, 4.8],
                    # Pad so silence_after_user lands in [1.5, 4.0]s.
                    "silence_between_turns_s": 2.0,
                },
                expected={"opening_emotion": "anxious"},
            ),
            # 2. Flat / withdrawn user, slightly rushed coach. Tests rate band.
            BenchmarkExample(
                id="vjs-flat-withdrawn",
                benchmark=self.name,
                payload={
                    "scenario": {
                        "situation": "user processing a recent loss",
                        "goal": "make space; resist solutioning",
                        "counterparty_role": "self",
                    },
                    "transcript": [
                        {
                            "speaker": "user",
                            "text": "I don't really know what to say about it.",
                            "description": "flat, withdrawn, low energy",
                        },
                        {
                            "speaker": "coach",
                            "text": (
                                "That's okay. We don't need to figure anything out "
                                "right now. What's here for you in this moment?"
                            ),
                            "description": "calm, unhurried, present",
                        },
                        {
                            "speaker": "user",
                            "text": "Just tired, mostly.",
                            "description": "exhausted, monotone",
                        },
                        {
                            "speaker": "coach",
                            "text": "Tired makes sense. Where do you feel that?",
                            "description": "soft, slow, attuned",
                        },
                    ],
                    # Coach turn 0: 22 words in 6.6s ≈ 200 wpm (rushed band).
                    # Coach turn 1: 8 words / 3.2s = 150 wpm (ideal).
                    "user_audio_durations_s": [3.0, 1.5],
                    "coach_audio_durations_s": [6.6, 3.2],
                    "silence_between_turns_s": 2.5,
                },
                expected={"opening_emotion": "withdrawn"},
            ),
            # 3. Escalating / frustrated user, slow coach. Tests long silence.
            BenchmarkExample(
                id="vjs-escalating-frustrated",
                benchmark=self.name,
                payload={
                    "scenario": {
                        "situation": "user prepping a confrontation with a manager",
                        "goal": "channel the heat without losing the ask",
                        "counterparty_role": "manager",
                    },
                    "transcript": [
                        {
                            "speaker": "user",
                            "text": (
                                "I'm so done with this. Every single week it's "
                                "the same thing and nothing ever changes."
                            ),
                            "description": "angry, escalating, fast-paced",
                        },
                        {
                            "speaker": "coach",
                            "text": "I hear how worn down you are.",
                            "description": "slow, grounding, deliberate",
                        },
                        {
                            "speaker": "user",
                            "text": "Worn down doesn't even cover it.",
                            "description": "intense, sharp, frustrated",
                        },
                        {
                            "speaker": "coach",
                            "text": (
                                "Let's slow down for a second. "
                                "If nothing changed, what's the cost to you in six months?"
                            ),
                            "description": "low and slow, redirecting energy",
                        },
                    ],
                    # Coach turn 0: 6 words / 4.0s = 90 wpm (slow band).
                    # Coach turn 1: 19 words / 7.5s ≈ 152 wpm (ideal).
                    "user_audio_durations_s": [4.5, 2.0],
                    "coach_audio_durations_s": [4.0, 7.5],
                    # Long silence — beyond ideal band, exercises silence_after_affect.
                    "silence_between_turns_s": 5.0,
                },
                expected={"opening_emotion": "angry"},
            ),
            # 4. Relieved / content user, gentle coach. Positive valence,
            #    low arousal — quiet settling after a tense week resolved
            #    well. Tight back-and-forth, brisk-but-warm pacing.
            BenchmarkExample(
                id="vjs-relieved-content",
                benchmark=self.name,
                payload={
                    "scenario": {
                        "situation": (
                            "user reflecting after a tense week ended better than expected"
                        ),
                        "goal": "let the relief land; notice what they did well",
                        "counterparty_role": "self",
                    },
                    "transcript": [
                        {
                            "speaker": "user",
                            "text": "It actually worked out. I can finally exhale.",
                            "description": "soft, settled, quietly relieved",
                        },
                        {
                            "speaker": "coach",
                            "text": "Good. Let it land.",
                            "description": "warm, unhurried, gentle",
                        },
                        {
                            "speaker": "user",
                            "text": "I forgot what this felt like.",
                            "description": "reflective, calm, faintly grateful",
                        },
                        {
                            "speaker": "coach",
                            "text": "What part of you got you here?",
                            "description": "curious, soft, present",
                        },
                    ],
                    # Coach turn 0: 4 words / 1.6s = 150 wpm (ideal).
                    # Coach turn 1: 7 words / 2.8s = 150 wpm (ideal).
                    "user_audio_durations_s": [3.0, 2.2],
                    "coach_audio_durations_s": [1.6, 2.8],
                    # Tight back-and-forth — short ideal silence.
                    "silence_between_turns_s": 1.6,
                },
                expected={"opening_emotion": "relieved"},
            ),
            # 5. Excited / eager user, energized coach. Positive valence,
            #    high arousal — exercises the opposite affect quadrant from #3
            #    (escalating-frustrated): same arousal, opposite valence.
            BenchmarkExample(
                id="vjs-excited-eager",
                benchmark=self.name,
                payload={
                    "scenario": {
                        "situation": "user just landed a job offer they thought was out of reach",
                        "goal": "let the win be felt; surface what made it possible",
                        "counterparty_role": "self",
                    },
                    "transcript": [
                        {
                            "speaker": "user",
                            "text": "They actually said yes. I can't believe it.",
                            "description": "elated, fast, breathless",
                        },
                        {
                            "speaker": "coach",
                            "text": (
                                "That's huge. Take a second — "
                                "what does it feel like right now?"
                            ),
                            "description": "warm, energized, present",
                        },
                        {
                            "speaker": "user",
                            "text": "Like I finally proved something to myself.",
                            "description": "eager, joyful, steadier",
                        },
                        {
                            "speaker": "coach",
                            "text": (
                                "Stay with that. What did you do this time "
                                "that you didn't before?"
                            ),
                            "description": "curious, encouraging, unhurried",
                        },
                    ],
                    # Coach turn 0: 13 words / 5.2s = 150 wpm (ideal).
                    # Coach turn 1: 13 words / 5.2s = 150 wpm (ideal).
                    "user_audio_durations_s": [2.6, 2.0],
                    "coach_audio_durations_s": [5.2, 5.2],
                    "silence_between_turns_s": 1.8,
                },
                expected={"opening_emotion": "excited"},
            ),
        ]
