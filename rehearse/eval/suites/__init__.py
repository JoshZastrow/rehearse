"""Eval suites registry.

A suite composes one Dataset, a scoring plan, and the environments it can run
against.
"""

from __future__ import annotations

from collections.abc import Callable

from rehearse.eval.suites.noop import NoopEval
from rehearse.eval.suites.production_voice_replay import ProductionVoiceReplayEval
from rehearse.eval.suites.voice_judges_smoke import VoiceJudgesSmokeEval
from rehearse.eval.suites.voice_rollout_judges import VoiceRolloutJudgesEval
from rehearse.eval.protocols import Eval

EVALS: dict[str, Callable[[], Eval]] = {
    "noop": NoopEval,
    "voice-judges-smoke": VoiceJudgesSmokeEval,
    "production-voice-replay": ProductionVoiceReplayEval,
    "voice-rollout-judges": VoiceRolloutJudgesEval,
}


def get_eval(name: str) -> Eval:
    if name not in EVALS:
        raise KeyError(f"unknown eval {name!r}. registered: {sorted(EVALS)}")
    return EVALS[name]()


def list_evals() -> list[str]:
    return sorted(EVALS)
