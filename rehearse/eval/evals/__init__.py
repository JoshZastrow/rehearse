"""Eval registry.

An Eval composes one Dataset, a scoring plan, and the environments it can run
against.
"""

from __future__ import annotations

from collections.abc import Callable

from rehearse.eval.evals.coach_dialogue_smoke import CoachDialogueSmokeEval
from rehearse.eval.evals.mme_emotion import MMEEmotionEval
from rehearse.eval.evals.mme_sandbox_rollout import MMESandboxRolloutEval
from rehearse.eval.evals.noop import NoopEval
from rehearse.eval.evals.production_voice_replay import ProductionVoiceReplayEval
from rehearse.eval.evals.voice_agent_smoke import VoiceAgentSmokeEval
from rehearse.eval.evals.voice_judges_smoke import VoiceJudgesSmokeEval
from rehearse.eval.evals.voice_rollout_judges import VoiceRolloutJudgesEval
from rehearse.eval.protocols import Eval

EVALS: dict[str, Callable[[], Eval]] = {
    "noop": NoopEval,
    "mme-emotion": MMEEmotionEval,
    "voice-agent-smoke": VoiceAgentSmokeEval,
    "coach-dialogue-smoke": CoachDialogueSmokeEval,
    "mme-sandbox-rollout": MMESandboxRolloutEval,
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
