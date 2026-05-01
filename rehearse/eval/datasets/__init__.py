"""Dataset registry.

Datasets only load examples. Evals compose datasets with scorers and compatible
environments.
"""

from __future__ import annotations

from collections.abc import Callable

from rehearse.eval.datasets.coach_dialogue_smoke import CoachDialogueSmokeDataset
from rehearse.eval.datasets.mme_emotion import MMEEmotionDataset
from rehearse.eval.datasets.mme_rollout_seeds import MMERolloutSeedDataset
from rehearse.eval.datasets.noop import NoopDataset
from rehearse.eval.datasets.voice_agent_smoke import VoiceAgentSmokeDataset
from rehearse.eval.protocols import Dataset

DATASETS: dict[str, Callable[[], Dataset]] = {
    "noop": NoopDataset,
    "mme-emotion": MMEEmotionDataset,
    "voice-agent-smoke": VoiceAgentSmokeDataset,
    "coach-dialogue-smoke": CoachDialogueSmokeDataset,
    "mme-emotion-rollout-seeds": MMERolloutSeedDataset,
}


def get_dataset(name: str) -> Dataset:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}. registered: {sorted(DATASETS)}")
    return DATASETS[name]()


def list_datasets() -> list[str]:
    return sorted(DATASETS)
