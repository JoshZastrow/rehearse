"""ProsodyService protocol and pluggable implementations.

One implementation runs per PipelineBackend session. The NullProsodyService
is safe for local dev where no GPU or external API is available.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rehearse.types import ProsodyScores


@runtime_checkable
class ProsodyService(Protocol):
    """Classify emotion scores from one completed user audio segment."""

    async def score(self, pcm16_16k: bytes) -> ProsodyScores:
        """Return emotion scores for the audio segment.

        Called once per final user utterance, after turn detection confirms
        the turn is complete. The segment is the full utterance audio.
        """
        ...


class NullProsodyService:
    """Emit zeroed prosody scores. For local dev where no classifier is available."""

    async def score(self, pcm16_16k: bytes) -> ProsodyScores:
        return ProsodyScores(arousal=0.0, valence=0.0, emotions={})
