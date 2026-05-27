"""voice-rollout-judges — full live rollout + audio + judges, no smoke.

Composes:
  - `VoiceRolloutJudgesDataset` — synthetic coaching scenarios
    (10 seed + grows toward 100). Loaded from JSONL on disk so the
    set can be edited without code changes.
  - `live-rollout-audio` environment — runs LLM customer + LLM coach,
    synthesizes per-turn Hume Octave audio, writes `timing.jsonl`.
  - `CompositeScorer` wrapping:
      * `ContentJudgeScorer` (DeepEval G-Eval, text)
      * `AffectPerceptionJudgeScorer` (Gemini multimodal)
      * `DeliveryJudgeScorer` (Gemini multimodal)
      * `NaturalnessScorer` (deterministic timing)
    aggregated by `AggregateScorer` into one `weighted_reward` row.

Live by default. Requires:
  - `ANTHROPIC_API_KEY` (customer + coach LLM)
  - `HUME_API_KEY` (per-turn TTS)
  - `GEMINI_API_KEY` (audio judges)
  - `OPENAI_API_KEY` (DeepEval G-Eval default)

Per-example weights are read from `payload["rubric_weights"]` so individual
scenarios can shift the balance. Defaults are content-heavy because the
text judge is the cheapest and most stable signal during early calibration.
"""

from __future__ import annotations

from collections.abc import Iterable

from rehearse.eval.datasets.voice_rollout_judges import VoiceRolloutJudgesDataset
from rehearse.eval.protocols import BenchmarkExample, Scorer
from rehearse.eval.scorers.aggregate import AggregateScorer
from rehearse.eval.scorers.audio_judges import AudioJudge, CalibratedAffectScorer, CalibratedDeliveryScorer
from rehearse.eval.scorers.composite import CompositeScorer
from rehearse.eval.scorers.intake_fidelity import IntakeFidelityScorer
from rehearse.eval.scorers.naturalness import NaturalnessScorer
from rehearse.eval.scorers.strict_content_judge import StrictContentJudgeScorer

# Existing weights scaled by 0.95 to make room for intake_fidelity at 0.05.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "content_quality": 0.38,
    "affect_perception": 0.3325,
    "delivery_quality": 0.1425,
    "naturalness.interruption_rate": 0.02375,
    "naturalness.silence_after_affect": 0.02375,
    "naturalness.speech_rate_band": 0.0475,
    "intake_fidelity": 0.05,
}


class VoiceRolloutJudgesEval:
    name = "voice-rollout-judges"
    version = "v0"
    supported_environments = frozenset(
        {"live-rollout-audio", "runtime-sandbox", "live-audio-sandbox"}
    )
    preferred_environment = "runtime-sandbox"

    def __init__(self) -> None:
        self.dataset = VoiceRolloutJudgesDataset()

    def load(self) -> Iterable[BenchmarkExample]:
        return self.dataset.load()

    def scoring_plan(self) -> list[Scorer]:
        children: list[Scorer] = [
            StrictContentJudgeScorer(),
            CalibratedAffectScorer(judge=AudioJudge()),
            CalibratedDeliveryScorer(judge=AudioJudge()),
            NaturalnessScorer(),
            IntakeFidelityScorer(),
        ]
        aggregator = AggregateScorer(weights=_DEFAULT_WEIGHTS)
        return [CompositeScorer(children=children, aggregator=aggregator)]

    def rollout_timeout_s(self) -> int:
        # 600s covers live-audio-sandbox rollouts (EVI round-trips + TTS per turn).
        # Text-only runtime-sandbox completes well within this budget (~90s typical).
        return 600
