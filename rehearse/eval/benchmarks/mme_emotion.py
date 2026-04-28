"""Compatibility wrapper for the MME-Emotion eval."""

from __future__ import annotations

from rehearse.eval.evals.mme_emotion import MMEEmotionEval


class MMEEmotionBenchmark(MMEEmotionEval):
    supported_targets = MMEEmotionEval.supported_environments
    preferred_target = MMEEmotionEval.preferred_environment
