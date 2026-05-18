"""Reward-model scorer — drop-in alternative to TrajectoryJudgeScorer.

Calls a self-hosted /score endpoint (FastAPI) instead of an LLM. Returns the
same list[RubricScore] shape as TrajectoryJudgeScorer so it can be swapped into
any eval's scoring_plan() without touching the runner.

Usage in a scoring_plan():

    from rehearse.eval.scorers.reward_model_scorer import RewardModelScorer

    def scoring_plan(self) -> list[Scorer]:
        return [RewardModelScorer()]  # reads REWARD_MODEL_URL env var

Or with an explicit URL and custom weights:

    RewardModelScorer(
        base_url="http://reward-model.internal:8000",
        weights={"emotion_responsiveness": 0.45, "coaching_trajectory_quality": 0.55},
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore

_DEFAULT_DIMENSIONS = ["emotion_responsiveness", "coaching_trajectory_quality"]

_DEFAULT_WEIGHTS = {
    "emotion_responsiveness": 0.45,
    "coaching_trajectory_quality": 0.55,
}


class RewardModelScorer:
    """Score a coaching transcript via a self-hosted reward-model endpoint.

    The endpoint contract (see infra/reward_model_server.py):

      POST {base_url}/score
      Body: {"transcript": [...], "scenario": {...}, "dimensions": [...]}
      Response: {"emotion_responsiveness": 0.8, "coaching_trajectory_quality": 0.7,
                 "weighted_reward": 0.74}

    On HTTP or connection error, returns zero scores with the error message as
    the rationale — same defensive pattern as TrajectoryJudgeScorer.
    """

    name = "reward_model"
    dimension = "weighted_reward"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        weights: dict[str, float] | None = None,
        http_client: Any = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("REWARD_MODEL_URL", "http://localhost:8000")).rstrip("/")
        self.weights = weights or dict(_DEFAULT_WEIGHTS)
        self._http_client = http_client  # injected in tests; None → create lazily

    def _get_client(self) -> Any:
        if self._http_client is not None:
            return self._http_client
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx is required for RewardModelScorer: pip install httpx") from exc
        self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        weights = dict(example.payload.get("rubric_weights") or self.weights)
        # Normalise to the two dimensions this scorer handles.
        er_w = weights.get("emotion_responsiveness", _DEFAULT_WEIGHTS["emotion_responsiveness"])
        ct_w = weights.get("coaching_trajectory_quality", _DEFAULT_WEIGHTS["coaching_trajectory_quality"])
        active_weights = {"emotion_responsiveness": er_w, "coaching_trajectory_quality": ct_w}

        if rollout.status != "ok" or not rollout.artifacts_dir:
            return self._zeros(example, run_id, active_weights, rationale=f"rollout {rollout.status}")

        transcript_path = Path(rollout.artifacts_dir) / "transcript.jsonl"
        if not transcript_path.exists():
            return self._zeros(
                example, run_id, active_weights,
                rationale=f"transcript not found at {transcript_path}",
            )

        transcript_lines = transcript_path.read_text().splitlines()
        if not transcript_lines:
            return self._zeros(example, run_id, active_weights, rationale="transcript is empty")

        transcript = _parse_transcript(transcript_lines)
        scenario = _build_scenario(example)

        try:
            response_data = await self._call_endpoint(transcript, scenario)
        except Exception as exc:
            return self._zeros(example, run_id, active_weights, rationale=f"endpoint error: {exc}")

        try:
            er_score = float(response_data["emotion_responsiveness"])
            ct_score = float(response_data["coaching_trajectory_quality"])
            weighted = float(
                response_data.get(
                    "weighted_reward",
                    er_w * er_score + ct_w * ct_score,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            return self._zeros(
                example, run_id, active_weights,
                rationale=f"malformed endpoint response: {exc}",
            )

        er_score = max(0.0, min(1.0, er_score))
        ct_score = max(0.0, min(1.0, ct_score))
        weighted = max(0.0, min(1.0, weighted))

        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension="emotion_responsiveness",
                value=er_score,
                scorer="llm_judge",
                rationale="reward_model endpoint",
            ),
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension="coaching_trajectory_quality",
                value=ct_score,
                scorer="llm_judge",
                rationale="reward_model endpoint",
            ),
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension="weighted_reward",
                value=weighted,
                scorer="llm_judge",
                rationale=(
                    f"{er_w:.2f}*{er_score:.2f} + "
                    f"{ct_w:.2f}*{ct_score:.2f} = {weighted:.3f} (reward_model)"
                ),
            ),
        ]

    async def _call_endpoint(
        self,
        transcript: list[dict[str, str]],
        scenario: dict[str, str],
    ) -> dict[str, Any]:
        client = self._get_client()
        payload = {
            "transcript": transcript,
            "scenario": scenario,
            "dimensions": _DEFAULT_DIMENSIONS,
        }
        resp = await client.post(f"{self.base_url}/score", json=payload)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _zeros(
        example: BenchmarkExample,
        run_id: str,
        weights: dict[str, float],
        rationale: str,
    ) -> list[RubricScore]:
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=dim,
                value=0.0,
                scorer="llm_judge",
                rationale=rationale,
            )
            for dim in (
                "emotion_responsiveness",
                "coaching_trajectory_quality",
                "weighted_reward",
            )
        ]


def _parse_transcript(jsonl_lines: list[str]) -> list[dict[str, str]]:
    """Convert transcript.jsonl lines to the flat [{speaker, text}] shape."""
    rows: list[dict[str, str]] = []
    for line in jsonl_lines:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        speaker = frame.get("speaker", "")
        text = frame.get("text", "").strip()
        if speaker and text:
            rows.append({"speaker": speaker, "text": text})
    return rows


def _build_scenario(example: BenchmarkExample) -> dict[str, str]:
    """Extract the four scenario keys from the example payload."""
    scenario = example.payload.get("scenario") or {}
    opening_emotion = (
        example.expected.get("opening_emotion")
        or example.payload.get("opening_emotion")
        or scenario.get("emotional_state")
        or "unknown"
    )
    return {
        "situation": scenario.get("situation", ""),
        "goal": scenario.get("goal", ""),
        "counterparty_role": scenario.get("counterparty_role", ""),
        "emotional_state": opening_emotion,
    }
