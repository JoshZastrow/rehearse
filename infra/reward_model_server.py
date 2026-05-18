"""Self-hosted reward model server — inference boundary for PPO/GRPO training.

Fine-tune by:
  1. Harvest training labels:
       python infra/harvest_judge_labels.py
       # writes data/reward_model_train.jsonl

  2. Convert to (input_ids, score) pairs and train a sequence regression head on
     top of a small encoder model, e.g.:
       - deberta-v3-small  (~180M params, good for sequence classification)
       - qwen2.5-1.5b      (~1.5B params, stronger but heavier)

     Each training example is:
       input  = tokenize(scenario_prefix + transcript_text)
       labels = {"emotion_responsiveness": 0.8, "coaching_trajectory_quality": 0.7}

     Use MSELoss per dimension; optionally add a weighted_reward head.

  3. Save checkpoint:
       model.save_pretrained("data/reward_model_ckpt")

  4. Start this server pointing at the checkpoint:
       REWARD_MODEL_PATH=data/reward_model_ckpt uvicorn infra.reward_model_server:app

  5. Run eval with RewardModelScorer to generate rewards.

  6. Use the weighted_reward as the scalar reward signal in GRPO/PPO:
       - Policy = coaching LLM
       - Reward = this server's /score output (weighted_reward field)
       - Rollout loop: generate transcript → POST /score → backprop via PPO/GRPO

Deploy:
    uvicorn infra.reward_model_server:app --host 0.0.0.0 --port 8000

Or with a real model checkpoint:
    REWARD_MODEL_PATH=data/reward_model_ckpt uvicorn infra.reward_model_server:app
"""

from __future__ import annotations

import os
import random
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Rehearse Reward Model", version="0.1.0")

# ---------------------------------------------------------------------------
# Request / response shapes — mirror the RewardModelScorer contract
# ---------------------------------------------------------------------------


class TranscriptTurn(BaseModel):
    speaker: str  # "user" | "coach"
    text: str


class ScoreRequest(BaseModel):
    transcript: list[TranscriptTurn]
    scenario: dict[str, str]
    dimensions: list[str] = ["emotion_responsiveness", "coaching_trajectory_quality"]


class ScoreResponse(BaseModel):
    emotion_responsiveness: float
    coaching_trajectory_quality: float
    weighted_reward: float


# ---------------------------------------------------------------------------
# Model abstraction
# ---------------------------------------------------------------------------


class RewardModel:
    """Abstract reward model.

    Subclass and override `predict` with a real fine-tuned model.
    The stub implementation returns random scores — replace with a trained
    sequence classification head over deberta-v3-small or qwen2.5-1.5b.
    """

    async def predict(
        self,
        transcript: list[TranscriptTurn],
        scenario: dict[str, str],
        dimensions: list[str],
    ) -> dict[str, float]:
        raise NotImplementedError


class StubRewardModel(RewardModel):
    """Random-score stub. Skeleton for fine-tuning — replace predict() with real inference."""

    async def predict(
        self,
        transcript: list[TranscriptTurn],
        scenario: dict[str, str],
        dimensions: list[str],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for dim in dimensions:
            scores[dim] = round(random.uniform(0.3, 0.9), 3)
        er = scores.get("emotion_responsiveness", 0.5)
        ct = scores.get("coaching_trajectory_quality", 0.5)
        scores["weighted_reward"] = round(0.45 * er + 0.55 * ct, 3)
        return scores


class HuggingFaceRewardModel(RewardModel):
    """Loaded from a fine-tuned HuggingFace checkpoint.

    Expects a sequence classification model with num_labels == len(dimensions).
    Label order must match DIMENSION_ORDER below.

    To produce a compatible checkpoint:
      - Use AutoModelForSequenceClassification with num_labels=2
      - Train with MSELoss against (emotion_responsiveness, coaching_trajectory_quality)
      - model.save_pretrained(path) + tokenizer.save_pretrained(path)
    """

    DIMENSION_ORDER = ["emotion_responsiveness", "coaching_trajectory_quality"]

    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.torch = torch

    def _build_input_text(
        self, transcript: list[TranscriptTurn], scenario: dict[str, str]
    ) -> str:
        scenario_prefix = (
            f"Situation: {scenario.get('situation', '')} | "
            f"Goal: {scenario.get('goal', '')} | "
            f"Counterparty: {scenario.get('counterparty_role', '')} | "
            f"Emotional state: {scenario.get('emotional_state', '')} | "
        )
        turns = " ".join(
            f"[{t.speaker.upper()}] {t.text}" for t in transcript
        )
        return scenario_prefix + turns

    async def predict(
        self,
        transcript: list[TranscriptTurn],
        scenario: dict[str, str],
        dimensions: list[str],
    ) -> dict[str, float]:
        text = self._build_input_text(transcript, scenario)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0]  # shape: (num_labels,)
        # Clamp to [0, 1] — model was trained with MSELoss on [0,1] targets
        raw = self.torch.sigmoid(logits).tolist()

        scores: dict[str, float] = {}
        for i, dim in enumerate(self.DIMENSION_ORDER):
            scores[dim] = round(float(raw[i]) if i < len(raw) else 0.5, 3)

        er = scores.get("emotion_responsiveness", 0.5)
        ct = scores.get("coaching_trajectory_quality", 0.5)
        scores["weighted_reward"] = round(0.45 * er + 0.55 * ct, 3)
        return scores


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(path: str | None = None) -> RewardModel:
    """Load a reward model from disk or fall back to the stub.

    If `path` is None, reads REWARD_MODEL_PATH from the environment.
    If neither is set, returns StubRewardModel.
    """
    model_path = path or os.environ.get("REWARD_MODEL_PATH")
    if model_path:
        return HuggingFaceRewardModel(model_path)
    return StubRewardModel()


_model: RewardModel | None = None


def _get_model() -> RewardModel:
    global _model
    if _model is None:
        _model = load_model()
    return _model


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": type(_get_model()).__name__}


@app.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    if not request.transcript:
        raise HTTPException(status_code=422, detail="transcript must be non-empty")

    model = _get_model()
    try:
        scores = await model.predict(
            transcript=request.transcript,
            scenario=request.scenario,
            dimensions=request.dimensions,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"model inference failed: {exc}") from exc

    return ScoreResponse(
        emotion_responsiveness=scores.get("emotion_responsiveness", 0.0),
        coaching_trajectory_quality=scores.get("coaching_trajectory_quality", 0.0),
        weighted_reward=scores.get("weighted_reward", 0.0),
    )
