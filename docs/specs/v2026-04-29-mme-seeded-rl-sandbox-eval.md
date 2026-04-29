# rehearse — Spec: MME-Seeded RL Sandbox Eval

**Status**: draft (implementation handoff)
**Owner**: jz
**Depends on**:
- `docs/specs/v2026-04-27-eval-harness.md`
- `docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md`

---

## 0. One-line Summary

Add an RLAIF-style eval that uses an MME-Emotion clip as the initial emotional
state for a multi-turn sandbox conversation. Two sandbox agents play out a
conversation: a user simulator conditioned on the clip and a rehearse voice-agent
adapter under test. A judge agent extracts key moments from the transcript,
scores them against a rubric, and emits weighted aggregate metrics.

This does **not** replace the direct `multimodal-llm` MME eval. That eval asks:
"can this audio model classify emotion from speech?" This new eval asks: "given
an emotionally charged opening, does the rehearse agent handle the conversation
well?"

## 1. Why This Exists

MME-Emotion is single-turn emotion classification. Rehearse is a multi-turn
coaching product. We need a bridge between those:

1. MME provides a grounded emotional seed: anger, sadness, frustration, etc.
2. A sandbox rollout turns that seed into a conversation trajectory.
3. An AI judge converts the trajectory into reward-like scores.
4. Those scores can be used for eval dashboards now and preference/RLAIF data
   later.

The design mirrors the evaluator pattern in the Tinker evaluation tutorial:
sampling evaluators generate behavior, compute metrics, and return
`dict[str, float]`. In our harness, the equivalent is:

```text
Dataset example -> Environment rollout -> Scorers -> aggregate_scores
```

The RL interpretation is:

```text
initial_state = MME clip + label + scenario seed
policy = rehearse agent adapter
environment = sandbox user simulator
trajectory = transcript + turn metadata
reward = judge scores + weighted aggregate
```

## 2. Scope

### In Scope

- New eval: `mme-sandbox-rollout`.
- New environment: `sandbox-rollout`.
- New sandbox interface for a user simulator agent and a rehearse agent adapter.
- New trajectory artifact format.
- New judge scorer that extracts key moments and assigns rubric scores.
- Weighted aggregate metrics suitable for RLAIF-style reward tracking.
- CLI flow through existing `rehearse-eval run`.

### Out of Scope

- Real Twilio calls.
- Real Hume EVI streaming.
- Training updates, PPO, GRPO, DPO, or online RL.
- Audio generation for sandbox turns after the first clip.
- Human calibration set for judge reliability.

## 3. Current Direct-Provider Eval vs New Rollout Eval

| Eval | Input | Environment | Output | Answers |
|---|---|---|---|---|
| `mme-emotion` | One MME clip | `multimodal-llm` | emotion label JSON | Can this provider classify emotion? |
| `mme-sandbox-rollout` | One MME clip + scenario seed | `sandbox-rollout` | transcript + judged reward | Can the rehearse agent respond well over a conversation? |

Both use MME, but only the second is product-shaped.

## 4. Architecture

```text
rehearse-eval run --eval mme-sandbox-rollout --environment sandbox-rollout
        |
        v
MMEEmotionSeedDataset
  - clip_path
  - opening emotion label
  - scenario seed
  - rubric weights
        |
        v
SandboxRolloutEnvironment
  1. derives initial affect observation from MME clip
  2. starts sandbox transcript
  3. alternates:
       SandboxUserAgent -> RehearseAgentAdapter
  4. writes trajectory.json
        |
        v
TrajectoryJudgeScorer
  - extracts key moments
  - scores rubric dimensions
  - emits RubricScore rows
        |
        v
Runner aggregation
  - weighted_reward
  - emotion_responsiveness
  - coaching_trajectory_quality
```

## 5. Inputs

### 5.1 Dataset Manifest

Add a new dataset manifest:

```text
evals/datasets/mme-emotion/v0-rollout-seeds/manifest.json
```

Shape:

```json
{
  "_meta": {
    "name": "mme-emotion-rollout-seeds",
    "version": "v0",
    "source": "Karl28/MME-Emotion + rehearse-authored scenario seeds"
  },
  "label_set": [
    "Anger",
    "Sadness",
    "Surprise",
    "Happiness",
    "Excited",
    "Fear",
    "Frustration",
    "Neutral",
    "Other"
  ],
  "rubric_weights": {
    "emotion_responsiveness": 0.45,
    "coaching_trajectory_quality": 0.55
  },
  "examples": [
    {
      "id": "mme-rollout-001",
      "clip": {
        "path": "../v0-10clip/clips/Ses05M_script01_1_F034.mp4",
        "emotion": "Anger",
        "duration_s": 8.2,
        "source_id": "ER_Lab/Ses05M_script01_1_F034.mp4"
      },
      "scenario": {
        "category": "relationship_conflict",
        "user_goal": "raise a recurring issue without escalating",
        "counterparty_role": "partner",
        "counterparty_style": "defensive but reachable",
        "stakes": "user wants to be heard without damaging the relationship"
      },
      "rollout": {
        "max_turns": 8,
        "opening_mode": "clip_conditioned_user_state"
      }
    }
  ]
}
```

### 5.2 Loaded `BenchmarkExample`

The dataset adapter emits:

```python
BenchmarkExample(
    id="mme-rollout-001",
    benchmark="mme-sandbox-rollout",
    payload={
        "clip_path": "evals/datasets/mme-emotion/v0-10clip/clips/...",
        "opening_emotion": "Anger",
        "label_set": [...],
        "scenario": {
            "category": "relationship_conflict",
            "user_goal": "raise a recurring issue without escalating",
            "counterparty_role": "partner",
            "counterparty_style": "defensive but reachable",
            "stakes": "user wants to be heard without damaging the relationship",
        },
        "rollout": {
            "max_turns": 8,
            "opening_mode": "clip_conditioned_user_state",
        },
        "rubric_weights": {
            "emotion_responsiveness": 0.45,
            "coaching_trajectory_quality": 0.55,
        },
    },
    expected={
        "opening_emotion": "Anger",
        "desired_behaviors": [
            "recognize emotional intensity",
            "avoid escalation",
            "ask a grounding question",
            "offer concrete next-step coaching",
        ],
    },
    metadata={
        "source_eval": "mme-emotion",
        "source_id": "ER_Lab/Ses05M_script01_1_F034.mp4",
    },
)
```

## 6. Outputs

### 6.1 Rollout Artifact

The environment writes:

```text
evals/runs/{run_id}/sessions/{example_id}/trajectory.json
```

Shape:

```json
{
  "example_id": "mme-rollout-001",
  "environment": "sandbox-rollout",
  "version": "v0",
  "initial_observation": {
    "clip_path": "...",
    "expected_emotion": "Anger",
    "audio_observation": {
      "provider": "gemini",
      "label": "Anger",
      "reasoning": "The speaker sounds tense and clipped."
    }
  },
  "scenario": {
    "category": "relationship_conflict",
    "user_goal": "raise a recurring issue without escalating"
  },
  "turns": [
    {
      "index": 0,
      "speaker": "user",
      "text": "I am so tired of bringing this up and getting nowhere.",
      "state": {
        "emotion": "Anger",
        "intensity": 0.85
      }
    },
    {
      "index": 1,
      "speaker": "agent",
      "text": "Before you go into the conversation, name the core request in one sentence...",
      "agent_metadata": {
        "model": "claude-sonnet-4-6",
        "latency_ms": 812
      }
    }
  ],
  "completed_reason": "max_turns",
  "errors": []
}
```

### 6.2 RolloutResult Payload

The environment returns:

```python
RolloutResult(
    status="ok",
    artifacts_dir=run_dir,
    payload={
        "trajectory_path": "trajectory.json",
        "turn_count": 8,
        "opening_emotion": "Anger",
        "agent_model": "...",
        "user_simulator_model": "...",
    },
)
```

### 6.3 Scorer Output

The scorer emits one `RubricScore` per metric:

```text
emotion_responsiveness
coaching_trajectory_quality
weighted_reward
```

The runner already aggregates each dimension by mean across examples.

## 7. Top Two Evaluation Metrics

### 7.1 `emotion_responsiveness`

**Question**: Did the agent correctly perceive and adapt to the user's emotional
state?

Scale: 0.0 to 1.0.

Judge criteria:

- Recognizes the opening emotion or emotional intensity.
- Responds in a way that matches the affective state.
- Avoids invalidating, minimizing, or mislabeling the emotion.
- Adjusts when the simulated user escalates or softens.

Examples:

```text
1.0 — Names/uses the emotional signal, validates it, and adapts over turns.
0.5 — Generic empathy; not harmful, but not clearly grounded in the emotional state.
0.0 — Misreads the emotion, escalates, dismisses, or ignores the affective signal.
```

### 7.2 `coaching_trajectory_quality`

**Question**: Did the multi-turn conversation move the user toward a better
real-world conversation?

Scale: 0.0 to 1.0.

Judge criteria:

- Helps the user clarify the ask.
- De-escalates without suppressing the user's concern.
- Gives concrete, speakable phrasing.
- Maintains appropriate pacing and does not over-talk.
- Tracks the scenario goal across turns.

Examples:

```text
1.0 — The user ends with a clearer, calmer, actionable plan.
0.5 — Some useful coaching, but generic or uneven across turns.
0.0 — Rambling, unsafe, invalidating, or not useful for the scenario.
```

### 7.3 Weighted Reward

Default:

```text
weighted_reward =
  0.45 * emotion_responsiveness
+ 0.55 * coaching_trajectory_quality
```

Reasoning:

- Emotion responsiveness is necessary but not sufficient.
- The product outcome is better coaching over the trajectory.

## 8. Interface Between Eval Harness and Voice Agent Sandbox

The eval harness should not import Twilio, Hume, FastAPI, or live runtime IO.
It should depend on a narrow sandbox protocol.

### 8.1 Sandbox Protocols

New module:

```text
rehearse/eval/sandboxes/base.py
```

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SandboxScenario:
    category: str
    user_goal: str
    counterparty_role: str
    counterparty_style: str
    stakes: str


@dataclass(frozen=True)
class InitialAffect:
    clip_path: Path
    expected_emotion: str
    inferred_emotion: str | None = None
    reasoning: str | None = None
    intensity: float | None = None


@dataclass
class SandboxTurn:
    index: int
    speaker: str                  # "user" | "agent"
    text: str
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxTrajectory:
    example_id: str
    initial_affect: InitialAffect
    scenario: SandboxScenario
    turns: list[SandboxTurn]
    completed_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SandboxUserAgent(Protocol):
    name: str
    version: str

    async def next_user_turn(
        self,
        scenario: SandboxScenario,
        initial_affect: InitialAffect,
        turns: list[SandboxTurn],
    ) -> SandboxTurn: ...


@runtime_checkable
class RehearseSandboxAgent(Protocol):
    name: str
    version: str

    async def next_agent_turn(
        self,
        scenario: SandboxScenario,
        initial_affect: InitialAffect,
        turns: list[SandboxTurn],
    ) -> SandboxTurn: ...
```

### 8.2 Adapter Boundary

New module:

```text
rehearse/eval/sandboxes/rehearse_agent.py
```

Responsibilities:

- Wrap the current non-live rehearse agent logic.
- Accept text turns plus `InitialAffect`.
- Return one agent text turn.
- Avoid live phone IO.
- Avoid Hume/Twilio dependencies.

The adapter can initially call a text model directly with the same coaching
prompts used in runtime. Later, it can wrap more of the real session state
machinery as runtime stabilizes.

### 8.3 User Simulator Boundary

New module:

```text
rehearse/eval/sandboxes/user_simulator.py
```

Responsibilities:

- Generate user turns conditioned on:
  - scenario goal,
  - opening emotion,
  - previous agent response,
  - desired escalation/softening curve.
- Keep the user behavior stable enough for repeatable evals.
- Support deterministic temperature-0 mode by default.

## 9. Code Changes

### 9.1 Dataset

Add:

```text
rehearse/eval/datasets/mme_rollout_seeds.py
evals/datasets/mme-emotion/v0-rollout-seeds/manifest.json
```

Register in:

```text
rehearse/eval/datasets/__init__.py
```

### 9.2 Eval

Add:

```text
rehearse/eval/evals/mme_sandbox_rollout.py
```

```python
class MMESandboxRolloutEval:
    name = "mme-sandbox-rollout"
    version = "v0"
    supported_environments = frozenset({"sandbox-rollout"})
    preferred_environment = "sandbox-rollout"

    def __init__(self) -> None:
        self.dataset = MMERolloutSeedDataset()

    def scoring_plan(self) -> list[Scorer]:
        return [TrajectoryJudgeScorer()]

    def rollout_timeout_s(self) -> int:
        return 180
```

Register in:

```text
rehearse/eval/evals/__init__.py
```

### 9.3 Environment

Add:

```text
rehearse/eval/environments/sandbox_rollout.py
```

Responsibilities:

1. Parse the example payload.
2. Build `InitialAffect`.
3. Optionally run direct audio classifier to infer initial emotion.
4. Instantiate `SandboxUserAgent`.
5. Instantiate `RehearseSandboxAgent`.
6. Alternate turns up to `max_turns`.
7. Write `trajectory.json`.
8. Return `RolloutResult`.

Register in:

```text
rehearse/eval/environments/__init__.py
```

### 9.4 Scorer

Add:

```text
rehearse/eval/scorers/trajectory_judge.py
```

The scorer reads `trajectory.json`, calls an LLM judge, and emits:

```text
emotion_responsiveness
coaching_trajectory_quality
weighted_reward
```

Judge output schema:

```json
{
  "key_moments": [
    {
      "turn_index": 3,
      "speaker": "agent",
      "moment": "Agent redirects from blame to a concrete ask",
      "scores": {
        "emotion_responsiveness": 0.8,
        "coaching_trajectory_quality": 0.7
      },
      "rationale": "The response validates anger while moving toward a speakable request."
    }
  ],
  "aggregate": {
    "emotion_responsiveness": 0.8,
    "coaching_trajectory_quality": 0.72,
    "weighted_reward": 0.756
  }
}
```

### 9.5 CLI

No new CLI primitives are required.

Example:

```bash
uv run rehearse-eval run \
  --eval mme-sandbox-rollout \
  --environment sandbox-rollout \
  --limit 10 \
  --concurrency 2 \
  --model-slot sandbox_user=claude-sonnet-4-6 \
  --model-slot rehearse_agent=claude-sonnet-4-6 \
  --model-slot judge=claude-opus-4-7
```

## 10. Phasing

### Phase RLE1 — Skeleton Rollout, No Model Calls

- Add dataset/eval/environment registrations.
- Add fake deterministic user simulator.
- Add fake deterministic rehearse agent.
- Write `trajectory.json`.
- Scorer returns fixed scores.
- Tests prove runner can execute the full shape.

### Phase RLE2 — Real User Simulator + Rehearse Agent Adapter

- Implement LLM-backed `SandboxUserAgent`.
- Implement `RehearseSandboxAgent`.
- Add prompt fixtures.
- Tests use mocked model clients.
- Manual demo produces plausible 8-turn transcript.

### Phase RLE3 — Judge Scorer

- Implement `TrajectoryJudgeScorer`.
- Parse structured judge JSON.
- Emit the two top metrics plus weighted reward.
- Store judge artifact beside trajectory.

### Phase RLE4 — RLAIF Data Export

- Export trajectories + scores as JSONL:

```text
evals/runs/{run_id}/rlaif_trajectories.jsonl
```

- Future DPO export can pair two trajectories from the same seed.

## 11. Open Questions

| # | Question | Recommendation |
|---|---|---|
| Q1 | Should the opening clip be transcribed? | Yes, but store transcription as observation metadata, not ground truth. |
| Q2 | Should the user simulator know the exact MME label? | Yes in v0; it is the intended hidden state. Hide it from the rehearse agent except through inferred affect. |
| Q3 | Should the rehearse agent receive raw audio? | Not in RLE1/RLE2. It receives an affect observation. A later audio-native agent adapter can receive the clip. |
| Q4 | Which judge model? | Claude Opus by default for stability; make it a model slot. |
| Q5 | Can this train directly? | Not yet. First produce stable eval metrics and trajectory artifacts. Then export preferences/rewards. |

## 12. Success Criteria

The implementation is complete when:

1. `uv run rehearse-eval run --eval mme-sandbox-rollout --environment sandbox-rollout --limit 1`
   writes a transcript-like `trajectory.json`.
2. `summary.md` contains:
   - `emotion_responsiveness`
   - `coaching_trajectory_quality`
   - `weighted_reward`
3. The run is deterministic enough that repeated temperature-0 runs on the same
   seed produce comparable scores.
4. The direct `mme-emotion` provider eval still works unchanged.

