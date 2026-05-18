"""Harvest judge.json artifacts into a reward-model training dataset.

Walks evals/runs/*/sessions/*/judge.json, joins each with the sibling
transcript.jsonl and the run's example payload, and writes one JSONL line
per session to data/reward_model_train.jsonl.

Each output line:
  {
    "transcript": [{"speaker": "user"|"coach", "text": "..."}],
    "scenario":   {"situation": "...", "goal": "...",
                   "counterparty_role": "...", "emotional_state": "..."},
    "emotion_responsiveness":     0.8,   # or null
    "coaching_trajectory_quality": 0.7,  # or null
    "weighted_reward":             0.74,
    "source_run":  "20260511T122551-dbe32cfd",
    "example_id":  "vrj-s01-peer-feedback-anxious"
  }

Note: existing runs may use different dimension names
(content_quality / affect_perception / delivery_quality). Those sessions
are still included with emotion_responsiveness and coaching_trajectory_quality
set to null so the row is present but clearly unlabelled for the two
canonical PPO dimensions. The weighted_reward is always harvested.

Usage:
    python infra/harvest_judge_labels.py
    python infra/harvest_judge_labels.py --runs-dir path/to/evals/runs --out data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _load_transcript(session_dir: Path) -> list[dict[str, str]]:
    transcript_path = session_dir / "transcript.jsonl"
    if not transcript_path.exists():
        return []
    rows: list[dict[str, str]] = []
    for line in transcript_path.read_text().splitlines():
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        speaker = frame.get("speaker", "")
        text = frame.get("text", "").strip()
        if speaker and text:
            rows.append({"speaker": speaker, "text": text})
    return rows


def _load_scenario(session_dir: Path, run_dir: Path) -> dict[str, str]:
    """Best-effort extraction of scenario fields.

    Checks intake.json (live sessions) and falls back to provenance / empty.
    """
    intake_path = session_dir / "intake.json"
    if intake_path.exists():
        try:
            intake = json.loads(intake_path.read_text())
            return {
                "situation": intake.get("situation", ""),
                "goal": intake.get("user_goal", ""),
                "counterparty_role": intake.get("counterparty_relationship", ""),
                "emotional_state": "",
            }
        except (json.JSONDecodeError, KeyError):
            pass
    return {"situation": "", "goal": "", "counterparty_role": "", "emotional_state": ""}


def _extract_scores(judge: dict) -> dict[str, float | None]:
    """Extract dimension scores regardless of which judge schema was used.

    Two schemas exist in the corpus:
      A. Legacy LLM judge:
           {"judge_output": {"emotion_responsiveness": {"score": 0.8, ...}, ...},
            "weights": {...}, "weighted_reward": 0.74}

      B. Composite aggregate judge:
           {"per_dim": {"content_quality": 0.62, "affect_perception": 0.55, ...},
            "weights": {...}, "weighted_reward": 0.63, ...}
    """
    result: dict[str, float | None] = {
        "emotion_responsiveness": None,
        "coaching_trajectory_quality": None,
        "weighted_reward": None,
    }

    # weighted_reward is present in both schemas
    if "weighted_reward" in judge:
        try:
            result["weighted_reward"] = float(judge["weighted_reward"])
        except (TypeError, ValueError):
            pass

    # Schema A: judge_output block
    judge_output = judge.get("judge_output")
    if isinstance(judge_output, dict):
        for dim in ("emotion_responsiveness", "coaching_trajectory_quality"):
            block = judge_output.get(dim)
            if isinstance(block, dict) and "score" in block:
                try:
                    result[dim] = float(block["score"])
                except (TypeError, ValueError):
                    pass

    # Schema B: per_dim block — map affect_perception → emotion_responsiveness
    # (best structural analogue; not identical, but useful training signal)
    per_dim = judge.get("per_dim")
    if isinstance(per_dim, dict):
        if result["emotion_responsiveness"] is None and "affect_perception" in per_dim:
            try:
                result["emotion_responsiveness"] = float(per_dim["affect_perception"])
            except (TypeError, ValueError):
                pass
        if result["coaching_trajectory_quality"] is None and "content_quality" in per_dim:
            try:
                result["coaching_trajectory_quality"] = float(per_dim["content_quality"])
            except (TypeError, ValueError):
                pass

    return result


def harvest(runs_dir: Path, out_path: Path) -> None:
    judge_paths = sorted(runs_dir.glob("*/sessions/*/judge.json"))
    if not judge_paths:
        print(f"No judge.json files found under {runs_dir}", file=sys.stderr)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_no_transcript = 0
    n_no_weighted_reward = 0
    er_values: list[float] = []
    ct_values: list[float] = []
    wr_values: list[float] = []

    with out_path.open("w") as fout:
        for judge_path in judge_paths:
            session_dir = judge_path.parent
            run_dir = session_dir.parent.parent  # runs/<run_id>/sessions/<ex_id>/judge.json
            run_id = run_dir.name
            example_id = session_dir.name

            try:
                judge = json.loads(judge_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            scores = _extract_scores(judge)
            if scores["weighted_reward"] is None:
                n_no_weighted_reward += 1
                continue

            transcript = _load_transcript(session_dir)
            if not transcript:
                n_no_transcript += 1
                continue

            scenario = _load_scenario(session_dir, run_dir)

            row: dict = {
                "transcript": transcript,
                "scenario": scenario,
                "emotion_responsiveness": scores["emotion_responsiveness"],
                "coaching_trajectory_quality": scores["coaching_trajectory_quality"],
                "weighted_reward": scores["weighted_reward"],
                "source_run": run_id,
                "example_id": example_id,
            }
            fout.write(json.dumps(row) + "\n")
            n_total += 1

            if scores["emotion_responsiveness"] is not None:
                er_values.append(scores["emotion_responsiveness"])
            if scores["coaching_trajectory_quality"] is not None:
                ct_values.append(scores["coaching_trajectory_quality"])
            if scores["weighted_reward"] is not None:
                wr_values.append(scores["weighted_reward"])

    # Summary
    print(f"Harvested {n_total} examples → {out_path}")
    if n_no_transcript:
        print(f"  Skipped {n_no_transcript} sessions with no transcript.jsonl")
    if n_no_weighted_reward:
        print(f"  Skipped {n_no_weighted_reward} sessions with no weighted_reward")

    def _fmt(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        return (
            f"n={len(vals)}  "
            f"mean={statistics.mean(vals):.3f}  "
            f"median={statistics.median(vals):.3f}  "
            f"min={min(vals):.3f}  "
            f"max={max(vals):.3f}"
        )

    print(f"\nScore distributions:")
    print(f"  emotion_responsiveness:     {_fmt(er_values)}")
    print(f"  coaching_trajectory_quality:{_fmt(ct_values)}")
    print(f"  weighted_reward:            {_fmt(wr_values)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("evals/runs"),
        help="Root directory containing eval run subdirectories (default: evals/runs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reward_model_train.jsonl"),
        help="Output JSONL path (default: data/reward_model_train.jsonl)",
    )
    args = parser.parse_args()
    harvest(args.runs_dir, args.out)


if __name__ == "__main__":
    main()
