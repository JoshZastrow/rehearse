"""Inspect a rollout: per-example transcript, audio paths, scorer rationales.

Usage:
    uv run python scripts/inspect_rollout.py <run_id> [--example <id>]
                                              [--runs-root evals/runs]
                                              [--show-rationales/--no-rationales]

Prints:
  - Aggregate scores summary across examples (mean, p50, count by dimension).
  - Per-example block: transcript turns, audio file presence + duration,
    each scorer row with rationale, and the aggregate weighted_reward.

Pure read-only tooling — never mutates run artifacts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import wave
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--runs-root", default="evals/runs")
    ap.add_argument("--example", default=None)
    ap.add_argument("--no-rationales", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.runs_root) / args.run_id
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}", file=sys.stderr)
        return 2

    results = _read_results(run_dir / "results.jsonl")
    by_example: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_example[row["example_id"]].append(row)

    print(f"# Run {args.run_id}")
    _print_aggregate(results)
    if args.summary_only:
        return 0

    sessions = run_dir / "sessions"
    target_ids = (
        [args.example] if args.example else sorted(by_example)
    )
    for ex_id in target_ids:
        rows = by_example.get(ex_id) or []
        ex_dir = sessions / ex_id
        _print_example(ex_id, rows, ex_dir, show_rationales=not args.no_rationales)
    return 0


def _read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _print_aggregate(rows: list[dict]) -> None:
    by_dim: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_dim[r["dimension"]].append(float(r["value"]))
    print("\n## Aggregate")
    print(f"{'dimension':<38} {'count':>5} {'mean':>7} {'p50':>7} {'min':>6} {'max':>6}")
    for dim in sorted(by_dim):
        vs = by_dim[dim]
        if not vs:
            continue
        print(
            f"{dim:<38} {len(vs):>5} "
            f"{statistics.fmean(vs):>7.3f} "
            f"{statistics.median(vs):>7.3f} "
            f"{min(vs):>6.2f} {max(vs):>6.2f}"
        )


def _print_example(
    ex_id: str,
    rows: list[dict],
    ex_dir: Path,
    *,
    show_rationales: bool,
) -> None:
    print(f"\n## Example: {ex_id}")
    transcript = ex_dir / "transcript.jsonl"
    if transcript.exists():
        print("  Transcript:")
        for i, line in enumerate(transcript.read_text().splitlines()):
            if not line.strip():
                continue
            f = json.loads(line)
            spk = f.get("speaker", "?")
            text = (f.get("text") or "").replace("\n", " ")
            print(f"    [{i}] {spk:>5}: {text}")
    else:
        print("  (no transcript)")

    for role in ("user", "coach"):
        d = ex_dir / "audio" / role
        if not d.exists():
            continue
        wavs = sorted(d.glob("turn_*.wav"))
        durs = [_wav_duration_s(p) for p in wavs]
        joined = ", ".join(f"{p.name}={dur:.1f}s" for p, dur in zip(wavs, durs))
        print(f"  Audio[{role}]: {joined or '(none)'}")

    print("  Scores:")
    for r in sorted(rows, key=lambda r: r["dimension"]):
        flags = ",".join(r.get("flags") or [])
        flag_str = f"  [{flags}]" if flags else ""
        line = (
            f"    {r['dimension']:<38} "
            f"{float(r['value']):>6.3f}  "
            f"({r.get('scorer','?')}, {r.get('modality','?')}){flag_str}"
        )
        print(line)
        if show_rationales and r.get("rationale"):
            rationale = str(r["rationale"]).strip().replace("\n", " ")
            if len(rationale) > 200:
                rationale = rationale[:200] + "…"
            print(f"        ↳ {rationale}")


def _wav_duration_s(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate() or 1)
    except Exception:
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
