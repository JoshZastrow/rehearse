"""Generate synthetic coaching scenarios for `voice-rollout-judges`.

One Anthropic call produces N scenarios in JSONL format compatible with
`evals/datasets/voice-rollout-judges/v1/scenarios.jsonl`. The prompt asks
for diversity across counterparty type, valence, arousal, stakes, and
coaching difficulty so the resulting set has a healthy hard/easy mix.

Usage:
    set -a && source .env && set +a
    uv run python scripts/generate_scenarios.py --count 90 \
        --out evals/datasets/voice-rollout-judges/v1/scenarios.jsonl \
        --start-id 11 --append

The script never deletes existing rows. With `--append` it concatenates
new rows after a blank line; without it, the script writes to a new file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path


_PROMPT = """Generate {count} synthetic coaching-call scenarios as a JSON array. Each
scenario is a person about to have a hard real-life conversation, calling a
coach to rehearse it. The scenarios train an evaluation harness that judges
how well a coach handles the call — so we need a wide difficulty range.

Constraints:
- Diverse counterparty types: peer, manager, direct report, parent, sibling,
  spouse/partner, friend, ex, customer/vendor, child, in-law, therapist,
  self/inner deliberation, doctor, landlord, neighbor, board member.
- Diverse valences and arousals: mix anxious/avoidant, flat/withdrawn,
  escalating/frustrated, defensive, numb, grieving, conflicted, eager,
  relieved, ashamed, suspicious, brittle, tender.
- Diverse stakes: career, relationship, money, health, safety, identity,
  legal, family.
- ~30% should be HARD scenarios where competent-but-generic coaching would
  fail (defensive counterparty, conflicting goals, time pressure, the user
  is wrong about what they want). ~50% medium. ~20% easy.
- Specific, attributable scenarios — no generic "I need to talk to my boss".
  Include a concrete trigger, a concrete number, or a concrete detail.

For each scenario produce a JSON object with this exact shape:
  {{
    "id": "vrj-sNN-<3-word-slug>",          // NN matches array index, slug uses lowercase-hyphens
    "difficulty": "easy"|"medium"|"hard",
    "scenario": {{
      "situation": "<one sentence>",
      "goal": "<what the user wants out of the rehearsal>",
      "counterparty_role": "<who they'll talk to>",
      "counterparty_style": "<2 short clauses>",
      "stakes": "<what's on the line>",
      "emotional_state": "<3-5 word phrase, e.g. 'anxious, hesitant, soft-spoken'>"
    }},
    "expected": {{"opening_emotion": "<single word>"}}
  }}

The starting NN for ids is {start_id:02d}. Increment NN by 1 for each entry.

Respond with ONLY a JSON array of {count} objects, nothing else."""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=90)
    ap.add_argument("--start-id", type=int, default=11)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key)

    parsed: list = []
    remaining = args.count
    next_id = args.start_id
    while remaining > 0:
        chunk = min(args.batch_size, remaining)
        prompt = _PROMPT.format(count=chunk, start_id=next_id)
        resp = await client.messages.create(
            model=args.model,
            max_tokens=8000,
            system="You output only the JSON array. No prose, no markdown fences.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        try:
            batch = _parse_json_array(text)
        except Exception as exc:
            print(f"batch starting id {next_id} failed: {exc}", file=sys.stderr)
            print(text[:800], file=sys.stderr)
            return 1
        if not isinstance(batch, list):
            print(f"unexpected batch shape: {type(batch).__name__}", file=sys.stderr)
            return 1
        parsed.extend(batch)
        next_id += chunk
        remaining -= chunk
        print(f"  batch ok: +{len(batch)} (cumulative={len(parsed)})", file=sys.stderr)

    rows = []
    for scenario in parsed:
        if not isinstance(scenario, dict):
            continue
        scenario.setdefault(
            "rubric_weights",
            {
                "content_quality": 0.40,
                "affect_perception": 0.35,
                "delivery_quality": 0.15,
                "naturalness.interruption_rate": 0.025,
                "naturalness.silence_after_affect": 0.025,
                "naturalness.speech_rate_band": 0.05,
            },
        )
        rows.append(scenario)

    mode = "a" if args.append else "w"
    with args.out.open(mode) as f:
        if args.append and args.out.stat().st_size > 0:
            f.write("\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} scenarios to {args.out}")
    return 0


def _parse_json_array(text: str):
    text = text.strip()
    # Strip leading/trailing markdown fences if Claude added them anyway.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
