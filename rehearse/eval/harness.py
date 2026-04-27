"""Eval harness orchestrator.

  run_eval(example_ids: list[str], pipeline_version: str) → EvalRun

For each example:
  1. Build a SyntheticUser from the scenario's SyntheticUserProfile.
  2. Build the same pipeline production uses, but against a SimulatedTransport
     that consumes the synthetic user's transcript+prosody frame stream.
  3. Let the pipeline run to completion; freeze the session artifacts.
  4. Run deterministic scorers (scorers.py) and the LLM judge (judge.py).
  5. Emit RubricScore rows to evals/runs/{run_id}/results.jsonl.

Tier-1 (scripted prosody), tier-2 (TTS→Hume), and tier-3 (human recordings)
are selected per example based on its configuration.
"""
