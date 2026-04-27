"""Deterministic scorers — the load-bearing dimensions.

Each scorer is a pure function over a frozen session + an ExampleScenario:

  pacing_adherence(session)            → RubricScore
  feedback_groundedness(session)       → RubricScore   (citations resolvable)
  prosody_citation_accuracy(session)   → RubricScore   (cited moments exist)
  fault_recall(session, example)       → RubricScore   (taxonomy match in feedback)
  fault_precision(session, example)    → RubricScore   (no hallucinated faults)
  incongruence_detection(session, ex)  → RubricScore   (word/prosody mismatch cited)

These are the dimensions with objective ground truth. If every other scorer
breaks, these alone carry the signal.
"""
