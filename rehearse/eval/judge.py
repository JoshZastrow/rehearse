"""LLM-as-judge for subjective rubric dimensions.

  judge(session, example) → list[RubricScore]

Covers dimensions that lack deterministic ground truth:
  - INTAKE_FIDELITY
  - CHARACTER_PERSONA_FIDELITY
  - CHARACTER_BELIEVABILITY
  - USEFULNESS_HOLISTIC

The judge is given the example scenario (for ground truth context) and the
full session artifacts. Outputs structured JSON validated against RubricScore.

Calibration: judge-vs-gold correlation is computed on each run. Dimensions
below ρ=0.7 are flagged before scores are trusted.
"""
