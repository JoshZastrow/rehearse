"""PhaseProcessor — the one stateful runtime phase controller.

Owns the three-phase state machine (INTAKE → PRACTICE → FEEDBACK). Consumes:
  - wall-clock tick frames
  - transcript frames (for soft phase-transition cues)
  - control frames

Emits:
  - PhaseTransitionFrame on phase boundary
  - PersonaSwitchFrame when the voice must swap (coach ↔ character)

Tested in isolation by feeding a stream of input frames and asserting the
output frame sequence (see tests/test_phases.py). No external I/O — no Hume,
no Claude, no filesystem.
"""
