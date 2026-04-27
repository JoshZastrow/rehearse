"""PhaseProcessor frame-level tests.

Feed a scripted sequence of input frames (ticks + transcripts), assert the
output sequence (phase transitions, persona switches). No Hume, no Claude,
no filesystem — the processor must be fully testable in isolation.
"""
