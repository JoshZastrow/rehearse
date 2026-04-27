"""Persona definitions and compilation.

Two concerns:
  1. Coach persona — a constant system prompt for Phase 1 intake and Phase 3
     feedback. Single source, version-controlled, diffable.
  2. Character compilation — a pure function `compile_character(intake) →
     CounterpartyPersona` that renders an intake into a system prompt for the
     Phase 2 voice.

Pure data and pure functions. No model calls, no I/O. Easy to unit-test and
review without running the service.
"""
