"""Round-trip tests for pydantic contracts.

Establishes that every domain, eval, training, and telemetry type serializes
and deserializes without loss. These are the cheapest tests that assert the
schema exists and is self-consistent; they will run before any application
logic lands.
"""
