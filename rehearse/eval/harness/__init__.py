"""Eval harness infrastructure — run plumbing that is not the system under test.

executor  — in-process and subprocess rollout executors
report    — DuckDB persistence and Rich terminal reporting
stream    — append-only scores.jsonl writer for live score streaming
watch     — live score watcher (tails scores.jsonl)
worker    — subprocess worker entry point
"""
