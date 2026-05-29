# Rehearse Package Restructure

**Date:** 2026-05-26  
**Status:** Approved

## Problem

`rehearse/` has 25 flat Python files alongside 9 existing subdirectories. Files from unrelated domains (session orchestration, HTTP routing, conversation state machine, caller memory) sit side-by-side, making the package hard to navigate and the domain boundaries invisible.

## Goals

- Group flat files into thematic subpackages with clear, single-purpose boundaries
- Preserve all existing behavior — no logic changes
- Update all imports across `rehearse/`, `tests/`, and `scripts/` to match new paths
- Update `README.md` with the new directory structure and per-directory explanations
- Verify the test suite passes after the move

## Non-Goals

- Changing any business logic
- Restructuring the already-organized subdirectories (`agents/`, `audio/`, `backends/`, `eval/`, `personas/`, `services/`, `transports/`, `writers/`)
- Creating new directories for single files

## Proposed Structure

### New subpackages (from flat files)

| Package | Files | Purpose |
|---|---|---|
| `rehearse/session/` | `session.py`, `conversation.py`, `runtime.py`, `finalize_sweeper.py`, `synthesis.py` | Call lifecycle — from boot to finalization and post-call artifact generation |
| `rehearse/phases/` | `phases.py`, `phases_llm.py`, `intake.py`, `consent.py`, `outcome.py`, `survey.py` | Conversation flow state machine — all per-turn and phase-transition logic |
| `rehearse/memory/` | `memory.py`, `memory_manager.py` | Caller memory — protocol definitions and implementations |
| `rehearse/api/` | `app.py`, `telephony.py`, `viewer.py` | HTTP layer — FastAPI app wiring, Twilio webhooks, session viewer |

### Merged into existing dirs

| File | Destination | Reasoning |
|---|---|---|
| `new_clm_responder.py` | `agents/new_clm_responder.py` | CLM orchestration belongs with other agents |
| `personas.py` | `personas/builder.py` | Persona builder logic; renamed to resolve name conflict with the `personas/` directory |
| `transport.py` | `backends/transport.py` | RuntimeTransport is the duplex transport abstraction — same layer as LLM backends |
| `participants.py` | `audio/participants.py` | VoiceParticipant contracts are live audio actor interfaces |

### Stay flat

`bus.py`, `frames.py`, `types.py`, `config.py`, `storage.py`, `pipeline.py` — foundational files with no natural grouping partner, or too widely imported to reorganize without disproportionate churn.

## Import Update Strategy

For each moved file, update every `from rehearse.<old_module> import ...` reference in:
1. All files within `rehearse/` (internal imports)
2. All files in `tests/`
3. All files in `scripts/`
4. Any other top-level consumers (evals, infra)

Each new subpackage gets an `__init__.py`. No re-exports — callers import from the canonical new path.

`personas.py` → `personas/builder.py` requires extra care: the existing `personas/__init__.py` already imports from within the `personas/` package; the builder functions (`build_intake_record`, `compile_character`) need to be re-pointed there.

## README Update

Add a `## Project Structure` section documenting:
- Top-level directories (`rehearse/`, `tests/`, `evals/`, `scripts/`, `infra/`, `web/`, `docs/`)
- Within `rehearse/`, each subpackage with a one-line description

## Verification

1. Run `uv run pytest` after all moves — full suite must pass
2. Run `uv run mypy rehearse/` if configured — no new errors
3. Confirm no `from rehearse.<old_path>` references remain via grep
