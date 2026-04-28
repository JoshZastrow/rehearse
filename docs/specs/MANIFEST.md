# rehearse — Specs Manifest

This manifest is the routing table for specs. Before committing to work, check this
file first, then read only the specs marked `implementation` or explicitly named by
the phase you are building.

## Status Vocabulary

| Status | Meaning | Commit guidance |
|---|---|---|
| `acknowledged` | Accepted as project direction, but no committed implementation work yet. | Safe to plan against. Mark `wip` when a PR/branch starts implementing it. |
| `wip` | Committed implementation work exists, but the spec is not fully delivered. | Read before changing affected code. Update phase notes as work lands. |
| `done` | Delivered, frozen, or resolved. | Use as reference. Do not reopen without a new amendment spec. |
| `superseded` | Replaced by a newer spec or decision. | Do not implement from it. Read only for historical context. |

## Read Policy

| Policy | Meaning |
|---|---|
| `foundation` | Stable baseline that frames the whole project. |
| `implementation` | Active implementation source of truth. Read before committing related work. |
| `amendment` | Modifies one or more implementation specs. Read alongside the affected spec. |
| `historical` | Preserved decision record. Do not use as a build handoff. |

## Current Manifest

| Spec | Status | Policy | Applies to | Notes |
|---|---|---|---|---|
| [`../../SPEC.md`](../../SPEC.md) | `done` | `foundation` | Whole product | Foundational design, treated as frozen unless a new amendment says otherwise. |
| [`v2026-04-27-eval-harness.md`](v2026-04-27-eval-harness.md) | `wip` | `implementation` | Eval harness | Phases 1-2 have shipped. Later eval phases remain open. |
| [`v2026-04-28-mme-emotion-and-audio-targets.md`](v2026-04-28-mme-emotion-and-audio-targets.md) | `acknowledged` | `implementation` | Eval phases A1-A6 | Next active eval direction. Supersedes EQ-Bench as the primary eval path. |
| [`v2026-04-27-runtime.md`](v2026-04-27-runtime.md) | `acknowledged` | `implementation` | Runtime phases R1-R7 | Read with the Drop Pipecat amendment. Sections C3, C5, and C7 are no longer authoritative. |
| [`v2026-04-28-drop-pipecat.md`](v2026-04-28-drop-pipecat.md) | `acknowledged` | `amendment` | Runtime phases R2-R7, eval simulated transport | Authoritative replacement for Pipecat-shaped runtime pieces. |
| [`v2026-04-28-hume-evi-bridge.md`](v2026-04-28-hume-evi-bridge.md) | `superseded` | `historical` | Runtime R2 decision history | Kept only to explain the bridge decision. Do not implement from it. |

## Workstream Map

| Workstream | Active specs to read | Ignore for implementation |
|---|---|---|
| Eval harness maintenance | `v2026-04-27-eval-harness.md` | `v2026-04-28-hume-evi-bridge.md` |
| Audio-native eval work | `v2026-04-27-eval-harness.md`, `v2026-04-28-mme-emotion-and-audio-targets.md` | `v2026-04-28-hume-evi-bridge.md` |
| Runtime R1 | `v2026-04-27-runtime.md` | `v2026-04-28-hume-evi-bridge.md` |
| Runtime R2-R7 | `v2026-04-27-runtime.md`, `v2026-04-28-drop-pipecat.md` | `v2026-04-28-hume-evi-bridge.md`; superseded runtime sections C3, C5, C7 |
| ML data pipeline | `../../SPEC.md` | No dedicated spec yet. Write one before implementation. |

## Update Rules

1. Add every new spec to the manifest in the same PR that adds the spec.
2. Move a spec to `acknowledged` once it is accepted as build direction.
3. Move a spec to `wip` when committed implementation work begins.
4. Move a spec to `done` only when its acceptance criteria are delivered and verified.
5. Mark older specs `superseded` instead of deleting them when a decision record is useful.
6. When a spec amends another spec, name the superseded sections in both files and in this manifest.

