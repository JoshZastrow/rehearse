---
name: hume-config
description: Use when editing Hume EVI personas (voice, prompt, greeting, timeouts, turn detection, nudges, interruption) or when "rehearse-hume", "PERSONAS", "Hume config", "voice swap", or persona routing comes up. Walks through the declare-in-code → diff → sync workflow and the safe edit patterns.
---

# Hume EVI configs as code

The Hume workspace is **not** the source of truth for this repo. The registry in
`rehearse/services/hume_configs.py` is. The `rehearse-hume` CLI reconciles the
workspace against it. Edits made in the Hume console drift and will be
overwritten by the next `sync`.

## When to invoke this skill

- The user wants to change voice, greeting line, prompt text, timeouts, turn
  detection, nudges, or interruption sensitivity for the live coach.
- The user mentions `rehearse-hume`, `PERSONAS`, "Hume config", "voice swap",
  "EVI config", or persona routing.
- The user asks to add a new persona (e.g., "relationship coach", "career
  coach") to the runtime.
- The user reports CLI errors from `rehearse-hume diff` or `sync`.
- The user is editing `rehearse/services/hume_configs.py` or
  `rehearse/services/hume_configs_cli.py`.

## Mental model

```
rehearse/services/hume_configs.py  ─┐
                                    ├─→  rehearse-hume sync  ─→  Hume API
PERSONAS = {"default": ...}        ─┘                              │
                                                                   ▼
sessions/.hume_configs.json  ←──── persona_key → config_id mapping
                                   (read by the runtime via select_config_id)
```

- `PERSONAS` is a `dict[str, HumePersonaConfig]` — one entry per persona key.
- `plan_sync(personas, remote)` is pure: returns `Create` / `NewVersion` /
  `NoOp` actions.
- `apply_sync(client, actions)` calls Hume and writes the mapping file
  atomically.
- Match key between repo and Hume is `display_name` (exact, case-sensitive).
- Hume mutates by **versioning**, not in-place update. `sync` calls
  `create_config_version` for an existing config, never `update_config_*`.

## Core workflow: change something on the default persona

1. Edit `PERSONAS["default"]` in `rehearse/services/hume_configs.py`.
   Change the field you care about — voice name, prompt text, timeout secs,
   etc. Pre-existing fields stay the same.
2. Run `uv run rehearse-hume diff`. Confirm it prints
   `NEW_VERSION default (cfg_id) diff=[...field names...]` and the diff list
   is exactly the fields you changed (nothing extra).
3. Run `uv run rehearse-hume sync`. It posts a new version and rewrites
   `sessions/.hume_configs.json` with `synced_at` updated.
4. Place a test call to confirm the change took effect. Hume picks up the new
   version on the next call automatically — no service restart needed.

## Adding a new persona

1. Add an entry to `PERSONAS` keyed by a stable string (e.g.,
   `"relationship_coach"`). All fields in `HumePersonaConfig` are required
   except those with defaults; copy from `"default"` and override what
   differs.
2. Pick a unique `display_name` (e.g., `"Rehearse Coach (relationship)"`) —
   this is the join key against Hume.
3. `uv run rehearse-hume diff` — should print
   `CREATE relationship_coach (Rehearse Coach (relationship))`.
4. `uv run rehearse-hume sync` — creates the config and adds it to the
   mapping file.
5. Write a test in `tests/test_hume_configs.py` asserting the new key is in
   `PERSONAS` with the expected fields, mirroring `test_default_persona_is_registered`.

The runtime does not yet route to non-default personas — wiring
`select_config_id(persona_key)` into `HumeEVIClient._connect` is a separate
follow-up. Adding a persona to `PERSONAS` is safe; it just sits available.

## Reading the diff output

```
NOOP default (cfg_abc)                     # in sync, no action
CREATE relationship_coach (Rehearse Coach (relationship))   # not in workspace
NEW_VERSION default (cfg_abc) diff=['voice', 'prompt_text']  # drift detected
```

`diff` exits 1 if any line is `CREATE` or `NEW_VERSION`. Use this in CI to gate
on "workspace will drift if this PR lands".

## Schema reference

The fields available on `HumePersonaConfig` (see the module for canonical
defaults):

| Field | Type | Notes |
|---|---|---|
| `persona_key` | `str` | Stable key used in `PERSONAS` and the mapping file |
| `display_name` | `str` | Join key against Hume; must be unique per persona |
| `evi_version` | `str` | Default `"4-mini"` |
| `voice` | `HumeVoice(name, id, provider)` | One of `name` or `id`; provider `HUME_AI` or `CUSTOM_VOICE` |
| `language_model` | `HumeLanguageModel(provider, model, temperature)` | Hume routes assistant turns to this model. The repo's CLM webhook is bypassed unless provider is `CUSTOM_LANGUAGE_MODEL` |
| `prompt_text` | `str` | Inline prompt; rendered to a `PostedConfigPromptSpec` |
| `on_new_chat`, `on_max_duration_timeout`, `on_inactivity_timeout` | `HumeEventMessage \| None` | Each: `enabled` + `text` |
| `timeouts` | `HumeTimeouts(max_duration_secs, inactivity_secs)` | |
| `turn_detection` | `HumeTurnDetection(end_of_turn_silence_ms, prefix_padding_ms, speech_detection_threshold)` | |
| `interruption_min_ms` | `int` | Min duration of user speech that interrupts the assistant |
| `nudges_enabled`, `nudges_interval_secs` | `bool`, `int` | |
| `builtin_tools` | `list[str]` | Names like `"web_search"`, `"hang_up"` |

`on_resume_chat` is **not** in the schema — Hume's `PostedEventMessageSpecs`
write API doesn't accept it.

## Safe edit patterns

- **Voice swap:** change `voice=HumeVoice(name="Inspiring Woman")` →
  `voice=HumeVoice(name="<other-voice>")`. Names come from the Hume voice
  library; verify the new name exists before sync.
- **Prompt edit:** edit the `_DEFAULT_PROMPT` string at module top, not the
  inline registry entry. Keep `\n` newlines explicit so the prompt round-trips
  cleanly through Hume's API and the diff stays small.
- **Timeout change:** `timeouts=HumeTimeouts(max_duration_secs=240, ...)`.
  Note the 5-minute Hume cap (300s) is the model-provider's hard limit; never
  go above it. The time-aware CLM (`rehearse/agents/timecard.py`) reads
  `HARD_CAP_SECONDS=300`.
- **Turn detection tuning:** lowering `end_of_turn_silence_ms` makes the
  coach respond faster (more interruptions); raising it makes it more
  patient. Default 500ms.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `AttributeError: 'NoneType' object has no attribute '...'` from `_snapshot_from_remote` | A workspace config matches a declared `display_name` but has nullable subobjects (e.g. `turn_detection=None`) | The `fetch_remote_configs` filter only snapshots configs whose name matches a persona, so this only fires if a managed config has a partial shape. Check the live config in the Hume console. |
| `diff` keeps showing `NEW_VERSION` after `sync` | A schema field isn't being rendered in `_to_create_kwargs` | Inspect `_to_create_kwargs` — every field in `_COMPARED_FIELDS` must be POSTed. |
| `CREATE` instead of `NEW_VERSION` on first run | The Hume `display_name` doesn't match the registry's | Rename in the console (preserves the existing config id), or accept the duplicate. |
| `RuntimeError: Missing required env vars: HUME_API_KEY` | `.env` not loaded or key not set | `cp .env.example .env`, fill in `HUME_API_KEY`. |

## Don't

- Don't edit configs in the Hume console for managed personas. Changes will
  be overwritten by the next `sync`.
- Don't call `client.empathic_voice.configs.update_config_*` from runtime
  code. Only `name`/`description` are mutable in place; behavior changes go
  through `create_config_version`.
- Don't add a field to `HumePersonaConfig` without also adding it to
  `_COMPARED_FIELDS` and rendering it in `_to_create_kwargs`. Missing one
  causes drift loops; missing the other means `diff` won't catch real drift.
- Don't commit `sessions/.hume_configs.json` — it's gitignored under
  `sessions/`. Each developer/environment syncs into its own workspace and
  ends up with its own mapping.

## Spec and plan

- Spec: `docs/specs/v2026-05-06-hume-config-as-code.md`
- Plan: `docs/plans/2026-05-06-hume-config-as-code.md`
