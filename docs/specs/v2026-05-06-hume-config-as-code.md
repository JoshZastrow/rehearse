# Hume EVI configs as code

**Status:** draft
**Date:** 2026-05-06
**Owner:** Josh Zastrow
**Related code:** `rehearse/services/hume_evi.py`, `rehearse/config.py`

## Goal

Replace manual Hume console click-ops with a declarative Python registry of EVI
configs and a CLI that reconciles it against Hume's API. Iterating on voice,
prompt copy, timeouts, or turn detection becomes a code change + one CLI call.
The registry supports multiple persona keys so the runtime can select a config
per call later, but per-call selection wiring is out of scope for this lift.

## Non-goals

- Wiring intake/session metadata to a persona key at runtime. We ship the
  schema, the sync, and a lookup helper; runtime selection is a follow-up.
- Mid-call config swap (the `swap_config` stub in `hume_evi.py` stays as is).
- Migrating prompt text out of the Hume config. The CLM webhook-side prompt
  (`rehearse/personas.py`) is unrelated and stays where it is.
- Custom voice creation/upload. We reference voices that already exist in the
  Hume workspace, by name or id.
- A `pull` or `list` CLI command. Punted to a follow-up; in v1 the seed for
  the registry is the snapshot taken in this design from the live config.

## Design

### Schema

A new module `rehearse/services/hume_configs.py` defines:

```python
class HumeVoice(BaseModel):
    name: str | None = None
    id: str | None = None
    provider: Literal["HUME_AI", "CUSTOM_VOICE"] = "HUME_AI"

class HumeLanguageModel(BaseModel):
    provider: str
    model: str | None = None
    temperature: float | None = None

class HumeEventMessage(BaseModel):
    enabled: bool = True
    text: str | None = None

class HumeTimeouts(BaseModel):
    max_duration_secs: int = 300
    inactivity_secs: int = 122

class HumeTurnDetection(BaseModel):
    end_of_turn_silence_ms: int = 500
    prefix_padding_ms: int = 300
    speech_detection_threshold: float = 0.4

class HumePersonaConfig(BaseModel):
    persona_key: str
    display_name: str
    evi_version: str = "4-mini"
    voice: HumeVoice
    language_model: HumeLanguageModel
    prompt_text: str
    on_new_chat: HumeEventMessage | None = None
    on_max_duration_timeout: HumeEventMessage | None = None
    on_inactivity_timeout: HumeEventMessage | None = None
    timeouts: HumeTimeouts = HumeTimeouts()
    turn_detection: HumeTurnDetection = HumeTurnDetection()
    interruption_min_ms: int = 800
    nudges_enabled: bool = True
    nudges_interval_secs: int = 8
    builtin_tools: list[str] = []  # e.g. ["web_search", "hang_up"]
```

The same module exports `PERSONAS: dict[str, HumePersonaConfig]` with one
entry, `"default"`, populated from the snapshot pulled from the current live
config (id `1259711b-0cec-43f4-a729-fea57e20cd32`):

- display_name: `"Rehearse Coach (default)"` (renamed from the timestamped
  console name; see Migration below)
- voice: `name="Inspiring Woman"`, provider `HUME_AI`
- language_model: provider `ANTHROPIC`, model `claude-sonnet-4-20250514`
- prompt_text: the existing 4-paragraph coaching brief, verbatim
- on_new_chat: enabled, text `"Hey there, what's on the mind?"`
- on_max_duration_timeout: enabled, text `None`
- on_inactivity_timeout: disabled
- timeouts: defaults (300s max, 122s inactivity)
- turn_detection: defaults (500/300/0.4)
- interruption_min_ms: 800
- nudges_enabled: true, interval 8s
- builtin_tools: `["web_search", "hang_up"]`

### Sync semantics

Pure function `plan_sync(personas, remote_configs) -> list[SyncAction]`:

- `Create(persona)` if no remote config matches `display_name`.
- `NewVersion(persona, config_id, diff)` if a matching config exists but the
  declared fields differ from the latest version.
- `NoOp(persona, config_id)` if all declared fields match.

Match key: `display_name` (case-sensitive, exact). Diff is field-by-field on
the schema above; we ignore Hume's auto-managed fields (`created_on`,
`modified_on`, `version`, voice `reference_tokens`/`reference_signed_uri`,
`prompt.id`/`version`/`name`).

The applier calls `client.empathic_voice.configs.create_config(...)` for new
configs and `create_config_version(...)` for updates. Voice is rendered as
`VoiceName` when a name is set, `VoiceId` when an id is set, with the declared
provider passed through.

After sync, the applier writes `sessions/.hume_configs.json`:

```json
{
  "default": "1259711b-0cec-43f4-a729-fea57e20cd32",
  "synced_at": "2026-05-06T20:35:00Z"
}
```

`sessions/.hume_configs.json` is added to `.gitignore` (already there as
`sessions/` is ignored).

### CLI

A new entry point `rehearse-hume` registered in `pyproject.toml`:

- `rehearse-hume diff` — print the planned actions and exit 0; exit 1 if any
  action would be Create or NewVersion (so CI can gate on "in sync").
- `rehearse-hume sync` — apply Create and NewVersion actions, write the
  mapping file, print a summary.

Both commands use `RuntimeConfig.from_env()` for `hume_api_key` and read the
`PERSONAS` registry directly.

### Runtime helper

`hume_configs.select_config_id(persona_key, *, mapping_path) -> str` reads the
mapping file and returns the config id for the persona, falling back to the
env `HUME_CONFIG_ID` if the mapping file is missing or the key is absent. No
runtime call site is changed in this lift; the helper exists for follow-up
wiring.

### Migration

The current live config has display_name
`"Your smart companion (5/1/2026, 03:23:55 PM)"`. Sync would treat that as a
non-match and create a second config named `"Rehearse Coach (default)"`. To
avoid creating a duplicate, the first run will be:

1. Run `rehearse-hume diff`. It prints `Create("default")`.
2. Operator manually renames the existing config in the Hume console to
   `"Rehearse Coach (default)"` (or accepts the duplicate).
3. Re-run `rehearse-hume sync`. With the rename, sync sees a match and emits
   `NewVersion` if anything else drifted, or `NoOp`.

This one-time migration is documented in the CLI's docstring.

## Test plan

Unit tests in `tests/test_hume_configs.py`:

- `plan_sync` returns `Create` when remote is empty.
- `plan_sync` returns `NoOp` when remote matches declared fields exactly.
- `plan_sync` returns `NewVersion` with a non-empty diff when a single field
  differs (covers voice name, prompt text, timeout secs, turn-detection
  threshold).
- `plan_sync` ignores Hume auto-managed fields (`created_on`, `version`,
  `reference_tokens`).
- `select_config_id` returns the mapping value when present.
- `select_config_id` falls back to a provided default when mapping file is
  missing or key is absent.
- Schema round-trip: `HumePersonaConfig.model_validate(payload).model_dump()`
  is idempotent.

CLI tests in `tests/test_hume_configs_cli.py`:

- `diff` exits 0 with empty output when in sync (mock the Hume client).
- `diff` exits 1 and prints a summary when out of sync.
- `sync` calls the right Hume SDK methods (mocked) and writes the mapping
  file.

No live-Hume integration test in v1.

## Out of scope (follow-ups)

- `rehearse-hume pull <config_id>` to snapshot a console-edited config back
  into the registry as Python source.
- `rehearse-hume list` to dump all workspace configs.
- Wiring `select_config_id` into `HumeEVIClient._connect` so a per-call
  persona key actually changes which config is used.
- Persona inference from intake/session metadata.
- Tests against a live Hume sandbox.

## Implementation note: `on_resume_chat` removed

During implementation we found that the Hume `PostedEventMessageSpecs` type does not expose `on_resume_chat` for writes — the field is absent from the SDK's `model_fields`. Comparing it would cause infinite drift (the plan detects a mismatch but the POST never includes the field, so Hume never reflects it back). `on_resume_chat` has been removed from `HumePersonaConfig`, `RemoteConfigSnapshot`, `_COMPARED_FIELDS`, the default persona seed, and the test fixtures.
