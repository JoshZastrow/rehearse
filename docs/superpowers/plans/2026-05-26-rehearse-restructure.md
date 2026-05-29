# Rehearse Package Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the 25 flat Python files in `rehearse/` into thematic subpackages, update all imports, and document the new structure in the README.

**Architecture:** Create four new subpackages (`session/`, `phases/`, `memory/`, `api/`) from the flat files, merge four files into existing subdirectories (`agents/`, `personas/`, `backends/`, `audio/`), delete one shadowed file (`personas.py`), and leave foundational primitives flat (`types.py`, `bus.py`, `frames.py`, `config.py`, `storage.py`, `pipeline.py`). No re-exports — all callers update to the canonical new path.

**Tech Stack:** Python, `git mv`, `sed`/`find` for bulk import rewrites, `uv run pytest` for verification.

---

### Task 0: Set up isolated worktree

**Files:**
- No file changes

- [ ] **Step 1: Create the worktree**

```bash
git worktree add .worktrees/restructure -b restructure
cd .worktrees/restructure
```

- [ ] **Step 2: Establish a clean baseline**

```bash
uv run pytest -x -q 2>&1 | tail -20
```

Expected: all tests pass (or pre-existing failures only — note any failures here before proceeding).

---

### Task 1: Create `rehearse/session/` subpackage

Move `session.py`, `conversation.py`, `runtime.py`, `finalize_sweeper.py`, `synthesis.py` into a new `session/` package.

**Files:**
- Create: `rehearse/session/__init__.py`
- Move: `rehearse/session.py` → `rehearse/session/session.py`
- Move: `rehearse/conversation.py` → `rehearse/session/conversation.py`
- Move: `rehearse/runtime.py` → `rehearse/session/runtime.py`
- Move: `rehearse/finalize_sweeper.py` → `rehearse/session/finalize_sweeper.py`
- Move: `rehearse/synthesis.py` → `rehearse/session/synthesis.py`
- Modify: all files importing from `rehearse.session`, `rehearse.conversation`, `rehearse.runtime`, `rehearse.finalize_sweeper`, `rehearse.synthesis`

- [ ] **Step 1: Create the package and move files**

```bash
mkdir rehearse/session
touch rehearse/session/__init__.py
git mv rehearse/session.py rehearse/session/session.py
git mv rehearse/conversation.py rehearse/session/conversation.py
git mv rehearse/runtime.py rehearse/session/runtime.py
git mv rehearse/finalize_sweeper.py rehearse/session/finalize_sweeper.py
git mv rehearse/synthesis.py rehearse/session/synthesis.py
```

- [ ] **Step 2: Update all imports**

```bash
# rehearse.session → rehearse.session.session
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.session import/from rehearse.session.session import/g' {} +

# rehearse.conversation → rehearse.session.conversation
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.conversation import/from rehearse.session.conversation import/g' {} +

# rehearse.runtime → rehearse.session.runtime
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.runtime import/from rehearse.session.runtime import/g' {} +

# rehearse.finalize_sweeper → rehearse.session.finalize_sweeper
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.finalize_sweeper import/from rehearse.session.finalize_sweeper import/g' {} +

# rehearse.synthesis → rehearse.session.synthesis
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.synthesis import/from rehearse.session.synthesis import/g' {} +
```

- [ ] **Step 3: Verify no stale imports remain**

```bash
grep -r "from rehearse\.session import\|from rehearse\.conversation import\|from rehearse\.runtime import\|from rehearse\.finalize_sweeper import\|from rehearse\.synthesis import" \
  rehearse/ tests/ scripts/ --include="*.py" | grep -v __pycache__
```

Expected: no output (all old paths gone).

- [ ] **Step 4: Run tests**

```bash
uv run pytest -x -q 2>&1 | tail -20
```

Expected: same pass/fail count as baseline.

- [ ] **Step 5: Commit**

```bash
git add rehearse/session/ rehearse/conversation.py rehearse/runtime.py \
  rehearse/finalize_sweeper.py rehearse/synthesis.py
git add -u
git commit -m "refactor: move session-lifecycle files into rehearse/session/"
```

---

### Task 2: Create `rehearse/phases/` subpackage

Move `phases.py`, `phases_llm.py`, `intake.py`, `consent.py`, `outcome.py`, `survey.py` into a new `phases/` package.

**Files:**
- Create: `rehearse/phases/__init__.py`
- Move: `rehearse/phases.py` → `rehearse/phases/phases.py`
- Move: `rehearse/phases_llm.py` → `rehearse/phases/phases_llm.py`
- Move: `rehearse/intake.py` → `rehearse/phases/intake.py`
- Move: `rehearse/consent.py` → `rehearse/phases/consent.py`
- Move: `rehearse/outcome.py` → `rehearse/phases/outcome.py`
- Move: `rehearse/survey.py` → `rehearse/phases/survey.py`
- Modify: all files importing from these six modules

- [ ] **Step 1: Create the package and move files**

```bash
mkdir rehearse/phases
touch rehearse/phases/__init__.py
git mv rehearse/phases.py rehearse/phases/phases.py
git mv rehearse/phases_llm.py rehearse/phases/phases_llm.py
git mv rehearse/intake.py rehearse/phases/intake.py
git mv rehearse/consent.py rehearse/phases/consent.py
git mv rehearse/outcome.py rehearse/phases/outcome.py
git mv rehearse/survey.py rehearse/phases/survey.py
```

- [ ] **Step 2: Update all imports**

```bash
# rehearse.phases → rehearse.phases.phases
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.phases import/from rehearse.phases.phases import/g' {} +

# rehearse.phases_llm → rehearse.phases.phases_llm
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.phases_llm import/from rehearse.phases.phases_llm import/g' {} +

# rehearse.intake → rehearse.phases.intake
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.intake import/from rehearse.phases.intake import/g' {} +

# rehearse.consent → rehearse.phases.consent
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.consent import/from rehearse.phases.consent import/g' {} +

# rehearse.outcome → rehearse.phases.outcome
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.outcome import/from rehearse.phases.outcome import/g' {} +

# rehearse.survey → rehearse.phases.survey
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.survey import/from rehearse.phases.survey import/g' {} +
```

- [ ] **Step 3: Verify no stale imports remain**

```bash
grep -r "from rehearse\.phases import\|from rehearse\.phases_llm import\|from rehearse\.intake import\|from rehearse\.consent import\|from rehearse\.outcome import\|from rehearse\.survey import" \
  rehearse/ tests/ scripts/ --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 4: Run tests**

```bash
uv run pytest -x -q 2>&1 | tail -20
```

Expected: same pass/fail count as baseline.

- [ ] **Step 5: Commit**

```bash
git add rehearse/phases/
git add -u
git commit -m "refactor: move conversation-flow files into rehearse/phases/"
```

---

### Task 3: Create `rehearse/memory/` subpackage

Move `memory.py` and `memory_manager.py` into a new `memory/` package.

**Files:**
- Create: `rehearse/memory/__init__.py`
- Move: `rehearse/memory.py` → `rehearse/memory/memory.py`
- Move: `rehearse/memory_manager.py` → `rehearse/memory/memory_manager.py`
- Modify: all files importing from `rehearse.memory` or `rehearse.memory_manager`

- [ ] **Step 1: Create the package and move files**

```bash
mkdir rehearse/memory
touch rehearse/memory/__init__.py
git mv rehearse/memory.py rehearse/memory/memory.py
git mv rehearse/memory_manager.py rehearse/memory/memory_manager.py
```

- [ ] **Step 2: Update all imports**

```bash
# rehearse.memory → rehearse.memory.memory
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.memory import/from rehearse.memory.memory import/g' {} +

# rehearse.memory_manager → rehearse.memory.memory_manager
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.memory_manager import/from rehearse.memory.memory_manager import/g' {} +
```

- [ ] **Step 3: Verify no stale imports remain**

```bash
grep -r "from rehearse\.memory import\|from rehearse\.memory_manager import" \
  rehearse/ tests/ scripts/ --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 4: Run tests**

```bash
uv run pytest -x -q 2>&1 | tail -20
```

Expected: same pass/fail count as baseline.

- [ ] **Step 5: Commit**

```bash
git add rehearse/memory/
git add -u
git commit -m "refactor: move caller-memory files into rehearse/memory/"
```

---

### Task 4: Create `rehearse/api/` subpackage

Move `app.py`, `telephony.py`, `viewer.py` into a new `api/` package.

**Files:**
- Create: `rehearse/api/__init__.py`
- Move: `rehearse/app.py` → `rehearse/api/app.py`
- Move: `rehearse/telephony.py` → `rehearse/api/telephony.py`
- Move: `rehearse/viewer.py` → `rehearse/api/viewer.py`
- Modify: all files importing from `rehearse.app`, `rehearse.telephony`, `rehearse.viewer`

- [ ] **Step 1: Create the package and move files**

```bash
mkdir rehearse/api
touch rehearse/api/__init__.py
git mv rehearse/app.py rehearse/api/app.py
git mv rehearse/telephony.py rehearse/api/telephony.py
git mv rehearse/viewer.py rehearse/api/viewer.py
```

- [ ] **Step 2: Update all imports**

```bash
# rehearse.app → rehearse.api.app
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.app import/from rehearse.api.app import/g' {} +

# rehearse.telephony → rehearse.api.telephony
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.telephony import/from rehearse.api.telephony import/g' {} +

# rehearse.viewer → rehearse.api.viewer
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.viewer import/from rehearse.api.viewer import/g' {} +
```

- [ ] **Step 3: Verify no stale imports remain**

```bash
grep -r "from rehearse\.app import\|from rehearse\.telephony import\|from rehearse\.viewer import" \
  rehearse/ tests/ scripts/ --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 4: Run tests**

```bash
uv run pytest -x -q 2>&1 | tail -20
```

Expected: same pass/fail count as baseline.

- [ ] **Step 5: Commit**

```bash
git add rehearse/api/
git add -u
git commit -m "refactor: move HTTP-layer files into rehearse/api/"
```

---

### Task 5: Merge `new_clm_responder.py` into `rehearse/agents/`

**Files:**
- Move: `rehearse/new_clm_responder.py` → `rehearse/agents/new_clm_responder.py`
- Modify: `rehearse/agents/clm.py` (lazy import inside function), `tests/test_clm_responder.py`

- [ ] **Step 1: Move the file**

```bash
git mv rehearse/new_clm_responder.py rehearse/agents/new_clm_responder.py
```

- [ ] **Step 2: Update all imports**

```bash
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.new_clm_responder import/from rehearse.agents.new_clm_responder import/g' {} +
```

Note: `rehearse/agents/clm.py` uses a lazy import inside a function body:
```python
from rehearse.new_clm_responder import NewCLMResponder
```
The sed command above covers it, but verify manually:

```bash
grep -n "new_clm_responder" rehearse/agents/clm.py
```

Expected: line shows `from rehearse.agents.new_clm_responder import NewCLMResponder`.

- [ ] **Step 3: Verify no stale imports remain**

```bash
grep -r "from rehearse\.new_clm_responder import" rehearse/ tests/ --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 4: Run tests**

```bash
uv run pytest -x -q 2>&1 | tail -20
```

Expected: same pass/fail count as baseline.

- [ ] **Step 5: Commit**

```bash
git add rehearse/agents/new_clm_responder.py
git add -u
git commit -m "refactor: move new_clm_responder into rehearse/agents/"
```

---

### Task 6: Delete shadowed `rehearse/personas.py`

`rehearse/personas/__init__.py` already contains all the content from the flat `rehearse/personas.py`. Python resolves `from rehearse.personas import ...` to the package, not the flat file — so `personas.py` is dead code that was never being imported.

**Files:**
- Delete: `rehearse/personas.py`
- No import changes needed

- [ ] **Step 1: Confirm it's truly shadowed**

```bash
python -c "import rehearse.personas; print(rehearse.personas.__file__)"
```

Expected: prints a path ending in `rehearse/personas/__init__.py` (not `rehearse/personas.py`).

- [ ] **Step 2: Delete the dead file**

```bash
git rm rehearse/personas.py
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest -x -q 2>&1 | tail -20
```

Expected: same pass/fail count as baseline.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: delete shadowed rehearse/personas.py (package takes precedence)"
```

---

### Task 7: Move `transport.py` → `rehearse/backends/`

`transport.py` defines `RuntimeTransport` and `InMemoryTwoWayChannel`. `rehearse/eval/transports.py` is a backwards-compat shim re-exporting from `rehearse.transport`. `tests/test_transport_move.py` tests both the canonical and legacy import paths and must be updated to reflect the new canonical location.

**Files:**
- Move: `rehearse/transport.py` → `rehearse/backends/transport.py`
- Modify: `rehearse/eval/transports.py` (update its re-export source)
- Modify: `tests/test_transport_move.py` (update canonical import path in test)
- Modify: all other files importing `from rehearse.transport`

- [ ] **Step 1: Move the file**

```bash
git mv rehearse/transport.py rehearse/backends/transport.py
```

- [ ] **Step 2: Update all imports**

```bash
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.transport import/from rehearse.backends.transport import/g' {} +
```

- [ ] **Step 3: Fix the backward-compat shim**

`rehearse/eval/transports.py` previously imported from `rehearse.transport`. The sed in Step 2 already updated it to `rehearse.backends.transport`. Verify:

```bash
head -10 rehearse/eval/transports.py
```

Expected: `from rehearse.backends.transport import (`.

- [ ] **Step 4: Update `test_transport_move.py` to reflect new canonical path**

The test's `test_canonical_import` function must now use `rehearse.backends.transport`. Open `tests/test_transport_move.py` and replace the test body:

```python
def test_canonical_import() -> None:
    from rehearse.backends.transport import (
        InMemoryTwoWayChannel,
        TwoWayChannel,
        RuntimeTransport,
        TransportClosedError,
        TransportEvent,
        TransportEventKind,
    )

    assert InMemoryTwoWayChannel is not None
    assert TwoWayChannel is not None
    assert RuntimeTransport is TwoWayChannel


def test_legacy_import_still_works() -> None:
    from rehearse.eval.transports import (
        InMemoryTwoWayChannel,
        TwoWayChannel,
        RuntimeTransport,
        TransportClosedError,
        TransportEvent,
    )

    assert InMemoryTwoWayChannel is not None


def test_audio_bytes_round_trip() -> None:
    """Audio bytes via TransportEvent data field round-trip unchanged."""
    from rehearse.backends.transport import InMemoryTwoWayChannel, TransportEvent

    import asyncio

    async def _run() -> None:
        transport = InMemoryTwoWayChannel()
        audio_bytes = b"\x00\x01\x02\x03" * 1000
        event = await transport.customer.send("audio", data=audio_bytes)
        received = await transport.runtime.receive()
        assert received.kind == "audio"
        assert received.data == audio_bytes

    asyncio.run(_run())


def test_both_paths_same_class() -> None:
    from rehearse.eval.transports import InMemoryTwoWayChannel as Legacy
    from rehearse.backends.transport import InMemoryTwoWayChannel as Canonical

    assert Legacy is Canonical
```

- [ ] **Step 5: Verify no stale imports remain**

```bash
grep -r "from rehearse\.transport import" rehearse/ tests/ scripts/ --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 6: Run tests**

```bash
uv run pytest -x -q 2>&1 | tail -20
```

Expected: same pass/fail count as baseline.

- [ ] **Step 7: Commit**

```bash
git add rehearse/backends/transport.py
git add -u
git commit -m "refactor: move transport.py into rehearse/backends/"
```

---

### Task 8: Move `participants.py` → `rehearse/audio/`

**Files:**
- Move: `rehearse/participants.py` → `rehearse/audio/participants.py`
- Modify: all files importing `from rehearse.participants`

- [ ] **Step 1: Move the file**

```bash
git mv rehearse/participants.py rehearse/audio/participants.py
```

- [ ] **Step 2: Update all imports**

```bash
find . -name "*.py" ! -path "./.worktrees/*" ! -path "*/__pycache__/*" \
  -exec sed -i '' 's/from rehearse\.participants import/from rehearse.audio.participants import/g' {} +
```

- [ ] **Step 3: Verify no stale imports remain**

```bash
grep -r "from rehearse\.participants import" rehearse/ tests/ scripts/ --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 4: Run tests**

```bash
uv run pytest -x -q 2>&1 | tail -20
```

Expected: same pass/fail count as baseline.

- [ ] **Step 5: Commit**

```bash
git add rehearse/audio/participants.py
git add -u
git commit -m "refactor: move participants.py into rehearse/audio/"
```

---

### Task 9: Update README with new project structure

Add a `## Project Structure` section to `README.md` documenting top-level directories and the `rehearse/` subpackages.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the project structure section**

Find the `## Current Architecture` section in `README.md` and insert the following block before it:

```markdown
## Project Structure

```
rehearse/               # Core Python package
├── types.py            # Domain types and Pydantic models (widely imported)
├── bus.py              # FrameBus — in-process async event bus
├── frames.py           # Frame types published onto the bus
├── config.py           # RuntimeConfig loaded from environment
├── storage.py          # LocalFilesystemStore — session artifact persistence
├── pipeline.py         # Live-call assembly reference doc
│
├── session/            # Call lifecycle orchestration
│   ├── session.py      # SessionOrchestrator, SessionHandle — create/finalize calls
│   ├── conversation.py # run_session() — transport-agnostic session runner
│   ├── runtime.py      # RuntimeHost — boots one session against a transport
│   ├── finalize_sweeper.py  # Sweep stale in_progress sessions on restart
│   └── synthesis.py    # SessionSynthesizer — post-call artifact generation
│
├── phases/             # Conversation flow state machine
│   ├── phases.py       # PhaseProcessor, PhaseBudgets — phase timing and transitions
│   ├── phases_llm.py   # MeetingPhaseProcessor — LLM-driven phase detection
│   ├── intake.py       # IntakeProcessor — captures caller situation during intake
│   ├── consent.py      # ConsentGate — verbal recording-consent at call start
│   ├── outcome.py      # OutcomeProbe — post-feedback yes/no outcome capture
│   └── survey.py       # SurveyAgent — post-call satisfaction survey
│
├── memory/             # Caller memory across sessions
│   ├── memory.py       # CallerMemory protocol + implementations (Null, InMemory, Honcho)
│   └── memory_manager.py  # MemoryManager — per-turn recall and storage
│
├── api/                # HTTP layer
│   ├── app.py          # FastAPI app factory — wires routes, storage, orchestration
│   ├── telephony.py    # Twilio webhooks, outbound calls, media websocket
│   └── viewer.py       # /viewer page — renders session artifacts as HTML
│
├── agents/             # CLM agent roles and routing
│   ├── clm.py          # CLM entrypoint and route mounting
│   ├── new_clm_responder.py  # NewCLMResponder — per-turn CLM orchestration
│   ├── router.py       # AgentRouter — selects agent for each turn
│   ├── registry.py     # AgentRegistry — maps phase+intake to agent instances
│   └── roles/          # Individual agent role implementations
│
├── audio/              # Audio codecs and voice participant contracts
│   ├── participants.py # VoiceParticipant ABC and VoiceSpeaker protocol
│   ├── twilio_stream.py  # TwilioCallerParticipant and TwilioStream
│   ├── mulaw.py        # μ-law codec helpers
│   └── resample.py     # PCM resampling
│
├── backends/           # LLM and voice backend adapters
│   ├── transport.py    # RuntimeTransport — duplex transport abstraction for eval and serving
│   ├── pipeline.py     # PipelineBackend — local STT/TTS pipeline
│   ├── managed.py      # ManagedBackend — remote managed voice backend
│   ├── tts.py          # TTS adapter
│   └── factory.py      # Backend factory — creates the right backend from config
│
├── personas/           # Persona registry and prompt builders
│   ├── __init__.py     # Coach/character/feedback prompts, consent classifier, intake builder
│   ├── registry.py     # PersonaRegistry — maps intake to practice partner
│   └── souls/          # Named persona definitions
│
├── services/           # External service integrations
│   ├── hume_evi.py     # HumeEVIClient — Hume voice backend
│   ├── hume_configs.py # Hume EVI config management
│   └── memory_mcp_server.py  # MCP server exposing caller memory
│
├── transports/         # LLM API transport clients
│   ├── anthropic.py    # Anthropic streaming transport
│   └── openai_compat.py  # OpenAI-compatible streaming transport
│
├── writers/            # Session artifact writers
│   └── artifacts.py    # AudioRecorder, TranscriptWriter, ProsodyWriter, TimingWriter
│
└── eval/               # Evaluation harness
    ├── cli.py          # rehearse-eval entry point
    ├── runner.py       # Eval run orchestration
    ├── scorers/        # LLM and deterministic judges
    ├── providers/      # LLM provider adapters for eval
    ├── targets/        # Eval targets (echo, raw LLM)
    ├── environments/   # Sandbox environments (in-process, subprocess)
    ├── customers/      # Synthetic customer drivers
    └── executors/      # Task executors
```

Top-level directories:

| Directory | Purpose |
|---|---|
| `rehearse/` | Core Python package |
| `tests/` | Unit and integration tests (mirrors `rehearse/` structure) |
| `evals/` | Eval datasets, fixtures, and run artifacts |
| `scripts/` | Operational scripts (serving, diagnostics, scenario generation) |
| `infra/` | Deployment and infrastructure configuration |
| `web/` | Frontend assets |
| `docs/` | Specs, plans, and architecture documents |
| `dev/` | Local development tooling and lab configs |
```

- [ ] **Step 2: Run tests to confirm README change doesn't break anything**

```bash
uv run pytest -x -q 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add project structure section to README"
```

---

### Task 10: Final verification

- [ ] **Step 1: Full test suite**

```bash
uv run pytest -q 2>&1 | tail -30
```

Expected: same pass/fail count as the Task 0 baseline.

- [ ] **Step 2: Confirm no old import paths remain**

```bash
grep -r "from rehearse\.\(session\|conversation\|runtime\|finalize_sweeper\|synthesis\|phases\b\|phases_llm\|intake\|consent\|outcome\|survey\|memory\b\|memory_manager\|app\|telephony\|viewer\|new_clm_responder\|transport\b\|participants\) import" \
  rehearse/ tests/ scripts/ --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 3: Confirm new structure**

```bash
ls rehearse/session/ rehearse/phases/ rehearse/memory/ rehearse/api/
ls rehearse/agents/new_clm_responder.py rehearse/backends/transport.py rehearse/audio/participants.py
ls rehearse/personas.py 2>/dev/null && echo "ERROR: personas.py should be deleted" || echo "OK: personas.py deleted"
```

Expected: all new paths exist, `personas.py` absent.
