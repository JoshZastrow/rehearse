# rehearse — Runtime Workstream

**Status**: draft (implementation-facing)
**Owner**: jz
**Depends on**: `SPEC.md`, `rehearse/types.py`, `docs/specs/v2026-04-27-runtime.md`, `docs/specs/v2026-04-28-drop-pipecat.md`
**Separate from**: `docs/specs/v2026-04-27-eval-harness.md`, `docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md`

---

## 0. One-line summary

This doc isolates the live runtime work: the Twilio-to-Hume phone call path, session orchestration, artifact capture, CLM webhook, and post-call synthesis. The eval harness is a separate system and is not part of this workstream.

## 1. Runtime boundary

The runtime owns one thing: a real phone call with a human.

Concrete path:

```
Twilio number
  -> Twilio webhooks + Media Streams
  -> rehearse FastAPI service
  -> owned Twilio audio bridge
  -> Hume EVI realtime session
  -> Hume CLM webhook -> our LLM/agent
  -> artifact writers
  -> post-call synthesis
  -> SMS back to the user with the session link
```

The runtime does **not** own benchmark loading, provider bakeoffs, sandboxed rollout workers, or scoring. Those belong to eval.

## 2. Separation from eval

The runtime and eval harness share **schemas**, not infrastructure.

Shared:
- `rehearse/types.py`
- artifact shapes under `sessions/{id}/`
- pure logic that can run on frozen artifacts, especially synthesis and some scorers

Runtime-only:
- Twilio webhooks
- Twilio Media Streams WebSocket handling
- Hume EVI WebSocket session
- Hume CLM webhook endpoint
- live session state
- live phase timing
- audio capture and persistence during a real call

Directory decision:
- Do **not** add `rehearse/runtime/` yet.
- Keep runtime code under `rehearse/audio/`, `rehearse/services/`, `rehearse/frames.py`, and `rehearse/bus.py`.
- Reason: the runtime surface is still small; a second top-level namespace would only move files, not simplify the architecture.

Eval-only:
- benchmark registries
- targets
- scorers
- subprocess executor / worker model
- multimodal provider comparisons
- MME-Emotion dataset plumbing

Design rule: eval must be able to run with no Twilio account, no Hume live session, and no phone call. Runtime must be able to place and complete a call without importing the eval runner.

## 3. What the runtime team still has to build

Dropping Pipecat removed framework glue, not product responsibilities. The runtime still owns:

1. **Twilio WebSocket bridge to Hume EVI**
   - Accept Twilio Media Streams frames
   - Decode μ-law 8k audio
   - Resample to PCM16 16k for Hume
   - Send assistant audio back to Twilio in the reverse direction

2. **Session state and phase state**
   - Mint and track `session_id`
   - Maintain in-call lifecycle
   - Advance intake -> practice -> feedback
   - Finalize cleanly on hangup, error, or timeout

3. **Transcript / prosody / audio artifact writing**
   - Persist `transcript.jsonl`
   - Persist `prosody.jsonl`
   - Persist `audio.wav`
   - Persist `telemetry.jsonl`
   - Update `session.json`

4. **CLM webhook endpoint**
   - Receive Hume turn callbacks
   - Dispatch to the correct coach or character agent
   - Call the underlying LLM
   - Stream tokens back in Hume-compatible form

5. **Finalize / post-call synthesis flow**
   - Run story synthesis on frozen artifacts
   - Run feedback synthesis on frozen artifacts
   - Mark session complete or partial
   - Send the viewer link by SMS

## 4. Runtime architecture

After dropping Pipecat, the live call path is:

```
Twilio
  -> TwilioStream
  -> HumeEVIClient
  -> FrameBus
  -> writers / PhaseProcessor
  -> SessionOrchestrator.finalize()
```

Component roles:

| Component | Responsibility |
|---|---|
| `rehearse/telephony.py` | Twilio webhook handling, outbound calls, SMS, `/media/{session_id}` attach |
| `rehearse/audio/twilio_stream.py` | Twilio Media Streams protocol and audio conversion |
| `rehearse/services/hume_evi.py` | Hume realtime WebSocket session, event parsing, audio send/receive |
| `rehearse/bus.py` | one-to-many fanout inside a live call |
| `rehearse/phases.py` | phase timing and soft-cue transitions |
| `rehearse/writers/` | transcript, prosody, audio, telemetry persistence |
| `rehearse/agents/` | CLM webhook and coach/character agent dispatch |
| `rehearse/session.py` | session lifecycle, finalize, synthesis kickoff |
| `rehearse/synthesis.py` | post-call story + feedback generation |

## 5. Implementation order

Build in this order:

1. Twilio transport and audio primitives
   - `mulaw.py`
   - `resample.py`
   - `twilio_stream.py`

2. Live-call event plumbing
   - `frames.py`
   - `bus.py`
   - `hume_evi.py`

3. Runtime entrypoint wiring
   - `/media/{session_id}`
   - `SessionOrchestrator.attach_pipeline()` or equivalent runtime attach

4. Writers and persistence
   - transcript
   - prosody
   - audio
   - telemetry

5. Phase transitions
   - intake
   - practice
   - feedback

6. CLM webhook + agents
   - coach
   - character

7. Post-call synthesis and SMS delivery

The first meaningful live demo is earlier than full product completion:

**Demo A**: dial in -> talk -> Hume default behavior speaks back.

That proves:
- Twilio bridge works
- Hume audio loop works
- real-time latency is acceptable

Only after Demo A should the team spend time on richer coach behavior.

## 6. Runtime success criteria

The runtime workstream is successful when all of the following are true:

1. A user can trigger a real phone session through Twilio.
2. The call reaches our FastAPI service over Media Streams.
3. Hume EVI can hear the user and speak back.
4. The CLM webhook can substitute our own LLM-driven coach or character behavior.
5. Transcript, prosody, audio, and telemetry artifacts are written for the call.
6. Post-call synthesis produces `story.md` and `feedback.md`.
7. The user receives an SMS with a viewer link.

## 7. Non-goals for the runtime team

Not part of this workstream:
- benchmark adapter work
- MME-Emotion ingest
- Gemini vs vLLM provider comparison
- scorer development
- `rehearse-eval` CLI
- subprocess executor / worker isolation
- benchmark result reporting

Those tasks can consume runtime artifacts later, but they do not block the live call path.

## 8. Decision rule

When deciding whether code belongs in runtime or eval, ask:

**Does this code exist to complete a real phone call with a human right now?**

- If yes, it belongs in runtime.
- If no, and it exists to simulate, benchmark, score, compare, or replay behavior, it belongs in eval.
