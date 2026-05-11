# rehearse — Spec: VoiceParticipant Protocol (v1)

**Status**: draft
**Owner**: jz
**Date**: 2026-05-11
**Depends on**: `rehearse/frames.py`, `rehearse/bus.py`, `rehearse/services/hume_evi.py`,
`rehearse/telephony.py`, `rehearse/consent.py`, `rehearse/outcome.py`,
`rehearse/agents/persona_swap.py`, `rehearse/types.py`
**Supersedes**: nothing (extends existing participant model, does not replace any spec)

---

## 0. One-line summary

Extract the voice I/O boundary into a typed `VoiceParticipant` protocol and a
narrower `VoiceSpeaker` protocol, so coach and caller implementations are
swappable without touching telephony wiring or business logic.

---

## 1. Goal

A live call has two participants: a **caller** (user) and a **coach**. Today both
are hard-coded in `telephony.py:media_stream` — the caller is always a Twilio
WebSocket and the coach is always `HumeEVIClient`. Swapping either requires
surgery on `telephony.py`. Three things make this worse:

1. **`hume.say` is injected as a bare callable** into `ConsentGate`,
   `OutcomeProbe`, and `PersonaSwapCoordinator`. All three business-logic
   components know they are talking to Hume specifically.

2. **There is no protocol** describing what a coach participant must provide.
   When we want to run an internal voice model, there is nothing to implement
   against — the contract exists only as implicit knowledge of `HumeEVIClient`'s
   method signatures.

3. **`SyntheticCaller` speaks a different interface** (`TwoWayChannel`-based
   text transport) than production participants (audio, FrameBus). Bridging the
   two for real-time observation requires protocol alignment first.

This spec defines:

- `VoiceParticipant` — the full protocol for one side of a voice call.
- `VoiceSpeaker` — a narrower protocol consumed only by business logic that
  needs to inject deterministic speech.
- `SpeakRequest` — a Pydantic-typed payload for `say` calls, enabling richer
  control without changing caller sites.
- Adaptation notes for every current implementation.

After v1 ships:
- Swapping Hume EVI for an internal voice model requires implementing
  `VoiceParticipant` in one new class. `telephony.py` and all business logic
  are unchanged.
- `ConsentGate`, `OutcomeProbe`, and `PersonaSwapCoordinator` have no import
  or direct reference to Hume.
- The codebase has a documented seam for three-way call support (observer
  via `FrameBus`) without requiring `TransportSide` or `Speaker` changes in v1.

---

## 2. Non-goals

- **Replacing Hume EVI in v1.** This spec defines the seam; swapping the
  implementation is a follow-up.
- **Three-way calls.** The `FrameBus` observer pattern is documented as a
  design note. `TransportSide` and `Speaker` are not extended.
- **`TwilioBridgeTransport`.** Deliberately excluded. `RuntimeHost._run_loop`
  is a text-turn loop designed for eval. Wrapping Twilio audio into it would
  require rebuilding Hume's VAD/STT front-end. The production audio path stays
  event-driven. See §7 for why this decision stands.
- **Changing the CLM webhook.** `rehearse/agents/clm.py` and
  `rehearse/personas.py` are untouched.
- **Eval harness transport.** `SyntheticCaller`'s `TwoWayChannel` interface
  is unchanged. §6 documents a future `AudioCallerParticipant` adapter for
  real-time eval, but that is not committed in v1.

---

## 3. Design commitments

1. **Client and Participant are different layers.**
   A *client* is a network I/O wrapper — it knows the external service's wire
   protocol, auth, and reconnect mechanics. It has no domain knowledge.
   A *participant* is a domain actor — it knows about `FrameBus`, `SpeakRequest`,
   and frame types. It uses a client as a transport detail.
   `HumeEVIClient` stays as the Hume network layer. `HumeEVIParticipant` is
   the domain actor that wraps it.

2. **`VoiceParticipant` is an ABC, not a Protocol.**
   All participant implementations live in this codebase. `ABC` with
   `@abstractmethod` gives compile-time contract acknowledgement and runtime
   enforcement at instantiation. `typing.Protocol` (structural duck typing) is
   reserved for interfaces over third-party code you don't control.
   `VoiceSpeaker` — consumed by business logic — remains a `Protocol` because
   it is a narrow capability slice, not a full class contract.

3. **`FrameBus` belongs in the participant, not the client.**
   `HumeEVIClient` currently takes `bus: FrameBus` in `__init__` — domain
   bleeding into the network layer. In v1 this is tolerated; extracting the bus
   dependency fully into `HumeEVIParticipant` is a follow-up cleanup. The spec
   draws the line clearly so future work knows where to head.

4. **`VoiceSpeaker` is the only interface business logic receives.** No
   component below `telephony.py` imports `HumeEVIClient` or any concrete
   participant class.

5. **`FrameBus` is the coordination layer.** Participants publish frames to the
   bus and are decoupled from each other. The session coordinator in
   `telephony.py` owns the audio pump between participants; neither participant
   holds a reference to the other.

6. **v1 is a refactor, not a rewrite.** All existing behaviour is preserved.
   The diff is: new file `rehearse/participants.py`, new class
   `HumeEVIParticipant` in `hume_evi.py`, updated type annotations in four
   files, no logic changes outside those files.

---

## 4. Architecture

### 4.1 Today

```
telephony.py:media_stream
  │
  ├─ async with TwilioStream(ws) as twilio
  ├─ async with HumeEVIClient(..., bus=bus) as hume
  │
  ├─ ConsentGate(speak=hume.say)          ← hume.say injected as bare callable
  ├─ OutcomeProbe(speak=hume.say)         ← hume.say injected as bare callable
  ├─ PersonaSwapCoordinator(speak=hume.say) ← hume.say injected as bare callable
  │
  ├─ hume_task = create_task(hume.run_event_loop())
  ├─ assistant_task = create_task(_pump_assistant_audio(twilio, bus))
  │
  └─ async for chunk in twilio.inbound():
         await hume.send_audio(chunk)     ← hard-coded Hume call
         await bus.publish(AudioChunk(USER, chunk))
```

- `ConsentGate`, `OutcomeProbe`, `PersonaSwapCoordinator` all import or
  depend on `Callable[[str], Awaitable[None]]` bound to Hume.
- `hume.run_event_loop()` and `hume.send_audio()` are called by name; there
  is no protocol the coordinator dispatches through.

### 4.2 After v1

```
telephony.py:media_stream
  │
  ├─ caller: VoiceParticipant = TwilioCallerParticipant(ws, bus, session_id)
  ├─ coach:  VoiceParticipant = HumeEVIParticipant(...)  ← wraps HumeEVIClient
  │                              also satisfies VoiceSpeaker
  │
  ├─ ConsentGate(speaker=coach)           ← VoiceSpeaker, no Hume reference
  ├─ OutcomeProbe(speaker=coach)          ← VoiceSpeaker, no Hume reference
  ├─ PersonaSwapCoordinator(speaker=coach) ← VoiceSpeaker, no Hume reference
  │
  ├─ coach_task = create_task(coach.run(bus))
  ├─ caller_task = create_task(caller.run(bus))  ← audio pump is now inside
  │
  └─ async for chunk in caller.audio_stream():
         await coach.receive_audio(chunk)
         await bus.publish(AudioChunk(USER, chunk))
```

To swap the coach (e.g. internal voice model):

```python
coach: VoiceParticipant = InternalVoiceParticipant(model_config, bus, session_id)
```

`telephony.py` is otherwise unchanged. Business logic is unchanged.

---

## 5. Contracts

### 5.1 `SpeakRequest` — Pydantic request type

```python
# rehearse/participants.py

from typing import Literal
from rehearse.types import Strict  # Pydantic BaseModel with extra="forbid"

class SpeakRequest(Strict):
    """Typed payload for one deterministic speech injection."""
    text: str
    priority: Literal["normal", "interrupt"] = "normal"
    utterance_id: str | None = None
```

`priority="interrupt"` is a hint to the participant that this line should
preempt any in-progress coach utterance. `HumeEVIClient` uses
`send_assistant_input` which already interrupts; a future TTS-based coach
can use this flag to cancel ongoing synthesis.

`utterance_id` lets the caller correlate the injected turn with transcript
records when needed. `None` means the participant generates its own id.

### 5.2 `ParticipantConfig` — Pydantic participant identity

```python
class ParticipantConfig(Strict):
    """Stable identity for one call participant. Persisted in session.json."""
    participant_id: str
    role: Literal["caller", "coach", "observer"]
    display_name: str | None = None
    backend: str  # e.g. "hume_evi", "twilio_stream", "synthetic_llm"
```

`session.json` gains a `participants: list[ParticipantConfig]` field. In v1
there are always exactly two entries: one `caller` and one `coach`.

### 5.3 `VoiceSpeaker` — narrow protocol for business logic

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class VoiceSpeaker(Protocol):
    """Capability to inject deterministic speech on a live call.

    Business logic components (ConsentGate, OutcomeProbe,
    PersonaSwapCoordinator) receive this interface rather than a concrete
    participant or a bare callable.
    """

    async def say(self, request: SpeakRequest) -> None:
        """Speak `request.text` verbatim, interrupting if priority="interrupt"."""
        ...
```

**Migration for call sites:** The three business logic classes currently
accept `speak: Callable[[str], Awaitable[None]]`. They are updated to accept
`speaker: VoiceSpeaker`. Internal call sites change from:

```python
await self._speak(text)
```
to:
```python
await self._speaker.say(SpeakRequest(text=text))
```

This is the only logic change in those three files.

### 5.4 `VoiceParticipant` — abstract base class

```python
from abc import ABC, abstractmethod

class VoiceParticipant(ABC):
    """One side of a live voice call.

    Implementations:
    - HumeEVIParticipant (coach): STT + CLM + TTS via Hume EVI.
    - TwilioCallerParticipant (caller): raw PCM I/O via Twilio WebSocket.
    - InternalVoiceParticipant (future coach): internal STT/TTS pipeline.
    - SyntheticCallerParticipant (future eval caller): LLM + audio synthesis.

    The FrameBus is the coordination layer. Participants publish frames to it
    and are decoupled from each other.
    """

    @property
    @abstractmethod
    def config(self) -> ParticipantConfig:
        """Return stable identity for this participant."""
        ...

    @abstractmethod
    async def receive_audio(self, pcm16_16k: bytes) -> None:
        """Accept one chunk of PCM16/16kHz audio from the other participant.

        For the coach: forwards to STT pipeline.
        For the caller: plays back to the human (used when the coach
        participant is a synthetic LLM in eval mode).
        """
        ...

    @abstractmethod
    async def say(self, request: SpeakRequest) -> None:
        """Inject one deterministic utterance without an LLM round-trip.

        For the coach: calls send_assistant_input (Hume) or TTS-then-stream
        (internal voice model).
        For the caller: no-op in production (real human speaks for themselves).
        For the synthetic caller: injects a scripted turn into the transport.
        """
        ...

    @abstractmethod
    async def run(self, bus: FrameBus) -> None:
        """Run this participant's event loop until the call ends.

        Must publish the following frames during the loop:
        - AudioChunk(speaker=<own role>)       — every outgoing audio chunk
        - TranscriptDelta(speaker=<own role>)  — final + interim transcripts
        - ProsodyEvent(speaker=<own role>)     — if prosody data is available
        - EndOfCall(reason=...)                — on termination or error

        Must return when EndOfCall is received on the bus or when the
        underlying connection closes.
        """
        ...
```

`VoiceParticipant` is a superset of `VoiceSpeaker`. Any `VoiceParticipant`
satisfies `VoiceSpeaker` structurally.

---

## 6. Implementation notes per class

### 6.1 `HumeEVIParticipant` — new class in `rehearse/services/hume_evi.py`

`HumeEVIClient` stays unchanged as the Hume network layer. `HumeEVIParticipant`
is a new class in the same module that extends `VoiceParticipant` and wraps the
client:

```python
class HumeEVIParticipant(VoiceParticipant):
    """Domain actor wrapping HumeEVIClient for the coach side of a live call."""

    def __init__(
        self,
        *,
        api_key: str,
        config_id: str,
        session_id: str,
        persona_key: str = "default",
        # ... same optional knobs as HumeEVIClient
    ) -> None:
        self._session_id = session_id
        self._client = HumeEVIClient(
            api_key=api_key,
            config_id=config_id,
            session_id=session_id,
            persona_key=persona_key,
            bus=None,  # bus injected at run() time, not construction
        )

    @property
    def config(self) -> ParticipantConfig:
        return ParticipantConfig(
            participant_id=self._session_id,
            role="coach",
            backend="hume_evi",
        )

    async def receive_audio(self, pcm16_16k: bytes) -> None:
        await self._client.send_audio(pcm16_16k)

    async def say(self, request: SpeakRequest) -> None:
        await self._client.say(request.text)

    async def run(self, bus: FrameBus) -> None:
        self._client._bus = bus  # bind bus at run time
        await self._client.run_event_loop()
```

`HumeEVIClient` is not changed. Its method signatures, existing tests, and
internal structure remain stable. The participant owns the domain translation;
the client owns the wire protocol.

The `bus=None` construction followed by `_bus` assignment at `run()` time is
a temporary bridge. The follow-up cleanup (commitment 3 in §3) moves bus
handling fully out of `HumeEVIClient` — at that point `HumeEVIParticipant`
subscribes to client events directly and publishes frames itself, and
`HumeEVIClient` becomes a pure I/O class with no domain imports.

### 6.2 `TwilioCallerParticipant` — new class wrapping `TwilioStream`

```python
class TwilioCallerParticipant:
    """VoiceParticipant wrapping a Twilio WebSocket media stream.

    - `run(bus)` reads inbound audio chunks and publishes AudioChunk(USER).
    - `receive_audio(pcm)` sends coach audio back to the caller.
    - `say(request)` is a no-op (the real human speaks for themselves).
    """

    def __init__(
        self,
        stream: TwilioStream,
        session_id: str,
    ) -> None:
        self._stream = stream
        self._session_id = session_id

    @property
    def config(self) -> ParticipantConfig:
        return ParticipantConfig(
            participant_id=self._session_id,
            role="caller",
            backend="twilio_stream",
        )

    async def receive_audio(self, pcm16_16k: bytes) -> None:
        await self._stream.send(pcm16_16k)

    async def say(self, request: SpeakRequest) -> None:
        pass  # no-op: real caller speaks for themselves

    async def run(self, bus: FrameBus) -> None:
        async for chunk in self._stream.inbound():
            await bus.publish(
                AudioChunk(
                    session_id=self._session_id,
                    speaker=Speaker.USER,
                    pcm16_16k=chunk,
                    ts=0.0,
                )
            )
```

The audio pump from `run` to the coach (`receive_audio`) is owned by
`telephony.py:media_stream`, not by the participant itself. This keeps the two
participants decoupled. The coordinator reads from `caller.audio_stream()` (or
equivalently iterates the bus for `AudioChunk(USER)`) and calls
`coach.receive_audio(chunk)`.

### 6.3 `ConsentGate`, `OutcomeProbe`, `PersonaSwapCoordinator`

Three identical changes — one per file:

```python
# Before
speak: Callable[[str], Awaitable[None]]
...
await self._speak(text)

# After
speaker: VoiceSpeaker
...
await self._speaker.say(SpeakRequest(text=text))
```

No logic changes. The `Speak = Callable[[str], Awaitable[None]]` type alias
in `persona_swap.py` is removed.

### 6.4 `SyntheticCaller` — no v1 changes

`SyntheticCaller.run(transport, runtime_phase)` uses the `TwoWayChannel` text
transport and is not changed. For a future `AudioCallerParticipant` that drives
the caller side with LLM + real-time TTS in eval, implement `VoiceParticipant`
as a new class in `rehearse/eval/customers/`. Its `run(bus)` subscribes to
`AudioChunk(COACH)` frames, TTS-plays them through the channel, and publishes
`AudioChunk(USER)` + `TranscriptDelta(USER)` frames back.

### 6.5 Observer pattern for three-way calls (design note, not v1)

The `FrameBus` already supports N subscribers. An observer (human listener,
logging service, supervisor model) can join any call by subscribing to the bus
without modifying either participant:

```python
async def human_observer_bridge(bus: FrameBus, twilio_ws: WebSocket) -> None:
    """Stream live transcript and coach audio to a third phone leg."""
    async for frame in bus.subscribe():
        if isinstance(frame, AudioChunk) and frame.speaker == Speaker.COACH:
            await stream_to_observer(twilio_ws, frame.pcm16_16k)
        elif isinstance(frame, TranscriptDelta) and frame.is_final:
            await push_transcript_line(twilio_ws, frame.text)
        elif isinstance(frame, EndOfCall):
            return
```

This requires no changes to `Speaker`, `TransportSide`, or either participant.
When three-way calls graduate from observer-only to full participant, `Speaker`
gains an `OBSERVER` variant and `ParticipantConfig.role` already has `"observer"`
defined.

---

## 7. Why not `TwilioBridgeTransport`?

`RuntimeHost._run_loop` is a text turn-taking loop:

```
receive text event → coach.respond(text) → send text event
```

This model assumes discrete turns and a synchronous request-response cycle.
The production telephony path is continuous audio streaming — Hume owns VAD,
turn detection, and TTS. Wrapping Twilio audio into `RuntimeHost._run_loop`
via a bridge transport would require:

1. VAD to segment audio into turns (replaces Hume's VAD).
2. STT to convert audio turns to text (replaces Hume's STT).
3. Pacing logic to buffer turns correctly.

That is a rebuild of Hume's front-end, not a simplification. `TwilioBridgeTransport`
makes sense only when replacing Hume wholesale with a separate STT + LLM + TTS
pipeline — at which point `RuntimeHost` and its text loop become the right seam.
That work is separate from this spec and should be driven by that future decision,
not anticipated here.

---

## 8. File inventory

| File | Change |
|---|---|
| `rehearse/participants.py` | **New.** `SpeakRequest`, `ParticipantConfig`, `VoiceSpeaker`, `VoiceParticipant` |
| `rehearse/services/hume_evi.py` | **New class** `HumeEVIParticipant(VoiceParticipant)`. `HumeEVIClient` unchanged. |
| `rehearse/telephony.py` | Wire `HumeEVIParticipant` + `TwilioCallerParticipant`. Pass `coach` (not `hume.say`) to business logic constructors. |
| `rehearse/audio/twilio_stream.py` | **New class** `TwilioCallerParticipant` in same module, or separate `rehearse/audio/caller_participant.py`. |
| `rehearse/consent.py` | `speak: Callable` → `speaker: VoiceSpeaker`. Call site update. |
| `rehearse/outcome.py` | `speak: Callable` → `speaker: VoiceSpeaker`. Call site update. |
| `rehearse/agents/persona_swap.py` | Remove `Speak` type alias. `speak: Speak` → `speaker: VoiceSpeaker`. Call site update. |
| `rehearse/types.py` | No v1 changes. `Speaker.OBSERVER` added when three-way is committed. |

---

## 9. Acceptance criteria

1. `rehearse/participants.py` defines `SpeakRequest`, `ParticipantConfig`,
   `VoiceSpeaker`, and `VoiceParticipant` with the exact signatures in §5.

2. `HumeEVIParticipant` satisfies `isinstance(participant, VoiceParticipant)`
   at runtime (protocol is `@runtime_checkable`).

3. `TwilioCallerParticipant` satisfies `isinstance(participant, VoiceParticipant)`
   at runtime.

4. `ConsentGate`, `OutcomeProbe`, and `PersonaSwapCoordinator` have no import
   of `HumeEVIClient` and no `Callable[[str], Awaitable[None]]` type annotation
   on their `speaker` parameter.

5. Existing tests in `tests/test_consent.py`, `tests/test_outcome.py`,
   `tests/test_persona_swap.py` pass without modification (mock objects that
   previously satisfied the `Callable` signature satisfy `VoiceSpeaker`
   structurally; if not, update mocks to add `.say(request)` method).

6. A smoke test asserts that an `HumeEVIParticipant` constructed with a mock
   socket correctly routes `say(SpeakRequest(text="hello"))` to
   `hume_client.say("hello")`.

7. `mypy --strict rehearse/participants.py` passes clean.
