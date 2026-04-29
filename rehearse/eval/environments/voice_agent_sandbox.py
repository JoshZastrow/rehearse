"""Sandboxed voice-agent environment.

This is the no-Twilio eval entry point. It connects a simulated customer agent
to a sandboxed voice-agent runtime through ``InMemoryDuplexTransport`` and lets
the two agents run their own loops. The implementation is intentionally small
today, but the boundary matches where a Hume-backed runtime adapter will plug in.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.sandbox_agents import SandboxAgent, SandboxAgentRunner, SandboxAgentRunResult
from rehearse.eval.sandbox_connection import SandboxConnection
from rehearse.eval.sandboxes import CustomerAgentSandbox, SandboxHandle, VoiceAgentSandbox
from rehearse.eval.transports import RuntimeDuplexEndpoint, TransportEvent
from rehearse.types import ConsentState, Phase, Session, Speaker, TranscriptFrame


@dataclass
class SandboxAgentContext:
    handle: SandboxHandle
    example: BenchmarkExample
    transport: RuntimeDuplexEndpoint


class ScriptedCustomerAgent:
    """Customer agent that places the simulated call by sending scripted turns."""

    name = "scripted-customer-agent"
    version = "v0"

    async def run(
        self,
        input: Any,
        *,
        context: SandboxAgentContext | None = None,
        max_turns: int = 10,
    ) -> SandboxAgentRunResult:
        if context is None:
            raise ValueError("ScriptedCustomerAgent requires SandboxAgentContext")
        turns = _customer_turns(input)
        sent: list[str] = []
        for idx, text in enumerate(turns[:max_turns]):
            sent.append(text)
            await context.transport.send(
                "text",
                payload={"text": text, "turn_index": idx, "role": "customer"},
            )
        await context.transport.send("control", payload={"event": "customer_done"})
        return SandboxAgentRunResult(
            final_output={"turns_sent": len(sent)},
            metadata={"turns": sent},
        )


class StubVoiceAgent:
    """Minimal voice-agent stand-in used until the real runtime adapter lands."""

    name = "stub-voice-agent"
    version = "v0"

    async def run(
        self,
        input: Any,
        *,
        context: SandboxAgentContext | None = None,
        max_turns: int = 10,
    ) -> SandboxAgentRunResult:
        if context is None:
            raise ValueError("StubVoiceAgent requires SandboxAgentContext")
        heard: list[str] = []
        replies: list[str] = []
        while len(heard) < max_turns:
            event = await context.transport.receive()
            if event.kind == "control" and event.payload.get("event") == "customer_done":
                break
            if event.kind != "text":
                continue
            text = str(event.payload.get("text", ""))
            heard.append(text)
            reply = f"I heard: {text}"
            replies.append(reply)
            await context.transport.send(
                "text",
                payload={
                    "text": reply,
                    "turn_index": len(replies) - 1,
                    "role": "voice_agent",
                },
            )
        await context.transport.send("control", payload={"event": "runtime_done"})
        return SandboxAgentRunResult(
            final_output={"turns_received": len(heard), "turns_sent": len(replies)},
            metadata={"heard": heard, "replies": replies},
        )


class _AgentRuntimeAdapter:
    def __init__(self, agent: SandboxAgent, *, input_key: str, max_turns: int) -> None:
        self.agent = agent
        self.name = agent.name
        self.version = agent.version
        self.input_key = input_key
        self.max_turns = max_turns
        self.result: SandboxAgentRunResult | None = None
        self._task: asyncio.Task[SandboxAgentRunResult] | None = None

    async def start(
        self,
        handle: SandboxHandle,
        example: BenchmarkExample,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> None:
        if transport is None:
            raise ValueError(f"{self.name} requires a runtime transport endpoint")
        context = SandboxAgentContext(handle=handle, example=example, transport=transport)
        agent_input = example.payload.get(self.input_key, example.payload)
        self._task = asyncio.create_task(
            SandboxAgentRunner.run(
                self.agent,
                agent_input,
                context=context,
                max_turns=self.max_turns,
            )
        )

    async def wait(self, timeout_s: float) -> SandboxAgentRunResult:
        if self._task is None:
            raise RuntimeError(f"{self.name} has not been started")
        self.result = await asyncio.wait_for(self._task, timeout=timeout_s)
        return self.result

    async def close(self) -> None:
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task


class VoiceAgentSandboxEnvironment:
    name = "voice-agent-sandbox"
    version = "v0"

    def __init__(self, model_slots: dict[str, str] | None = None) -> None:
        self.model_slots = model_slots or {}

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult:
        started = datetime.now()
        run_dir.mkdir(parents=True, exist_ok=True)
        timeout_s = float(example.payload.get("timeout_s", self.model_slots.get("timeout_s", 10)))
        max_turns = int(example.payload.get("max_turns", self.model_slots.get("max_turns", 10)))

        customer_runtime = _AgentRuntimeAdapter(
            ScriptedCustomerAgent(),
            input_key="customer_script",
            max_turns=max_turns,
        )
        voice_runtime = _AgentRuntimeAdapter(
            StubVoiceAgent(),
            input_key="runtime_input",
            max_turns=max_turns,
        )
        connection = SandboxConnection(
            customer=CustomerAgentSandbox(customer_agent=customer_runtime),
            runtime=VoiceAgentSandbox(
                runtime=voice_runtime,
                model_slots=self.model_slots,
            ),
        )

        try:
            async with connection.lifecycle(example=example, run_dir=run_dir, rng_seed=rng_seed):
                customer_result, runtime_result = await asyncio.gather(
                    customer_runtime.wait(timeout_s),
                    voice_runtime.wait(timeout_s),
                )
        except TimeoutError:
            completed = datetime.now()
            await connection.close()
            return RolloutResult(
                example_id=example.id,
                target_name=self.name,
                target_version=self.version,
                status="timeout",
                started_at=started,
                completed_at=completed,
                duration_ms=int((completed - started).total_seconds() * 1000),
                artifacts_dir=run_dir,
                error=f"voice-agent sandbox exceeded {timeout_s}s",
            )
        except Exception as exc:
            completed = datetime.now()
            await connection.close()
            return RolloutResult(
                example_id=example.id,
                target_name=self.name,
                target_version=self.version,
                status="error",
                started_at=started,
                completed_at=completed,
                duration_ms=int((completed - started).total_seconds() * 1000),
                artifacts_dir=run_dir,
                error=f"{type(exc).__name__}: {exc}",
            )

        session_id = f"sandbox-{example.id}"
        _write_artifacts(
            run_dir=run_dir,
            session_id=session_id,
            events=connection.transport.events,
        )
        completed = datetime.now()
        return RolloutResult(
            example_id=example.id,
            target_name=self.name,
            target_version=self.version,
            status="ok",
            started_at=started,
            completed_at=completed,
            duration_ms=int((completed - started).total_seconds() * 1000),
            artifacts_dir=run_dir,
            payload={
                "session_id": session_id,
                "transport_events": len(connection.transport.events),
                "customer_output": customer_result.final_output,
                "runtime_output": runtime_result.final_output,
            },
        )


def _customer_turns(input: Any) -> list[str]:
    if isinstance(input, list):
        return [str(turn) for turn in input]
    if isinstance(input, str):
        return [input]
    if isinstance(input, dict):
        raw = input.get("customer_script") or input.get("script")
        if isinstance(raw, list):
            return [str(turn) for turn in raw]
        prompt = input.get("prompt") or input.get("situation") or input.get("text")
        if prompt:
            return [str(prompt)]
    return ["Hello, I am ready to rehearse."]


def _write_artifacts(
    *,
    run_dir: Path,
    session_id: str,
    events: list[TransportEvent],
) -> None:
    conversation_path = run_dir / "conversation.jsonl"
    transcript_path = run_dir / "transcript.jsonl"
    with conversation_path.open("w") as f:
        for event in events:
            f.write(
                json.dumps(
                    {
                        "id": event.id,
                        "source": event.source,
                        "kind": event.kind,
                        "payload": event.payload,
                        "created_at": event.created_at.isoformat(),
                    }
                )
                + "\n"
            )

    utterance_idx = 0
    with transcript_path.open("w") as f:
        for event in events:
            if event.kind != "text":
                continue
            speaker = Speaker.USER if event.source == "customer" else Speaker.COACH
            frame = TranscriptFrame(
                session_id=session_id,
                utterance_id=f"utt-{utterance_idx:04d}",
                ts_start=float(utterance_idx),
                ts_end=float(utterance_idx + 1),
                speaker=speaker,
                phase=Phase.PRACTICE,
                text=str(event.payload.get("text", "")),
            )
            f.write(frame.model_dump_json() + "\n")
            utterance_idx += 1

    session = Session(
        id=session_id,
        created_at=datetime.now(),
        consent=ConsentState.GRANTED,
        completion_status="complete",
        artifact_paths={
            "conversation": str(conversation_path),
            "transcript": str(transcript_path),
        },
        pipeline_version=f"{VoiceAgentSandboxEnvironment.name}@{VoiceAgentSandboxEnvironment.version}",
    )
    (run_dir / "session.json").write_text(session.model_dump_json(indent=2))
