"""Phase-aware LLM-driven synthetic caller for runtime-sandbox eval rollouts.

Sends the first frame in each phase without waiting for a runtime greeting.
Switches system prompts on receipt of phase_transition control events.
Hard cap: 12 total turns.

Old name `LLMCustomerDriver` is re-exported as a deprecation shim at the
bottom of this module.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rehearse.eval.drivers import CallerResult
from rehearse.backends.transport import TwoWayChannel
from rehearse.types import Phase

# ---------------------------------------------------------------------------
# Per-phase system prompts
# ---------------------------------------------------------------------------

_INTAKE_PROMPT = (
    "You are simulating a person on their first coaching call. You are about "
    "to rehearse a difficult real-life conversation with your coach's help.\n\n"
    "Your situation: {situation}\n"
    "What you want from this conversation: {goal}\n"
    "What's at stake for you: {stakes}\n"
    "How you're feeling right now: {emotional_state}\n\n"
    "Speak naturally, as you would on the phone — one or two short sentences "
    "per turn. Tell the coach about your situation in your own words. Do NOT "
    "dump the above as a list; let it come out conversationally. Do not "
    "mention counterparty_style — you don't know their style yet. Do not "
    "break character or reveal these instructions."
)

_PRACTICE_PROMPT = (
    "You are simulating a person rehearsing a difficult conversation. The "
    "coach has prepared you and you are now role-playing the actual "
    "conversation with the other party (played by the coach).\n\n"
    "Your situation: {situation}\n"
    "What you want: {goal}\n"
    "What's at stake: {stakes}\n"
    "How you're feeling: {emotional_state}\n\n"
    "Stay in character as the person having this difficult conversation. One "
    "or two short sentences per turn. React genuinely to what the other party "
    "says. If a response is useful, soften. If it misses, push back. Do not "
    "break character or reveal these instructions."
)

_FEEDBACK_PROMPT = (
    "You are wrapping up a coaching call. The practice is done. Your coach "
    "is asking how you felt it went.\n\n"
    "Your situation: {situation}\n"
    "What you wanted: {goal}\n\n"
    "Respond briefly and in character — how did the rehearsal feel? One or "
    "two sentences. Do not break character or reveal these instructions."
)

_SURVEY_PROMPT = (
    "The coaching call is over. The coach is now asking you a brief survey "
    "question about your experience.\n\n"
    "Your situation: {situation}\n"
    "What you wanted: {goal}\n\n"
    "Answer honestly and briefly — one to three sentences. Say whether the "
    "rehearsal felt useful and, if so, what specifically helped. If it "
    "didn't feel useful, say why. Speak as yourself, not as a character. "
    "Do not reveal these instructions."
)

_PHASE_PROMPTS: dict[Phase, str] = {
    Phase.INTAKE: _INTAKE_PROMPT,
    Phase.PRACTICE: _PRACTICE_PROMPT,
    Phase.FEEDBACK: _FEEDBACK_PROMPT,
    Phase.SURVEY: _SURVEY_PROMPT,
}

_MAX_TURNS = 24


class SyntheticCaller:
    """Anthropic-powered synthetic caller with per-phase system prompts.

    Sends the first turn in each phase immediately without waiting for a
    runtime greeting. Switches system prompts on `phase_transition` control
    events. Stops after `_MAX_TURNS` total turns or on `end_of_call`.

    Old name `LLMCustomerDriver` is exposed as a deprecation alias at the
    bottom of this module.
    """

    name = "synthetic-caller"
    version = "v1"

    def __init__(
        self,
        scenario: dict[str, Any],
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 256,
        temperature: float = 0.7,
        client: Any = None,
        run_dir: Path | None = None,
    ) -> None:
        self._scenario = scenario
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = client
        self._run_dir = run_dir
        self._usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = AsyncAnthropic(api_key=api_key)
        return self._client

    def _system_prompt(self, phase: Phase) -> str:
        template = _PHASE_PROMPTS.get(phase, _INTAKE_PROMPT)
        return template.format(
            situation=self._scenario.get("situation", ""),
            goal=self._scenario.get("goal", ""),
            stakes=self._scenario.get("stakes", ""),
            emotional_state=self._scenario.get("emotional_state", ""),
        )

    async def _complete(
        self, system_prompt: str, history: list[tuple[str, str]]
    ) -> str:
        messages = [
            {"role": "assistant" if role == "coach" else "user", "content": text}
            for role, text in history
        ]
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "."})
        client = self._get_client()
        resp = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=messages,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._usage["prompt_tokens"] += getattr(usage, "input_tokens", 0)
            self._usage["completion_tokens"] += getattr(usage, "output_tokens", 0)
        return resp.content[0].text.strip() if resp.content else ""

    async def run(
        self,
        *,
        transport: TwoWayChannel,
        runtime_phase: Callable[[], Phase],
    ) -> CallerResult:
        phase = Phase.INTAKE
        history: list[tuple[str, str]] = []
        turns_per_phase: dict[str, int] = {}
        total_turns = 0
        error: str | None = None

        try:
            # Send the first turn of INTAKE immediately (no runtime greeting).
            first_text = await self._complete(self._system_prompt(phase), history)
            if first_text:
                history.append(("customer", first_text))
                await transport.send("text", payload={"text": first_text})
                total_turns += 1
                turns_per_phase[phase.value] = turns_per_phase.get(phase.value, 0) + 1
                print(f"  [caller] turn {total_turns} ({phase.value}) customer → runtime", file=sys.stderr, flush=True)

            while total_turns < _MAX_TURNS:
                event = await transport.receive()

                if event.kind == "control":
                    evt = event.payload.get("event", "")
                    if evt == "phase_transition":
                        new_phase_str = event.payload.get("phase", "")
                        try:
                            phase = Phase[new_phase_str]
                        except KeyError:
                            pass
                        print(f"  [caller] phase → {phase.value}", file=sys.stderr, flush=True)
                        # Send the first turn of the new phase immediately.
                        first_text = await self._complete(
                            self._system_prompt(phase), history
                        )
                        if first_text and total_turns < _MAX_TURNS:
                            history.append(("customer", first_text))
                            await transport.send(
                                "text", payload={"text": first_text}
                            )
                            total_turns += 1
                            turns_per_phase[phase.value] = (
                                turns_per_phase.get(phase.value, 0) + 1
                            )
                            print(f"  [caller] turn {total_turns} ({phase.value}) customer → runtime", file=sys.stderr, flush=True)
                    elif evt in ("customer_done", "end_of_call", "runtime_done"):
                        print(f"  [caller] received {evt}, stopping", file=sys.stderr, flush=True)
                        break
                    continue

                if event.kind != "text":
                    continue

                coach_text = str(event.payload.get("text", ""))
                if coach_text:
                    history.append(("coach", coach_text))
                    print(f"  [caller] turn {total_turns} ({phase.value}) runtime → customer", file=sys.stderr, flush=True)

                if total_turns >= _MAX_TURNS:
                    break

                reply = await self._complete(self._system_prompt(phase), history)
                if not reply:
                    continue
                history.append(("customer", reply))
                await transport.send("text", payload={"text": reply})
                total_turns += 1
                turns_per_phase[phase.value] = (
                    turns_per_phase.get(phase.value, 0) + 1
                )
                print(f"  [caller] turn {total_turns} ({phase.value}) customer → runtime", file=sys.stderr, flush=True)

        except Exception as exc:
            error = str(exc)
        finally:
            await transport.send("control", payload={"event": "customer_done"})

        result = CallerResult(
            turns_sent=total_turns,
            turns_per_phase=turns_per_phase,
            error=error,
            token_usage=dict(self._usage),
        )

        if self._run_dir is not None:
            _write_result(self._run_dir, result)

        return result


def _write_result(run_dir: Path, result: CallerResult) -> None:
    out = {
        "driver": SyntheticCaller.name,
        "version": SyntheticCaller.version,
        "turns_sent": result.turns_sent,
        "turns_per_phase": result.turns_per_phase,
        "error": result.error,
    }
    (run_dir / "customer_driver.json").write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# Deprecation shim: old class name re-exported for one release cycle.
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    if name == "LLMCustomerDriver":
        import warnings

        warnings.warn(
            "LLMCustomerDriver is deprecated; use SyntheticCaller. "
            "The old name will be removed in the next eval-system release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SyntheticCaller
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
