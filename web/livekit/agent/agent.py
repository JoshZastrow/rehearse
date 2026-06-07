"""LiveKit voice agent for Rehearse — prototype backend.

Connects to a LiveKit room as an agent, using:
  STT: Deepgram
  VAD: Silero
  LLM: OpenAI gpt-4o-mini
  TTS: Cartesia
  Turn detection: multilingual model

Environment variables required:
  LIVEKIT_URL          wss://your-project.livekit.cloud
  LIVEKIT_API_KEY      your api key
  LIVEKIT_API_SECRET   your api secret
  OPENAI_API_KEY       openai key
  DEEPGRAM_API_KEY     deepgram key
  CARTESIA_API_KEY     cartesia key
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai, silero

load_dotenv()
logging.basicConfig(level=logging.INFO)


SYSTEM_PROMPT = """You are a supportive AI coach for Rehearse, a platform that helps people
practice difficult conversations. Your role is to:
- Play the role of the person the user wants to practice talking to
- Give realistic, empathetic responses
- After each exchange, briefly note what went well

Keep responses concise and natural — this is a voice conversation."""


class RehearsalAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(
        ctx.room,
        agent=RehearsalAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=agents.noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and ask what kind of conversation they'd like to practice today."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
