"""Pipecat pipeline construction.

`build_pipeline(transport, session)` returns a configured Pipecat Pipeline:

    Transport (in)
      → HumeEVIService (speech-to-speech; emits transcript + prosody frames)
      → PhaseProcessor (owns phase state, emits transition/persona frames)
      → ArtifactWriters (transcript, prosody, audio, telemetry)
      → HumeEVIService (persona config reconfigured on PersonaSwitchFrame)
      → Transport (out)

The same builder is used in production (Twilio transport) and in eval
(SimulatedTransport that feeds synthetic frames). This is the load-bearing
decoupling that makes eval validate the real system.
"""
