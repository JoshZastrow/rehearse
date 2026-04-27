"""FastAPI application.

Hosts:
  - POST /sms/inbound    — Twilio inbound SMS webhook; triggers a new session
                           and places an outbound call.
  - POST /voice/outbound — Twilio voice webhook returning TwiML that opens a
                           bidirectional Media Streams WebSocket to this host.
  - WS   /voice/stream   — Twilio Media Streams endpoint. Bridges to the
                           Pipecat pipeline built by `rehearse.pipeline`.
  - GET  /sessions/{id}  — Static artifact viewer (serves web/viewer.html).

No business logic here. Wires routes to handlers and pipelines.
"""
