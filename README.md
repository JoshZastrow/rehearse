# rehearse

Voice agent inference engine for real-time conversation coaching.

A 5-minute phone call: 1 minute intake, 3 minutes live practice with an AI counterparty, 1 minute feedback. Built as a prototype ML system — live sessions are the data source for continual improvement of a purpose-built voice model.

See [SPEC.md](SPEC.md) for the full design.

## Status

Scaffold only. No application logic yet.

## Stack

- Pipecat for the voice pipeline (frames, processors, transports)
- Hume EVI for speech-to-speech with prosody as first-class output
- Twilio for phone telephony
- Anthropic Claude for intake synthesis and feedback (v0)
- Pydantic for all data interfaces
- FastAPI for webhook endpoints
