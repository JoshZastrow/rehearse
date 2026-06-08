# LiveKit WebRTC Architecture — Rehearse Prototype

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User's Browser                              │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  React + Vite App  (web/livekit/app/)                        │  │
│   │                                                              │  │
│   │  / ──────────────── Design Selector                         │  │
│   │  /design/gemini ─── Design 1: Gemini Style                  │  │
│   │  /design/minimal ── Design 2: Minimal Orb                   │  │
│   │  /design/waveform ─ Design 3: Waveform Split                │  │
│   │                                                              │  │
│   │  useVoiceSession() hook                                      │  │
│   │   ├─ livekit-client  ←──── WebRTC (DTLS/SRTP/ICE)           │  │
│   │   └─ transcript state ←─── DataChannel (JSON events)        │  │
│   └───────────────────────────────────┬──────────────────────────┘  │
└───────────────────────────────────────│─────────────────────────────┘
                                        │  WebRTC / WebSocket
                              ┌─────────▼──────────┐
                              │   LiveKit Server    │
                              │  (Cloud or self-    │
                              │   hosted)           │
                              └──────┬──────────────┘
                                     │  Worker SDK
                              ┌──────▼──────────────────────────────┐
                              │  Python Agent  (web/livekit/agent/) │
                              │                                     │
                              │  entrypoint(ctx: JobContext)        │
                              │   │                                 │
                              │   ├─ Deepgram STT ─── PCM audio →   │
                              │   │                   transcript    │
                              │   ├─ Silero VAD ─── voice activity  │
                              │   ├─ OpenAI LLM ─── GPT-4o-mini     │
                              │   ├─ Cartesia TTS ── audio output   │
                              │   └─ Turn Detector ─ natural turns  │
                              └─────────────────────────────────────┘
```

## Data Flow — Single Session

```
Browser mic (PCM16/48kHz)
        │
        │  WebRTC audio track
        ▼
LiveKit Server ──────────────── Agent receives audio
                                        │
                                   Silero VAD
                                        │ voice detected
                                   Deepgram STT
                                        │ transcript chunk
                                   OpenAI LLM
                                        │ text response
                                   Cartesia TTS
                                        │ PCM audio
                LiveKit Server ◄─────────┘
        │
        │  WebRTC audio track
        ▼
Browser speaker                  DataChannel ──► transcript JSON
                                                  → useVoiceSession()
                                                  → UI transcript state
```

## Token Flow

```
Browser                  FastAPI (rehearse server)           LiveKit Cloud
   │                              │                               │
   │── GET /api/livekit/token ──►│                               │
   │                              │── POST /api/token ──────────►│
   │                              │◄─ {token: "eyJ..."} ─────────│
   │◄─ {token: "eyJ..."} ─────────│                               │
   │                              │                               │
   │──── room.connect(url, token) ──────────────────────────────►│
   │◄─── WebRTC negotiation ────────────────────────────────────►│
```

## Component Map

```
web/livekit/
├── agent/
│   ├── agent.py          ← LiveKit WorkerOptions entrypoint
│   ├── requirements.txt  ← livekit-agents + plugins
│   └── .env.example      ← LIVEKIT_URL, API keys
└── app/
    ├── src/
    │   ├── hooks/
    │   │   └── useVoiceSession.ts   ← shared LiveKit room state
    │   ├── designs/
    │   │   ├── Design1Gemini.tsx    ← streaming transcript + orb bar
    │   │   ├── Design2MinimalOrb.tsx ← canvas morphing blob
    │   │   └── Design3Waveform.tsx  ← split transcript + waveform bars
    │   ├── App.tsx                  ← design selector home
    │   └── main.tsx                 ← React Router root
    ├── package.json
    └── vite.config.ts
```
