/// <reference types="vite/client" />
import { useCallback, useEffect, useRef, useState } from 'react'
import { Room, RoomEvent, Track } from 'livekit-client'
import { useAuth } from '@clerk/clerk-react'

export type ConnectionState = 'idle' | 'connecting' | 'connected' | 'error'
export type Speaker = 'user' | 'agent' | null

export interface TranscriptEntry {
  id: string
  speaker: 'user' | 'agent'
  text: string
  final: boolean
}

export interface VoiceSession {
  connectionState: ConnectionState
  activeSpeaker: Speaker
  transcript: TranscriptEntry[]
  audioLevel: number
  connect: () => Promise<void>
  disconnect: () => void
  toggleMic: () => void
  isMicOn: boolean
}

const LIVEKIT_URL = import.meta.env.VITE_LIVEKIT_URL ?? ''
const TOKEN_ENDPOINT = import.meta.env.VITE_TOKEN_ENDPOINT ?? '/api/token'

interface TokenResponse {
  token: string
  room?: string
  url?: string
}

/** Fetch a LiveKit token from the gated token server using the Clerk JWT. */
async function fetchToken(clerkToken: string | null): Promise<TokenResponse> {
  const headers: Record<string, string> = {}
  if (clerkToken) headers.Authorization = `Bearer ${clerkToken}`
  const res = await fetch(TOKEN_ENDPOINT, { headers })
  if (!res.ok) {
    // 401 = not signed in, 402 = billing not set up, 429 = rate limited.
    throw new Error(`Token fetch failed: ${res.status}`)
  }
  return (await res.json()) as TokenResponse
}

export function useVoiceSession(): VoiceSession {
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle')
  const [activeSpeaker, setActiveSpeaker] = useState<Speaker>(null)
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([])
  const [audioLevel, setAudioLevel] = useState(0)
  const [isMicOn, setIsMicOn] = useState(true)

  const { getToken } = useAuth()
  const roomRef = useRef<Room | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const levelFrameRef = useRef<number>(0)

  const connect = useCallback(async () => {
    setConnectionState('connecting')
    try {
      const clerkToken = await getToken()
      const { token, url } = await fetchToken(clerkToken)
      const room = new Room({ adaptiveStream: true, dynacast: true })
      roomRef.current = room

      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) {
          track.attach()
        }
      })

      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        track.detach()
      })

      room.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
        const local = room.localParticipant
        if (speakers.some((s) => s.identity === local.identity)) {
          setActiveSpeaker('user')
        } else if (speakers.length > 0) {
          setActiveSpeaker('agent')
        } else {
          setActiveSpeaker(null)
        }
      })

      room.on(RoomEvent.DataReceived, (payload) => {
        try {
          const msg = JSON.parse(new TextDecoder().decode(payload)) as {
            type: string
            text?: string
            speaker?: string
            final?: boolean
            id?: string
          }
          if (msg.type === 'transcript') {
            setTranscript((prev) => {
              const id = msg.id ?? crypto.randomUUID()
              const existing = prev.findIndex((e) => e.id === id)
              const entry: TranscriptEntry = {
                id,
                speaker: (msg.speaker ?? 'agent') as 'user' | 'agent',
                text: msg.text ?? '',
                final: msg.final ?? false,
              }
              if (existing >= 0) {
                const updated = [...prev]
                updated[existing] = entry
                return updated
              }
              return [...prev, entry]
            })
          }
        } catch {
          // ignore malformed data
        }
      })

      await room.connect(url ?? LIVEKIT_URL, token)
      const micPub = await room.localParticipant.setMicrophoneEnabled(true)

      // Measure local mic level via Web Audio API
      const mediaTrack = micPub?.track?.mediaStreamTrack
      if (mediaTrack) {
        const audioCtx = new AudioContext()
        const source = audioCtx.createMediaStreamSource(new MediaStream([mediaTrack]))
        const analyser = audioCtx.createAnalyser()
        analyser.fftSize = 256
        source.connect(analyser)
        analyserRef.current = analyser
        const buf = new Uint8Array(analyser.frequencyBinCount)

        function pollLevel() {
          analyser.getByteFrequencyData(buf)
          const rms = Math.sqrt(buf.reduce((s, v) => s + v * v, 0) / buf.length) / 128
          setAudioLevel(rms)
          levelFrameRef.current = requestAnimationFrame(pollLevel)
        }
        levelFrameRef.current = requestAnimationFrame(pollLevel)
      }

      setConnectionState('connected')
    } catch (err) {
      console.error('LiveKit connect error:', err)
      setConnectionState('error')
    }
  }, [getToken])

  const disconnect = useCallback(() => {
    cancelAnimationFrame(levelFrameRef.current)
    analyserRef.current = null
    roomRef.current?.disconnect()
    roomRef.current = null
    setConnectionState('idle')
    setActiveSpeaker(null)
    setAudioLevel(0)
  }, [])

  const toggleMic = useCallback(() => {
    const room = roomRef.current
    if (!room) return
    const next = !isMicOn
    void room.localParticipant.setMicrophoneEnabled(next)
    setIsMicOn(next)
  }, [isMicOn])

  useEffect(() => () => disconnect(), [disconnect])

  return { connectionState, activeSpeaker, transcript, audioLevel, connect, disconnect, toggleMic, isMicOn }
}
