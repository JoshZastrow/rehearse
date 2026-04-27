"""Session artifact I/O.

The asset layer. Reads and writes session directories as defined in SPEC §5.2:

    sessions/{session_id}/
      intake.json        IntakeRecord
      story.md           markdown
      transcript.jsonl   TranscriptFrame per line
      prosody.jsonl      ProsodyFrame per line
      audio.wav          PCM audio
      feedback.md        markdown
      session.json       Session (the manifest)
      telemetry.jsonl    InferenceLogEntry per line

Every writer is append-only where possible so a mid-call crash leaves a
replayable partial session.
"""
