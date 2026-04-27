"""Tier-2 prosody: synthetic user → emotion-controlled TTS → Hume EVI.

Used for the ~20 of 100 validation examples. Synthesizes audio from the
synthetic user's planned utterances using a TTS that accepts emotion tags
(Hume Octave or ElevenLabs v3), pipes audio into a Hume EVI session, and
surfaces the real prosody events back to the eval pipeline.

The point is not to replace tier-1; it is to keep tier-1 honest.
"""
