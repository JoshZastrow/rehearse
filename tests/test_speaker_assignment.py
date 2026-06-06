"""Unit tests for speaker assignment helpers in annotate.py.

Tests the three paths:
1. Diarization segments + speaker ID mapping → word labels
2. Transcript timestamp fallback → word labels
3. _map_speaker_ids_to_roles anchor logic

Roles are 'provider' (the coach model) and 'caller' (the user). The helpers
accept both the legacy transcript labels ("coach"/"user") and the current ones
("provider"/"caller") on input, and always emit the current labels.
"""

from __future__ import annotations

from train.pipeline.annotate import (
    _assign_speakers_from_segments,
    _assign_speakers_from_transcript,
    _map_speaker_ids_to_roles,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

_SEGMENTS = [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
    {"speaker": "SPEAKER_01", "start": 3.5, "end": 7.0},
    {"speaker": "SPEAKER_00", "start": 7.5, "end": 10.0},
]

# Legacy "coach"/"user" labels on input exercise the back-compat mapping.
_TRANSCRIPT = [
    {"ts_start": 0.0, "ts_end": 3.0, "speaker": "coach", "is_interim": False},
    {"ts_start": 3.5, "ts_end": 7.0, "speaker": "user",  "is_interim": False},
    {"ts_start": 7.5, "ts_end": 10.0, "speaker": "coach", "is_interim": False},
]

_WORDS = [
    {"text": "Hello",   "start": 0.5,  "end": 0.9},   # inside SPEAKER_00 / provider
    {"text": "there",   "start": 1.0,  "end": 1.3},   # inside SPEAKER_00 / provider
    {"text": "Yes",     "start": 4.0,  "end": 4.3},   # inside SPEAKER_01 / caller
    {"text": "exactly", "start": 8.0,  "end": 8.5},   # inside SPEAKER_00 / provider
]


# ─── _map_speaker_ids_to_roles ────────────────────────────────────────────────

def test_map_ids_uses_first_provider_utterance_as_anchor():
    id_to_role = _map_speaker_ids_to_roles(_SEGMENTS, _TRANSCRIPT)
    # First provider utterance is 0.0–3.0, which overlaps SPEAKER_00
    assert id_to_role["SPEAKER_00"] == "provider"
    assert id_to_role["SPEAKER_01"] == "caller"


def test_map_ids_fallback_when_no_transcript():
    id_to_role = _map_speaker_ids_to_roles(_SEGMENTS, [])
    assert id_to_role["SPEAKER_00"] == "provider"
    assert id_to_role["SPEAKER_01"] == "caller"


# ─── _assign_speakers_from_segments ──────────────────────────────────────────

def test_assign_from_segments_correct_labels():
    id_to_role = _map_speaker_ids_to_roles(_SEGMENTS, _TRANSCRIPT)
    labels = _assign_speakers_from_segments(_WORDS, _SEGMENTS, id_to_role)
    assert labels == ["provider", "provider", "caller", "provider"]


def test_assign_from_segments_no_overlap_defaults_to_caller():
    words = [{"text": "gap", "start": 3.1, "end": 3.2}]  # between segments
    id_to_role = {"SPEAKER_00": "provider", "SPEAKER_01": "caller"}
    labels = _assign_speakers_from_segments(words, _SEGMENTS, id_to_role)
    assert labels == ["caller"]


# ─── _assign_speakers_from_transcript ────────────────────────────────────────

def test_assign_from_transcript_correct_labels():
    labels = _assign_speakers_from_transcript(_WORDS, _TRANSCRIPT)
    assert labels == ["provider", "provider", "caller", "provider"]


def test_assign_from_transcript_no_match_defaults_to_caller():
    words = [{"text": "silence", "start": 11.0, "end": 11.5}]  # past all utterances
    labels = _assign_speakers_from_transcript(words, _TRANSCRIPT)
    assert labels == ["caller"]


def test_assign_from_transcript_unknown_speaker_defaults_to_caller():
    transcript = [{"ts_start": 0.0, "ts_end": 5.0, "speaker": "unknown", "is_interim": False}]
    words = [{"text": "word", "start": 1.0, "end": 1.5}]
    labels = _assign_speakers_from_transcript(words, transcript)
    assert labels == ["caller"]
