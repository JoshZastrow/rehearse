"""Tests for annotate_session_async — the post-synthesis annotation step.

Modal is never contacted. process_remote.spawn is replaced with a fake that
returns a FunctionCall-shaped object whose .get() returns a canned alignment
dict. The test verifies that audio.json is written with the expected content.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from train.pipeline.annotate import annotate_session_async

_FAKE_RESULT = {
    "alignments": [
        ["Hello", [0.0, 0.4], "SPEAKER_MAIN"],
        ["world", [0.5, 0.9], "SPEAKER_MAIN"],
    ]
}

SESSION_ID = "test-session-abc123"


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    session_dir = tmp_path / SESSION_ID
    session_dir.mkdir()
    # Write a minimal valid WAV header so read_bytes() has something real.
    wav = session_dir / "audio.wav"
    wav.write_bytes(_minimal_wav())
    return session_dir


def _minimal_wav() -> bytes:
    """44-byte WAV with a single silent sample — enough for read_bytes()."""
    import struct
    num_samples = 1
    sample_rate = 16_000
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    chunk_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE",
        b"fmt ", 16, 1, num_channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    ) + b"\x00\x00"


@pytest.mark.asyncio
async def test_annotate_session_async_writes_audio_json(audio_dir: Path) -> None:
    audio_path = audio_dir / "audio.wav"
    fake_call = MagicMock()
    fake_call.get.return_value = _FAKE_RESULT

    with patch("train.pipeline.annotate.process_remote") as mock_remote:
        mock_remote.spawn.return_value = fake_call
        await annotate_session_async(SESSION_ID, audio_path)

    out = audio_dir / "audio.json"
    assert out.exists(), "audio.json was not written"
    written = json.loads(out.read_text())
    assert written == _FAKE_RESULT
    assert len(written["alignments"]) == 2
    assert written["alignments"][0] == ["Hello", [0.0, 0.4], "SPEAKER_MAIN"]


@pytest.mark.asyncio
async def test_annotate_session_async_skips_if_already_annotated(audio_dir: Path) -> None:
    audio_path = audio_dir / "audio.wav"
    existing = {"alignments": [["already", [0.0, 0.1], "SPEAKER_MAIN"]]}
    (audio_dir / "audio.json").write_text(json.dumps(existing))

    with patch("train.pipeline.annotate.process_remote") as mock_remote:
        await annotate_session_async(SESSION_ID, audio_path)
        mock_remote.spawn.assert_not_called()

    assert json.loads((audio_dir / "audio.json").read_text()) == existing


@pytest.mark.asyncio
async def test_annotate_session_async_skips_missing_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "no-audio.wav"  # does not exist

    with patch("train.pipeline.annotate.process_remote") as mock_remote:
        await annotate_session_async(SESSION_ID, audio_path)
        mock_remote.spawn.assert_not_called()
