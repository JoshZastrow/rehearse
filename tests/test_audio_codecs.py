from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from rehearse.audio.mulaw import decode_mulaw, encode_pcm16
from rehearse.audio.resample import downsample_16k_to_8k, resample_pcm16, upsample_8k_to_16k


def _pcm16_sine(*, sample_rate: int, hz: float, frames: int, amplitude: int = 10_000) -> bytes:
    samples = [
        int(amplitude * math.sin(2 * math.pi * hz * idx / sample_rate))
        for idx in range(frames)
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


def test_mulaw_silence_round_trip() -> None:
    pcm = struct.pack("<4h", 0, 0, 0, 0)
    encoded = encode_pcm16(pcm)
    decoded = decode_mulaw(encoded)

    assert encoded == b"\xff\xff\xff\xff"
    assert decoded == pcm


def test_mulaw_round_trip_preserves_signal_shape() -> None:
    pcm = _pcm16_sine(sample_rate=8_000, hz=440, frames=160)
    decoded = decode_mulaw(encode_pcm16(pcm))

    original = np.frombuffer(pcm, dtype=np.int16).astype(np.int32)
    reconstructed = np.frombuffer(decoded, dtype=np.int16).astype(np.int32)
    mae = np.mean(np.abs(original - reconstructed))

    assert mae < 400


def test_mulaw_rejects_odd_length_pcm() -> None:
    with pytest.raises(ValueError):
        encode_pcm16(b"\x00")


def test_resample_identity_returns_original_bytes() -> None:
    pcm = _pcm16_sine(sample_rate=16_000, hz=440, frames=64)
    assert resample_pcm16(pcm, src_rate=16_000, dst_rate=16_000) == pcm


def test_resample_upsample_and_downsample_round_trip() -> None:
    pcm_8k = _pcm16_sine(sample_rate=8_000, hz=330, frames=80)
    pcm_16k = upsample_8k_to_16k(pcm_8k)
    round_trip = downsample_16k_to_8k(pcm_16k)

    assert len(pcm_16k) == len(pcm_8k) * 2
    original = np.frombuffer(pcm_8k, dtype=np.int16).astype(np.int32)
    reconstructed = np.frombuffer(round_trip, dtype=np.int16).astype(np.int32)
    mae = np.mean(np.abs(original - reconstructed))
    assert mae < 250


def test_resample_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        resample_pcm16(b"\x00", src_rate=8_000, dst_rate=16_000)
    with pytest.raises(ValueError):
        resample_pcm16(b"\x00\x00", src_rate=0, dst_rate=16_000)
