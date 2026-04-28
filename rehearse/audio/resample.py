"""PCM16 mono resampling helpers."""

from __future__ import annotations

import numpy as np


def resample_pcm16(pcm16: bytes, *, src_rate: int, dst_rate: int) -> bytes:
    """Resample little-endian PCM16 mono audio with linear interpolation."""

    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if len(pcm16) % 2 != 0:
        raise ValueError("pcm16 input must contain an even number of bytes")
    if src_rate == dst_rate or not pcm16:
        return pcm16

    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float64)
    src_positions = np.arange(len(samples), dtype=np.float64)
    dst_length = max(1, int(round(len(samples) * dst_rate / src_rate)))
    dst_positions = np.linspace(0, len(samples) - 1, num=dst_length)
    resampled = np.interp(dst_positions, src_positions, samples)
    clipped = np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)
    return clipped.tobytes()


def upsample_8k_to_16k(pcm16_8k: bytes) -> bytes:
    """Upsample PCM16 mono audio from 8kHz to 16kHz."""

    return resample_pcm16(pcm16_8k, src_rate=8_000, dst_rate=16_000)


def downsample_16k_to_8k(pcm16_16k: bytes) -> bytes:
    """Downsample PCM16 mono audio from 16kHz to 8kHz."""

    return resample_pcm16(pcm16_16k, src_rate=16_000, dst_rate=8_000)
