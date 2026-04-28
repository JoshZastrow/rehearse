"""Encode and decode G.711 mu-law audio in pure Python.

This file handles the audio format Twilio Media Streams uses on the wire. It
lets the runtime convert between Twilio mu-law bytes and normal PCM16 samples.
"""

from __future__ import annotations

import struct

_BIAS = 0x84
_CLIP = 32635
_SEGMENT_END = (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF)


def encode_pcm16(pcm16: bytes) -> bytes:
    """Convert PCM16 mono audio bytes into mu-law bytes."""

    if len(pcm16) % 2 != 0:
        raise ValueError("pcm16 input must contain an even number of bytes")
    samples = struct.unpack(f"<{len(pcm16) // 2}h", pcm16)
    return bytes(_encode_sample(sample) for sample in samples)


def decode_mulaw(mulaw: bytes) -> bytes:
    """Convert mu-law bytes into PCM16 mono audio bytes."""

    samples = [_decode_sample(byte) for byte in mulaw]
    return struct.pack(f"<{len(samples)}h", *samples)


def _encode_sample(sample: int) -> int:
    """Convert one PCM16 sample into one mu-law byte."""
    if sample < 0:
        sign = 0x80
        sample = -sample
    else:
        sign = 0x00
    sample = min(sample, _CLIP) + _BIAS

    exponent = 7
    for idx, threshold in enumerate(_SEGMENT_END):
        if sample <= threshold:
            exponent = idx
            break

    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def _decode_sample(byte: int) -> int:
    """Convert one mu-law byte into one PCM16 sample."""
    ulaw = (~byte) & 0xFF
    sign = ulaw & 0x80
    exponent = (ulaw >> 4) & 0x07
    mantissa = ulaw & 0x0F
    sample = ((mantissa << 3) + _BIAS) << exponent
    sample -= _BIAS
    return -sample if sign else sample
