"""Pure-Python G.711 mu-law codec helpers."""

from __future__ import annotations

import struct

_BIAS = 0x84
_CLIP = 32635
_SEGMENT_END = (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF)


def encode_pcm16(pcm16: bytes) -> bytes:
    """Encode little-endian PCM16 mono audio to mu-law bytes."""

    if len(pcm16) % 2 != 0:
        raise ValueError("pcm16 input must contain an even number of bytes")
    samples = struct.unpack(f"<{len(pcm16) // 2}h", pcm16)
    return bytes(_encode_sample(sample) for sample in samples)


def decode_mulaw(mulaw: bytes) -> bytes:
    """Decode mu-law bytes to little-endian PCM16 mono audio."""

    samples = [_decode_sample(byte) for byte in mulaw]
    return struct.pack(f"<{len(samples)}h", *samples)


def _encode_sample(sample: int) -> int:
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
    ulaw = (~byte) & 0xFF
    sign = ulaw & 0x80
    exponent = (ulaw >> 4) & 0x07
    mantissa = ulaw & 0x0F
    sample = ((mantissa << 3) + _BIAS) << exponent
    sample -= _BIAS
    return -sample if sign else sample
