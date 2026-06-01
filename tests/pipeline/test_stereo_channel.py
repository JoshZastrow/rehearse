import json
import math
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def patch_cuda(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)


def _make_mock_mimi(dep_q: int = 8, frame_rate: float = 12.5, sample_rate: int = 24000):
    """Mock Mimi that records the input it was called with and returns zero tokens."""
    mimi = MagicMock()
    mimi.frame_rate = frame_rate
    mimi.sample_rate = sample_rate
    calls = []

    def encode(x):
        calls.append(x.clone().cpu())
        B, C, T = x.shape
        T_frames = max(1, math.ceil(T * frame_rate / sample_rate))
        return torch.zeros(B, dep_q, T_frames)

    mimi.encode.side_effect = encode
    mimi._calls = calls
    return mimi


def _make_mock_interleaver(zero_padding: int = 0, frame_rate: float = 12.5):
    iv = MagicMock()
    iv.zero_padding = zero_padding
    iv.audio_frame_rate = frame_rate
    iv.prepare_item.return_value = torch.zeros(1, 1, 13)  # ceil(1.0 * 12.5) = 13
    return iv


def _make_tokenizer(channel: int):
    from finetune.data.interleaver import InterleavedTokenizer
    mimi = _make_mock_mimi()
    interleaver = _make_mock_interleaver()
    tok = InterleavedTokenizer(mimi, interleaver, duration_sec=1.0, channel=channel)
    return tok, mimi


def test_stereo_channel0_selects_left(tmp_path):
    """Channel 0 encodes only the left (provider) track."""
    tok, mimi = _make_tokenizer(channel=0)

    # Stereo wav: left=1.0, right=-1.0
    T = 24000
    wav = np.stack([np.ones(T), -np.ones(T)])  # [2, T]

    info = {"alignments": []}
    p = tmp_path / "audio_stereo.wav"
    p.touch()
    (tmp_path / "audio_stereo.json").write_text(json.dumps(info))

    tok(wav, 0.0, str(p))

    assert len(mimi._calls) == 1
    encoded_input = mimi._calls[0]  # [1, 1, T]
    assert encoded_input.shape[0] == 1, "must be batch size 1 (mono)"
    assert float(encoded_input.mean()) == pytest.approx(1.0, abs=1e-5), \
        "channel 0 (all 1.0) should be encoded, not channel 1 (all -1.0)"


def test_stereo_channel1_selects_right(tmp_path):
    """Channel 1 encodes only the right (caller) track."""
    tok, mimi = _make_tokenizer(channel=1)

    T = 24000
    wav = np.stack([np.ones(T), -np.ones(T)])  # [2, T]; right=-1.0

    info = {"alignments": []}
    p = tmp_path / "audio_stereo.wav"
    p.touch()
    (tmp_path / "audio_stereo.json").write_text(json.dumps(info))

    tok(wav, 0.0, str(p))

    assert len(mimi._calls) == 1
    encoded_input = mimi._calls[0]
    assert float(encoded_input.mean()) == pytest.approx(-1.0, abs=1e-5), \
        "channel 1 (all -1.0) should be encoded"


def test_mono_input_unaffected_by_channel(tmp_path):
    """Mono wav [1, T] passes through correctly regardless of channel setting."""
    tok, mimi = _make_tokenizer(channel=0)

    T = 24000
    wav = np.ones((1, T))  # [1, T] mono

    info = {"alignments": []}
    p = tmp_path / "audio.wav"
    p.touch()
    (tmp_path / "audio.json").write_text(json.dumps(info))

    tok(wav, 0.0, str(p))

    assert len(mimi._calls) == 1
    encoded_input = mimi._calls[0]
    # The mock captures the raw input to encode() before any downsampling,
    # so T is the full sample count, not the number of audio frames.
    assert encoded_input.shape == (1, 1, T), "mono input shape to encode must be [1, 1, T]"


def test_stereo_produces_correct_code_shape(tmp_path):
    """Stereo input with channel selection must produce codes with K=9 (1 text + 8 audio)."""
    tok, _ = _make_tokenizer(channel=0)

    T = 24000
    wav = np.random.randn(2, T).astype(np.float32)

    info = {"alignments": []}
    p = tmp_path / "audio_stereo.wav"
    p.touch()
    (tmp_path / "audio_stereo.json").write_text(json.dumps(info))

    sample = tok(wav, 0.0, str(p))
    # codes shape: [1, K, T_frames] where K = 1 (text) + 8 (audio) = 9
    assert sample.codes.shape[1] == 9, \
        f"Expected 9 codebooks (1 text + 8 audio), got {sample.codes.shape[1]}"


def test_channel_out_of_bounds_raises(tmp_path):
    """Selecting a channel index beyond the number of channels raises ValueError."""
    tok, _ = _make_tokenizer(channel=5)
    T = 24000
    wav = np.stack([np.ones(T), -np.ones(T)])  # [2, T], valid channels: 0, 1
    info = {"alignments": []}
    p = tmp_path / "audio_stereo.wav"
    p.touch()
    (tmp_path / "audio_stereo.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="channel=5 out of range"):
        tok(wav, 0.0, str(p))
