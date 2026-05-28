"""Tests for MoshiBackend components."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Pre-seed sys.modules with lightweight stubs for heavy optional dependencies
# (torch, moshi, sentencepiece) so that moshi_loader can be imported without
# these packages being installed in the test environment.
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> MagicMock:
    stub = MagicMock(spec=ModuleType)
    stub.__name__ = name
    stub.__spec__ = None
    return stub


_HEAVY_MODULES = [
    "torch",
    "moshi",
    "moshi.models",
    "moshi.models.loaders",
    "moshi.models.lm",
    "moshi.models.compression",
    "sentencepiece",
]

for _mod_name in _HEAVY_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _make_stub(_mod_name)  # type: ignore[assignment]

# Ensure the stubs expose the names that moshi_loader imports at module level.
_moshi_models = sys.modules["moshi.models"]
_moshi_loaders = sys.modules["moshi.models.loaders"]

if not isinstance(getattr(_moshi_models, "loaders", None), MagicMock):
    _moshi_models.loaders = _moshi_loaders  # type: ignore[attr-defined]
if not isinstance(getattr(_moshi_models, "LMGen", None), MagicMock):
    _moshi_models.LMGen = MagicMock(name="LMGen")  # type: ignore[attr-defined]
if not isinstance(getattr(_moshi_loaders, "CheckpointInfo", None), MagicMock):
    _moshi_loaders.CheckpointInfo = MagicMock(name="CheckpointInfo")  # type: ignore[attr-defined]
if not hasattr(_moshi_loaders, "get_mimi"):
    _moshi_loaders.get_mimi = MagicMock(name="get_mimi")  # type: ignore[attr-defined]
if not hasattr(_moshi_loaders, "get_moshi_lm"):
    _moshi_loaders.get_moshi_lm = MagicMock(name="get_moshi_lm")  # type: ignore[attr-defined]
# Constants used by _load_local to build file paths.
if not hasattr(_moshi_loaders, "MIMI_NAME"):
    _moshi_loaders.MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"  # type: ignore[attr-defined]
if not hasattr(_moshi_loaders, "MOSHI_NAME"):
    _moshi_loaders.MOSHI_NAME = "model.safetensors"  # type: ignore[attr-defined]
if not hasattr(_moshi_loaders, "TEXT_TOKENIZER_NAME"):
    _moshi_loaders.TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"  # type: ignore[attr-defined]

_sentencepiece = sys.modules["sentencepiece"]
if not hasattr(_sentencepiece, "SentencePieceProcessor"):
    _sentencepiece.SentencePieceProcessor = MagicMock(name="SentencePieceProcessor")  # type: ignore[attr-defined]


def test_loader_returns_models_from_local_path(tmp_path):
    """load_models() should use get_mimi and get_moshi_lm when given a local checkpoint dir."""
    from rehearse.backends.moshi_loader import load_models

    fake_mimi = MagicMock(name="mimi")
    fake_mimi.frame_size = 1920
    fake_lm = MagicMock(name="lm")
    fake_lm_gen = MagicMock(name="lm_gen")
    fake_tokenizer = MagicMock(name="tokenizer")

    checkpoint_dir = tmp_path / "moshi"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"fake")
    (checkpoint_dir / "tokenizer-e351c8d8-checkpoint125.safetensors").write_bytes(b"fake")
    (checkpoint_dir / "tokenizer_spm_32k_3.model").write_bytes(b"fake")

    with (
        patch("rehearse.backends.moshi_loader.loaders.get_mimi", return_value=fake_mimi) as mock_mimi,
        patch("rehearse.backends.moshi_loader.loaders.get_moshi_lm", return_value=fake_lm) as mock_lm,
        patch("rehearse.backends.moshi_loader.LMGen", return_value=fake_lm_gen),
        patch("sentencepiece.SentencePieceProcessor") as mock_sp,
    ):
        mock_sp.return_value = fake_tokenizer
        mimi, lm_gen, tokenizer = load_models(
            checkpoint_path=str(checkpoint_dir),
            hf_repo="kyutai/moshiko-pytorch-bf16",
            device="cpu",
        )

    assert mimi is fake_mimi
    assert lm_gen is fake_lm_gen
    assert tokenizer is fake_tokenizer
    mock_mimi.assert_called_once()
    mock_lm.assert_called_once()


def test_loader_falls_back_to_hf_when_no_local_path():
    """load_models() should use CheckpointInfo.from_hf_repo when checkpoint_path is empty."""
    from rehearse.backends.moshi_loader import load_models

    fake_info = MagicMock(name="checkpoint_info")
    fake_mimi = MagicMock(name="mimi")
    fake_mimi.frame_size = 1920
    fake_lm = MagicMock(name="lm")
    fake_lm_gen = MagicMock(name="lm_gen")
    fake_tokenizer = MagicMock(name="tokenizer")
    fake_info.get_mimi.return_value = fake_mimi
    fake_info.get_moshi.return_value = fake_lm
    fake_info.get_text_tokenizer.return_value = fake_tokenizer
    fake_info.lm_gen_config = {}

    with (
        patch("rehearse.backends.moshi_loader.CheckpointInfo.from_hf_repo", return_value=fake_info),
        patch("rehearse.backends.moshi_loader.LMGen", return_value=fake_lm_gen),
    ):
        mimi, lm_gen, tokenizer = load_models(
            checkpoint_path="",
            hf_repo="kyutai/moshiko-pytorch-bf16",
            device="cpu",
        )

    assert mimi is fake_mimi
    assert lm_gen is fake_lm_gen
    fake_info.get_mimi.assert_called_once_with(device="cpu")


def test_moshi_asr_returns_empty_on_empty_buffer():
    from unittest.mock import patch
    from rehearse.backends.moshi_asr import MoshiASR
    with patch("rehearse.backends.moshi_asr.WhisperModel"):
        asr = MoshiASR(model_size="tiny")
    result = asr.transcribe_and_reset()
    assert result == ""


def test_moshi_asr_transcribes_buffered_audio():
    import numpy as np
    from unittest.mock import MagicMock, patch
    from rehearse.backends.moshi_asr import MoshiASR

    mock_model = MagicMock()
    seg1 = MagicMock(); seg1.text = " Hello"
    seg2 = MagicMock(); seg2.text = " world"
    mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())

    with patch("rehearse.backends.moshi_asr.WhisperModel", return_value=mock_model):
        asr = MoshiASR(model_size="tiny")

    # 100ms of silence at 16kHz = 1600 samples × 2 bytes
    pcm16 = (np.zeros(1600, dtype=np.int16)).tobytes()
    asr.push_audio(pcm16)

    text = asr.transcribe_and_reset()
    assert text == "Hello world"
    mock_model.transcribe.assert_called_once()

    # Buffer is cleared after transcription
    assert asr.transcribe_and_reset() == ""
