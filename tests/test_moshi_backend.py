"""Tests for MoshiBackend components."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
        patch("rehearse.backends.moshi_loader.sentencepiece.SentencePieceProcessor") as mock_sp,
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
