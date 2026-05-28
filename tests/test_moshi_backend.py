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
    "torchaudio",
    "torchaudio.functional",
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

# Wire torchaudio.functional into the torchaudio stub.
_torchaudio = sys.modules["torchaudio"]
_torchaudio_functional = sys.modules["torchaudio.functional"]
if not isinstance(getattr(_torchaudio, "functional", None), MagicMock):
    _torchaudio.functional = _torchaudio_functional  # type: ignore[attr-defined]

# Make torch attrs usable in the stub (spec=ModuleType blocks unknown attrs).
_torch = sys.modules["torch"]

from contextlib import contextmanager as _cm

@_cm
def _no_grad():
    yield

_torch.no_grad = _no_grad  # type: ignore[attr-defined]
# from_numpy: return a MagicMock that supports .float().unsqueeze()
_torch.from_numpy = MagicMock(name="torch.from_numpy")  # type: ignore[attr-defined]
_torch.int16 = MagicMock(name="torch.int16")  # type: ignore[attr-defined]

# torchaudio.functional.resample is also spec-blocked — add it explicitly.
_torchaudio_functional.resample = MagicMock(name="torchaudio.functional.resample")  # type: ignore[attr-defined]

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


# ---- Task 4: MoshiBackend tests ----

import threading
import queue as _queue
import concurrent.futures as _futures
from contextlib import contextmanager


@contextmanager
def _noop_ctx():
    yield


@pytest.mark.asyncio
async def test_moshi_backend_satisfies_protocol():
    from rehearse.backends.base import ConversationBackend
    from rehearse.backends.moshi import MoshiBackend

    fake_mimi = MagicMock(); fake_mimi.frame_size = 1920
    fake_lm_gen = MagicMock()
    fake_tokenizer = MagicMock()
    fake_asr = MagicMock()

    with (
        patch("rehearse.backends.moshi.load_models", return_value=(fake_mimi, fake_lm_gen, fake_tokenizer)),
        patch("rehearse.backends.moshi.MoshiASR", return_value=fake_asr),
    ):
        backend = MoshiBackend(
            checkpoint_path="",
            hf_repo="kyutai/moshiko-pytorch-bf16",
            device="cpu",
            asr_model="tiny",
        )

    assert isinstance(backend, ConversationBackend)


@pytest.mark.asyncio
async def test_moshi_backend_start_launches_task():
    import asyncio
    from rehearse.backends.moshi import MoshiBackend
    from rehearse.bus import FrameBus

    fake_mimi = MagicMock(); fake_mimi.frame_size = 1920
    fake_lm_gen = MagicMock()
    fake_tokenizer = MagicMock()
    fake_asr = MagicMock()

    with (
        patch("rehearse.backends.moshi.load_models", return_value=(fake_mimi, fake_lm_gen, fake_tokenizer)),
        patch("rehearse.backends.moshi.MoshiASR", return_value=fake_asr),
    ):
        backend = MoshiBackend(
            checkpoint_path="",
            hf_repo="kyutai/moshiko-pytorch-bf16",
            device="cpu",
            asr_model="tiny",
        )

    fake_mimi.streaming_forever = MagicMock(return_value=_noop_ctx())
    fake_lm_gen.streaming_forever = MagicMock(return_value=_noop_ctx())
    fake_asr.transcribe_and_reset.return_value = ""

    bus = FrameBus(session_id="test-start")
    await backend.start("test-start", bus)
    assert backend._task is not None
    assert not backend._task.done()
    await backend.close()


@pytest.mark.asyncio
async def test_send_caller_audio_puts_to_queue():
    from rehearse.backends.moshi import MoshiBackend
    from rehearse.bus import FrameBus

    fake_mimi = MagicMock(); fake_mimi.frame_size = 1920
    fake_lm_gen = MagicMock()
    fake_tokenizer = MagicMock()
    fake_asr = MagicMock()

    with (
        patch("rehearse.backends.moshi.load_models", return_value=(fake_mimi, fake_lm_gen, fake_tokenizer)),
        patch("rehearse.backends.moshi.MoshiASR", return_value=fake_asr),
    ):
        backend = MoshiBackend(
            checkpoint_path="",
            hf_repo="kyutai/moshiko-pytorch-bf16",
            device="cpu",
            asr_model="tiny",
        )

    fake_mimi.streaming_forever = MagicMock(return_value=_noop_ctx())
    fake_lm_gen.streaming_forever = MagicMock(return_value=_noop_ctx())
    fake_asr.transcribe_and_reset.return_value = ""

    bus = FrameBus(session_id="test-queue")
    await backend.start("test-queue", bus)

    pcm = b"\x00" * 640
    await backend.send_caller_audio(pcm)
    assert backend._audio_q.qsize() == 1

    await backend.close()


def test_create_backend_moshi_returns_moshi_backend():
    from pathlib import Path
    from rehearse.backends.factory import create_backend
    from rehearse.backends.moshi import MoshiBackend
    from rehearse.config import RuntimeConfig

    cfg = RuntimeConfig(
        twilio_account_sid="x", twilio_auth_token="x",
        twilio_from_number="x", public_base_url="x",
        hume_api_key="x", hume_config_id="x",
        session_root=Path("/tmp"),
        backend_type="moshi",
        moshi_checkpoint_path="",
        moshi_hf_repo="kyutai/moshiko-pytorch-bf16",
        moshi_device="cpu",
        moshi_asr_model="tiny",
    )
    fake_mimi = MagicMock(); fake_mimi.frame_size = 1920

    with (
        patch("rehearse.backends.moshi.load_models",
              return_value=(fake_mimi, MagicMock(), MagicMock())),
        patch("rehearse.backends.moshi.MoshiASR"),
    ):
        backend = create_backend(cfg)

    assert isinstance(backend, MoshiBackend)
