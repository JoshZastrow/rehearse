"""Tests for InteractiveBackend components."""
from __future__ import annotations

import sys
from contextlib import contextmanager as _cm
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Heavy-dependency stubs: installed via a session-scoped autouse fixture so
# that sys.modules entries are cleaned up after this module's tests run and
# do not pollute other test modules.
# ---------------------------------------------------------------------------

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


def _make_stub(name: str) -> MagicMock:
    stub = MagicMock(spec=ModuleType)
    stub.__name__ = name
    stub.__spec__ = None
    return stub


def _install_stubs() -> dict:
    """Insert lightweight stubs into sys.modules; return mapping of what was added."""
    added: dict = {}
    for mod_name in _HEAVY_MODULES:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _make_stub(mod_name)  # type: ignore[assignment]
            added[mod_name] = sys.modules[mod_name]

    _torchaudio = sys.modules["torchaudio"]
    _torchaudio_functional = sys.modules["torchaudio.functional"]
    if not isinstance(getattr(_torchaudio, "functional", None), MagicMock):
        _torchaudio.functional = _torchaudio_functional  # type: ignore[attr-defined]

    _torch = sys.modules["torch"]

    @_cm
    def _no_grad():
        yield

    _torch.no_grad = _no_grad  # type: ignore[attr-defined]
    _torch.from_numpy = MagicMock(name="torch.from_numpy")  # type: ignore[attr-defined]
    _torch.int16 = MagicMock(name="torch.int16")  # type: ignore[attr-defined]

    _torchaudio_functional.resample = MagicMock(  # type: ignore[attr-defined]
        name="torchaudio.functional.resample"
    )

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
    if not hasattr(_moshi_loaders, "MIMI_NAME"):
        _moshi_loaders.MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"  # type: ignore[attr-defined]
    if not hasattr(_moshi_loaders, "MOSHI_NAME"):
        _moshi_loaders.MOSHI_NAME = "model.safetensors"  # type: ignore[attr-defined]
    if not hasattr(_moshi_loaders, "TEXT_TOKENIZER_NAME"):
        _moshi_loaders.TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"  # type: ignore[attr-defined]

    _sentencepiece = sys.modules["sentencepiece"]
    if not hasattr(_sentencepiece, "SentencePieceProcessor"):
        _sentencepiece.SentencePieceProcessor = MagicMock(  # type: ignore[attr-defined]
            name="SentencePieceProcessor"
        )

    return added


@pytest.fixture(scope="module", autouse=True)
def _interactive_stubs():
    """Install heavy-dep stubs before any test in this module; remove them after."""
    added = _install_stubs()
    yield
    for mod_name in added:
        sys.modules.pop(mod_name, None)
    for key in list(sys.modules):
        if key.startswith("rehearse.backends.interactive"):
            sys.modules.pop(key, None)


def test_loader_returns_models_from_local_path(tmp_path):
    """load_models() should use get_mimi and get_moshi_lm when given a local checkpoint dir."""
    from rehearse.backends.interactive.loader import load_models

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
        patch("rehearse.backends.interactive.loader.loaders.get_mimi", return_value=fake_mimi) as mock_mimi,
        patch("rehearse.backends.interactive.loader.loaders.get_moshi_lm", return_value=fake_lm) as mock_lm,
        patch("rehearse.backends.interactive.loader.LMGen", return_value=fake_lm_gen),
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
    from rehearse.backends.interactive.loader import load_models

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
        patch("rehearse.backends.interactive.loader.CheckpointInfo.from_hf_repo", return_value=fake_info),
        patch("rehearse.backends.interactive.loader.LMGen", return_value=fake_lm_gen),
    ):
        mimi, lm_gen, tokenizer = load_models(
            checkpoint_path="",
            hf_repo="kyutai/moshiko-pytorch-bf16",
            device="cpu",
        )

    assert mimi is fake_mimi
    assert lm_gen is fake_lm_gen
    fake_info.get_mimi.assert_called_once_with(device="cpu")


def test_interactive_asr_returns_empty_on_empty_buffer():
    from rehearse.backends.interactive.asr import InteractiveASR
    with patch("rehearse.backends.interactive.asr.WhisperModel"):
        asr = InteractiveASR(model_size="tiny")
    result = asr.transcribe_and_reset()
    assert result == ""


def test_interactive_asr_transcribes_buffered_audio():
    import numpy as np
    from rehearse.backends.interactive.asr import InteractiveASR

    mock_model = MagicMock()
    seg1 = MagicMock(); seg1.text = " Hello"
    seg2 = MagicMock(); seg2.text = " world"
    mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())

    with patch("rehearse.backends.interactive.asr.WhisperModel", return_value=mock_model):
        asr = InteractiveASR(model_size="tiny")

    pcm16 = (__import__("numpy").zeros(1600, dtype=__import__("numpy").int16)).tobytes()
    asr.push_audio(pcm16)

    text = asr.transcribe_and_reset()
    assert text == "Hello world"
    mock_model.transcribe.assert_called_once()
    assert asr.transcribe_and_reset() == ""


# ---- InteractiveBackend tests ----

from contextlib import contextmanager


@contextmanager
def _noop_ctx():
    yield


@pytest.mark.asyncio
async def test_interactive_backend_satisfies_protocol():
    from rehearse.backends.base import ConversationBackend
    from rehearse.backends.interactive import InteractiveBackend

    fake_mimi = MagicMock(); fake_mimi.frame_size = 1920

    with (
        patch("rehearse.backends.interactive.backend.load_models",
              return_value=(fake_mimi, MagicMock(), MagicMock())),
        patch("rehearse.backends.interactive.backend.InteractiveASR"),
    ):
        backend = InteractiveBackend(
            checkpoint_path="",
            hf_repo="kyutai/moshiko-pytorch-bf16",
            device="cpu",
            asr_model="tiny",
        )

    assert isinstance(backend, ConversationBackend)


@pytest.mark.asyncio
async def test_interactive_backend_start_launches_task():
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.bus import FrameBus

    fake_mimi = MagicMock(); fake_mimi.frame_size = 1920
    fake_asr = MagicMock()
    fake_asr.transcribe_and_reset.return_value = ""

    with (
        patch("rehearse.backends.interactive.backend.load_models",
              return_value=(fake_mimi, MagicMock(), MagicMock())),
        patch("rehearse.backends.interactive.backend.InteractiveASR", return_value=fake_asr),
    ):
        backend = InteractiveBackend(
            checkpoint_path="",
            hf_repo="kyutai/moshiko-pytorch-bf16",
            device="cpu",
            asr_model="tiny",
        )

    bus = FrameBus(session_id="test-start")
    await backend.start("test-start", bus)
    assert backend._task is not None
    assert not backend._task.done()
    await backend.close()


@pytest.mark.asyncio
async def test_send_caller_audio_puts_to_queue():
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.bus import FrameBus

    fake_mimi = MagicMock(); fake_mimi.frame_size = 1920
    fake_asr = MagicMock()
    fake_asr.transcribe_and_reset.return_value = ""

    with (
        patch("rehearse.backends.interactive.backend.load_models",
              return_value=(fake_mimi, MagicMock(), MagicMock())),
        patch("rehearse.backends.interactive.backend.InteractiveASR", return_value=fake_asr),
    ):
        backend = InteractiveBackend(
            checkpoint_path="",
            hf_repo="kyutai/moshiko-pytorch-bf16",
            device="cpu",
            asr_model="tiny",
        )

    bus = FrameBus(session_id="test-queue")
    await backend.start("test-queue", bus)

    await backend.send_caller_audio(b"\x00" * 640)
    assert backend._audio_q.qsize() == 1

    await backend.close()


def test_create_backend_interactive_returns_interactive_backend():
    from rehearse.backends.factory import create_backend
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.config import RuntimeConfig

    cfg = RuntimeConfig(
        twilio_account_sid="x", twilio_auth_token="x",
        twilio_from_number="x", public_base_url="x",
        hume_api_key="x", hume_config_id="x",
        session_root=Path("/tmp"),
        backend_type="interactive",
        interactive_checkpoint_path="",
        interactive_model_repo="kyutai/moshiko-pytorch-bf16",
        interactive_device="cpu",
        interactive_asr_model="tiny",
    )
    fake_mimi = MagicMock(); fake_mimi.frame_size = 1920

    with (
        patch("rehearse.backends.interactive.backend.load_models",
              return_value=(fake_mimi, MagicMock(), MagicMock())),
        patch("rehearse.backends.interactive.backend.InteractiveASR"),
    ):
        backend = create_backend(cfg)

    assert isinstance(backend, InteractiveBackend)
