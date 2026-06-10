"""Tests for the PersonaPlex model option on the interactive backend.

PersonaPlex (https://github.com/NVIDIA/personaplex) is a Moshi finetune with
voice + role conditioning, served through the same wire protocol as the Moshi
interactive servers. These tests cover the pure helpers and source-level
wiring; GPU behaviour is verified by the live_modal suite.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[1]
_INFRA_PY = _REPO_ROOT / "infra" / "interactive.py"
_BACKEND_PY = _REPO_ROOT / "rehearse" / "backends" / "interactive" / "backend.py"


# ---------------------------------------------------------------------------
# personaplex_loader pure helpers
# ---------------------------------------------------------------------------


def test_wrap_system_tags_wraps_plain_text():
    from rehearse.backends.interactive.personaplex_loader import wrap_system_tags

    assert wrap_system_tags("You are a coach.") == "<system> You are a coach. <system>"


def test_wrap_system_tags_is_idempotent():
    from rehearse.backends.interactive.personaplex_loader import wrap_system_tags

    wrapped = "<system> You are a coach. <system>"
    assert wrap_system_tags(wrapped) == wrapped


def test_resolve_voice_prompt_returns_exact_match(tmp_path: Path):
    from rehearse.backends.interactive.personaplex_loader import resolve_voice_prompt

    (tmp_path / "NATF0.pt").write_bytes(b"x")
    (tmp_path / "NATM0.pt").write_bytes(b"x")
    assert resolve_voice_prompt(str(tmp_path), "NATM0.pt") == str(tmp_path / "NATM0.pt")


def test_resolve_voice_prompt_falls_back_to_first_pt(tmp_path: Path):
    from rehearse.backends.interactive.personaplex_loader import resolve_voice_prompt

    (tmp_path / "VARM1.pt").write_bytes(b"x")
    (tmp_path / "NATF2.pt").write_bytes(b"x")
    # Requested voice is missing — fall back to the first .pt alphabetically.
    assert resolve_voice_prompt(str(tmp_path), "GONE.pt") == str(tmp_path / "NATF2.pt")


def test_resolve_voice_prompt_raises_when_dir_empty(tmp_path: Path):
    from rehearse.backends.interactive.personaplex_loader import resolve_voice_prompt

    with pytest.raises(FileNotFoundError):
        resolve_voice_prompt(str(tmp_path), "NATF0.pt")


# ---------------------------------------------------------------------------
# load_personaplex_models — fork detection + loading
# ---------------------------------------------------------------------------


class _UpstreamLMGen:
    """Stands in for upstream kyutai LMGen — lacks the PersonaPlex APIs."""

    def __init__(self, *args, **kwargs):
        pass


class _ForkLMGen:
    """Stands in for the NVIDIA fork's LMGen."""

    def __init__(self, lm_model, device=None, **kwargs):
        self.lm_model = lm_model
        self.device = device
        self.kwargs = kwargs
        self.voice_prompt = None
        self.text_prompt_tokens = None

    async def step_system_prompts_async(self, mimi, is_alive=None):
        pass

    def load_voice_prompt_embeddings(self, path):
        self.voice_prompt = path

    def load_voice_prompt(self, path):
        self.voice_prompt = path


class _FakeMimi:
    # No frame_size — the fork's MimiModel lacks the property; the loader
    # must backfill it (sample_rate / frame_rate = 1920).
    sample_rate = 24_000
    frame_rate = 12.5


def _install_fake_moshi(monkeypatch, *, lm_gen_cls):
    """Install a fake `moshi` package + `sentencepiece` into sys.modules."""
    calls: dict[str, object] = {}

    fake_loaders = types.ModuleType("moshi.models.loaders")
    fake_loaders.MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
    fake_loaders.MOSHI_NAME = "model.safetensors"
    fake_loaders.TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"

    def get_mimi(path, device):
        calls["mimi_path"] = str(path)
        calls["mimi_device"] = device
        return _FakeMimi()

    class _FakeLM:
        def eval(self):
            calls["lm_eval"] = True

    def get_moshi_lm(path, device=None):
        calls["moshi_path"] = str(path)
        calls["moshi_device"] = device
        return _FakeLM()

    fake_loaders.get_mimi = get_mimi
    fake_loaders.get_moshi_lm = get_moshi_lm

    fake_models = types.ModuleType("moshi.models")
    fake_models.loaders = fake_loaders
    fake_models.LMGen = lm_gen_cls

    fake_moshi = types.ModuleType("moshi")
    fake_moshi.models = fake_models

    fake_sp = types.ModuleType("sentencepiece")

    class _FakeTokenizer:
        def __init__(self, path):
            calls["tokenizer_path"] = str(path)

        def encode(self, text):
            return [1, 2, 3]

    fake_sp.SentencePieceProcessor = _FakeTokenizer

    monkeypatch.setitem(sys.modules, "moshi", fake_moshi)
    monkeypatch.setitem(sys.modules, "moshi.models", fake_models)
    monkeypatch.setitem(sys.modules, "moshi.models.loaders", fake_loaders)
    monkeypatch.setitem(sys.modules, "sentencepiece", fake_sp)
    return calls


def test_load_personaplex_models_rejects_upstream_moshi(monkeypatch, tmp_path: Path):
    from rehearse.backends.interactive.personaplex_loader import load_personaplex_models

    _install_fake_moshi(monkeypatch, lm_gen_cls=_UpstreamLMGen)
    with pytest.raises(RuntimeError, match="moshi-personaplex"):
        load_personaplex_models(
            checkpoint_path=str(tmp_path),
            hf_repo="nvidia/personaplex-7b-v1",
            device="cpu",
        )


def test_load_personaplex_models_loads_from_checkpoint_dir(monkeypatch, tmp_path: Path):
    from rehearse.backends.interactive.personaplex_loader import load_personaplex_models

    calls = _install_fake_moshi(monkeypatch, lm_gen_cls=_ForkLMGen)
    mimi, lm_gen, tokenizer = load_personaplex_models(
        checkpoint_path=str(tmp_path),
        hf_repo="nvidia/personaplex-7b-v1",
        device="cuda",
    )

    assert calls["mimi_path"] == str(tmp_path / "tokenizer-e351c8d8-checkpoint125.safetensors")
    assert calls["moshi_path"] == str(tmp_path / "model.safetensors")
    assert calls["tokenizer_path"] == str(tmp_path / "tokenizer_spm_32k_3.model")
    assert calls["lm_eval"] is True
    assert isinstance(lm_gen, _ForkLMGen)
    # PersonaPlex server settings: 0.5 s of silence padding after prompts,
    # mimi's sample rate / frame rate threaded through.
    assert lm_gen.kwargs["audio_silence_frame_cnt"] == 6
    assert lm_gen.kwargs["sample_rate"] == 24_000
    assert lm_gen.kwargs["frame_rate"] == 12.5
    # Backfilled — the fork's MimiModel has no frame_size property.
    assert mimi.frame_size == 1920


def test_apply_persona_loads_pt_embeddings_and_text_prompt(monkeypatch, tmp_path: Path):
    from rehearse.backends.interactive.personaplex_loader import apply_persona

    _install_fake_moshi(monkeypatch, lm_gen_cls=_ForkLMGen)
    lm_gen = _ForkLMGen(lm_model=None)

    class _Tok:
        def encode(self, text):
            assert text == "<system> Be kind. <system>"
            return [7, 8]

    voice = tmp_path / "NATF0.pt"
    voice.write_bytes(b"x")
    apply_persona(lm_gen, _Tok(), voice_prompt_path=str(voice), text_prompt="Be kind.")
    assert lm_gen.voice_prompt == str(voice)
    assert lm_gen.text_prompt_tokens == [7, 8]


def test_apply_persona_empty_text_prompt_clears_tokens(tmp_path: Path):
    from rehearse.backends.interactive.personaplex_loader import apply_persona

    lm_gen = _ForkLMGen(lm_model=None)
    voice = tmp_path / "NATF0.pt"
    voice.write_bytes(b"x")
    apply_persona(lm_gen, object(), voice_prompt_path=str(voice), text_prompt="")
    assert lm_gen.text_prompt_tokens is None


# ---------------------------------------------------------------------------
# infra/interactive.py — Modal server wiring (source-level, no GPU needed)
# ---------------------------------------------------------------------------


def _infra_module():
    spec = importlib.util.spec_from_file_location("infra_interactive", _INFRA_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _class_def(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in infra/interactive.py")


def test_infra_defines_personaplex_server_classes():
    tree = ast.parse(_INFRA_PY.read_text())
    src = _INFRA_PY.read_text()

    # Names skip the "Server" suffix so the Modal web-endpoint label stays
    # under the 63-char DNS limit (<workspace>--rehearse-interactive-<cls>-serve).
    for cls_name, role in [
        ("PersonaplexProvider", "provider"),
        ("PersonaplexCaller", "caller"),
    ]:
        node = _class_def(tree, cls_name)
        attrs = {
            t.targets[0].id: getattr(t.value, "value", None)
            for t in node.body
            if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name)
        }
        assert attrs.get("model_type") == "personaplex", cls_name
        assert attrs.get("speaker_role") == role, cls_name
        # Gated HF repo — the class must mount the HF_TOKEN secret.
        decorator_src = "".join(
            ast.get_source_segment(src, d) or "" for d in node.decorator_list
        )
        assert "HF_TOKEN" in decorator_src, f"{cls_name} missing HF_TOKEN secret"


def test_infra_moshi_servers_keep_moshi_model_type():
    mod = _infra_module()
    assert mod._InteractiveServerBase.model_type == "moshi"


def test_infra_personaplex_image_pins_fork_compatible_torch():
    """The NVIDIA fork pins torch>=2.2,<2.5 — the personaplex image must comply."""
    src = _INFRA_PY.read_text()
    assert "moshi-personaplex" in src
    assert "github.com/NVIDIA/personaplex" in src
    start = src.index("personaplex_image")
    end = src.index("hf_cache_vol")
    image_src = src[start:end]
    assert 'torch==2.4' in image_src, "personaplex image must pin torch<2.5"


def test_infra_decodes_first_eight_codebooks():
    """PersonaPlex has dep_q=16; only the first 8 codebooks are output audio.

    Decoding tokens[:, 1:] would feed 16 codebooks to an 8-codebook mimi and
    crash — both warmup and session must slice [:, 1:9].
    """
    src = _INFRA_PY.read_text()
    assert "tokens[:, 1:9" in src
    assert "tokens[:, 1:, :]" not in src and "tokens[:, 1:])" not in src


def test_infra_resolve_persona_defaults_and_handshake_override(monkeypatch):
    mod = _infra_module()
    monkeypatch.delenv("PERSONAPLEX_TEXT_PROMPT", raising=False)
    monkeypatch.delenv("PERSONAPLEX_VOICE_PROMPT", raising=False)

    text, voice = mod._resolve_persona({}, "provider")
    assert text and voice.endswith(".pt")

    text2, voice2 = mod._resolve_persona(
        {"text_prompt": "You are Ayelen from CitySan.", "voice_prompt": "VARF3.pt"},
        "provider",
    )
    assert text2 == "You are Ayelen from CitySan."
    assert voice2 == "VARF3.pt"

    # Provider and caller get distinct default personas.
    provider_text, _ = mod._resolve_persona({}, "provider")
    caller_text, _ = mod._resolve_persona({}, "caller")
    assert provider_text != caller_text


def test_infra_resolve_persona_env_override(monkeypatch):
    mod = _infra_module()
    monkeypatch.setenv("PERSONAPLEX_TEXT_PROMPT", "Custom role.")
    monkeypatch.setenv("PERSONAPLEX_VOICE_PROMPT", "NATM3.pt")
    text, voice = mod._resolve_persona({}, "provider")
    assert text == "Custom role."
    assert voice == "NATM3.pt"


# ---------------------------------------------------------------------------
# Local InteractiveBackend — model_type dispatch (source-level; importing the
# module needs torch + faster-whisper + vendored moshi, so inspect the AST)
# ---------------------------------------------------------------------------


def test_local_backend_accepts_personaplex_model_type():
    tree = ast.parse(_BACKEND_PY.read_text())
    init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    kwonly = {a.arg for a in init.args.kwonlyargs}
    assert "model_type" in kwonly, "InteractiveBackend.__init__ must accept model_type"

    src = _BACKEND_PY.read_text()
    assert "personaplex" in src
    assert "load_personaplex_models" in src
    assert "step_system_prompts" in src, "personaplex requires persona prefill before streaming"


# ---------------------------------------------------------------------------
# Eval rollout provenance — records which interactive model produced the run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_audio_sandbox_provenance_records_interactive_model(
    tmp_path: Path, monkeypatch
):
    import json
    from unittest.mock import patch

    from rehearse.eval.environments.live_audio_sandbox import LiveAudioSandboxEnvironment
    from tests.test_live_audio_sandbox_rollout import (
        _EndingBackend,
        _StubLLM,
        _StubTTS,
        _example,
    )

    monkeypatch.setenv("BACKEND_TYPE", "interactive")
    monkeypatch.setenv("INTERACTIVE_MODEL_TYPE", "personaplex")
    monkeypatch.setenv(
        "INTERACTIVE_PROVIDER_ENDPOINT", "wss://example--personaplex.modal.run/ws"
    )

    env = LiveAudioSandboxEnvironment(
        tts_provider=_StubTTS(),
        llm_client=_StubLLM(),
        customer_max_turns=1,
    )
    with patch(
        "rehearse.eval.environments.live_audio_sandbox.create_backend",
        return_value=_EndingBackend(),
    ):
        result = await env.rollout(_example(), tmp_path / "run", rng_seed=0)

    assert result.status == "ok", result.error
    provenance = json.loads((result.artifacts_dir / "provenance.json").read_text())
    assert provenance["backend_type"] == "interactive"
    assert provenance["interactive_model_type"] == "personaplex"
    assert provenance["interactive_provider_endpoint"] == (
        "wss://example--personaplex.modal.run/ws"
    )
