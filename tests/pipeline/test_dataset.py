import json
import wave
from pathlib import Path

import pytest
import numpy as np

import rehearse.train.modal as _modal_mod


def _write_wav(path: Path, duration_sec: float = 2.0, sample_rate: int = 16000) -> None:
    """Write a minimal valid WAV file."""
    n_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((np.zeros(n_samples, dtype=np.int16)).tobytes())


@pytest.fixture
def sessions_dir(tmp_path):
    """Two sessions, each with audio.wav and audio.json."""
    for sid in ("session_a", "session_b"):
        d = tmp_path / sid
        d.mkdir()
        _write_wav(d / "audio.wav")
        (d / "audio.json").write_text(json.dumps({"alignments": []}))
    return tmp_path


def test_push_to_volume_calls_push_data(sessions_dir, tmp_path, monkeypatch):
    """When push_to_volume=True, push_data is called with the built files."""
    from train.pipeline.dataset import ManifestConfig, _run

    push_calls = []
    monkeypatch.setattr(_modal_mod, "push_data", lambda files, manifest: push_calls.append((files, manifest)))

    cfg = ManifestConfig(
        sessions_root=sessions_dir,
        out=tmp_path / "sessions.jsonl",
        push_to_volume=True,
        require_annotation=True,
    )
    _run(cfg)

    assert len(push_calls) == 1
    files, manifest_content = push_calls[0]
    remote_paths = [f[0] for f in files]
    assert any("/data/data/sessions/session_a/audio.wav" in p for p in remote_paths)
    assert any("/data/data/sessions/session_b/audio.wav" in p for p in remote_paths)
    assert any("/data/data/sessions/session_a/audio.json" in p for p in remote_paths)

    rewritten = [json.loads(line) for line in manifest_content.decode().splitlines() if line]
    assert all(e["path"].startswith("/data/data/sessions/") for e in rewritten)


def test_push_to_volume_false_does_not_call_push_data(sessions_dir, tmp_path, monkeypatch):
    """When push_to_volume=False, push_data is never called."""
    from train.pipeline.dataset import ManifestConfig, _run

    push_calls = []
    monkeypatch.setattr(_modal_mod, "push_data", lambda files, manifest: push_calls.append(1))

    cfg = ManifestConfig(
        sessions_root=sessions_dir,
        out=tmp_path / "sessions.jsonl",
        push_to_volume=False,
        require_annotation=True,
    )
    _run(cfg)

    assert len(push_calls) == 0


def test_local_manifest_unchanged_after_push(sessions_dir, tmp_path, monkeypatch):
    """Local manifest still has original absolute paths even after push."""
    from train.pipeline.dataset import ManifestConfig, _run

    monkeypatch.setattr(_modal_mod, "push_data", lambda files, manifest: None)

    out = tmp_path / "sessions.jsonl"
    cfg = ManifestConfig(
        sessions_root=sessions_dir,
        out=out,
        push_to_volume=True,
        require_annotation=True,
    )
    _run(cfg)

    local_entries = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert all(e["path"].startswith(str(sessions_dir)) for e in local_entries)
