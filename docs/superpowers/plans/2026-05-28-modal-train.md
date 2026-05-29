# Modal Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `rehearse-train` to run moshi fine-tuning on Modal GPU with a `with_modal=true` flag (default), and add `push_to_volume=true` to `dataset.py` to sync session data to a Modal Volume before training.

**Architecture:** Three file changes — new `rehearse/train/modal.py` isolates all Modal infrastructure (app, Volume, GPU function, data push); `cli.py` gets a `with_modal: bool = True` field that routes `_run` to Modal or local torchrun; `dataset.py` gets `push_to_volume: bool = True` that syncs the built manifest and audio files to the Volume after writing.

**Tech Stack:** `modal`, `chz`, `pyyaml`, `subprocess`, `torchrun` (via Modal A10G GPU)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `rehearse/train/modal.py` | Create | Modal app, Volume, GPU train function, data push function, thin wrappers |
| `rehearse/train/cli.py` | Modify | Add `with_modal: bool = True`; route to `modal.py` when True |
| `train/pipeline/dataset.py` | Modify | Add `push_to_volume: bool = True`; sync manifest + audio to Volume after build |
| `tests/pipeline/test_modal.py` | Create | Unit tests for pure helper functions in `modal.py` |
| `tests/pipeline/test_train_cli.py` | Modify | Add routing tests for `with_modal` flag |
| `tests/pipeline/test_dataset.py` | Create | Tests for `push_to_volume` flag in `dataset.py` |

---

### Task 1: Create `rehearse/train/modal.py`

**Files:**
- Create: `rehearse/train/modal.py`
- Create: `tests/pipeline/test_modal.py`

The Modal functions (`train_on_modal`, `push_to_volume`) require live Modal auth and a GPU — they are not unit-testable. The pure path-rewriting helpers (`_rewrite_config_paths`, `_to_volume_path`) are pure functions tested here. The thin wrappers (`run_training`, `push_data`) are tested indirectly via cli/dataset tests.

- [ ] **Step 1: Write failing tests for path helpers**

Create `tests/pipeline/test_modal.py`:

```python
import pytest
from rehearse.train.modal import _rewrite_config_paths, _to_volume_path


def test_rewrite_sets_train_data():
    cfg = {"data": {"train_data": "/local/sessions.jsonl", "eval_data": ""}, "run_dir": "runs/exp1"}
    result = _rewrite_config_paths(cfg)
    assert result["data"]["train_data"] == "/data/data/sessions.jsonl"


def test_rewrite_sets_run_dir():
    cfg = {"data": {"train_data": ""}, "run_dir": "runs/exp1"}
    result = _rewrite_config_paths(cfg)
    assert result["run_dir"] == "/data/runs/exp1"


def test_rewrite_uses_run_dir_stem():
    cfg = {"data": {"train_data": ""}, "run_dir": "runs/moshi_7B-20260528-120000"}
    result = _rewrite_config_paths(cfg)
    assert result["run_dir"] == "/data/runs/moshi_7B-20260528-120000"


def test_rewrite_preserves_other_fields():
    cfg = {"data": {"train_data": ""}, "run_dir": "runs/x", "max_steps": 500, "lora": {"rank": 64}}
    result = _rewrite_config_paths(cfg)
    assert result["max_steps"] == 500
    assert result["lora"]["rank"] == 64


def test_rewrite_does_not_mutate_input():
    cfg = {"data": {"train_data": "/local/sessions.jsonl"}, "run_dir": "runs/x"}
    _rewrite_config_paths(cfg)
    assert cfg["data"]["train_data"] == "/local/sessions.jsonl"


def test_rewrite_creates_data_section_if_missing():
    cfg = {"run_dir": "runs/x"}
    result = _rewrite_config_paths(cfg)
    assert result["data"]["train_data"] == "/data/data/sessions.jsonl"


def test_to_volume_path_wav():
    result = _to_volume_path("/Users/josh/sessions/abc123/audio.wav", "abc123")
    assert result == "/data/data/sessions/abc123/audio.wav"


def test_to_volume_path_json():
    result = _to_volume_path("/local/sessions/xyz/audio.json", "xyz")
    assert result == "/data/data/sessions/xyz/audio.json"
```

- [ ] **Step 2: Run tests — confirm they fail (module not found)**

```bash
pytest tests/pipeline/test_modal.py -v
```

Expected: `ImportError: cannot import name '_rewrite_config_paths' from 'rehearse.train.modal'`

- [ ] **Step 3: Create `rehearse/train/modal.py`**

```python
"""
Modal infrastructure for moshi fine-tuning.

Defines the Modal app, persistent Volume, GPU training function, and data
upload function. Called by rehearse/train/cli.py when with_modal=True, and
by train/pipeline/dataset.py when push_to_volume=True.

Volume 'rehearse-training' is mounted at /data inside Modal containers:
    /data/data/sessions.jsonl       — rewritten manifest (Volume paths)
    /data/data/sessions/<id>/       — audio.wav + audio.json per session
    /data/runs/<run_name>/          — checkpoints and logs
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import modal
import yaml

app = modal.App("rehearse-train")
volume = modal.Volume.from_name("rehearse-training", create_if_missing=True)

_FINETUNE_DIR = Path(__file__).parents[2] / "lib" / "moshi-finetune"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "sentencepiece",
        "simple-parsing",
        "fire",
        "safetensors",
        "tensorboard",
        "sphn==0.1.12",
        "pyyaml",
        "chz",
    )
    .pip_install(
        "moshi @ git+https://github.com/kyutai-labs/moshi.git#subdirectory=moshi"
    )
)

_finetune_mount = modal.Mount.from_local_dir(
    _FINETUNE_DIR,
    remote_path="/moshi-finetune",
)


# ─── Pure helpers (testable without Modal auth) ───────────────────────────────


def _rewrite_config_paths(config_dict: dict) -> dict:
    """Rewrite local paths in a merged config dict to Volume paths."""
    out = dict(config_dict)
    run_name = Path(out.get("run_dir", "run")).name
    out["data"] = dict(out.get("data", {}))
    out["data"]["train_data"] = "/data/data/sessions.jsonl"
    out["run_dir"] = f"/data/runs/{run_name}"
    return out


def _to_volume_path(local_path: str, session_id: str) -> str:
    """Map a local session file path to its Volume counterpart."""
    filename = Path(local_path).name
    return f"/data/data/sessions/{session_id}/{filename}"


# ─── Modal functions ──────────────────────────────────────────────────────────


@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={"/data": volume},
    image=image,
    mounts=[_finetune_mount],
)
def train_on_modal(config_dict: dict) -> None:
    rewritten = _rewrite_config_paths(config_dict)

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key and "wandb" in rewritten:
        rewritten.setdefault("wandb", {})["key"] = wandb_key

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(rewritten, f)
        temp_yaml = f.name

    env = {**os.environ, "PYTHONPATH": "/moshi-finetune"}
    subprocess.run(
        ["torchrun", "--nproc-per-node", "1", "/moshi-finetune/train.py", temp_yaml],
        check=True,
        env=env,
    )
    volume.commit()


@app.function(
    timeout=3600,
    volumes={"/data": volume},
    image=image,
)
def push_to_volume(files: list[tuple[str, bytes]], manifest_content: bytes) -> None:
    for remote_path, content in files:
        Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
        Path(remote_path).write_bytes(content)
    dest = Path("/data/data/sessions.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(manifest_content)
    volume.commit()


# ─── Thin wrappers (importable from cli.py and dataset.py) ───────────────────


def run_training(config_dict: dict) -> None:
    with app.run():
        train_on_modal.remote(config_dict)


def push_data(files: list[tuple[str, bytes]], manifest_content: bytes) -> None:
    with app.run():
        push_to_volume.remote(files, manifest_content)
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
pytest tests/pipeline/test_modal.py -v
```

Expected:
```
tests/pipeline/test_modal.py::test_rewrite_sets_train_data PASSED
tests/pipeline/test_modal.py::test_rewrite_sets_run_dir PASSED
tests/pipeline/test_modal.py::test_rewrite_uses_run_dir_stem PASSED
tests/pipeline/test_modal.py::test_rewrite_preserves_other_fields PASSED
tests/pipeline/test_modal.py::test_rewrite_does_not_mutate_input PASSED
tests/pipeline/test_modal.py::test_rewrite_creates_data_section_if_missing PASSED
tests/pipeline/test_modal.py::test_to_volume_path_wav PASSED
tests/pipeline/test_modal.py::test_to_volume_path_json PASSED
8 passed
```

- [ ] **Step 5: Commit**

```bash
git add rehearse/train/modal.py tests/pipeline/test_modal.py
git commit -m "feat: add modal.py with GPU train function, Volume push, and path helpers"
```

---

### Task 2: Add `with_modal` flag to `rehearse/train/cli.py`

**Files:**
- Modify: `rehearse/train/cli.py`
- Modify: `tests/pipeline/test_train_cli.py`

- [ ] **Step 1: Write failing routing tests**

In `tests/pipeline/test_train_cli.py`, add these imports to the existing import block at the top of the file:

```python
import rehearse.train.modal as _modal_mod
from rehearse.train.cli import TrainConfig, _merge_yaml, _run
```

Then add these tests at the end of the file (after the existing 9 tests):

```python
import rehearse.train.modal as _modal_mod
from rehearse.train.cli import _run


def test_run_routes_to_modal_when_flag_true(base_yaml, monkeypatch):
    """_run calls modal.run_training when with_modal=True."""
    called = []
    monkeypatch.setattr(_modal_mod, "run_training", lambda d: called.append(d))
    tc = TrainConfig(config=base_yaml, run_dir="runs/test", with_modal=True)
    _run(tc)
    assert len(called) == 1
    assert called[0]["run_dir"] == "runs/test"


def test_run_routes_to_local_when_flag_false(base_yaml, monkeypatch):
    """_run calls subprocess.run when with_modal=False."""
    calls = []
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: calls.append(cmd))
    tc = TrainConfig(config=base_yaml, run_dir="runs/test", with_modal=False)
    _run(tc)
    assert len(calls) == 1
    assert calls[0][0] == "torchrun"


def test_dry_run_shows_routing_modal(base_yaml, capsys):
    """dry_run prints routing decision for Modal."""
    tc = TrainConfig(config=base_yaml, run_dir="runs/test", with_modal=True, dry_run=True)
    _run(tc)
    out = capsys.readouterr().out
    assert "Modal" in out


def test_dry_run_shows_routing_local(base_yaml, capsys):
    """dry_run prints routing decision for local torchrun."""
    tc = TrainConfig(config=base_yaml, run_dir="runs/test", with_modal=False, dry_run=True)
    _run(tc)
    out = capsys.readouterr().out
    assert "local" in out
```

- [ ] **Step 2: Run new tests — confirm they fail**

```bash
pytest tests/pipeline/test_train_cli.py::test_run_routes_to_modal_when_flag_true \
       tests/pipeline/test_train_cli.py::test_run_routes_to_local_when_flag_false \
       tests/pipeline/test_train_cli.py::test_dry_run_shows_routing_modal \
       tests/pipeline/test_train_cli.py::test_dry_run_shows_routing_local -v
```

Expected: `TypeError` or `unexpected keyword argument 'with_modal'`

- [ ] **Step 3: Update `rehearse/train/cli.py`**

The full updated file (add `with_modal` field to `TrainConfig` and update `_run`):

```python
"""
Training CLI for moshi fine-tuning.

Wraps torchrun with a chz-based configuration layer. Accepts a base YAML
and override flags; merges them and delegates to lib/moshi-finetune/train.py.
By default runs on Modal A10G GPU; set with_modal=false for local torchrun.

── Usage ──────────────────────────────────────────────────────────────────────
    rehearse-train
    rehearse-train run_dir=runs/my-run max_steps=500
    rehearse-train with_modal=false run_dir=runs/local max_steps=5
    rehearse-train dry_run=true
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import chz
import yaml

_TRAIN_PY = Path(__file__).parents[2] / "lib" / "moshi-finetune" / "train.py"

_OVERRIDES: dict[str, tuple[str, ...]] = {
    "run_dir":      ("run_dir",),
    "max_steps":    ("max_steps",),
    "batch_size":   ("batch_size",),
    "duration_sec": ("duration_sec",),
    "lora_rank":    ("lora", "rank"),
}

_DEFAULT_CONFIG = Path(__file__).parents[2] / "rehearse" / "models" / "moshi_7B" / "config.yaml"


@chz.chz
class TrainConfig:
    config: Path = _DEFAULT_CONFIG
    """Base training YAML. Defines data paths, LoRA settings, optimizer params.
    Defaults to the bundled moshi-7B config."""

    run_dir: str = ""
    """Output directory for checkpoints and logs.
    Defaults to runs/<config-stem>-<YYYYMMDD-HHMMSS> if empty."""

    gpus: int = 1
    """Number of GPUs passed to torchrun --nproc-per-node.
    Use 1 for a single GPU; set to match available device count."""

    max_steps: int = 0
    """Training steps. 0 = use value from YAML (default 2000).
    Total tokens ≈ max_steps × gpus × batch_size × duration_sec × 9 × 12.5."""

    batch_size: int = 0
    """Examples per GPU per step. 0 = use value from YAML (default 16).
    Reduce if you hit OOM; also try lowering duration_sec."""

    lora_rank: int = 0
    """LoRA adapter rank. 0 = use YAML value (default 128). Keep ≤ 128."""

    duration_sec: float = 0.0
    """Max sequence length in seconds. 0 = use YAML value (default 100).
    Lowering reduces memory but may degrade quality."""

    dry_run: bool = False
    """Print the resolved torchrun command without executing."""

    with_modal: bool = True
    """Run training on Modal A10G GPU (default). Set to false for local torchrun (requires CUDA)."""


def _set_nested(d: dict, keys: tuple[str, ...], value: object) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _merge_yaml(config: TrainConfig) -> str:
    with open(config.config) as f:
        data: dict = yaml.safe_load(f) or {}

    run_dir = config.run_dir
    if not run_dir:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = f"runs/{config.config.stem}-{stamp}"

    field_values = {
        "run_dir":      run_dir,
        "max_steps":    config.max_steps,
        "batch_size":   config.batch_size,
        "duration_sec": config.duration_sec,
        "lora_rank":    config.lora_rank,
    }

    for field, keys in _OVERRIDES.items():
        value = field_values[field]
        if field == "run_dir" or value:
            _set_nested(data, keys, value)

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key and "wandb" in data:
        _set_nested(data, ("wandb", "key"), wandb_key)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="rehearse-train-"
    ) as f:
        yaml.dump(data, f)
        return f.name


def _run(config: TrainConfig) -> None:
    temp_yaml = _merge_yaml(config)
    finetune_dir = str(_TRAIN_PY.parent)
    cmd = [
        "torchrun",
        "--nproc-per-node", str(config.gpus),
        str(_TRAIN_PY),
        temp_yaml,
    ]
    if config.dry_run:
        route = "Modal (A10G GPU)" if config.with_modal else "local torchrun"
        print(f"Routing: {route}")
        print(f"PYTHONPATH={finetune_dir} {' '.join(cmd)}")
        print(f"\nResolved config written to: {temp_yaml}")
        return
    if config.with_modal:
        from rehearse.train.modal import run_training
        config_dict = yaml.safe_load(Path(temp_yaml).read_text())
        run_training(config_dict)
    else:
        env = {**os.environ, "PYTHONPATH": finetune_dir}
        subprocess.run(cmd, check=True, env=env)


def main() -> None:
    chz.nested_entrypoint(_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all train CLI tests**

```bash
pytest tests/pipeline/test_train_cli.py -v
```

Expected: all 13 tests pass (9 existing + 4 new routing tests)

- [ ] **Step 5: Commit**

```bash
git add rehearse/train/cli.py tests/pipeline/test_train_cli.py
git commit -m "feat: add with_modal flag to TrainConfig; route to Modal or local torchrun"
```

---

### Task 3: Add `push_to_volume` flag to `train/pipeline/dataset.py`

**Files:**
- Modify: `train/pipeline/dataset.py`
- Modify: `pyproject.toml`
- Create: `tests/pipeline/test_dataset.py`

- [ ] **Step 1: Add `pythonpath` to pytest config**

`train/pipeline/dataset.py` is not an installed package. Without this, `from train.pipeline.dataset import ...` will fail in tests.

In `pyproject.toml`, update `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-m 'not live_api and not live_modal'"
markers = [
    "live_api: hits real provider APIs; requires sourced .env. Deselected by default. Run with `pytest -m live_api`.",
    "live_modal: hits a deployed Modal endpoint; requires VLLM_BASE_URL. Deselected by default. Run with `pytest -m live_modal`.",
    "slow: loads ML models, takes >10s",
    "pipeline: requires faster-whisper + pocket-tts installed",
]
```

- [ ] **Step 2: Write failing tests**

Create `tests/pipeline/test_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
pytest tests/pipeline/test_dataset.py -v
```

Expected: `TypeError: ManifestConfig() got an unexpected keyword argument 'push_to_volume'`

- [ ] **Step 3: Update `train/pipeline/dataset.py`**

The full updated file:

```python
"""
Build a training manifest (JSONL) from session audio files.

Scans a sessions root directory for audio.wav files that have a sibling
audio.json annotation, computes duration for each, and writes a manifest
in the same format as DailyTalk's dailytalk.jsonl.

── Usage ────────────────────────────────────────────────────────────────────
    python train/components/build_manifest.py \\
        sessions_root=/path/to/sessions \\
        out=/path/to/sessions.jsonl

    # Only include sessions that already have audio.json annotations:
    python train/components/build_manifest.py \\
        sessions_root=/path/to/sessions \\
        out=/path/to/sessions.jsonl \\
        require_annotation=true

    # Push data to Modal Volume after building (default: true):
    python train/pipeline/dataset.py sessions_root=sessions/ out=runs/sessions.jsonl

── Output format ────────────────────────────────────────────────────────────
    Each line: {"path": "/abs/path/to/audio.wav", "duration": 12.34}

    This is identical to DailyTalk's dailytalk.jsonl and can be passed
    directly to data.train_data in the training config.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import chz
import sphn

logger = logging.getLogger(__name__)


@chz.chz
class ManifestConfig:
    sessions_root: Path
    """Root directory containing session subdirectories, each with audio.wav."""

    out: Path
    """Output JSONL path."""

    require_annotation: bool = True
    """Only include sessions that have a sibling audio.json annotation file."""

    min_duration: float = 1.0
    """Skip audio files shorter than this many seconds."""

    push_to_volume: bool = True
    """After writing the manifest, sync session audio files to Modal Volume
    'rehearse-training'. The local manifest is unchanged; a rewritten copy
    with /data/data/... paths is written to the Volume at
    /data/data/sessions.jsonl."""

    verbose: bool = False


def _init_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _run(config: ManifestConfig) -> None:
    _init_logging(config.verbose)

    wavs = sorted(config.sessions_root.rglob("audio.wav"))
    logger.info("Found %d audio.wav files under %s", len(wavs), config.sessions_root)

    entries = []
    skipped = 0
    for wav in wavs:
        annotation = wav.with_suffix(".json")
        if config.require_annotation and not annotation.exists():
            logger.debug("Skipping (no annotation): %s", wav)
            skipped += 1
            continue

        try:
            x, sr = sphn.read(str(wav))
            duration = x.shape[-1] / sr
        except Exception as exc:
            logger.warning("Failed to read %s: %s", wav, exc)
            skipped += 1
            continue

        if duration < config.min_duration:
            logger.debug("Skipping short file (%.2fs): %s", duration, wav)
            skipped += 1
            continue

        entries.append({"path": str(wav), "duration": duration})
        logger.debug("%.2fs  %s", duration, wav)

    logger.info("%d entries, %d skipped", len(entries), skipped)

    config.out.parent.mkdir(parents=True, exist_ok=True)
    with open(config.out, "w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")

    logger.info("Wrote %s", config.out)

    if config.push_to_volume and entries:
        files: list[tuple[str, bytes]] = []
        rewritten_entries = []
        for entry in entries:
            wav = Path(entry["path"])
            session_id = wav.parent.name
            remote_wav = f"/data/data/sessions/{session_id}/audio.wav"
            files.append((remote_wav, wav.read_bytes()))
            ann = wav.with_suffix(".json")
            if ann.exists():
                remote_ann = f"/data/data/sessions/{session_id}/audio.json"
                files.append((remote_ann, ann.read_bytes()))
            rewritten_entries.append({"path": remote_wav, "duration": entry["duration"]})
        manifest_content = "\n".join(json.dumps(e) for e in rewritten_entries).encode()
        from rehearse.train.modal import push_data
        push_data(files, manifest_content)
        logger.info("Pushed %d files to Modal Volume 'rehearse-training'", len(files))


if __name__ == "__main__":
    chz.nested_entrypoint(_run)
```

- [ ] **Step 4: Run all dataset tests**

```bash
pytest tests/pipeline/test_dataset.py -v
```

Expected:
```
tests/pipeline/test_dataset.py::test_push_to_volume_calls_push_data PASSED
tests/pipeline/test_dataset.py::test_push_to_volume_false_does_not_call_push_data PASSED
tests/pipeline/test_dataset.py::test_local_manifest_unchanged_after_push PASSED
3 passed
```

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/pipeline/ -v
```

Expected: all tests pass (test_modal.py × 8, test_train_cli.py × 13, test_dataset.py × 3)

- [ ] **Step 6: Commit**

```bash
git add train/pipeline/dataset.py tests/pipeline/test_dataset.py
git commit -m "feat: add push_to_volume flag to dataset.py; syncs data to Modal Volume after manifest build"
```

---

### Task 4: End-to-end dry run verification

**Files:** none

- [ ] **Step 1: Verify dry run shows Modal routing**

```bash
rehearse-train dry_run=true run_dir=runs/smoke
```

Expected output contains:
```
Routing: Modal (A10G GPU)
PYTHONPATH=.../lib/moshi-finetune torchrun --nproc-per-node 1 .../train.py /tmp/rehearse-train-XXXXX.yaml

Resolved config written to: /tmp/rehearse-train-XXXXX.yaml
```

- [ ] **Step 2: Verify dry run shows local routing**

```bash
rehearse-train dry_run=true with_modal=false run_dir=runs/smoke
```

Expected output contains:
```
Routing: local torchrun
```

- [ ] **Step 3: Verify resolved config paths**

```bash
# copy the temp yaml path from step 1 output, then:
cat /tmp/rehearse-train-<the-path>.yaml
```

Expected: YAML with `run_dir: runs/smoke`, `data.train_data` pointing to the base config's empty value (path rewriting happens inside `train_on_modal` in the container, not locally).

- [ ] **Step 4: Commit spec doc**

```bash
git add docs/superpowers/specs/2026-05-28-modal-train-design.md
git commit -m "docs: add Modal training design spec"
```
