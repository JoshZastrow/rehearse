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
import sys
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
    .pip_install("chz", "pyyaml")
    .pip_install_from_pyproject(str(_FINETUNE_DIR / "pyproject.toml"))
    .add_local_dir(
        _FINETUNE_DIR,
        remote_path="/moshi-finetune",
    )
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
)
def train_on_modal(config_dict: dict) -> None:
    rewritten = _rewrite_config_paths(config_dict)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(rewritten, f)
        temp_yaml = f.name

    try:
        env = {**os.environ, "PYTHONPATH": "/moshi-finetune"}
        cmd = ["torchrun", "--nproc-per-node", "1", "/moshi-finetune/train.py", temp_yaml]
        proc = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, text=True)
        stderr_buf: list[str] = []
        assert proc.stderr is not None
        for line in proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            stderr_buf.append(line)
        proc.wait()
        if proc.returncode != 0:
            tail = "".join(stderr_buf[-100:])
            raise RuntimeError(
                f"torchrun failed (exit {proc.returncode}):\n{tail}"
            )
        volume.commit()
    finally:
        Path(temp_yaml).unlink(missing_ok=True)


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
    with modal.enable_output():
        with app.run():
            train_on_modal.remote(config_dict)


def push_data(files: list[tuple[str, bytes]], manifest_content: bytes) -> None:
    with modal.enable_output():
        with app.run():
            push_to_volume.remote(files, manifest_content)
