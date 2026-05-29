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
