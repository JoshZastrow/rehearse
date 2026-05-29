# Modal Training Design

**Date:** 2026-05-28
**Status:** Approved

## Overview

Extend `rehearse-train` to run moshi fine-tuning on Modal GPU infrastructure with a single `with_modal=true` flag (default). Local torchrun remains available via `with_modal=false`. Session data is synced to a Modal Volume before training; checkpoints are written back to the same Volume.

## Architecture

Three files change:

| File | Change |
|------|--------|
| `rehearse/train/cli.py` | Add `with_modal: bool = True`; route `_run` to `modal.py` when True |
| `rehearse/train/modal.py` | New. Modal app, Volume, GPU train function, data push function |
| `train/pipeline/dataset.py` | Add `push_to_volume: bool = True`; after writing manifest, sync data to Volume |

Infrastructure (Modal vs local) is a CLI flag. Application logic (`_merge_yaml`, config resolution) stays in `cli.py` and is reused by both paths.

## Volume Structure

Volume name: `rehearse-training`  
Mounted at: `/data` inside Modal containers

```
/data/
├── data/
│   ├── sessions.jsonl          ← manifest (absolute paths rewritten to /data/data/...)
│   └── sessions/
│       └── <session_id>/
│           ├── audio.wav
│           └── audio.json
└── runs/
    └── <run_name>/
        ├── args.yaml           ← written by moshi-finetune after training starts
        └── checkpoints/
```

Local sessions remain the canonical store. The Volume is a training mirror populated by `push_to_volume`.

## `rehearse/train/modal.py`

```python
app = modal.App("rehearse-train")
volume = modal.Volume.from_name("rehearse-training", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("torch", "torchaudio", "moshi", "simple-parsing", "fire",
                 "safetensors", "tensorboard", "sphn", "pyyaml", "chz")
)
```

**`train_on_modal(config_dict: dict) -> None`**
- `@app.function(gpu="A10G", timeout=3600, volumes={"/data": volume}, image=image)`
- Receives the merged config as a plain dict (serializable)
- Rewrites `data.train_data` → `/data/data/sessions.jsonl`
- Rewrites `run_dir` → `/data/runs/<run_name>` (keeps the stem, prepends `/data/runs/`)
- Writes the rewritten config to a temp YAML file inside the container
- Sets `PYTHONPATH` to the moshi-finetune directory (cloned into the image or installed)
- Runs torchrun via `subprocess.run`
- Commits Volume after training (volume.commit())

**`push_to_volume(manifest_path: Path, sessions_root: Path) -> None`**
- `@app.function(timeout=3600, volumes={"/data": volume}, image=image)`
- Reads manifest lines; for each `{"path": ..., "duration": ...}` entry, uploads `audio.wav` + sibling `audio.json` (if present) to `/data/data/sessions/<session_id>/`
- Writes a rewritten manifest to `/data/data/sessions.jsonl` with paths remapped to `/data/data/...`
- Commits Volume after all uploads

## `rehearse/train/cli.py` Changes

Add one field to `TrainConfig`:

```python
with_modal: bool = True
"""Run training on Modal GPU (default). Set to false for local torchrun."""
```

In `_run`:
```python
if config.with_modal:
    from rehearse.train import modal as modal_train
    config_dict = yaml.safe_load(Path(temp_yaml).read_text())
    with app.run():
        modal_train.train_on_modal.remote(config_dict)
else:
    # existing torchrun path
    subprocess.run(cmd, check=True, env=env)
```

Dry run prints the resolved config and the routing decision regardless of `with_modal`.

## `train/pipeline/dataset.py` Changes

Add one field to `ManifestConfig`:

```python
push_to_volume: bool = True
"""After writing the manifest, sync data to Modal Volume 'rehearse-training'."""
```

In `_run`, after writing the manifest:
```python
if config.push_to_volume:
    from rehearse.train import modal as modal_train
    with modal_train.app.run():
        modal_train.push_to_volume.remote(config.out, config.sessions_root)
```

## Path Rewriting

`push_to_volume` rewrites each path from the local absolute path to its Volume counterpart:

```
/Users/josh/sessions/<id>/audio.wav  →  /data/data/sessions/<id>/audio.wav
```

The rewritten manifest at `/data/data/sessions.jsonl` is what `train_on_modal` reads. The original local manifest is unchanged.

## Secrets

`WANDB_API_KEY` is injected from the environment at runtime by `_merge_yaml`. It is never committed to a config file. The wandb section in `rehearse/models/moshi_7B/config.yaml` is and must remain commented out.

## Usage

```sh
# Default: build manifest, sync to Volume, train on Modal A10G
python train/pipeline/dataset.py sessions_root=sessions/ out=runs/sessions.jsonl
rehearse-train run_dir=runs/my-run max_steps=500

# Explicit flags
rehearse-train with_modal=true run_dir=runs/exp1 max_steps=200 batch_size=4

# Local torchrun (requires CUDA)
rehearse-train with_modal=false run_dir=runs/local max_steps=5

# Dry run — print resolved config, routing decision, no execution
rehearse-train dry_run=true
```

## Out of Scope

- Downloading checkpoints from Volume back to local disk (manual `modal volume get` for now)
- Multi-GPU (nproc_per_node > 1) on Modal (single A10G, `gpus=1` default)
- Modal inference serving
- Automated eval → training trigger (future continual learning loop)
- Switching canonical session store to Modal Volume (future: when inference runs on Modal)
