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
