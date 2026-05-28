#!/usr/bin/env python3
"""Download Moshi model weights from HuggingFace Hub.

Saves weights to rehearse/models/kyutai/<repo-name>/ by default.
The directory is gitignored; weights are fetched once and reused.

Usage:
    uv run python scripts/download_moshi_weights.py
    uv run python scripts/download_moshi_weights.py --repo kyutai/moshiko-pytorch-bf16
    uv run python scripts/download_moshi_weights.py --out rehearse/models/kyutai/moshiko-pytorch-bf16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_REPO = "kyutai/moshiko-pytorch-bf16"
_DEFAULT_OUT = _REPO_ROOT / "rehearse" / "models" / "kyutai" / "moshiko-pytorch-bf16"

# Files needed for inference (skip large training-only artefacts)
_ALLOW_PATTERNS = [
    "model.safetensors",
    "tokenizer-e351c8d8-checkpoint125.safetensors",
    "tokenizer_spm_32k_3.model",
    "config.json",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=_DEFAULT_REPO, help="HuggingFace repo id")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="Local output directory")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: uv pip install huggingface_hub", file=sys.stderr)
        return 1

    print(f"Downloading {args.repo} → {out}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
        allow_patterns=_ALLOW_PATTERNS,
    )

    missing = [f for f in _ALLOW_PATTERNS if not (out / f).exists()]
    if missing:
        print(f"WARNING: expected files not found: {missing}", file=sys.stderr)
        return 1

    print(f"\nWeights ready at: {out}")
    print("Set in .env:  MOSHI_CHECKPOINT_PATH=" + str(out.relative_to(_REPO_ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
