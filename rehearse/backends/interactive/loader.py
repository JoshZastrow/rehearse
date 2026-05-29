"""Load Moshi + Mimi models from a local directory or HuggingFace Hub.

Caller passes checkpoint_path (directory containing model.safetensors, mimi
safetensors, and SPM tokenizer). If empty, falls back to HF Hub download.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

# moshi lib is vendored at lib/moshi/moshi — ensure it is importable
_MOSHI_LIB = Path(__file__).parents[3] / "lib" / "moshi" / "moshi"
if str(_MOSHI_LIB) not in sys.path:
    sys.path.insert(0, str(_MOSHI_LIB))

from moshi.models import loaders, LMGen
from moshi.models.loaders import CheckpointInfo

if TYPE_CHECKING:
    from moshi.models.compression import MimiModel
    from moshi.models.lm import LMModel

log = structlog.get_logger(__name__)

# Cache keyed by (checkpoint_path, hf_repo, device) — weights are large and
# loading takes several seconds, so we reuse across calls in the same process.
_cache: dict[tuple[str, str, str], tuple["MimiModel", "LMGen", Any]] = {}


def load_models(
    checkpoint_path: str,
    hf_repo: str,
    device: str,
) -> tuple["MimiModel", "LMGen", Any]:
    """Return (mimi, lm_gen, tokenizer) ready for streaming inference.

    Results are cached in-process: the first call loads ~14 GB of weights;
    subsequent calls with the same arguments return instantly.
    """
    key = (checkpoint_path, hf_repo, device)
    if key in _cache:
        log.debug("interactive_loader.cache_hit")
        return _cache[key]
    result = _load_local(Path(checkpoint_path), device) if checkpoint_path else _load_hf(hf_repo, device)
    _cache[key] = result
    return result


def _load_local(
    ckpt_dir: Path,
    device: str,
) -> tuple["MimiModel", "LMGen", Any]:
    mimi_path = ckpt_dir / loaders.MIMI_NAME
    moshi_path = ckpt_dir / loaders.MOSHI_NAME
    tokenizer_path = ckpt_dir / loaders.TEXT_TOKENIZER_NAME

    log.info("interactive_loader.loading_local", ckpt_dir=str(ckpt_dir))
    mimi = loaders.get_mimi(str(mimi_path), device=device)
    mimi.set_num_codebooks(8)

    lm_model = loaders.get_moshi_lm(str(moshi_path), device=device)
    lm_gen = LMGen(lm_model, temp=0.8, temp_text=0.7)

    import sentencepiece as _sp  # noqa: PLC0415 — deferred to avoid import-time failure
    tokenizer = _sp.SentencePieceProcessor()
    tokenizer.Load(str(tokenizer_path))

    log.info("interactive_loader.loaded_local")
    return mimi, lm_gen, tokenizer


def _load_hf(
    hf_repo: str,
    device: str,
) -> tuple["MimiModel", "LMGen", Any]:
    log.info("interactive_loader.loading_hf", repo=hf_repo)
    info = CheckpointInfo.from_hf_repo(hf_repo)

    mimi = info.get_mimi(device=device)
    mimi.set_num_codebooks(8)

    lm_model = info.get_moshi(device=device)
    lm_gen = LMGen(lm_model, temp=0.8, temp_text=0.7, **info.lm_gen_config)

    tokenizer = info.get_text_tokenizer()
    log.info("interactive_loader.loaded_hf")
    return mimi, lm_gen, tokenizer
