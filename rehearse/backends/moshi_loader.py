"""Load Moshi + Mimi models from a local directory or HuggingFace Hub.

Caller passes checkpoint_path (directory containing model.safetensors, mimi
safetensors, and SPM tokenizer). If empty, falls back to HF Hub download.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import sentencepiece
import structlog

# moshi lib is vendored at lib/moshi/moshi — ensure it is importable
_MOSHI_LIB = Path(__file__).parents[2] / "lib" / "moshi" / "moshi"
if str(_MOSHI_LIB) not in sys.path:
    sys.path.insert(0, str(_MOSHI_LIB))

from moshi.models import loaders, LMGen
from moshi.models.loaders import CheckpointInfo

if TYPE_CHECKING:
    from moshi.models.compression import MimiModel
    from moshi.models.lm import LMModel

log = structlog.get_logger(__name__)


def load_models(
    checkpoint_path: str,
    hf_repo: str,
    device: str,
) -> tuple["MimiModel", "LMGen", sentencepiece.SentencePieceProcessor]:
    """Return (mimi, lm_gen, tokenizer) ready for streaming inference.

    If checkpoint_path is a non-empty string, loads weights from that
    directory. Otherwise downloads from hf_repo.
    """
    if checkpoint_path:
        return _load_local(Path(checkpoint_path), device)
    return _load_hf(hf_repo, device)


def _load_local(
    ckpt_dir: Path,
    device: str,
) -> tuple["MimiModel", "LMGen", sentencepiece.SentencePieceProcessor]:
    mimi_path = ckpt_dir / loaders.MIMI_NAME
    moshi_path = ckpt_dir / loaders.MOSHI_NAME
    tokenizer_path = ckpt_dir / loaders.TEXT_TOKENIZER_NAME

    log.info("moshi_loader.loading_local", ckpt_dir=str(ckpt_dir))
    mimi = loaders.get_mimi(str(mimi_path), device=device)
    mimi.set_num_codebooks(8)

    lm_model = loaders.get_moshi_lm(str(moshi_path), device=device)
    lm_gen = LMGen(lm_model, temp=0.8, temp_text=0.7)

    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.Load(str(tokenizer_path))

    log.info("moshi_loader.loaded_local")
    return mimi, lm_gen, tokenizer


def _load_hf(
    hf_repo: str,
    device: str,
) -> tuple["MimiModel", "LMGen", sentencepiece.SentencePieceProcessor]:
    log.info("moshi_loader.loading_hf", repo=hf_repo)
    info = CheckpointInfo.from_hf_repo(hf_repo)

    mimi = info.get_mimi(device=device)
    mimi.set_num_codebooks(8)

    lm_model = info.get_moshi(device=device)
    lm_gen = LMGen(lm_model, temp=0.8, temp_text=0.7, **info.lm_gen_config)

    tokenizer = info.get_text_tokenizer()
    log.info("moshi_loader.loaded_hf")
    return mimi, lm_gen, tokenizer
