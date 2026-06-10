"""Load PersonaPlex models + voice prompts (NVIDIA's Moshi fork).

PersonaPlex (https://github.com/NVIDIA/personaplex) is a Moshi finetune with
voice + role conditioning: a text prompt sets the persona, a voice prompt sets
the speaker identity, and both are prefilled into the LM before streaming.

It requires NVIDIA's fork of the moshi library (package `moshi-personaplex`,
module name still `moshi`) — upstream kyutai moshi lacks the voice/text prompt
APIs. Imports are deferred so this module stays importable without torch.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

DEFAULT_PERSONAPLEX_REPO = "nvidia/personaplex-7b-v1"

_FORK_INSTALL_HINT = (
    "personaplex requires NVIDIA's moshi fork (moshi-personaplex). Install with: "
    "pip install 'moshi-personaplex @ "
    "git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi'"
)


def wrap_system_tags(text: str) -> str:
    """Wrap a role prompt in the <system> tags the model expects, if missing."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def load_personaplex_models(
    checkpoint_path: str,
    hf_repo: str,
    device: str,
) -> tuple[Any, Any, Any]:
    """Return (mimi, lm_gen, tokenizer) ready for streaming inference.

    Mirrors PersonaPlex's server.py model setup: get_mimi/get_moshi_lm from the
    fork's loaders (dep_q=16 weights with 8 output codebooks) and an LMGen with
    0.5 s of audio silence padding after the voice/text prompt prefill.
    """
    from moshi.models import LMGen, loaders

    if not hasattr(LMGen, "step_system_prompts_async"):
        raise RuntimeError(_FORK_INSTALL_HINT)

    import sentencepiece

    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        mimi_weight = str(ckpt / loaders.MIMI_NAME)
        moshi_weight = str(ckpt / loaders.MOSHI_NAME)
        tokenizer_path = str(ckpt / loaders.TEXT_TOKENIZER_NAME)
        log.info("personaplex_loader.loading_local", ckpt_dir=checkpoint_path)
    else:
        from huggingface_hub import hf_hub_download

        log.info("personaplex_loader.loading_hf", repo=hf_repo)
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)

    mimi = loaders.get_mimi(mimi_weight, device)
    if not hasattr(mimi, "frame_size"):
        # The fork's MimiModel predates upstream's frame_size property — its
        # server computes sample_rate / frame_rate by hand. Backfill it so the
        # streaming loop can treat both model types uniformly.
        mimi.frame_size = int(mimi.sample_rate / mimi.frame_rate)
    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    lm = loaders.get_moshi_lm(moshi_weight, device=device)
    lm.eval()
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
    )
    log.info("personaplex_loader.loaded")
    return mimi, lm_gen, tokenizer


def ensure_voice_prompt_dir(hf_repo: str) -> str:
    """Download + extract voices.tgz from the HF repo; return the directory."""
    from huggingface_hub import hf_hub_download

    voices_tgz = Path(hf_hub_download(hf_repo, "voices.tgz"))
    voices_dir = voices_tgz.parent / "voices"
    if not voices_dir.exists():
        log.info("personaplex_loader.extracting_voices", tgz=str(voices_tgz))
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)
    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")
    return str(voices_dir)


def resolve_voice_prompt(voice_prompt_dir: str, name: str) -> str:
    """Return the path for a voice prompt; fall back to the first .pt present.

    The voices directory comes from an external tarball, so a guessed default
    filename may not exist — falling back keeps the server bootable while the
    warning makes the substitution visible.
    """
    path = Path(voice_prompt_dir) / name
    if path.exists():
        return str(path)
    candidates = sorted(Path(voice_prompt_dir).glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No voice prompts found in {voice_prompt_dir}")
    log.warning(
        "personaplex_loader.voice_prompt_fallback",
        requested=name,
        using=candidates[0].name,
    )
    return str(candidates[0])


def apply_persona(lm_gen: Any, tokenizer: Any, *, voice_prompt_path: str, text_prompt: str) -> None:
    """Set the voice + role conditioning on lm_gen for the next prefill.

    Must be followed by lm_gen.step_system_prompts(_async)(mimi) inside a
    streaming context, then mimi.reset_streaming() — matching server.py.
    """
    if lm_gen.voice_prompt != voice_prompt_path:
        if voice_prompt_path.endswith(".pt"):
            lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
        else:
            lm_gen.load_voice_prompt(voice_prompt_path)
    lm_gen.text_prompt_tokens = (
        tokenizer.encode(wrap_system_tags(text_prompt)) if text_prompt else None
    )
