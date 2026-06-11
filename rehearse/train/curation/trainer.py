"""Trainer protocol for ablation runs.

Implementations dispatch one budget-matched training run over the curated
corpus minus excluded_ids and return the held-out loss. Tests use FakeTrainer
(tests/curation/conftest.py); the live Modal adapter lives here too.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_DEFAULT_BASE_CONFIG = Path(__file__).parents[1] / "models" / "moshi_7B" / "config.yaml"
_HELDOUT_MANIFEST_VOLUME = "/data/data/heldout_sessions.jsonl"


class Trainer(Protocol):
    def run(self, *, excluded_ids: list[str], seed: int, max_steps: int, kind: str) -> float:
        """Run training without excluded_ids; return held-out loss."""
        ...


class ModalTrainer:
    """Budget-matched runs on the rehearse-training volume.

    Builds an ablated manifest from the local curated manifest minus
    excluded_ids, pushes it to data/ablation/<run_id>.jsonl, dispatches
    rehearse.train.modal.run_training with eval over the held-out manifest,
    and reads the final eval loss from <run_dir>/metrics.eval.jsonl.
    """

    def __init__(
        self,
        *,
        manifest_path: Path,
        base_config: Path = _DEFAULT_BASE_CONFIG,
        volume_client: Any = None,
        volume_name: str = "rehearse-training",
    ):
        from rehearse.train.curation.volume import TrainingVolumeClient

        self._manifest_path = manifest_path
        self._base_config = base_config
        self._volume_client = volume_client or TrainingVolumeClient(volume_name)
        self._volume_name = volume_name

    def run(self, *, excluded_ids: list[str], seed: int, max_steps: int, kind: str) -> float:
        import yaml

        from rehearse.train.modal import run_training

        run_id = f"abl-{uuid.uuid4().hex[:8]}"
        entries = [
            json.loads(line)
            for line in self._manifest_path.read_text().splitlines()
            if line.strip()
        ]
        kept = [e for e in entries if Path(e["path"]).parent.name not in excluded_ids]
        remote_entries = [
            {**e, "path": f"/data/data/sessions/{Path(e['path']).parent.name}/audio_stereo.wav"}
            for e in kept
        ]
        manifest_remote = f"data/ablation/{run_id}.jsonl"
        manifest = ("\n".join(json.dumps(e) for e in remote_entries) + "\n").encode()
        self._volume_client.push([(manifest_remote, manifest)])

        config = yaml.safe_load(self._base_config.read_text())
        config.setdefault("data", {})
        config["data"]["train_data"] = f"/data/{manifest_remote}"
        config["data"]["eval_data"] = _HELDOUT_MANIFEST_VOLUME
        config["do_eval"] = True
        config["eval_freq"] = max_steps  # evaluate at the final step
        config["max_steps"] = max_steps
        config["seed"] = seed
        config["run_dir"] = f"runs/{run_id}"
        logger.info(
            "%s run %s: %d sessions (excluded %s), %d steps, seed %d",
            kind, run_id, len(kept), excluded_ids, max_steps, seed,
        )
        run_training(config)
        return self._read_heldout_loss(run_id)

    def _read_heldout_loss(self, run_id: str) -> float:
        import modal

        volume = modal.Volume.from_name(self._volume_name)
        raw = b"".join(volume.read_file(f"runs/{run_id}/metrics.eval.jsonl"))
        lines = [line for line in raw.decode().splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"no eval metrics written for run {run_id}")
        last = json.loads(lines[-1])
        for key in ("eval_loss", "audio_eval_loss", "train_loss"):
            if key in last:
                return float(last[key])
        raise RuntimeError(f"no loss key in eval metrics for run {run_id}: {last}")
