"""Client-side push of curated artifacts to the Modal training volume.

Unlike rehearse.train.modal.push_to_volume (which runs inside a container),
this uploads directly from the client via Volume.batch_upload — no GPU app run.
Paths are volume-root-relative (e.g. "data/sessions/<id>/audio_stereo.wav",
which containers see under /data/...).
"""

from __future__ import annotations

import io


class TrainingVolumeClient:
    def __init__(self, volume_name: str = "rehearse-training"):
        self._volume_name = volume_name

    def push(self, files: list[tuple[str, bytes]]) -> None:
        import modal

        volume = modal.Volume.from_name(self._volume_name, create_if_missing=True)
        with volume.batch_upload(force=True) as batch:
            for remote_path, content in files:
                batch.put_file(io.BytesIO(content), "/" + remote_path.lstrip("/"))
