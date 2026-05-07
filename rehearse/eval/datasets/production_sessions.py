"""ProductionSessionsDataset — enumerate completed production sessions.

Yields one `BenchmarkExample` per session bundle on disk. The replay
environment consumes the example's payload to locate the bundle and
re-materialize per-turn artifacts in the run directory.

Consent gate: by default only sessions with `consent == "granted"` in
`session.json` are yielded. Pass `require_consent=False` for ad-hoc
dogfooding of the operator's own pre-consent-flow sessions; this should
never be used to score data that will train a model.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample

DEFAULT_SESSIONS_ROOT = Path("sessions")


def _consent_required_default() -> bool:
    """Allow operator dogfood via `REHEARSE_REQUIRE_CONSENT=0`.

    Default is `True`. Setting the env var to `0`/`false`/`no` bypasses
    the consent gate so pre-consent-flow sessions can be scored locally.
    Never use in any pipeline whose output trains a model.
    """
    raw = os.environ.get("REHEARSE_REQUIRE_CONSENT", "1").lower()
    return raw not in ("0", "false", "no", "off")


class ProductionSessionsDataset:
    name = "production-sessions"
    version = "v1"

    def __init__(
        self,
        sessions_root: Path | str = DEFAULT_SESSIONS_ROOT,
        *,
        require_consent: bool | None = None,
        session_ids: list[str] | None = None,
    ) -> None:
        self._root = Path(sessions_root)
        self._require_consent = (
            _consent_required_default() if require_consent is None else require_consent
        )
        self._session_ids = session_ids

    def load(self) -> Iterable[BenchmarkExample]:
        if not self._root.exists():
            return []
        examples: list[BenchmarkExample] = []
        candidates = (
            sorted(self._session_ids)
            if self._session_ids is not None
            else sorted(p.name for p in self._root.iterdir() if p.is_dir())
        )
        for sid in candidates:
            session_dir = self._root / sid
            session_json = session_dir / "session.json"
            transcript = session_dir / "transcript.jsonl"
            audio = session_dir / "audio.wav"
            if not (session_json.exists() and transcript.exists() and audio.exists()):
                continue
            # Skip bundles whose transcript never got written to (early
            # disconnect, etc.) — they have no scoreable content.
            if transcript.stat().st_size == 0:
                continue
            try:
                manifest = json.loads(session_json.read_text())
            except json.JSONDecodeError:
                continue
            consent = manifest.get("consent", "pending")
            if self._require_consent and consent != "granted":
                continue
            examples.append(
                BenchmarkExample(
                    id=sid,
                    benchmark=self.name,
                    payload={
                        "session_id": sid,
                        "session_dir": str(session_dir),
                        "consent": consent,
                    },
                    expected={},
                    metadata={
                        "created_at": manifest.get("created_at"),
                        "phone_number_hash": manifest.get("phone_number_hash"),
                    },
                )
            )
        return examples
