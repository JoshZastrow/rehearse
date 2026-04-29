"""Render a live session viewer from stored runtime artifacts.

This file serves the `/viewer` page that the post-call SMS links to. It reads
the session manifest plus any captured artifacts, then renders one plain HTML
page with the most important runtime evidence.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from rehearse.storage import LocalFilesystemStore
from rehearse.types import ProsodyFrame, Session, TranscriptFrame


def mount_viewer_routes(app: FastAPI, store: LocalFilesystemStore) -> None:
    """Register the runtime-backed viewer route on the FastAPI app."""

    @app.get("/viewer", response_class=HTMLResponse)
    async def viewer(session_id: str) -> HTMLResponse:
        """Render one session viewer page for the requested session id."""
        session = await _load_session(store, session_id)
        transcript = await _load_jsonl(store, session, "transcript", TranscriptFrame)
        prosody = await _load_jsonl(store, session, "prosody", ProsodyFrame)
        story = await _load_text(store, session, "story")
        feedback = await _load_text(store, session, "feedback")
        return HTMLResponse(
            _render_viewer_html(
                store=store,
                session=session,
                transcript=transcript,
                prosody=prosody,
                story=story,
                feedback=feedback,
            )
        )


async def _load_session(store: LocalFilesystemStore, session_id: str) -> Session:
    """Read and validate `session.json` for one runtime session."""
    try:
        payload = await store.read(session_id, "session.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="unknown session") from exc
    return Session.model_validate_json(payload)


async def _load_jsonl(
    store: LocalFilesystemStore,
    session: Session,
    artifact_key: str,
    model,
) -> list:
    """Read one JSONL artifact file and return validated rows, or an empty list."""
    path = session.artifact_paths.get(artifact_key)
    if not path:
        return []
    try:
        raw = (await store.read(session.id, path)).decode("utf-8").strip()
    except FileNotFoundError:
        return []
    if not raw:
        return []
    return [model.model_validate_json(line) for line in raw.splitlines()]


async def _load_text(
    store: LocalFilesystemStore,
    session: Session,
    artifact_key: str,
) -> str | None:
    """Read one text artifact, or return `None` when it is missing."""
    path = session.artifact_paths.get(artifact_key)
    if not path:
        return None
    try:
        return (await store.read(session.id, path)).decode("utf-8")
    except FileNotFoundError:
        return None


def _render_viewer_html(
    *,
    store: LocalFilesystemStore,
    session: Session,
    transcript: Sequence[TranscriptFrame],
    prosody: Sequence[ProsodyFrame],
    story: str | None,
    feedback: str | None,
) -> str:
    """Build the full HTML document for one stored session."""
    artifact_links = _artifact_links(store, session)
    phone_hash = _escape(session.phone_number_hash or "Not captured")
    audio_src = (
        store.public_url(session.id, session.artifact_paths["audio"])
        if "audio" in session.artifact_paths
        else None
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Rehearse Viewer</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f5efe5;
        --panel: #fffaf2;
        --ink: #1e1a16;
        --muted: #6a6258;
        --line: #d9cdbd;
        --accent: #8a4b12;
        --accent-soft: #f1dfc7;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background:
          radial-gradient(circle at top right, #f6dcc0 0, transparent 30%),
          linear-gradient(180deg, #f8f1e7 0, var(--bg) 100%);
        color: var(--ink);
        font: 16px/1.5 Georgia, "Times New Roman", serif;
      }}
      main {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 32px 20px 56px;
      }}
      h1, h2 {{ margin: 0 0 12px; line-height: 1.1; }}
      h1 {{ font-size: 40px; }}
      h2 {{ font-size: 24px; }}
      p, li, td, th, pre {{ font-size: 15px; }}
      .hero {{
        display: grid;
        gap: 12px;
        margin-bottom: 24px;
      }}
      .badge {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font: 600 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .grid {{
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      }}
      section {{
        margin-top: 16px;
        padding: 18px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: color-mix(in srgb, var(--panel) 92%, white);
        box-shadow: 0 10px 30px rgb(77 49 19 / 0.06);
      }}
      dl {{ margin: 0; display: grid; gap: 10px; }}
      dt {{ font-weight: 700; }}
      dd {{ margin: 2px 0 0; color: var(--muted); }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        text-align: left;
        padding: 8px 0;
        border-bottom: 1px solid var(--line);
        vertical-align: top;
      }}
      ul {{ margin: 0; padding-left: 18px; }}
      pre {{
        overflow-x: auto;
        white-space: pre-wrap;
        padding: 14px;
        border-radius: 14px;
        background: #fbf6ef;
        border: 1px solid var(--line);
      }}
      audio {{ width: 100%; margin-top: 8px; }}
      .muted {{ color: var(--muted); }}
      .artifact-links a {{ color: var(--accent); }}
    </style>
  </head>
  <body>
    <main>
      <div class="hero">
        <span class="badge">{_escape(session.completion_status)}</span>
        <h1>Rehearse Session Viewer</h1>
        <p class="muted">Session <code>{_escape(session.id)}</code></p>
      </div>

      <div class="grid">
        <section>
          <h2>Overview</h2>
          <dl>
            <div><dt>Created</dt><dd>{_escape(session.created_at.isoformat())}</dd></div>
            <div><dt>Phone Hash</dt><dd>{phone_hash}</dd></div>
            <div><dt>Consent</dt><dd>{_escape(session.consent.value)}</dd></div>
            <div><dt>Artifacts</dt><dd class="artifact-links">{artifact_links}</dd></div>
          </dl>
        </section>
        <section>
          <h2>Audio</h2>
          {_render_audio(audio_src)}
        </section>
      </div>

      <div class="grid">
        <section>
          <h2>Intake</h2>
          {_render_structured_block(session.intake)}
        </section>
        <section>
          <h2>Persona</h2>
          {_render_structured_block(session.persona)}
        </section>
      </div>

      <section>
        <h2>Phase Timing</h2>
        {_render_phase_timings(session)}
      </section>

      <section>
        <h2>Transcript</h2>
        {_render_transcript(transcript)}
      </section>

      <section>
        <h2>Prosody Highlights</h2>
        {_render_prosody(prosody)}
      </section>

      <div class="grid">
        <section>
          <h2>Story</h2>
          {_render_markdown_text(story)}
        </section>
        <section>
          <h2>Feedback</h2>
          {_render_markdown_text(feedback)}
        </section>
      </div>
    </main>
  </body>
</html>"""


def _artifact_links(store: LocalFilesystemStore, session: Session) -> str:
    """Render public artifact links from the session manifest."""
    if not session.artifact_paths:
        return '<span class="muted">No artifacts yet.</span>'
    links = [
        f'<a href="{_escape(store.public_url(session.id, path))}">{_escape(name)}</a>'
        for name, path in sorted(session.artifact_paths.items())
    ]
    return " · ".join(links)


def _render_audio(audio_src: str | None) -> str:
    """Render the audio player or an empty-state note."""
    if not audio_src:
        return '<p class="muted">No call recording has been written yet.</p>'
    return (
        f'<p class="muted">Captured call audio from the runtime session.</p>'
        f'<audio controls preload="none" src="{_escape(audio_src)}"></audio>'
    )


def _render_structured_block(value: object | None) -> str:
    """Render a pydantic object as pretty JSON, or show an empty-state note."""
    if value is None:
        return '<p class="muted">Not captured yet.</p>'
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return f"<pre>{_escape(json.dumps(payload, indent=2, sort_keys=True))}</pre>"


def _render_phase_timings(session: Session) -> str:
    """Render the persisted phase timing rows as a simple table."""
    if not session.phase_timings:
        return '<p class="muted">No live phase timing has been captured yet.</p>'
    rows = "".join(
        "<tr>"
        f"<td>{_escape(item.phase.value)}</td>"
        f"<td>{_escape(item.started_at.isoformat())}</td>"
        f"<td>{_escape(item.ended_at.isoformat() if item.ended_at else 'Still active')}</td>"
        f"<td>{item.budget_seconds}s</td>"
        f"<td>{'yes' if item.overran else 'no'}</td>"
        "</tr>"
        for item in session.phase_timings
    )
    return (
        "<table><thead><tr><th>Phase</th><th>Started</th><th>Ended</th>"
        "<th>Budget</th><th>Overran</th></tr></thead><tbody>"
        f"{rows}</tbody></table>"
    )


def _render_transcript(transcript: Sequence[TranscriptFrame]) -> str:
    """Render transcript rows in reading order with speaker and phase labels."""
    if not transcript:
        return '<p class="muted">No transcript has been captured yet.</p>'
    rows = "".join(
        "<tr>"
        f"<td>{_escape(frame.speaker.value)}</td>"
        f"<td>{_escape(frame.phase.value)}</td>"
        f"<td>{frame.ts_start:.2f}-{frame.ts_end:.2f}</td>"
        f"<td>{_escape(frame.text)}</td>"
        "</tr>"
        for frame in transcript
    )
    return (
        "<table><thead><tr><th>Speaker</th><th>Phase</th><th>Time</th><th>Text</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _render_prosody(prosody: Sequence[ProsodyFrame]) -> str:
    """Render the strongest captured emotion labels for each prosody frame."""
    if not prosody:
        return '<p class="muted">No prosody measurements have been captured yet.</p>'
    rows = "".join(
        "<tr>"
        f"<td>{_escape(frame.speaker.value)}</td>"
        f"<td>{frame.scores.arousal:.2f}</td>"
        f"<td>{frame.scores.valence:.2f}</td>"
        f"<td>{_escape(_top_emotions(frame))}</td>"
        "</tr>"
        for frame in prosody
    )
    return (
        "<table><thead><tr><th>Speaker</th><th>Arousal</th><th>Valence</th>"
        f"<th>Top emotions</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _top_emotions(frame: ProsodyFrame) -> str:
    """Return a short comma-separated summary of the strongest emotion scores."""
    pairs = sorted(
        frame.scores.emotions.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    if not pairs:
        return "No emotion labels"
    return ", ".join(f"{name} {score:.2f}" for name, score in pairs)


def _render_markdown_text(value: str | None) -> str:
    """Render a stored markdown artifact as escaped preformatted text."""
    if not value:
        return '<p class="muted">Not generated yet.</p>'
    return f"<pre>{_escape(value)}</pre>"


def _escape(value: str) -> str:
    """Escape user-controlled text before inserting it into HTML."""
    return html.escape(value, quote=True)
