"""Eval experiment report: DuckDB persistence + Rich terminal table.

Tracks every run in `evals/runs/runs.duckdb` and renders a colour-coded
experiment table with per-metric scores, duration, and token usage.

Layout: the legend uses human-readable short names; the table uses 4-char
column headers so scores (0.00–1.00) always fit without truncation.
Minimum rendered width is 160 chars — use a wide terminal or pipe to `less -S`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
from rich import box
from rich.console import Console
from rich.table import Table

# ── metric metadata ──────────────────────────────────────────────────────────

# Legend abbreviations: human-readable, shown above the table.
_METRIC_ABBREV: dict[str, str] = {
    "weighted_reward":                  "reward",
    "content_quality":                  "content",
    "affect_perception":                "affect",
    "delivery_quality":                 "delivery",
    "intake_fidelity":                  "intake",
    "naturalness.interruption_rate":    "no-int",
    "naturalness.silence_after_affect": "silence",
    "naturalness.speech_rate_band":     "speech",
}

# Table column headers: exactly 4 chars, never truncated by Rich.
_TABLE_COL: dict[str, str] = {
    "weighted_reward":                  "rwrd",
    "content_quality":                  "cont",
    "affect_perception":                "afct",
    "delivery_quality":                 "dlvr",
    "intake_fidelity":                  "intk",
    "naturalness.interruption_rate":    "nint",
    "naturalness.silence_after_affect": "slnc",
    "naturalness.speech_rate_band":     "spch",
}

_METRIC_DESC: dict[str, str] = {
    "reward":   "Weighted composite across all scored dimensions",
    "content":  "Did the coach move the user toward clearer, more effective phrasing?",
    "affect":   "Did the coach correctly read the user's emotional state?",
    "delivery": "Did the coach's prosody and pacing match the emotional moment?",
    "intake":   "Did the runtime correctly capture situation, relationship, and stakes?",
    "no-int":   "Interruption rate — 0 interruptions per turn is ideal",
    "silence":  "Silence after affect events — 1.5–4.0s is ideal",
    "speech":   "Coach speech rate band — 130–170 wpm is ideal",
}

# Minimum console width to render without column compression.
_MIN_WIDTH = 160

# ── DuckDB ───────────────────────────────────────────────────────────────────

_DB_FILE = "runs.duckdb"


def _connect(runs_root: Path) -> duckdb.DuckDBPyConnection:
    runs_root.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(runs_root / _DB_FILE))
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id       VARCHAR PRIMARY KEY,
            eval_name    VARCHAR,
            environment  VARCHAR,
            run_date     TIMESTAMP,
            n_examples   INTEGER,
            duration_s   FLOAT,
            total_tokens INTEGER,
            scores       JSON
        )
    """)
    return con


def record_run(
    *,
    run_id: str,
    eval_name: str,
    environment: str,
    run_date: datetime,
    n_examples: int,
    duration_s: float,
    total_tokens: int,
    scores: dict[str, float],
    runs_root: Path,
) -> None:
    """Upsert one run into the experiment DB."""
    con = _connect(runs_root)
    con.execute(
        """
        INSERT OR REPLACE INTO runs
            (run_id, eval_name, environment, run_date, n_examples, duration_s, total_tokens, scores)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id, eval_name, environment, run_date,
            n_examples, duration_s, total_tokens,
            json.dumps(scores),
        ],
    )
    con.close()


def ensure_run_recorded(run_id: str, runs_root: Path) -> None:
    """Reconstruct a run from disk and record it if not already in the DB.

    Used by `rehearse-eval show` for runs that pre-date the report system.
    """
    con = _connect(runs_root)
    existing = con.execute(
        "SELECT run_id FROM runs WHERE run_id = ?", [run_id]
    ).fetchone()
    con.close()
    if existing:
        return

    run_dir = runs_root / run_id
    run_json_path = run_dir / "run.json"
    if not run_json_path.exists():
        return

    data = json.loads(run_json_path.read_text())
    pipeline = data.get("pipeline_version", "/")
    eval_name = pipeline.split("/")[0].split("@")[0]
    environment = pipeline.split("/")[1].split("@")[0] if "/" in pipeline else ""
    started = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else datetime.now()
    completed = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else started
    duration_s = (completed - started).total_seconds()
    n_examples = len(data.get("example_ids", []))
    scores = data.get("aggregate_scores") or {}

    record_run(
        run_id=run_id,
        eval_name=eval_name,
        environment=environment,
        run_date=started,
        n_examples=n_examples,
        duration_s=duration_s,
        total_tokens=0,
        scores=scores,
        runs_root=runs_root,
    )


# ── Rich renderer ─────────────────────────────────────────────────────────────

def render_report(runs_root: Path, highlight_run_id: str | None = None) -> None:
    """Print metric legend then a colour-coded experiment table to the terminal."""
    con = _connect(runs_root)
    rows = con.execute("""
        SELECT run_id, eval_name, environment, run_date,
               n_examples, duration_s, total_tokens, scores
        FROM runs
        ORDER BY run_date DESC
        LIMIT 50
    """).fetchall()
    con.close()

    # Use actual terminal width, but guarantee enough room for all columns.
    width = max(Console().width, _MIN_WIDTH)
    console = Console(width=width)

    if not rows:
        console.print("[dim]No runs recorded yet.[/dim]")
        return

    # Determine which metric keys appear across all runs, in preferred order.
    all_score_dicts: list[dict[str, float]] = [
        json.loads(r[7]) if r[7] else {} for r in rows
    ]
    seen_keys: list[str] = []
    for k in _METRIC_ABBREV:
        if any(k in s for s in all_score_dicts):
            seen_keys.append(k)
    for s in all_score_dicts:
        for k in s:
            if k not in seen_keys:
                seen_keys.append(k)

    legend_abbrevs = [_METRIC_ABBREV.get(k, k) for k in seen_keys]
    col_headers    = [_TABLE_COL.get(k, k[:4]) for k in seen_keys]

    # ── legend ────────────────────────────────────────────────────────────────
    console.print()
    console.print("[bold]Metrics[/bold]")
    for col, abbr, key in zip(col_headers, legend_abbrevs, seen_keys):
        desc = _METRIC_DESC.get(abbr, "")
        console.print(f"  [cyan]{col}[/cyan]  {abbr:<8}  {desc}")
    console.print()

    # ── table ─────────────────────────────────────────────────────────────────
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        pad_edge=False,
    )

    # Fixed columns — id uses just the 8-char hash for compactness.
    table.add_column("id",    style="dim", no_wrap=True, min_width=8)
    table.add_column("date",  no_wrap=True, min_width=11)
    table.add_column("eval",  no_wrap=False, min_width=10, max_width=24)
    table.add_column("n",     justify="right", min_width=1)
    table.add_column("secs",  justify="right", min_width=4, no_wrap=True)
    table.add_column("tok",   justify="right", min_width=6, no_wrap=True)

    # Score columns — 4-char header, min_width=4 ensures 0.00 never truncates.
    for col in col_headers:
        table.add_column(col, justify="right", min_width=4, no_wrap=True)

    for row in rows:
        run_id, eval_name, _env, run_date, n_examples, duration_s, total_tokens, scores_json = row
        scores = json.loads(scores_json) if scores_json else {}
        is_highlight = run_id == highlight_run_id

        # 8-char hash: last segment after final '-', else last 8 chars.
        short_id = run_id.rsplit("-", 1)[-1] if run_id and "-" in run_id else (run_id or "")[-8:]
        date_str = run_date.strftime("%m-%d %H:%M") if run_date else ""
        dur_str  = f"{int(duration_s)}s" if duration_s else "-"
        tok_str  = f"{int(total_tokens):,}" if total_tokens else "-"

        score_cells: list[str] = []
        for k in seen_keys:
            v = scores.get(k)
            if v is None:
                score_cells.append("[dim] -  [/dim]")
            else:
                color = "green" if v >= 0.7 else ("yellow" if v >= 0.4 else "red")
                score_cells.append(f"[{color}]{v:.2f}[/{color}]")

        row_style = "bold yellow" if is_highlight else ""
        table.add_row(
            short_id, date_str, eval_name or "",
            str(n_examples or ""), dur_str, tok_str,
            *score_cells,
            style=row_style,
        )

    console.print(table)


# ── list-runs ─────────────────────────────────────────────────────────────────

def _load_session_data(session_dir: Path) -> dict:
    """Read the lightweight per-session files we need for list-runs display."""
    data: dict = {}

    session_json = session_dir / "session.json"
    if session_json.exists():
        try:
            s = json.loads(session_json.read_text())
            data["completion_status"] = s.get("completion_status", "unknown")
            data["phase_timings"] = s.get("phase_timings", [])
        except Exception:
            pass

    judge_json = session_dir / "judge.json"
    if judge_json.exists():
        try:
            j = json.loads(judge_json.read_text())
            scores: dict[str, float] = dict(j.get("per_dim") or {})
            if "weighted_reward" in j:
                scores["weighted_reward"] = j["weighted_reward"]
            data["scores"] = scores
        except Exception:
            pass

    audio_path = session_dir / "audio.wav"
    if audio_path.exists():
        data["audio"] = audio_path

    return data


def _score_cell(v: float | None) -> str:
    if v is None:
        return "[dim]-[/dim]"
    color = "green" if v >= 0.7 else ("yellow" if v >= 0.4 else "red")
    return f"[{color}]{v:.2f}[/{color}]"


def list_runs(
    runs_root: Path,
    *,
    n: int = 10,
    eval_filter: str | None = None,
    scenario_filter: str | None = None,
    play_session: str | None = None,
) -> None:
    """Print recent runs with their per-rollout scores and audio paths."""
    console = Console()

    # Collect run directories, newest first.
    run_dirs = sorted(
        [d for d in runs_root.iterdir() if d.is_dir() and (d / "run.json").exists()],
        key=lambda d: d.name,
        reverse=True,
    )

    if eval_filter:
        run_dirs = [
            d for d in run_dirs
            if eval_filter in (json.loads((d / "run.json").read_text()).get("pipeline_version", ""))
        ]

    run_dirs = run_dirs[:n]

    if not run_dirs:
        console.print("[dim]No runs found.[/dim]")
        return

    # If --play, find the most recent audio for that session and open it.
    if play_session:
        for run_dir in sorted(runs_root.iterdir(), key=lambda d: d.name, reverse=True):
            if not run_dir.is_dir():
                continue
            sessions_root = run_dir / "sessions"
            if not sessions_root.exists():
                continue
            # Match by exact id or suffix (e.g. "vrj-s01-peer-feedback-anxious"
            # matches "eval-vrj-s01-peer-feedback-anxious").
            for session_dir in sessions_root.iterdir():
                if play_session in session_dir.name:
                    audio = session_dir / "audio.wav"
                    if audio.exists():
                        console.print(f"Opening [cyan]{audio}[/cyan]")
                        if sys.platform == "darwin":
                            subprocess.run(["open", str(audio)], check=False)
                        else:
                            subprocess.run(["xdg-open", str(audio)], check=False)
                        return
        console.print(f"[yellow]No audio found for session matching {play_session!r}[/yellow]")
        return

    score_keys = list(_TABLE_COL.keys())  # preferred order

    for run_dir in run_dirs:
        try:
            run_data = json.loads((run_dir / "run.json").read_text())
        except Exception:
            continue

        pipeline = run_data.get("pipeline_version", "")
        eval_name = pipeline.split("/")[0].split("@")[0] if pipeline else "?"
        short_id = run_dir.name.rsplit("-", 1)[-1]

        started_raw = run_data.get("started_at", "")
        try:
            started = datetime.fromisoformat(started_raw)
            date_str = started.strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_str = started_raw[:16]

        completed_raw = run_data.get("completed_at")
        try:
            duration_s = int(
                (datetime.fromisoformat(completed_raw) - datetime.fromisoformat(started_raw)).total_seconds()
            )
            dur_str = f"{duration_s}s"
        except Exception:
            dur_str = "?"

        example_ids: list[str] = run_data.get("example_ids", [])
        agg = run_data.get("aggregate_scores") or {}
        rwrd = agg.get("weighted_reward")
        rwrd_str = _score_cell(rwrd)

        n_examples = len(example_ids)
        console.print(
            f"\n[bold cyan]{short_id}[/bold cyan]  "
            f"[dim]{date_str}[/dim]  "
            f"[bold]{eval_name}[/bold]  "
            f"{n_examples} rollout{'s' if n_examples != 1 else ''}  "
            f"[dim]{dur_str}[/dim]  "
            f"rwrd {rwrd_str}"
        )

        sessions_root = run_dir / "sessions"
        if not sessions_root.exists():
            console.print("  [dim](no sessions directory)[/dim]")
            continue

        # Build session list: from run.json example_ids in order,
        # plus any sessions directory entries not in example_ids.
        session_names = list(example_ids)
        for entry in sorted(sessions_root.iterdir()):
            bare = entry.name.removeprefix("eval-")
            if bare not in session_names and entry.name not in session_names:
                session_names.append(entry.name)

        if scenario_filter:
            session_names = [s for s in session_names if scenario_filter in s]

        for i, example_id in enumerate(session_names):
            is_last = i == len(session_names) - 1
            branch = "└─" if is_last else "├─"
            pad    = "   " if is_last else "│  "

            # Session dir may be stored as "eval-<id>" or "<id>".
            session_dir = sessions_root / f"eval-{example_id}"
            if not session_dir.exists():
                session_dir = sessions_root / example_id
            if not session_dir.exists():
                console.print(f"  {branch} [dim]{example_id}[/dim]  [dim](no session dir)[/dim]")
                continue

            sd = _load_session_data(session_dir)

            # Phase timing summary.
            timing_parts: list[str] = []
            for pt in sd.get("phase_timings", []):
                phase = pt.get("phase", "?")
                try:
                    secs = int(
                        (datetime.fromisoformat(pt["ended_at"]) - datetime.fromisoformat(pt["started_at"])).total_seconds()
                    )
                    chunk = f"{phase} {secs}s"
                except Exception:
                    chunk = phase
                if pt.get("overran"):
                    chunk += " [yellow]⚠ overran[/yellow]"
                timing_parts.append(chunk)
            timing_str = "  ".join(timing_parts) if timing_parts else ""

            # Per-session scores from judge.json.
            scores = sd.get("scores") or {}
            score_parts: list[str] = []
            for key in score_keys:
                abbr = _TABLE_COL.get(key, key[:4])
                v = scores.get(key)
                if v is not None:
                    score_parts.append(f"{abbr} {_score_cell(float(v))}")
            score_str = "  ".join(score_parts)

            audio: Path | None = sd.get("audio")

            console.print(f"  {branch} [bold]{example_id}[/bold]")
            if timing_str or score_str:
                console.print(f"  {pad}  {timing_str}{'  ' if timing_str and score_str else ''}{score_str}")
            if audio:
                console.print(f"  {pad}  ", end="")
                console.print(str(audio), style="dim", highlight=False, overflow="fold")
            else:
                console.print(f"  {pad}  [dim](no audio)[/dim]")

    console.print()
