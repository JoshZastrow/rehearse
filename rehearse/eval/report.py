"""Eval experiment report: DuckDB persistence + Rich terminal table.

Tracks every run in `evals/runs/runs.duckdb` and renders a colour-coded
experiment table with per-metric scores, duration, and token usage.

Layout: the legend uses human-readable short names; the table uses 4-char
column headers so scores (0.00–1.00) always fit without truncation.
Minimum rendered width is 160 chars — use a wide terminal or pipe to `less -S`.
"""

from __future__ import annotations

import json
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
