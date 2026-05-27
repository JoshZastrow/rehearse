"""`rehearse-eval watch <run_dir>` — live aggregate readout from `scores.jsonl`.

Tails `<run_dir>/scores.jsonl`, parses each appended `RubricScore`, and
re-renders an aggregate table using `rich`. Composite recompute waits
for every child dimension of an example to land before that example is
folded into the aggregate (decision per spec §13.2).

The watcher exits cleanly when `<run_dir>/scores.jsonl.done` appears.

Decoupled from the runner — runner writes JSONL, watcher reads. Either
can run in any order; if the watcher starts after the runner finishes,
it reads the existing file in one pass and exits on the sentinel.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from statistics import fmean
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table


def watch(run_dir: Path, *, poll_interval_s: float = 0.2) -> int:
    """Tail `run_dir/scores.jsonl` and re-render until the sentinel appears.

    Returns 0 on clean exit (sentinel found), 1 if the run dir is missing.
    """
    if not run_dir.exists():
        Console(stderr=True).print(f"[red]run_dir not found: {run_dir}[/red]")
        return 1

    scores_path = run_dir / "scores.jsonl"
    sentinel_path = run_dir / "scores.jsonl.done"

    state = AggregateState()
    console = Console()

    with Live(state.render(run_dir), console=console, refresh_per_second=4) as live:
        for line in _follow(scores_path, sentinel_path, poll_interval_s):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate partial / malformed lines
            state.add(row)
            live.update(state.render(run_dir))

        # Final render after sentinel.
        live.update(state.render(run_dir, done=True))

    return 0


class AggregateState:
    """Accumulate scores by (example_id, dimension) and render a `rich` table.

    The composite (`weighted_reward`) is published by the runner once all
    child dimensions for an example have landed, so we treat `weighted_reward`
    rows the same as any other dimension — no recompute logic in the watcher.
    """

    def __init__(self) -> None:
        # dimension -> list of values (one per example completed for that dim)
        self._values: dict[str, list[float]] = defaultdict(list)
        # example_id -> set of dimensions seen
        self._dims_per_example: dict[str, set[str]] = defaultdict(set)
        self._n_total: int = 0

    def add(self, row: dict[str, Any]) -> None:
        dim = row.get("dimension")
        val = row.get("value")
        ex_id = row.get("example_id")
        if dim is None or val is None or ex_id is None:
            return
        self._values[dim].append(float(val))
        self._dims_per_example[ex_id].add(dim)
        self._n_total += 1

    def render(self, run_dir: Path, *, done: bool = False) -> Table:
        title = f"Eval run [bold cyan]{run_dir.name}[/bold cyan]"
        if done:
            title += " — [green]done[/green]"
        else:
            title += f" — {self._n_total} scores landed"

        table = Table(title=title, header_style="bold")
        table.add_column("Dimension", style="cyan")
        table.add_column("Mean", justify="right", style="green")
        table.add_column("N", justify="right")

        # weighted_reward (composite aggregate) at the bottom; everything else
        # alphabetical for stable ordering across redraws.
        dims = sorted(d for d in self._values if d != "weighted_reward")
        if "weighted_reward" in self._values:
            dims = dims + ["weighted_reward"]

        for dim in dims:
            vals = self._values[dim]
            mean = fmean(vals) if vals else 0.0
            style = "bold yellow" if dim == "weighted_reward" else ""
            table.add_row(
                f"[{style}]{dim}[/{style}]" if style else dim,
                f"{mean:.3f}",
                str(len(vals)),
            )

        if not dims:
            table.add_row("[dim]waiting for first score…[/dim]", "", "")

        return table


def _follow(
    scores_path: Path,
    sentinel_path: Path,
    poll_interval_s: float,
) -> Iterator[str]:
    """Yield one line at a time from `scores_path`, polling for new lines.

    Stops yielding when `sentinel_path` exists AND the file is fully consumed.
    Reads existing content first if the file already has lines.
    """
    # Wait for the file to exist (runner may not have started yet).
    while not scores_path.exists():
        if sentinel_path.exists():
            return
        time.sleep(poll_interval_s)

    with open(scores_path, "r", encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if line:
                stripped = line.rstrip("\n")
                if stripped:
                    yield stripped
                continue
            if sentinel_path.exists():
                # Drain anything still buffered after sentinel appeared.
                tail = fh.read()
                for tail_line in tail.splitlines():
                    if tail_line:
                        yield tail_line
                return
            time.sleep(poll_interval_s)
