"""CLI entry point for managing Hume EVI configs declaratively.

Examples:
    # Show what sync would change without writing anything.
    rehearse-hume diff

    # Reconcile the live workspace against PERSONAS, then write
    # `sessions/.hume_configs.json` with persona_key -> config_id.
    rehearse-hume sync

The first time you run `sync`, Hume will likely contain a manually-created
config under a different display_name (e.g. with a timestamp). To avoid
creating a duplicate, rename that config in the Hume console to match the
declared `display_name` (e.g. `Rehearse Coach (default)`) before running
sync. After that, every change to `PERSONAS` is one `sync` away from being
live.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from hume.client import AsyncHumeClient

from rehearse.config import RuntimeConfig
from rehearse.services import hume_configs as _hume_configs
from rehearse.services.hume_configs import (
    MAPPING_PATH_DEFAULT,
    PERSONAS,
    Create,
    NewVersion,
    NoOp,
    apply_sync,
    plan_sync,
)


async def run_diff(client) -> int:
    """Print planned actions; exit 1 if any Create/NewVersion is needed."""
    remote = await _hume_configs.fetch_remote_configs(client)
    actions = plan_sync(PERSONAS, remote_configs=remote)
    drift = False
    for action in actions:
        if isinstance(action, Create):
            print(f"CREATE {action.persona.persona_key} ({action.persona.display_name})")
            drift = True
        elif isinstance(action, NewVersion):
            print(
                f"NEW_VERSION {action.persona.persona_key} "
                f"({action.config_id}) diff={action.diff}"
            )
            drift = True
        elif isinstance(action, NoOp):
            print(f"NOOP {action.persona.persona_key} ({action.config_id})")
    return 1 if drift else 0


async def run_sync(client, *, mapping_path: Path = MAPPING_PATH_DEFAULT) -> int:
    """Execute pending actions and write the persona->config_id mapping."""
    remote = await _hume_configs.fetch_remote_configs(client)
    actions = plan_sync(PERSONAS, remote_configs=remote)
    mapping = await apply_sync(client, actions, mapping_path=mapping_path)
    print(f"Wrote {mapping_path} with {len(mapping)} persona(s).")
    return 0


def main() -> None:
    """Argparse entry point for `rehearse-hume`."""
    parser = argparse.ArgumentParser(prog="rehearse-hume")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("diff", help="Show planned reconcile actions; exit 1 if drifted.")
    sub.add_parser("sync", help="Apply reconcile actions and write the id mapping.")
    args = parser.parse_args()

    cfg = RuntimeConfig.from_env()
    client = AsyncHumeClient(api_key=cfg.hume_api_key)

    if args.command == "diff":
        sys.exit(asyncio.run(run_diff(client)))
    if args.command == "sync":
        sys.exit(asyncio.run(run_sync(client)))
