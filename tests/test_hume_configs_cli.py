"""Verify the rehearse-hume CLI's applier wiring against a mocked Hume client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from rehearse.services.hume_configs import (
    PERSONAS,
    Create,
    NewVersion,
    apply_sync,
)


@pytest.mark.asyncio
async def test_apply_sync_creates_new_config(tmp_path: Path):
    fake_client = MagicMock()
    fake_client.empathic_voice.configs.create_config = AsyncMock(
        return_value=MagicMock(id="cfg_new")
    )
    fake_client.empathic_voice.configs.create_config_version = AsyncMock()

    actions = [Create(persona=PERSONAS["default"])]
    mapping_path = tmp_path / "mapping.json"

    mapping = await apply_sync(fake_client, actions, mapping_path=mapping_path)

    assert mapping["default"] == "cfg_new"
    assert fake_client.empathic_voice.configs.create_config.await_count == 1
    assert fake_client.empathic_voice.configs.create_config_version.await_count == 0
    on_disk = json.loads(mapping_path.read_text())
    assert on_disk["default"] == "cfg_new"
    assert "synced_at" in on_disk


@pytest.mark.asyncio
async def test_apply_sync_appends_new_version(tmp_path: Path):
    fake_client = MagicMock()
    fake_client.empathic_voice.configs.create_config = AsyncMock()
    fake_client.empathic_voice.configs.create_config_version = AsyncMock(
        return_value=MagicMock(id="cfg_existing")
    )

    actions = [
        NewVersion(
            persona=PERSONAS["default"],
            config_id="cfg_existing",
            diff=["voice", "prompt_text"],
        )
    ]
    mapping_path = tmp_path / "mapping.json"

    mapping = await apply_sync(fake_client, actions, mapping_path=mapping_path)

    assert mapping["default"] == "cfg_existing"
    assert fake_client.empathic_voice.configs.create_config.await_count == 0
    assert fake_client.empathic_voice.configs.create_config_version.await_count == 1
    call_kwargs = fake_client.empathic_voice.configs.create_config_version.await_args.kwargs
    assert call_kwargs["id"] == "cfg_existing"
