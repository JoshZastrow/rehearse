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
from rehearse.services.hume_configs_cli import run_diff, run_sync


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



@pytest.mark.asyncio
async def test_run_diff_returns_zero_when_in_sync(tmp_path: Path, monkeypatch, capsys):
    from rehearse.services import hume_configs

    async def _fake_fetch(client):
        snap = hume_configs.RemoteConfigSnapshot(
            id="cfg_match",
            display_name=PERSONAS["default"].display_name,
            evi_version=PERSONAS["default"].evi_version,
            voice=PERSONAS["default"].voice.model_copy(),
            language_model=PERSONAS["default"].language_model.model_copy(),
            prompt_text=PERSONAS["default"].prompt_text,
            on_new_chat=PERSONAS["default"].on_new_chat.model_copy(),
            on_max_duration_timeout=PERSONAS["default"].on_max_duration_timeout.model_copy(),
            on_inactivity_timeout=PERSONAS["default"].on_inactivity_timeout.model_copy(),
            timeouts=PERSONAS["default"].timeouts.model_copy(),
            turn_detection=PERSONAS["default"].turn_detection.model_copy(),
            interruption_min_ms=PERSONAS["default"].interruption_min_ms,
            nudges_enabled=PERSONAS["default"].nudges_enabled,
            nudges_interval_secs=PERSONAS["default"].nudges_interval_secs,
            builtin_tools=list(PERSONAS["default"].builtin_tools),
        )
        return [snap]

    monkeypatch.setattr(hume_configs, "fetch_remote_configs", _fake_fetch)
    fake_client = MagicMock()
    exit_code = await run_diff(fake_client)
    assert exit_code == 0


@pytest.mark.asyncio
async def test_run_diff_returns_one_when_drifted(monkeypatch, capsys):
    from rehearse.services import hume_configs

    async def _fake_fetch(client):
        return []

    monkeypatch.setattr(hume_configs, "fetch_remote_configs", _fake_fetch)
    fake_client = MagicMock()
    exit_code = await run_diff(fake_client)
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "CREATE" in captured.out


@pytest.mark.asyncio
async def test_run_sync_writes_mapping(tmp_path: Path, monkeypatch):
    from rehearse.services import hume_configs

    async def _fake_fetch(client):
        return []

    monkeypatch.setattr(hume_configs, "fetch_remote_configs", _fake_fetch)
    fake_client = MagicMock()
    fake_client.empathic_voice.configs.create_config = AsyncMock(
        return_value=MagicMock(id="cfg_new")
    )
    fake_client.empathic_voice.configs.create_config_version = AsyncMock()

    mapping_path = tmp_path / "mapping.json"
    exit_code = await run_sync(fake_client, mapping_path=mapping_path)
    assert exit_code == 0
    on_disk = json.loads(mapping_path.read_text())
    assert on_disk["default"] == "cfg_new"
