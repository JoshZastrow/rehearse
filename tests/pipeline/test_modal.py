import pytest
from rehearse.train.modal import _rewrite_config_paths, _to_volume_path


def test_rewrite_sets_train_data():
    cfg = {"data": {"train_data": "/local/sessions.jsonl", "eval_data": ""}, "run_dir": "runs/exp1"}
    result = _rewrite_config_paths(cfg)
    assert result["data"]["train_data"] == "/data/data/sessions.jsonl"


def test_rewrite_sets_run_dir():
    cfg = {"data": {"train_data": ""}, "run_dir": "runs/exp1"}
    result = _rewrite_config_paths(cfg)
    assert result["run_dir"] == "/data/runs/exp1"


def test_rewrite_uses_run_dir_stem():
    cfg = {"data": {"train_data": ""}, "run_dir": "runs/moshi_7B-20260528-120000"}
    result = _rewrite_config_paths(cfg)
    assert result["run_dir"] == "/data/runs/moshi_7B-20260528-120000"


def test_rewrite_preserves_other_fields():
    cfg = {"data": {"train_data": ""}, "run_dir": "runs/x", "max_steps": 500, "lora": {"rank": 64}}
    result = _rewrite_config_paths(cfg)
    assert result["max_steps"] == 500
    assert result["lora"]["rank"] == 64


def test_rewrite_does_not_mutate_input():
    cfg = {"data": {"train_data": "/local/sessions.jsonl"}, "run_dir": "runs/x"}
    _rewrite_config_paths(cfg)
    assert cfg["data"]["train_data"] == "/local/sessions.jsonl"


def test_rewrite_creates_data_section_if_missing():
    cfg = {"run_dir": "runs/x"}
    result = _rewrite_config_paths(cfg)
    assert result["data"]["train_data"] == "/data/data/sessions.jsonl"


def test_to_volume_path_wav():
    result = _to_volume_path("/Users/josh/sessions/abc123/audio.wav", "abc123")
    assert result == "/data/data/sessions/abc123/audio.wav"


def test_to_volume_path_json():
    result = _to_volume_path("/local/sessions/xyz/audio.json", "xyz")
    assert result == "/data/data/sessions/xyz/audio.json"
