"""Tests for rehearse.train.modal image build and config helpers.

These tests run locally without Modal auth — they inspect the modal.py source
for image build ordering and call the pure helper functions directly.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_MODAL_PY = Path(__file__).parents[1] / "rehearse" / "train" / "modal.py"


def _image_chain_calls() -> list[str]:
    """Return the method names in the `image = (...)` builder chain in modal.py.

    Parses the AST so we are immune to comment changes and whitespace.
    """
    source = _MODAL_PY.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "image" not in targets:
            continue
        # Collect all chained .method() calls on the rhs
        calls: list[str] = []
        expr = node.value
        while isinstance(expr, ast.Call):
            func = expr.func
            if isinstance(func, ast.Attribute):
                calls.append(func.attr)
                expr = func.value
            else:
                break
        calls.reverse()
        return calls

    return []


# ── image structure ──────────────────────────────────────────────────────────


def test_image_add_local_dir_is_last_step():
    """add_local_dir without copy=True must be the final image build step.

    Modal raises InvalidError at deploy time if run_commands (or any other
    bake step) follows an add_local_* call that doesn't use copy=True, because
    Modal mounts those files at container startup rather than baking them into
    the image layer.

    Parsing the image builder chain from the AST catches this before any
    Modal API call.
    """
    source = _MODAL_PY.read_text()
    tree = ast.parse(source)

    # Locate the image assignment and parse the chained calls WITH keyword args.
    add_local_indices: list[int] = []
    bake_after_add_local: list[tuple[int, str]] = []

    calls = _image_chain_calls()

    ADD_LOCAL_METHODS = {"add_local_dir", "add_local_file", "add_local_python_source"}
    BAKE_METHODS = {"run_commands", "pip_install", "apt_install", "run_function",
                    "pip_install_from_pyproject", "pip_install_from_requirements"}

    # Walk the chain looking for add_local without copy=True followed by bake steps.
    # We treat add_local with copy=True as a bake step (safe to have things after it).
    # Since we can't easily extract kw args from the chain in _image_chain_calls,
    # use a regex scan for `add_local_dir(...copy=True...)` to detect the copy flag.
    copy_true_pattern = re.compile(r'add_local_\w+\s*\([^)]*copy\s*=\s*True')
    has_copy_true = bool(copy_true_pattern.search(source))

    for i, method in enumerate(calls):
        if method in ADD_LOCAL_METHODS and not has_copy_true:
            add_local_indices.append(i)

    for i, method in enumerate(calls):
        if method in BAKE_METHODS:
            for al_idx in add_local_indices:
                if i > al_idx:
                    bake_after_add_local.append((i, method))

    assert not bake_after_add_local, (
        f"Build step(s) {bake_after_add_local} appear after add_local_dir "
        f"without copy=True. Either move add_local_dir to the end of the chain "
        f"or set copy=True. Chain: {calls}"
    )


def test_image_uses_pyproject_for_finetune_deps():
    """The Modal image must derive moshi-finetune deps from its pyproject.toml.

    Previously the image manually listed a subset of pip packages and missed
    deps like torch==2.6, triton, and tqdm declared in pyproject.toml.
    The fix uses pip_install_from_pyproject so pyproject.toml is the single
    source of truth.
    """
    source = _MODAL_PY.read_text()
    assert "pip_install_from_pyproject" in source, (
        "Modal image must call pip_install_from_pyproject pointing at "
        "lib/moshi-finetune/pyproject.toml. Without it, the manually listed "
        "packages may diverge from the declared dependencies."
    )


# ── config path rewriting ────────────────────────────────────────────────────


def test_rewrite_config_paths_sets_volume_train_data():
    from rehearse.train.modal import _rewrite_config_paths

    cfg = {"run_dir": "runs/my-run", "data": {"train_data": "/local/sessions.jsonl", "shuffle": True}}
    out = _rewrite_config_paths(cfg)
    assert out["data"]["train_data"] == "/data/data/sessions.jsonl"


def test_rewrite_config_paths_extracts_run_name():
    from rehearse.train.modal import _rewrite_config_paths

    cfg = {"run_dir": "runs/smoke-test", "data": {}}
    out = _rewrite_config_paths(cfg)
    assert out["run_dir"] == "/data/runs/smoke-test"


def test_rewrite_config_paths_does_not_mutate_input():
    from rehearse.train.modal import _rewrite_config_paths

    cfg = {"run_dir": "runs/x", "data": {"train_data": "local.jsonl"}}
    _rewrite_config_paths(cfg)
    assert cfg["data"]["train_data"] == "local.jsonl"


def test_to_volume_path():
    from rehearse.train.modal import _to_volume_path

    result = _to_volume_path("/local/sessions/abc123/audio_stereo.wav", "abc123")
    assert result == "/data/data/sessions/abc123/audio_stereo.wav"


# ── dep compatibility ─────────────────────────────────────────────────────────

_FINETUNE_PYPROJECT = Path(__file__).parents[1] / "lib" / "moshi-finetune" / "pyproject.toml"


def test_sphn_pin_compatible_with_moshi():
    """moshi-finetune's sphn pin must satisfy moshi's sphn>=0.2.0,<0.3.0 requirement.

    moshi 0.2.13 (from git HEAD) requires sphn>=0.2.0,<0.3.0. The old pin
    sphn==0.1.12 causes a ResolutionImpossible error during Modal image build.
    """
    import re
    from packaging.version import Version
    from packaging.specifiers import SpecifierSet

    source = _FINETUNE_PYPROJECT.read_text()
    match = re.search(r'"sphn([^"]*)"', source)
    assert match, "Could not find sphn dependency in moshi-finetune/pyproject.toml"

    spec_str = match.group(1).strip()
    spec = SpecifierSet(spec_str) if spec_str else SpecifierSet("")

    # moshi 0.2.13 requires sphn>=0.2.0,<0.3.0
    moshi_requires = SpecifierSet(">=0.2.0,<0.3.0")
    compatible = [v for v in ["0.1.12", "0.2.0", "0.2.1"] if v in spec]
    assert any(v in moshi_requires for v in compatible), (
        f"sphn pin '{spec_str}' in pyproject.toml has no versions satisfying "
        f"moshi's >=0.2.0,<0.3.0 requirement. Update to e.g. '>=0.2.0,<0.3.0'."
    )
    assert "0.1.12" not in spec, (
        "sphn==0.1.12 is incompatible with moshi>=0.2.13. Update the pin."
    )
