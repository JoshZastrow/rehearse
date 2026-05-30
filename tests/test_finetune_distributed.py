"""Tests for finetune.distributed — CUDA_VISIBLE_DEVICES handling.

Modal (and some GPU environments) do not set CUDA_VISIBLE_DEVICES.
These tests verify that visible_devices() and set_device() degrade gracefully
when the env var is absent, rather than raising KeyError.

The finetune package is added to sys.path from train/ so these
tests run without installing the heavy training deps into the rehearse venv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make `finetune` importable without installing its deps.
_FINETUNE_ROOT = Path(__file__).parents[1] / "train"
if str(_FINETUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FINETUNE_ROOT))


def _mock_torch_cuda(device_count: int = 1, available: bool = True):
    """Context manager that stubs out torch in finetune.distributed."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        fake_cuda = MagicMock()
        fake_cuda.is_available.return_value = available
        fake_cuda.device_count.return_value = device_count
        fake_cuda.set_device = MagicMock()
        with patch.dict("sys.modules", {"torch": MagicMock(cuda=fake_cuda),
                                         "torch.distributed": MagicMock()}):
            # Re-import after patching so the module uses the stub
            if "finetune.distributed" in sys.modules:
                del sys.modules["finetune.distributed"]
            import finetune.distributed as dist_mod
            dist_mod.torch = MagicMock(cuda=fake_cuda)
            yield dist_mod

    return _ctx()


def test_visible_devices_without_env_var():
    """visible_devices() must not raise KeyError when CUDA_VISIBLE_DEVICES is unset.

    Modal does not set CUDA_VISIBLE_DEVICES. Without the fix, visible_devices()
    raises KeyError immediately, crashing train.py before any training code runs.
    """
    env_without = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    with patch.dict(os.environ, env_without, clear=True):
        with _mock_torch_cuda(device_count=1) as dist_mod:
            result = dist_mod.visible_devices()
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


def test_visible_devices_with_env_var():
    """visible_devices() must still work correctly when CUDA_VISIBLE_DEVICES is set."""
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
        with _mock_torch_cuda(device_count=2) as dist_mod:
            assert dist_mod.visible_devices() == [0, 1]


def test_set_device_without_cuda_visible_devices():
    """set_device() must not raise KeyError when CUDA_VISIBLE_DEVICES is unset.

    Regression: line 30 in distributed.py accessed os.environ['CUDA_VISIBLE_DEVICES']
    for a log line, raising KeyError in Modal where the var is absent.
    """
    env_without = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env_without["LOCAL_RANK"] = "0"
    with patch.dict(os.environ, env_without, clear=True):
        with _mock_torch_cuda(device_count=1) as dist_mod:
            # Should not raise
            dist_mod.set_device()
