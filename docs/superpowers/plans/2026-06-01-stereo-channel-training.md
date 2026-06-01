# Stereo Channel Training Implementation Plan

> **Status: COMPLETE** — All three tasks shipped and merged to `main` on 2026-06-01. PR: [#30](https://github.com/JoshZastrow/rehearse/pull/30). Default training target is now the caller (channel 1); provider model uses `config_provider.yaml`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `audio_stereo.wav` per-channel audio encoding into moshi fine-tuning so provider and caller models each train on their own speaker's audio only.

**Architecture:** Three changes: (1) `InterleavedTokenizer` gains a `channel` param that selects a mono channel from stereo input before Mimi encoding; (2) `TrainArgs` and `train.py` expose `channel` and `main_speaker_label` as config fields; (3) Provider and caller YAML configs set these fields correctly. `train/pipeline/dataset.py` already prefers `audio_stereo.wav` paths in the manifest — no changes needed there.

**Tech Stack:** PyTorch, moshi (Mimi encoder, LMModel), simple_parsing (Serializable dataclasses), chz (CLI config), sphn (audio I/O). Files are in `lib/moshi-finetune/` (the fine-tuning library) and `rehearse/models/moshi_7B/` (our model configs).

---

## Background: data flow

The training data pipeline produces `audio_stereo.wav` (stereo, 2-channel WAV):
- **Left channel (index 0):** provider speech only (coach/AI agent)
- **Right channel (index 1):** caller speech only (person being coached)

`train/pipeline/dataset.py` already writes `audio_stereo.wav` paths to the manifest when the file exists.

`InterleavedTokenizer.__call__` receives `wav: np.ndarray` of shape `[C, T]` from `sphn.dataset_jsonl`. For stereo, `C=2`. Currently `audio_tensor[:, None]` produces `[2, 1, T]`, which Mimi encodes as a batch of 2 → `[2, 8, T_frames]`. Then `.view(1, -1, T)` produces `[1, 16, T_frames]`, breaking the model's `K == 9` assertion. The fix is to select one channel before encoding.

The interleaver text masking already works: `keep_main_only=True` + `main_speaker_label="provider"` drops caller text tokens from the loss. `main_speaker_label` is currently hardcoded in `train.py`; this plan makes it configurable.

---

## File structure

| File | Change |
|------|--------|
| `lib/moshi-finetune/finetune/data/interleaver.py` | Add `channel: int = 0` to `InterleavedTokenizer`, select mono channel in `__call__` |
| `lib/moshi-finetune/finetune/args.py` | Add `channel: int = 0` and `main_speaker_label: str = "provider"` to `TrainArgs` |
| `lib/moshi-finetune/train.py` | Pass `args.channel` to `InterleavedTokenizer`, `args.main_speaker_label` to `Interleaver` |
| `rehearse/models/moshi_7B/config.yaml` | Add `channel: 0` and `main_speaker_label: provider` |
| `rehearse/models/moshi_7B/config_caller.yaml` | New file — caller model config with `channel: 1`, `main_speaker_label: caller` |
| `tests/pipeline/test_stereo_channel.py` | New — unit tests for channel selection behavior |

---

### Task 1: `InterleavedTokenizer` channel selection

**Files:**
- Modify: `lib/moshi-finetune/finetune/data/interleaver.py:247-289`
- Create: `tests/pipeline/test_stereo_channel.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_stereo_channel.py`:

```python
import math
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock


def _make_mock_mimi(dep_q: int = 8, frame_rate: float = 12.5, sample_rate: int = 24000):
    """Mock Mimi that records the input it was called with and returns zero tokens."""
    mimi = MagicMock()
    mimi.frame_rate = frame_rate
    mimi.sample_rate = sample_rate
    calls = []

    def encode(x):
        calls.append(x.clone().cpu())
        B, C, T = x.shape
        T_frames = max(1, math.ceil(T * frame_rate / sample_rate))
        return torch.zeros(B, dep_q, T_frames)

    mimi.encode.side_effect = encode
    mimi._calls = calls
    return mimi


def _make_mock_interleaver(zero_padding: int = 0, frame_rate: float = 12.5):
    iv = MagicMock()
    iv.zero_padding = zero_padding
    iv.audio_frame_rate = frame_rate
    iv.prepare_item.return_value = torch.zeros(1, 1, 16)
    return iv


def _make_tokenizer(channel: int):
    from finetune.data.interleaver import InterleavedTokenizer
    mimi = _make_mock_mimi()
    interleaver = _make_mock_interleaver()
    tok = InterleavedTokenizer(mimi, interleaver, duration_sec=1.0, channel=channel)
    return tok, mimi


def test_stereo_channel0_selects_left(tmp_path):
    """Channel 0 encodes only the left (provider) track."""
    import json
    tok, mimi = _make_tokenizer(channel=0)

    # Stereo wav: left=1.0, right=-1.0
    T = 24000
    wav = np.stack([np.ones(T), -np.ones(T)])  # [2, T]

    info = {"alignments": []}
    p = tmp_path / "audio_stereo.wav"
    p.touch()
    (tmp_path / "audio_stereo.json").write_text(json.dumps(info))

    tok(wav, 0.0, str(p))

    assert len(mimi._calls) == 1
    encoded_input = mimi._calls[0]  # [1, 1, T]
    assert encoded_input.shape[0] == 1, "must be batch size 1 (mono)"
    assert float(encoded_input.mean()) == pytest.approx(1.0, abs=1e-5), \
        "channel 0 (all 1.0) should be encoded, not channel 1 (all -1.0)"


def test_stereo_channel1_selects_right(tmp_path):
    """Channel 1 encodes only the right (caller) track."""
    import json
    tok, mimi = _make_tokenizer(channel=1)

    T = 24000
    wav = np.stack([np.ones(T), -np.ones(T)])  # [2, T]; right=-1.0

    info = {"alignments": []}
    p = tmp_path / "audio_stereo.wav"
    p.touch()
    (tmp_path / "audio_stereo.json").write_text(json.dumps(info))

    tok(wav, 0.0, str(p))

    assert len(mimi._calls) == 1
    encoded_input = mimi._calls[0]
    assert float(encoded_input.mean()) == pytest.approx(-1.0, abs=1e-5), \
        "channel 1 (all -1.0) should be encoded"


def test_mono_input_unaffected_by_channel(tmp_path):
    """Mono wav [1, T] passes through correctly regardless of channel setting."""
    import json
    tok, mimi = _make_tokenizer(channel=0)

    T = 24000
    wav = np.ones((1, T))  # [1, T] mono

    info = {"alignments": []}
    p = tmp_path / "audio.wav"
    p.touch()
    (tmp_path / "audio.json").write_text(json.dumps(info))

    tok(wav, 0.0, str(p))

    assert len(mimi._calls) == 1
    encoded_input = mimi._calls[0]
    assert encoded_input.shape == (1, 1, T), "mono input shape must be [1, 1, T]"


def test_stereo_produces_correct_code_shape(tmp_path):
    """Stereo input with channel selection must produce codes with K=9 (1 text + 8 audio)."""
    import json
    tok, _ = _make_tokenizer(channel=0)

    T = 24000
    wav = np.random.randn(2, T).astype(np.float32)

    info = {"alignments": []}
    p = tmp_path / "audio_stereo.wav"
    p.touch()
    (tmp_path / "audio_stereo.json").write_text(json.dumps(info))

    sample = tok(wav, 0.0, str(p))
    # codes shape: [1, K, T_frames] where K = 1 (text) + 8 (audio) = 9
    assert sample.codes.shape[1] == 9, \
        f"Expected 9 codebooks (1 text + 8 audio), got {sample.codes.shape[1]}"
```

- [ ] **Step 2: Verify tests fail**

```bash
cd /Users/joshuazastrow/Github/rehearse
uv run pytest tests/pipeline/test_stereo_channel.py -v 2>&1 | tail -20
```

Expected: FAIL — `InterleavedTokenizer` does not accept `channel` kwarg yet.

- [ ] **Step 3: Add `channel` to `InterleavedTokenizer`**

Edit `lib/moshi-finetune/finetune/data/interleaver.py`.

In `InterleavedTokenizer.__init__` (around line 248), add `channel: int = 0`:

```python
class InterleavedTokenizer:
    def __init__(self, mimi, interleaver, duration_sec: float, channel: int = 0):
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        self.channel = channel
```

In `InterleavedTokenizer.__call__` (around line 254), replace:

```python
        audio_tensor = torch.Tensor(wav).cuda()
        audio_tokens = self.mimi.encode(audio_tensor[:, None])
```

with:

```python
        audio_tensor = torch.Tensor(wav).cuda()  # [C, T]
        if audio_tensor.shape[0] > 1:
            # Stereo input: select provider (0) or caller (1) channel only.
            audio_tensor = audio_tensor[self.channel : self.channel + 1]  # [1, T]
        audio_tokens = self.mimi.encode(audio_tensor[:, None])  # [1, 1, T]
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
uv run pytest tests/pipeline/test_stereo_channel.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git -C lib/moshi-finetune add finetune/data/interleaver.py
git -C lib/moshi-finetune commit -m "feat: add channel selection to InterleavedTokenizer for stereo training"
git add tests/pipeline/test_stereo_channel.py
git commit -m "test: add stereo channel selection tests for InterleavedTokenizer"
```

---

### Task 2: Wire `channel` and `main_speaker_label` through training args

**Files:**
- Modify: `lib/moshi-finetune/finetune/args.py:67-134`
- Modify: `lib/moshi-finetune/train.py:155-166`

- [ ] **Step 1: Write the failing test**

Add to `tests/pipeline/test_stereo_channel.py`:

```python
def test_train_args_has_channel_field():
    """TrainArgs must have channel and main_speaker_label with correct defaults."""
    import sys
    sys.path.insert(0, "lib/moshi-finetune")
    from finetune.args import TrainArgs
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(TrainArgs)}
    assert "channel" in fields, "TrainArgs missing 'channel' field"
    assert fields["channel"].default == 0, "channel must default to 0 (provider)"
    assert "main_speaker_label" in fields, "TrainArgs missing 'main_speaker_label' field"
    assert fields["main_speaker_label"].default == "provider"
```

- [ ] **Step 2: Verify test fails**

```bash
uv run pytest tests/pipeline/test_stereo_channel.py::test_train_args_has_channel_field -v
```

Expected: FAIL — `TrainArgs` has no `channel` field.

- [ ] **Step 3: Add fields to `TrainArgs`**

Edit `lib/moshi-finetune/finetune/args.py`. In `TrainArgs` (after `duration_sec: float = 10` on line 81), add:

```python
    duration_sec: float = 10
    channel: int = 0
    """Audio channel to encode with Mimi. 0 = provider (left), 1 = caller (right).
    Only meaningful when training from audio_stereo.wav (stereo input)."""
    main_speaker_label: str = "provider"
    """Speaker label to supervise text loss on. Must match labels in audio.json
    alignments. Options: 'provider' (coach/AI agent) or 'caller' (person being coached)."""
```

- [ ] **Step 4: Update `train.py` to use both new args**

Edit `lib/moshi-finetune/train.py`.

Replace the `Interleaver` instantiation (lines ~155-163):

```python
    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
        main_speaker_label="provider",
    )
```

with:

```python
    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
        main_speaker_label=args.main_speaker_label,
    )
```

Replace the `InterleavedTokenizer` instantiation (~line 164):

```python
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )
```

with:

```python
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec, channel=args.channel
    )
```

- [ ] **Step 5: Run the new test plus full test suite**

```bash
uv run pytest tests/pipeline/test_stereo_channel.py -v
uv run pytest tests/pipeline/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git -C lib/moshi-finetune add finetune/args.py train.py
git -C lib/moshi-finetune commit -m "feat: add channel and main_speaker_label to TrainArgs, wire into Interleaver and InterleavedTokenizer"
git add tests/pipeline/test_stereo_channel.py
git commit -m "test: add TrainArgs channel/main_speaker_label field assertions"
```

---

### Task 3: Provider and caller model config files

**Files:**
- Modify: `rehearse/models/moshi_7B/config.yaml`
- Create: `rehearse/models/moshi_7B/config_caller.yaml`

- [ ] **Step 1: Write the failing tests**

Add to `tests/pipeline/test_stereo_channel.py`:

```python
def test_provider_config_has_channel_and_label():
    """Provider config must set channel=0 and main_speaker_label=provider."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        Path("rehearse/models/moshi_7B/config.yaml").read_text()
    )
    assert cfg.get("channel") == 0, "provider config must have channel: 0"
    assert cfg.get("main_speaker_label") == "provider"


def test_caller_config_has_channel_and_label():
    """Caller config must set channel=1 and main_speaker_label=caller."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        Path("rehearse/models/moshi_7B/config_caller.yaml").read_text()
    )
    assert cfg.get("channel") == 1, "caller config must have channel: 1"
    assert cfg.get("main_speaker_label") == "caller"
```

- [ ] **Step 2: Verify tests fail**

```bash
uv run pytest tests/pipeline/test_stereo_channel.py::test_provider_config_has_channel_and_label tests/pipeline/test_stereo_channel.py::test_caller_config_has_channel_and_label -v
```

Expected: FAIL — config fields don't exist yet.

- [ ] **Step 3: Add fields to the provider config**

Edit `rehearse/models/moshi_7B/config.yaml`. Add two lines after `duration_sec: 100`:

```yaml
duration_sec: 100
channel: 0
main_speaker_label: provider
```

- [ ] **Step 4: Create the caller config**

Create `rehearse/models/moshi_7B/config_caller.yaml`. This is a minimal override — inherits everything from `config.yaml` implicitly (the operator provides both base paths and a `--config` override in CLI, or passes this file directly):

```yaml
# Caller model training config.
# Train on the right channel (caller speech) only.
# Pass this file to rehearse-train instead of the default config.yaml.
#
# Usage:
#   rehearse-train config_path=rehearse/models/moshi_7B/config_caller.yaml \
#       data.train_data=runs/sessions.jsonl \
#       run_dir=runs/caller-v1
data:
  train_data: ''
  eval_data: ''
lora:
  enable: true
  rank: 128
  scaling: 2.
optim:
  lr: 2.0e-6
  weight_decay: 0.1
  pct_start: 0.05
batch_size: 16
max_steps: 2000
duration_sec: 100
channel: 1
main_speaker_label: caller
do_eval: false
first_codebook_weight_multiplier: 100.
text_padding_weight: 0.5
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/pipeline/test_stereo_channel.py -v
uv run pytest tests/pipeline/ -v
```

Expected: all 7 tests PASS (4 tokenizer + 1 TrainArgs + 2 config).

- [ ] **Step 6: Commit**

```bash
git add rehearse/models/moshi_7B/config.yaml \
        rehearse/models/moshi_7B/config_caller.yaml \
        tests/pipeline/test_stereo_channel.py
git commit -m "feat: add channel/main_speaker_label to training configs for provider and caller models"
```

---

## Verification

After all tasks, do a dry-run of both model configs to confirm the routing is correct:

```bash
# Provider model (default)
rehearse-train dry_run=true \
    config_path=rehearse/models/moshi_7B/config.yaml \
    data.train_data=runs/sessions.jsonl \
    run_dir=runs/provider-v1

# Caller model
rehearse-train dry_run=true \
    config_path=rehearse/models/moshi_7B/config_caller.yaml \
    data.train_data=runs/sessions.jsonl \
    run_dir=runs/caller-v1
```

Both should print the resolved config. Confirm `channel: 0` / `main_speaker_label: provider` and `channel: 1` / `main_speaker_label: caller` respectively.

## What remains out of scope

- FD-V1.5 / FD-V3 evaluation harness — requires a LiveKit-compatible Modal inference endpoint (separate plan)
- Downloading trained checkpoints from Modal Volume (manual `modal volume get` for now)
- `eval_data` path rewriting in `_rewrite_config_paths` for caller eval set (no eval sessions designated yet)
