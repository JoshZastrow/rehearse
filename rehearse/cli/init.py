"""rehearse-init — interactive onboarding wizard.

Walks through required env vars, optional infra (memory backend, Modal judge,
Modal interactive backend), and writes/patches the project .env file.

Usage:
    rehearse-init            # full wizard
    rehearse-init --force    # re-enter all values, even already-set ones
    rehearse-init --env-only # skip infra deploys, just collect env vars
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RED = "\033[31m"


def _h(text: str) -> str:
    return f"{_BOLD}{_CYAN}{text}{_RESET}"


def _ok(text: str) -> str:
    return f"{_GREEN}✓{_RESET} {text}"


def _warn(text: str) -> str:
    return f"{_YELLOW}!{_RESET} {text}"


def _err(text: str) -> str:
    return f"{_RED}✗{_RESET} {text}"


def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"


def _ask(prompt: str, default: str = "", secret: bool = False) -> str:
    """Prompt the user for a value, returning default if they press Enter."""
    hint = f" [{_dim('****' if (secret and default) else (default or 'none'))}]" if default else ""
    try:
        if secret and default:
            import getpass
            value = getpass.getpass(f"  {prompt}{hint}: ").strip()
        else:
            value = input(f"  {prompt}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return value or default


def _confirm(prompt: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"  {prompt} {_dim(hint)}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    if not answer:
        return default
    return answer in ("y", "yes")


def _section(title: str) -> None:
    print(f"\n{_h('─── ' + title + ' ───')}")


def _load_dotenv(path: Path) -> dict[str, str]:
    """Read .env into a dict, preserving only KEY=VALUE lines (skipping comments)."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def _patch_dotenv(path: Path, updates: dict[str, str]) -> None:
    """Write/patch .env: updates existing keys in-place, appends new ones.

    Preserves all comments and unset keys from the existing file.
    """
    lines: list[str] = path.read_text().splitlines() if path.exists() else []
    written: set[str] = set()

    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Check if this line sets one of our keys (including commented-out ones)
        m = re.match(r"^#?\s*([A-Z0-9_]+)\s*=", stripped)
        if m:
            key = m.group(1)
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                written.add(key)
                continue
        new_lines.append(line)

    # Append keys not already present in the file
    remaining = {k: v for k, v in updates.items() if k not in written}
    if remaining:
        new_lines.append("")
        new_lines.append("# Added by rehearse-init")
        for k, v in remaining.items():
            new_lines.append(f"{k}={v}")

    path.write_text("\n".join(new_lines) + "\n")


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command, streaming output. Returns exit code."""
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def _modal_token() -> str:
    """Return the active Modal token ID from `modal token info`, or empty string."""
    try:
        result = subprocess.run(
            ["modal", "token", "info"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Token:"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return ""


def _modal_workspace() -> str:
    """Return the active Modal workspace slug from `modal token info`, or empty string."""
    try:
        result = subprocess.run(
            ["modal", "token", "info"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Workspace:"):
                    # "Workspace: joshzastrow (ac-xxxxx)" → "joshzastrow"
                    return line.split(":", 1)[1].strip().split()[0]
    except FileNotFoundError:
        pass
    return ""


def _modal_app_endpoint(app_name: str, class_name: str, method: str, path: str) -> str:
    """Derive the Modal ASGI WebSocket endpoint URL for a class-based app.

    Pattern: wss://<workspace>--<app-name>-<classname>-<method>.modal.run/<path>
    """
    workspace = _modal_workspace()
    if not workspace:
        return ""
    slug = f"{workspace}--{app_name}-{class_name}-{method}".replace("_", "-")
    return f"wss://{slug}.modal.run/{path.lstrip('/')}"


def _repo_root() -> Path:
    """Find the project root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------

def _collect_core(existing: dict[str, str], force: bool) -> dict[str, str]:
    """Prompt for required API keys and server settings."""
    _section("Core API Keys")
    print(_dim("  Required for all backend types.\n"))

    updates: dict[str, str] = {}

    fields = [
        ("TWILIO_ACCOUNT_SID", "Twilio account SID", False),
        ("TWILIO_AUTH_TOKEN", "Twilio auth token", True),
        ("TWILIO_PHONE_NUMBER", "Twilio phone number (e.g. +15551234567)", False),
        ("HUME_API_KEY", "Hume API key", True),
        ("HUME_CONFIG_ID", "Hume voice config ID", False),
        ("ANTHROPIC_API_KEY", "Anthropic API key", True),
        ("BASE_URL", "Public base URL (ngrok or fly.io, e.g. https://abc.ngrok.io)", False),
    ]

    for key, label, secret in fields:
        current = existing.get(key, "")
        if current and not force:
            print(_ok(f"{key} already set"))
            updates[key] = current
        else:
            value = _ask(label, default=current, secret=secret)
            if value:
                updates[key] = value
            elif current:
                updates[key] = current
            else:
                print(_warn(f"{key} left blank — server will fail to start without it"))

    return updates


def _collect_backend(existing: dict[str, str], force: bool, env_only: bool, root: Path) -> dict[str, str]:
    """Ask which backend to use and collect backend-specific settings."""
    _section("Backend Type")
    current_backend = existing.get("BACKEND_TYPE", "interactive")

    print("  Available backends:")
    print("    1) managed     — Hume EVI handles voice, STT, TTS")
    print("    2) pipeline    — Open-source STT (Whisper) + TTS (Kokoro) + LiteLLM")
    print("    3) interactive — Full-duplex PersonaPlex model (local GPU or Modal cloud) (default)")
    print()

    choice_map = {"1": "managed", "2": "pipeline", "3": "interactive",
                  "managed": "managed", "pipeline": "pipeline", "interactive": "interactive"}
    default_num = {"managed": "1", "pipeline": "2", "interactive": "3"}.get(current_backend, "3")

    raw = _ask(f"Backend (1/2/3 or name)", default=default_num)
    backend = choice_map.get(raw.strip(), current_backend)

    updates: dict[str, str] = {"BACKEND_TYPE": backend}

    if backend == "interactive":
        updates.update(_collect_interactive(existing, force, env_only, root))

    return updates


def _collect_interactive(existing: dict[str, str], force: bool, env_only: bool, root: Path) -> dict[str, str]:
    """Ask local-vs-Modal for the interactive backend."""
    print()
    current_provider = existing.get("INTERACTIVE_PROVIDER_ENDPOINT", "")
    current_caller = existing.get("INTERACTIVE_CALLER_ENDPOINT", "")

    if current_provider:
        print(_ok(f"Modal provider endpoint already set: {current_provider}"))
        if not force and not _confirm("Re-deploy interactive servers?", default=False):
            return {
                "INTERACTIVE_PROVIDER_ENDPOINT": current_provider,
                "INTERACTIVE_CALLER_ENDPOINT": current_caller,
            }

    print("  How should the interactive backend run?")
    print("    1) Local GPU   — loads Moshi weights on this machine")
    print("    2) Modal cloud — deploys provider + caller to A10G GPUs via `modal deploy`")
    print()
    mode = _ask("Mode (1/2)", default="2" if current_provider else "1")

    updates: dict[str, str] = {}

    if mode == "1" or mode.lower().startswith("l"):
        ckpt = _ask(
            "Checkpoint path (local dir with model weights, empty = HF Hub download)",
            default=existing.get("INTERACTIVE_CHECKPOINT_PATH", ""),
        )
        updates["INTERACTIVE_CHECKPOINT_PATH"] = ckpt
        updates["INTERACTIVE_PROVIDER_ENDPOINT"] = ""
        updates["INTERACTIVE_CALLER_ENDPOINT"] = ""
        updates["INTERACTIVE_DEVICE"] = _ask("Device", default=existing.get("INTERACTIVE_DEVICE", "cuda"))
    else:
        if env_only:
            print(_warn(
                "Skipping Modal deploy (--env-only). Set INTERACTIVE_PROVIDER_ENDPOINT "
                "and INTERACTIVE_CALLER_ENDPOINT manually."
            ))
            updates["INTERACTIVE_PROVIDER_ENDPOINT"] = current_provider
            updates["INTERACTIVE_CALLER_ENDPOINT"] = current_caller
        else:
            print()
            print(_dim("  Deploying interactive inference servers to Modal (A10G)..."))
            rc = _run(["modal", "deploy", "infra/interactive.py"], cwd=root)
            if rc != 0:
                print(_err(
                    "modal deploy failed. Set INTERACTIVE_PROVIDER_ENDPOINT and "
                    "INTERACTIVE_CALLER_ENDPOINT manually in .env."
                ))
                updates["INTERACTIVE_PROVIDER_ENDPOINT"] = current_provider
                updates["INTERACTIVE_CALLER_ENDPOINT"] = current_caller
            else:
                provider_url = _modal_app_endpoint("rehearse-interactive", "providerserver", "serve", "ws")
                caller_url = _modal_app_endpoint("rehearse-interactive", "callerserver", "serve", "ws")
                if provider_url:
                    print(_ok(f"Provider endpoint derived from workspace: {provider_url}"))
                if caller_url:
                    print(_ok(f"Caller endpoint derived from workspace: {caller_url}"))
                provider = _ask(
                    "Provider wss:// endpoint" if not provider_url else "Confirm or override provider endpoint",
                    default=provider_url or current_provider,
                )
                caller = _ask(
                    "Caller wss:// endpoint" if not caller_url else "Confirm or override caller endpoint",
                    default=caller_url or current_caller,
                )
                updates["INTERACTIVE_PROVIDER_ENDPOINT"] = provider
                updates["INTERACTIVE_CALLER_ENDPOINT"] = caller

    return updates


def _setup_judge(existing: dict[str, str], env_only: bool, root: Path) -> dict[str, str]:
    """Optionally deploy the LLM judge to Modal."""
    _section("LLM Judge (optional)")
    print(_dim("  Gemma-based audio judge for evals. Requires Modal.\n"))

    current_url = existing.get("VLLM_BASE_URL", "")
    if current_url:
        print(_ok(f"VLLM_BASE_URL already set: {current_url}"))
        if not _confirm("Re-deploy judge?", default=False):
            return {}

    if not _confirm("Deploy judge to Modal?", default=not env_only):
        return {}

    if env_only:
        print(_warn("Skipping deploy (--env-only)."))
        return {}

    # Ensure modal CLI is available
    if subprocess.run(["modal", "--version"], capture_output=True).returncode != 0:
        print(_warn("modal not found — running: uv pip install modal && modal setup"))
        _run(["uv", "pip", "install", "modal"], cwd=root)
        _run(["modal", "setup"], cwd=root)

    print(_dim("  Deploying judge..."))
    rc = _run(["modal", "deploy", "infra/judge.py"], cwd=root)
    if rc != 0:
        print(_err("modal deploy infra/judge.py failed. Set VLLM_BASE_URL manually."))
        return {}

    print()
    vllm_url = _ask("Paste the VLLM_BASE_URL printed above", default=current_url)

    auto_token = _modal_token()
    current_key = existing.get("VLLM_API_KEY", "")
    if auto_token and not current_key:
        print(_ok(f"Modal token detected automatically: {auto_token[:12]}..."))
        vllm_key = auto_token
    else:
        vllm_key = _ask("VLLM_API_KEY (Modal token)", default=current_key or auto_token, secret=True)

    updates: dict[str, str] = {}
    if vllm_url:
        updates["VLLM_BASE_URL"] = vllm_url
    if vllm_key:
        updates["VLLM_API_KEY"] = vllm_key
    return updates


def _setup_memory(existing: dict[str, str], env_only: bool, root: Path) -> dict[str, str]:
    """Optionally set up caller memory (persists consent + history across sessions)."""
    _section("Caller Memory (optional)")
    print(_dim("  Persists caller consent and history across sessions.\n"))

    memory_dir = root / "lib" / "honcho"
    current_url = existing.get("HONCHO_BASE_URL", "")
    current_key = existing.get("HONCHO_API_KEY", "")
    current_mcp = existing.get("MEMORY_MCP_URL", "")

    if memory_dir.exists():
        print(_ok("Self-hosted memory backend already installed"))
        print(_dim("  `make serve` will start it automatically."))
        return {}

    if current_mcp:
        print(_ok(f"MEMORY_MCP_URL set: {current_mcp}"))
        return {}

    if current_url:
        print(_ok(f"Memory backend URL set: {current_url}"))
        return {}

    if current_key:
        print(_ok("Memory cloud API key set"))
        return {}

    print("  Memory options:")
    print("    1) Self-hosted  — installs a local memory server, no account needed")
    print("    2) Cloud        — provide an API key for the hosted memory service")
    print("    3) Skip         — no caller memory (every caller treated as new)")
    print()
    choice = _ask("Choice (1/2/3)", default="3")

    if choice == "1":
        if env_only:
            print(_warn("Skipping install (--env-only). Run `make setup-honcho` manually."))
            return {}
        print(_dim("  Installing self-hosted memory backend..."))
        rc = _run(["make", "setup-honcho"], cwd=root)
        if rc != 0:
            print(_err("Memory backend setup failed. Check output above."))
        else:
            print(_ok("Memory backend installed. `make serve` will start it automatically."))
        return {}

    elif choice == "2":
        key = _ask("Memory service API key", secret=True)
        if key:
            return {"HONCHO_API_KEY": key}

    return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import fire
    fire.Fire(_wizard)


def _wizard(force: bool = False, env_only: bool = False) -> None:
    """Interactive setup wizard for Rehearse.

    Args:
        force: Re-prompt for all values, even ones already set in .env.
        env_only: Collect env vars only; skip all `modal deploy` and `make` commands.
    """
    root = _repo_root()
    env_path = root / ".env"

    print(f"\n{_BOLD}Rehearse setup wizard{_RESET}")
    print(_dim(f"  Project root : {root}"))
    print(_dim(f"  .env path    : {env_path}"))
    if env_only:
        print(_dim("  Mode         : env-only (no deploys)"))
    if force:
        print(_dim("  Mode         : force (re-enter all values)"))

    existing = _load_dotenv(env_path)
    updates: dict[str, str] = {}

    # Phase 1: core keys
    updates.update(_collect_core(existing, force))

    # Phase 2: backend
    updates.update(_collect_backend(existing, force, env_only, root))

    # Phase 3: optional infra
    updates.update(_setup_judge(existing, env_only, root))
    updates.update(_setup_memory(existing, env_only, root))

    # Write .env
    _section("Writing .env")
    _patch_dotenv(env_path, updates)
    print(_ok(f"Updated {env_path}"))

    # Summary
    print(f"\n{_BOLD}Setup complete.{_RESET} Next steps:\n")
    backend = updates.get("BACKEND_TYPE", existing.get("BACKEND_TYPE", "interactive"))
    if backend == "interactive":
        endpoint = updates.get("INTERACTIVE_PROVIDER_ENDPOINT", "")
        if endpoint:
            print("  • INTERACTIVE_PROVIDER_ENDPOINT is set → calls route to Modal")
        else:
            print("  • INTERACTIVE_PROVIDER_ENDPOINT is empty → inference runs locally (needs GPU)")
    print("  • Run `make serve` to start the server")
    print("  • Run `make test` to verify the setup")
    print()
