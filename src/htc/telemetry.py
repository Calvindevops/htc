"""Opt-in, anonymous usage telemetry.

Telemetry is **OFF by default**. On the first run of any `htc` command (no
stored preference yet), `ensure_preference()` prints a one-time notice and
persists `telemetry: false` to `~/.htc/config.json` — it never turns
telemetry on by itself. `HTC_TELEMETRY=0/1` always overrides the stored
preference.

When (and only when) telemetry is enabled AND `HTC_POSTHOG_KEY` is set,
`track()` posts a single anonymized event to PostHog's capture endpoint.
Sending is best-effort: any network/library failure is swallowed and never
propagates to the caller.

NEVER include in event props: file contents, repo names/paths, questions,
answers, or keys. Only coarse, pre-bucketed values (command name, size
bucket, provider, duration bucket, score bucket, counts).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx

POSTHOG_CAPTURE_URL = "https://us.i.posthog.com/capture/"

NOTICE = (
    "htc: anonymous usage telemetry is OFF by default.\n"
    "  If enabled, htc sends only anonymized event counts (command name,\n"
    "  size/score buckets) — never file contents, paths, questions,\n"
    "  answers, or keys.\n"
    "  Opt in any time:  export HTC_TELEMETRY=1\n"
    "  This notice will not show again.\n"
)


def _config_dir() -> Path:
    override = os.environ.get("HTC_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".htc"


def _config_path() -> Path:
    return _config_dir() / "config.json"


def _load_config() -> dict[str, Any]:
    path = _config_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_config(config: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))


def _env_override() -> bool | None:
    raw = os.environ.get("HTC_TELEMETRY")
    if raw is None:
        return None
    return raw not in ("0", "false", "False", "")


def ensure_preference(print_notice: bool = True) -> bool:
    """Ensure a telemetry preference is stored; return whether telemetry is
    enabled for this run. First run only: print the opt-in notice and
    persist a default-OFF preference (never auto-enables)."""
    config = _load_config()
    if "telemetry" not in config:
        if print_notice:
            print(NOTICE)
        config["telemetry"] = False
        config.setdefault("install_id", uuid.uuid4().hex)
        _save_config(config)

    override = _env_override()
    if override is not None:
        return override
    return bool(config.get("telemetry", False))


def is_enabled() -> bool:
    """Whether telemetry is currently enabled, honoring the env override.
    Does not print the notice or write config (see `ensure_preference`)."""
    override = _env_override()
    if override is not None:
        return override
    return bool(_load_config().get("telemetry", False))


def _install_id() -> str:
    config = _load_config()
    install_id = config.get("install_id")
    if not install_id:
        install_id = uuid.uuid4().hex
        config["install_id"] = install_id
        _save_config(config)
    return install_id


def track(event: str, props: dict[str, Any] | None = None) -> None:
    """Send one anonymized event, if telemetry is enabled and a PostHog key
    is configured. No-op otherwise. Never raises."""
    if not is_enabled():
        return
    key = os.environ.get("HTC_POSTHOG_KEY")
    if not key:
        return
    try:
        httpx.post(
            POSTHOG_CAPTURE_URL,
            json={
                "api_key": key,
                "event": event,
                "properties": {**(props or {}), "distinct_id": _install_id()},
            },
            timeout=2.0,
        )
    except Exception:
        pass


def bucket_repo_size(num_files: int) -> str:
    """Anonymized order-of-magnitude bucket for a file count."""
    if num_files <= 0:
        return "0"
    if num_files < 50:
        return "1-49"
    if num_files < 500:
        return "50-499"
    if num_files < 5000:
        return "500-4999"
    return "5000+"


def bucket_duration(seconds: float) -> str:
    """Anonymized coarse bucket for a wall-clock duration."""
    if seconds < 5:
        return "<5s"
    if seconds < 30:
        return "5-30s"
    if seconds < 120:
        return "30-120s"
    if seconds < 600:
        return "2-10m"
    return "10m+"


def bucket_score(score: float) -> str:
    """Anonymized coarse bucket for an Agent-Ready score (0-100)."""
    if score >= 90:
        return "90-100"
    if score >= 75:
        return "75-89"
    if score >= 50:
        return "50-74"
    if score >= 25:
        return "25-49"
    return "0-24"
