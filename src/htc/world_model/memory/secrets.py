"""Secret storage for HTC — API keys never touch plaintext `~/.htc/config.json`.

Prefers the OS credential store via the optional `keyring` library (macOS
Keychain / Windows Credential Manager / Linux Secret Service, `pip install
htc[embed]`). Falls back to a `~/.htc/credentials.json` file with restrictive
`0600` permissions when keyring is unavailable (not installed, no backend, or
raises for any other reason). Best-effort only — never raises.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

_SERVICE = "htc"
_CREDENTIALS_PATH = Path.home() / ".htc" / "credentials.json"


def _load_credentials_file() -> dict:
    if _CREDENTIALS_PATH.is_file():
        try:
            return json.loads(_CREDENTIALS_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_credentials_file(credentials: dict) -> None:
    _CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CREDENTIALS_PATH.write_text(json.dumps(credentials, indent=2, sort_keys=True) + "\n")
    os.chmod(_CREDENTIALS_PATH, stat.S_IRUSR | stat.S_IWUSR)


def save_secret(name: str, value: str) -> str:
    """Save `value` under `name`, preferring the OS credential store.

    Returns which method was used: `"keyring"` or `"file"`. Never raises —
    on any keyring failure it falls back to the `0600` file and prints a
    one-line notice.
    """
    try:
        import keyring

        keyring.set_password(_SERVICE, name, value)
        return "keyring"
    except Exception:
        print(
            "htc: no OS keyring available — saving credential to "
            "~/.htc/credentials.json (permissions 0600)"
        )
        credentials = _load_credentials_file()
        credentials[name] = value
        _save_credentials_file(credentials)
        return "file"


def load_secret(name: str) -> str | None:
    """Read `name` from the OS credential store first, then the `0600` file.
    Returns `None` if absent from both."""
    try:
        import keyring

        value = keyring.get_password(_SERVICE, name)
        if value is not None:
            return value
    except Exception:
        pass
    return _load_credentials_file().get(name)
