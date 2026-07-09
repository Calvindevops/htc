"""Per-source provenance manifest — content hash + chunk ids, persisted to
`<root>/.htc/memory/manifest.json`.

Keyed on content hash (SHA-256 of the file's bytes), never mtime — mtime
isn't deterministic across checkouts, git clones, or test fixtures, so it
can't be asserted on and shouldn't drive staleness decisions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_REL_PATH = Path(".htc") / "memory" / "manifest.json"


def manifest_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / MANIFEST_REL_PATH


def content_hash(path: Path) -> str:
    """SHA-256 hex digest of `path`'s bytes — the sole staleness key."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(root: str | Path) -> dict[str, dict]:
    """Load the persisted manifest, or `{}` if none exists yet / it's corrupt."""
    path = manifest_path(root)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def save_manifest(root: str | Path, manifest: dict[str, dict]) -> None:
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
