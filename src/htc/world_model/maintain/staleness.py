"""Staleness detection — diff the current on-disk state of `sources` against
a persisted manifest, deterministically, with no wall-clock/mtime involved.
"""

from __future__ import annotations

from pathlib import Path

from ...adapters.base import Source
from ..ingest.corpus import iter_ingestible_files
from .state import content_hash, load_manifest, manifest_path


def check_staleness(
    sources: list[Source], root: str | Path, manifest: dict[str, dict]
) -> dict[str, list[str]]:
    """Classify every currently-ingestible source path (under `sources`)
    against `manifest`:

    - "new": ingestible now, absent from the manifest
    - "changed": present in both, but the content hash differs
    - "deleted": present in the manifest, absent now
    - "fresh": present in both with an identical content hash

    Returns source paths (relative to `root`, posix-style), each list sorted
    for determinism.
    """
    root_path = Path(root).expanduser().resolve()
    current_by_path = {
        f.relative_to(root_path).as_posix(): f for f in iter_ingestible_files(sources, root_path)
    }

    new: list[str] = []
    changed: list[str] = []
    fresh: list[str] = []
    for rel_path, file_path in current_by_path.items():
        entry = manifest.get(rel_path)
        if entry is None:
            new.append(rel_path)
        elif content_hash(file_path) != entry.get("hash"):
            changed.append(rel_path)
        else:
            fresh.append(rel_path)

    deleted = [rel_path for rel_path in manifest if rel_path not in current_by_path]

    return {
        "new": sorted(new),
        "changed": sorted(changed),
        "deleted": sorted(deleted),
        "fresh": sorted(fresh),
    }


def count_stale(sources: list[Source], root: str | Path) -> int | None:
    """Count of sources that have drifted from the persisted manifest
    (changed + deleted). Returns `None` if no manifest exists yet — nothing
    to compare against, so a caller (e.g. an eval run record) can omit the
    field entirely rather than reporting a misleading zero. Lets a
    displayed Agent-Ready score honestly note when it may be stale."""
    root_path = Path(root).expanduser().resolve()
    if not manifest_path(root_path).is_file():
        return None
    manifest = load_manifest(root_path)
    staleness = check_staleness(sources, root_path, manifest)
    return len(staleness["changed"]) + len(staleness["deleted"])
