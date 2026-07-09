"""Self-maintenance for HTC memory — gBrain's self-healing loop: staleness
detection, incremental refresh, and near-duplicate collapsing, all
deterministic (content-hash keyed, no wall-clock/mtime)."""

from .dedup import dedup_chunks
from .refresh import refresh_memory
from .staleness import check_staleness, count_stale
from .state import content_hash, load_manifest, manifest_path, save_manifest

__all__ = [
    "check_staleness",
    "content_hash",
    "count_stale",
    "dedup_chunks",
    "load_manifest",
    "manifest_path",
    "refresh_memory",
    "save_manifest",
]
