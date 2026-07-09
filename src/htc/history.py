"""Local results/report history — the user's own data, always on.

Every `goldens`/`eval`/`onboard`/`handbook` run appends one JSON line to
`<root>/.htc/history/runs.jsonl`. No privacy concern: this never leaves the
machine and holds nothing the user didn't already write to `.htc/`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

HISTORY_REL_PATH = Path(".htc") / "history" / "runs.jsonl"


def _history_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / HISTORY_REL_PATH


def record_run(
    root: str | Path,
    kind: str,
    summary: dict[str, Any],
    now: float | None = None,
) -> dict[str, Any]:
    """Append one history entry and return it.

    `now` is an injectable unix timestamp for deterministic tests; real
    callers may omit it, in which case the current wall-clock time is used.
    The entry's `index` is a monotonic counter (count of prior entries).
    """
    path = _history_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    if path.is_file():
        with path.open() as f:
            index = sum(1 for _ in f)
    entry = {
        "index": index,
        "kind": kind,
        "timestamp": time.time() if now is None else now,
        "summary": summary,
    }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_history(root: str | Path) -> list[dict[str, Any]]:
    """Load all recorded runs, oldest first."""
    path = _history_path(root)
    if not path.is_file():
        return []
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def score_trend(root: str | Path) -> list[float]:
    """Eval scores over time, in run order."""
    return [
        entry["summary"]["score"]
        for entry in load_history(root)
        if entry.get("kind") == "eval" and "score" in entry.get("summary", {})
    ]
