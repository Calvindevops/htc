"""Task bank load/save — the pre-registered set of tasks the study runs
every agent against."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from .model import Task

MIN_RECOMMENDED_TASKS = 8


def load_bank(path: str | Path) -> list[Task]:
    data = json.loads(Path(path).expanduser().read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of tasks")
    tasks = [Task(**item) for item in data]
    if len(tasks) < MIN_RECOMMENDED_TASKS:
        print(
            f"warning: task bank has {len(tasks)} tasks, "
            f"fewer than the recommended {MIN_RECOMMENDED_TASKS}",
            file=sys.stderr,
        )
    return tasks


def save_bank(tasks: list[Task], path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(t) for t in tasks], indent=2) + "\n")
    return out
