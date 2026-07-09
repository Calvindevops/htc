"""The blind grading workflow.

A human must not know which agent produced which output while scoring —
that's what keeps the correlation study honest. `make_grading_sheet` shuffles
attempts with an explicit seed, assigns opaque `blind_id`s, and splits the
result into TWO artifacts:

- the GRADER-FACING sheet: only `blind_id` / `task_prompt` / `output` columns
  — this is the file a human actually opens and scores.
- the PRIVATE key: `blind_id` -> `task_id` / `agent_id`, used solely by
  `ingest_grades` to reverse a human's filled-in scores back to the real
  attempt. It must never be shown to the grader.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .model import Attempt, Grade, Task


def make_grading_sheet(
    attempts: list[Attempt], tasks: list[Task], seed: int
) -> tuple[list[dict], list[dict]]:
    """Shuffle attempts deterministically (given `seed`), assign opaque
    `blind_id`s, and pair each with its task prompt for human scoring.

    Returns `(grader_sheet, key)`:
    - `grader_sheet`: rows with only `blind_id` / `task_prompt` / `output` —
      safe to hand to a human grader.
    - `key`: rows with `blind_id` / `task_id` / `agent_id` — private, used by
      `ingest_grades` to reverse the mapping. Never show this to the grader.
    """
    prompts_by_id = {task.id: task.prompt for task in tasks}
    order = list(range(len(attempts)))
    random.Random(seed).shuffle(order)
    grader_sheet = []
    key = []
    for i, index in enumerate(order):
        attempt = attempts[index]
        blind_id = f"blind-{i:03d}"
        grader_sheet.append(
            {
                "blind_id": blind_id,
                "task_prompt": prompts_by_id.get(attempt.task_id, ""),
                "output": attempt.output,
            }
        )
        key.append(
            {
                "blind_id": blind_id,
                "task_id": attempt.task_id,
                "agent_id": attempt.agent_id,
            }
        )
    return grader_sheet, key


def save_grading_sheet(sheet: list[dict], path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sheet, indent=2) + "\n")
    return out


def load_grading_sheet(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).expanduser().read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of sheet rows")
    return data


def ingest_grades(key: list[dict], filled_scores: dict[str, int], grader_id: str) -> list[Grade]:
    """Map a human's filled-in {blind_id: score} back to (task_id, agent_id)
    via the private `key` (see `make_grading_sheet`)."""
    rows_by_blind_id = {row["blind_id"]: row for row in key}
    grades = []
    for blind_id, score in filled_scores.items():
        row = rows_by_blind_id.get(blind_id)
        if row is None:
            raise ValueError(f"unknown blind_id in filled_scores: {blind_id}")
        grades.append(
            Grade(
                task_id=row["task_id"],
                agent_id=row["agent_id"],
                grader_id=grader_id,
                score=score,
            )
        )
    return grades
