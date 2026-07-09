"""The blind grading workflow.

A human must not know which agent produced which output while scoring —
that's what keeps the correlation study honest. `make_grading_sheet` shuffles
attempts with an explicit seed and assigns opaque `blind_id`s; a human works
from the `blind_id` / `task_prompt` / `output` columns only. The `task_id` /
`agent_id` fields also present in each row are the reversal mapping consumed
by `ingest_grades` after grading — a grading UI built on top of this sheet
should render only the blind columns to the human.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .model import Attempt, Grade, Task


def make_grading_sheet(attempts: list[Attempt], tasks: list[Task], seed: int) -> list[dict]:
    """Shuffle attempts deterministically (given `seed`), assign opaque
    `blind_id`s, and pair each with its task prompt for human scoring."""
    prompts_by_id = {task.id: task.prompt for task in tasks}
    order = list(range(len(attempts)))
    random.Random(seed).shuffle(order)
    sheet = []
    for i, index in enumerate(order):
        attempt = attempts[index]
        sheet.append(
            {
                "blind_id": f"blind-{i:03d}",
                "task_id": attempt.task_id,
                "agent_id": attempt.agent_id,
                "task_prompt": prompts_by_id.get(attempt.task_id, ""),
                "output": attempt.output,
            }
        )
    return sheet


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


def ingest_grades(sheet: list[dict], filled_scores: dict[str, int], grader_id: str) -> list[Grade]:
    """Map a human's filled-in {blind_id: score} back to (task_id, agent_id)
    via the sheet's saved mapping."""
    rows_by_blind_id = {row["blind_id"]: row for row in sheet}
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
