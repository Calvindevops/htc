"""Run every agent on the ladder against every task in the bank.

Reuses the `_cmd_agent` pattern from `htc.evaluation.runner`: pipe the task
prompt to the agent's CLI command, run inside the repo root. Human agents
(`agent_cmd is None`) are skipped here — their attempts are supplied
externally (see `htc.study.grading`).
"""

from __future__ import annotations

from pathlib import Path

from ..evaluation.runner import _cmd_agent
from .model import AgentSpec, Attempt, Task


def run_attempts(bank: list[Task], agents: list[AgentSpec], root: str | Path) -> list[Attempt]:
    """Run each non-human agent against each task, in bank order then agent
    order, for a deterministic result list."""
    root_path = Path(root).expanduser().resolve()
    attempts: list[Attempt] = []
    for agent in agents:
        if agent.agent_cmd is None:
            continue
        for task in bank:
            output = _cmd_agent(agent.agent_cmd, root_path, task.prompt)
            attempts.append(Attempt(task_id=task.id, agent_id=agent.id, output=output))
    return attempts
