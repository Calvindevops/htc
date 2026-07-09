"""Data model for the correlation study: agent-ladder x task-bank, blind
grading, and the resulting statistics.

This is the machinery that VALIDATES the Agent-Ready score — it does not
compute the score itself (see `htc.evaluation`). A human runs and grades the
study later; these types are the shared contract between the run, grading,
and stats stages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """One study task: a real event drawn from company history."""

    id: str
    prompt: str
    category: str
    provenance: str


@dataclass(frozen=True)
class AgentSpec:
    """One rung on the agent ladder, e.g. base / partial-handbook / full-handbook / human."""

    id: str
    label: str
    agent_cmd: str | None  # None for human — attempts supplied externally


@dataclass(frozen=True)
class Attempt:
    """One agent's output for one task."""

    task_id: str
    agent_id: str
    output: str


@dataclass(frozen=True)
class Grade:
    """A human grader's score for one attempt. 0=harmful 1=redo 2=major-edits
    3=minor-edits 4=ship-as-is."""

    task_id: str
    agent_id: str
    grader_id: str
    score: int
