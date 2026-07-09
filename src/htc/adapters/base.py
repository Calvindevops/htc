"""The OSS extension point.

A `CompanyAdapter` tells HTC where a company's knowledge lives, which repo to
sandbox, and (optionally) the held-out Q&A used to prove the trained agent beats
base. Core ships `FilesystemAdapter`; private deployments implement their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

SourceKind = str  # "repo" | "docs" | "iac" | "schema" | "tickets"

# Directories the twin (and any agent under evaluation) must never read from —
# `.htc/` holds the answer key (goldens.json, results.json); leaking it invalidates
# the eval. Shared by evaluation/runner.py and twin/mcp_server.py.
EXCLUDED_DIRS = {".git", "node_modules", ".venv", ".htc"}


@dataclass(frozen=True)
class Source:
    """One ingestible location for the world-model."""

    path: str  # directory or glob
    kind: SourceKind = "repo"


@dataclass(frozen=True)
class WorkspaceSpec:
    """The repo to git-worktree into a resettable sandbox for the twin."""

    repo_path: str
    writable: bool = False  # P1 read-only; P2 flips this on for mutating tasks


@dataclass(frozen=True)
class QAItem:
    """A held-out evaluation question with the artifact a correct answer must cite."""

    question: str
    answer: str
    artifact: str  # file path / graph node id / env var / table name


@runtime_checkable
class CompanyAdapter(Protocol):
    """The contract core depends on. Implement this to point HTC at any company."""

    def name(self) -> str:
        """Stable identifier for this company (used in artifact paths)."""
        ...

    def sources(self) -> list[Source]:
        """Locations to ingest into the world-model."""
        ...

    def workspace_spec(self) -> WorkspaceSpec:
        """Which repo to sandbox for the twin."""
        ...

    def golden_qa(self) -> list[QAItem] | None:
        """Optional held-out Q&A with reference answers for deterministic eval."""
        ...

    def rubric(self) -> str:
        """Natural-language RULER rubric for the task family."""
        ...
