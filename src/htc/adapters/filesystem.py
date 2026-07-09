"""Reference adapter: point HTC at any local repository.

This is what makes the OSS story real — a stranger clones htc-core, points
`FilesystemAdapter` at their repo, and runs `htc twin` with zero private code.
"""

from __future__ import annotations

from pathlib import Path

from .base import QAItem, Source, WorkspaceSpec

_DEFAULT_RUBRIC = """\
Rank the trajectories by how well the final answer answers the question about
this codebase. Reward answers that are correct, grounded in real files the agent
actually read (via tool calls), and concise. An answer that does not cite the
correct file/symbol/value, or that was not supported by a tool call, must rank
below any answer that did. Penalize confident-but-unsupported prose."""


class FilesystemAdapter:
    """A `CompanyAdapter` over a single local repo directory."""

    def __init__(self, root: str):
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(f"root is not a directory: {resolved}")
        self._root = resolved

    @property
    def root(self) -> Path:
        return self._root

    def name(self) -> str:
        return self._root.name

    def sources(self) -> list[Source]:
        return [Source(path=str(self._root), kind="repo")]

    def workspace_spec(self) -> WorkspaceSpec:
        return WorkspaceSpec(repo_path=str(self._root), writable=False)

    def golden_qa(self) -> list[QAItem] | None:
        return None  # no curated eval set for a generic repo

    def rubric(self) -> str:
        return _DEFAULT_RUBRIC
