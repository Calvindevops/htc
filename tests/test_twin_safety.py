"""The builtin agent's tools must stay confined to the repo root."""

from __future__ import annotations

import pytest

from htc.evaluation.runner import _twin_tools


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "inside.txt").write_text("safe")
    (tmp_path.parent / "outside.txt").write_text("SECRET")
    return tmp_path


class TestPathConfinement:
    def test_read_inside_works(self, repo):
        _, impls = _twin_tools(repo)
        assert impls["read_file"]("inside.txt") == "safe"

    def test_traversal_is_blocked(self, repo):
        _, impls = _twin_tools(repo)
        with pytest.raises(ValueError, match="escapes repo root"):
            impls["read_file"]("../outside.txt")

    def test_absolute_path_stays_confined(self, repo):
        _, impls = _twin_tools(repo)
        # Path("/etc/passwd") joined onto root resolves to /etc/passwd — must be blocked.
        with pytest.raises(ValueError, match="escapes repo root"):
            impls["read_file"]("/etc/passwd")

    def test_missing_file_is_soft_error(self, repo):
        _, impls = _twin_tools(repo)
        assert "not a file" in impls["read_file"]("ghost.txt")

    def test_search_skips_git(self, repo):
        (repo / ".git").mkdir()
        (repo / ".git" / "config-inside.txt").write_text("x")
        _, impls = _twin_tools(repo)
        assert "config-inside" not in impls["search_files"]("config-inside")


class TestHtcAnswerKeyExcluded:
    """The `.htc/` dir holds the answer key (goldens.json, results.json) — the
    agent under evaluation must never be able to read it."""

    @pytest.fixture
    def repo_with_htc(self, tmp_path):
        htc_dir = tmp_path / ".htc"
        htc_dir.mkdir()
        (htc_dir / "goldens.json").write_text('[{"question": "leaked?"}]')
        (tmp_path / "app.py").write_text("print('hi')")
        return tmp_path

    def test_read_file_rejects_htc(self, repo_with_htc):
        _, impls = _twin_tools(repo_with_htc)
        with pytest.raises(ValueError, match="excluded directory"):
            impls["read_file"](".htc/goldens.json")

    def test_search_files_does_not_surface_htc(self, repo_with_htc):
        _, impls = _twin_tools(repo_with_htc)
        assert "goldens" not in impls["search_files"]("goldens")

    def test_grep_content_does_not_surface_htc(self, repo_with_htc):
        _, impls = _twin_tools(repo_with_htc)
        assert "leaked" not in impls["grep_content"]("leaked")
