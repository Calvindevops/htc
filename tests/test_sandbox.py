"""Docker sandbox: argv construction, preflight, and non-raising failure modes."""

from __future__ import annotations

import subprocess

import pytest

from htc.sandbox import SandboxConfig, SandboxError, build_docker_argv, run_in_sandbox


class TestBuildDockerArgv:
    def test_default_config_shape(self, tmp_path):
        cfg = SandboxConfig()
        argv = build_docker_argv("claude -p", tmp_path, cfg)

        assert argv[:4] == ["docker", "run", "--rm", "-i"]
        mount_index = argv.index("-v")
        mount = argv[mount_index + 1]
        assert mount.endswith(":/repo:ro")
        assert mount.split(":")[0] == str(tmp_path.resolve())
        assert tmp_path.resolve().is_absolute()
        assert "-w" in argv and argv[argv.index("-w") + 1] == "/repo"
        assert argv[argv.index("--network") + 1] == "bridge"
        assert argv[argv.index("--memory") + 1] == "2g"
        assert argv[argv.index("--cpus") + 1] == "2"
        assert argv[-4:] == [cfg.image, "sh", "-c", "claude -p"]

    def test_network_none(self, tmp_path):
        cfg = SandboxConfig(network="none")
        argv = build_docker_argv("cmd", tmp_path, cfg)
        assert argv[argv.index("--network") + 1] == "none"

    def test_env_passthrough(self, tmp_path):
        cfg = SandboxConfig(env_passthrough=("OPENAI_API_KEY", "ANTHROPIC_API_KEY"))
        argv = build_docker_argv("cmd", tmp_path, cfg)
        assert argv.count("-e") == 2
        e_indices = [i for i, a in enumerate(argv) if a == "-e"]
        forwarded = [argv[i + 1] for i in e_indices]
        assert forwarded == ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]


class TestRunInSandbox:
    def test_raises_sandbox_error_when_docker_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("htc.sandbox.docker_available", lambda: False)
        with pytest.raises(SandboxError, match="Docker not found"):
            run_in_sandbox("claude -p", tmp_path, "question", SandboxConfig())

    def test_happy_path_returns_stdout(self, tmp_path, monkeypatch):
        monkeypatch.setattr("htc.sandbox.docker_available", lambda: True)
        monkeypatch.setattr(
            "htc.sandbox.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=[], returncode=0, stdout="the answer\n", stderr=""
            ),
        )
        result = run_in_sandbox("claude -p", tmp_path, "question", SandboxConfig())
        assert result == "the answer"

    def test_nonzero_exit_returns_failure_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr("htc.sandbox.docker_available", lambda: True)
        monkeypatch.setattr(
            "htc.sandbox.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="boom"
            ),
        )
        result = run_in_sandbox("claude -p", tmp_path, "question", SandboxConfig())
        assert "exit 1" in result
        assert "boom" in result

    def test_timeout_returns_timeout_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr("htc.sandbox.docker_available", lambda: True)

        def _raise_timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="docker", timeout=300)

        monkeypatch.setattr("htc.sandbox.subprocess.run", _raise_timeout)
        result = run_in_sandbox("claude -p", tmp_path, "question", SandboxConfig())
        assert result == "(sandboxed agent timed out)"
