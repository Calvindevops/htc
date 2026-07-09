"""Docker sandbox for `--agent-cmd` eval mode.

Piping a golden's question to an external agent CLI via `shell=True` on the
host gives that agent full filesystem access. Sandbox mode isolates it: the
agent command runs inside a container with the repo mounted read-only and no
other host access. This only applies to `--agent-cmd` — the builtin agent
runs in-process, already path-confined via `_twin_tools`, and is unaffected
by `--sandbox`.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class SandboxError(RuntimeError):
    """Raised when the sandbox cannot run (e.g. Docker missing)."""


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "python:3.12-slim"
    network: str = "bridge"
    env_passthrough: tuple[str, ...] = field(default_factory=tuple)
    memory: str = "2g"
    cpus: str = "2"
    timeout: int = 300


def docker_available() -> bool:
    """True if the `docker` CLI is present and responds."""
    try:
        out = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0


def build_docker_argv(cmd: str, root: Path, cfg: SandboxConfig) -> list[str]:
    """Build the `docker run` argv for executing `cmd` against the repo."""
    abs_root = Path(root).expanduser().resolve()
    argv = [
        "docker",
        "run",
        "--rm",
        "-i",
        "-v",
        f"{abs_root}:/repo:ro",
        "-w",
        "/repo",
        "--network",
        cfg.network,
        "--memory",
        cfg.memory,
        "--cpus",
        cfg.cpus,
    ]
    for name in cfg.env_passthrough:
        argv += ["-e", name]
    argv += [cfg.image, "sh", "-c", cmd]
    return argv


def run_in_sandbox(cmd: str, root: Path, question: str, cfg: SandboxConfig) -> str:
    """Run `cmd` inside a Docker container, piping `question` on stdin.

    Mirrors the return-string style of the host `_cmd_agent`: timeouts and
    nonzero exits produce a descriptive string rather than raising, so a
    per-item sandbox failure behaves the same as a per-item agent failure.
    """
    if not docker_available():
        raise SandboxError("Docker not found — install Docker or run without --sandbox")
    argv = build_docker_argv(cmd, root, cfg)
    try:
        out = subprocess.run(
            argv,
            input=question,
            capture_output=True,
            text=True,
            timeout=cfg.timeout,
        )
    except subprocess.TimeoutExpired:
        return "(sandboxed agent timed out)"
    if out.returncode != 0:
        return f"(sandboxed agent failed, exit {out.returncode}; stderr: {out.stderr[:300]})"
    answer = out.stdout.strip()
    return answer or f"(sandboxed agent produced no output; stderr: {out.stderr[:300]})"
