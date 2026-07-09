"""Read-only company twin, exposed as an MCP stdio server.

Launched as a subprocess by `twin/server.py` (the ART rollout connects to it over
stdio, exactly like the mcp-rl AlphaVantage example). Tools are read-only and
path-confined to `HTC_TWIN_ROOT`. An optional graphify `graph.json` at
`HTC_TWIN_GRAPH` enables `query_graph`.

Run directly:  HTC_TWIN_ROOT=/path python -m htc.twin.mcp_server
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ..adapters.base import EXCLUDED_DIRS

ROOT = Path(os.environ["HTC_TWIN_ROOT"]).expanduser().resolve()
GRAPH_PATH = os.environ.get("HTC_TWIN_GRAPH", "")
MAX_CHARS = 16_000  # keep tool output well under the rollout's 20k cap

mcp = FastMCP("htc-twin")


def _safe(rel: str) -> Path:
    """Resolve a path and confine it inside ROOT (block traversal)."""
    target = (ROOT / rel).resolve()
    if target != ROOT and ROOT not in target.parents:
        raise ValueError(f"path escapes twin root: {rel}")
    if set(target.relative_to(ROOT).parts) & EXCLUDED_DIRS:
        raise ValueError(f"path is inside an excluded directory: {rel}")
    return target


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n...[truncated]"


def _is_excluded(p: Path) -> bool:
    return bool(set(p.relative_to(ROOT).parts) & EXCLUDED_DIRS)


def _rg(args: list[str]) -> str | None:
    """Run ripgrep if available; return None if rg is missing."""
    try:
        globs = [f"!{d}/**" for d in EXCLUDED_DIRS]
        out = subprocess.run(
            [
                "rg",
                "--no-config",
                "-n",
                "--max-columns",
                "300",
                *(f"-g{g}" for g in globs),
                *args,
                str(ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout or "(no matches)"
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return "(search timed out)"


@mcp.tool()
def search_files(name: str) -> str:
    """Find files whose path contains `name` (case-insensitive)."""
    hits = [
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file() and name.lower() in p.name.lower() and not _is_excluded(p)
    ]
    return _clip("\n".join(sorted(hits)[:200]) or "(no files matched)")


@mcp.tool()
def grep(pattern: str) -> str:
    """Search file contents for `pattern` (regex). Returns path:line:match."""
    rg = _rg(["--", pattern])
    if rg is not None:
        return _clip(rg)
    # Fallback: naive Python scan (no ripgrep installed)
    import re

    rx = re.compile(pattern)
    out: list[str] = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or _is_excluded(p):
            continue
        try:
            for i, line in enumerate(p.read_text("utf-8", "ignore").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{p.relative_to(ROOT)}:{i}:{line.strip()[:200]}")
                    if len(out) >= 200:
                        return _clip("\n".join(out))
        except OSError:
            continue
    return _clip("\n".join(out) or "(no matches)")


@mcp.tool()
def read_file(path: str) -> str:
    """Read a file (relative to the company root)."""
    target = _safe(path)
    if not target.is_file():
        return f"(not a file: {path})"
    return _clip(target.read_text("utf-8", "ignore"))


@mcp.tool()
def query_graph(query: str) -> str:
    """Look up nodes in the company knowledge graph whose id/label contains `query`."""
    if not GRAPH_PATH or not Path(GRAPH_PATH).is_file():
        return "(no knowledge graph configured for this twin)"
    try:
        data = json.loads(Path(GRAPH_PATH).read_text("utf-8", "ignore"))
    except json.JSONDecodeError:
        return "(graph unreadable)"
    nodes = data.get("nodes", [])
    q = query.lower()
    matches = [
        n
        for n in nodes
        if q in str(n.get("id", "")).lower() or q in str(n.get("label", "")).lower()
    ]
    return _clip(json.dumps(matches[:40], indent=2) or "(no nodes matched)")


if __name__ == "__main__":
    mcp.run()
