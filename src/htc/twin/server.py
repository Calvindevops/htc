"""Build the twin's stdio launch params from a CompanyAdapter, and list its tools.

The returned `StdioServerParameters` is what the ART rollout connects to (same
shape as the mcp-rl examples). `list_tools` is a convenience for the CLI.
"""

from __future__ import annotations

import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..adapters.base import CompanyAdapter


def twin_server_params(
    adapter: CompanyAdapter, graph_path: str | None = None
) -> StdioServerParameters:
    """Launch params for the read-only twin MCP server over this adapter's repo."""
    root = adapter.workspace_spec().repo_path
    env = {"HTC_TWIN_ROOT": root}
    if graph_path:
        env["HTC_TWIN_GRAPH"] = graph_path
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "htc.twin.mcp_server"],
        env=env,
    )


async def list_tools(params: StdioServerParameters) -> list[str]:
    """Connect to the twin and return the names of its exposed tools."""
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [t.name for t in result.tools]
