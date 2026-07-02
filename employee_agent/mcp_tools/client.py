import json

from mcp.shared.memory import create_connected_server_and_client_session

from employee_agent.mcp_tools.server import build_mcp_server


class MCPToolError(Exception):
    """Raised when an MCP tool call is disallowed or fails."""


class MCPToolClient:
    def __init__(self, server=None, allowlist: set[str] | None = None):
        self._server = server or build_mcp_server()
        self._allowlist = set(allowlist or [])

    async def call(self, tool: str, args: dict) -> dict:
        if tool not in self._allowlist:
            raise MCPToolError(f"tool not allowlisted: {tool}")
        async with create_connected_server_and_client_session(self._server) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
        if result.isError:
            raise MCPToolError(f"tool {tool} returned an error")
        return json.loads(result.content[0].text)
