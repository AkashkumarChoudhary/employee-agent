import pytest

from employee_agent.mcp_tools.client import MCPToolClient, MCPToolError


async def test_call_allowlisted_tool_returns_dict():
    client = MCPToolClient(allowlist={"verify_certification"})
    out = await client.call("verify_certification", {"name": "Ada", "certification": "CISSP"})
    assert out["verified"] is True
    assert out["name"] == "Ada"


async def test_unknown_cert_returns_not_verified():
    client = MCPToolClient(allowlist={"verify_certification"})
    out = await client.call("verify_certification", {"name": "Ada", "certification": "nope"})
    assert out["verified"] is False


async def test_non_allowlisted_tool_raises():
    client = MCPToolClient(allowlist=set())
    with pytest.raises(MCPToolError):
        await client.call("verify_certification", {"name": "Ada", "certification": "PMP"})
