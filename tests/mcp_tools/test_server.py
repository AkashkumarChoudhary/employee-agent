import json

from mcp.shared.memory import create_connected_server_and_client_session

from employee_agent.mcp_tools.server import build_mcp_server


async def _call(cert: str) -> dict:
    server = build_mcp_server()
    async with create_connected_server_and_client_session(server) as session:
        await session.initialize()
        res = await session.call_tool(
            "verify_certification", {"name": "Ada", "certification": cert}
        )
    assert res.isError is False
    return json.loads(res.content[0].text)


async def test_known_certification_verified():
    out = await _call("PMP")
    assert out["verified"] is True
    assert out["source"] == "mock-registry"


async def test_unknown_certification_not_verified():
    out = await _call("underwater basket weaving")
    assert out["verified"] is False
