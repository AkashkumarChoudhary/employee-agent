from mcp.server.fastmcp import FastMCP

_KNOWN_CERTIFICATIONS = {
    "pmp", "cfa", "cissp", "aws certified solutions architect",
    "pmp certification", "python", "scrum master",
}


def build_mcp_server() -> FastMCP:
    server = FastMCP("employee-agent-tools")

    @server.tool()
    def verify_certification(name: str, certification: str) -> dict:
        """Verify a candidate certification against a mock registry."""
        verified = certification.strip().lower() in _KNOWN_CERTIFICATIONS
        return {
            "name": name,
            "certification": certification,
            "verified": verified,
            "source": "mock-registry",
        }

    return server
