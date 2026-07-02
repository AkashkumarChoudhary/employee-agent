from fastapi import Header, HTTPException, Request


async def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> str:
    allowed = request.app.state.settings.allowed_api_keys()
    if not x_api_key or x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return x_api_key
