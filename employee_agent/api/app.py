from fastapi import APIRouter, Depends, FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from employee_agent.api.auth import require_api_key
from employee_agent.api.store import JobStore
from employee_agent.config import Settings, get_settings
from employee_agent.providers.factory import build_provider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore

router = APIRouter()


def _rate_key(request: Request) -> str:
    return request.headers.get("x-api-key") or (
        request.client.host if request.client else "anon"
    )


def _limit_value(request: Request) -> str:
    return request.app.state.settings.rate_limit


limiter = Limiter(key_func=_rate_key)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/whoami")
async def whoami(api_key: str = Depends(require_api_key)):
    return {"api_key": api_key}


def create_app(settings: Settings | None = None, provider=None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="Employee Agent API")
    prov = provider or build_provider(s)
    app.state.settings = s
    app.state.provider = prov
    app.state.retriever = Retriever(prov, VectorStore(path=s.chroma_path))
    app.state.jobs = JobStore()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(router)
    return app
