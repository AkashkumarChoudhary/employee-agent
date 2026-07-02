import os
import tempfile
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from langgraph.types import Command
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from employee_agent.api.auth import require_api_key
from employee_agent.api.schemas import (
    ApproveRequest,
    CreateJobResponse,
    JobResponse,
    TraceResponse,
    TraceStep,
)
from employee_agent.api.store import JobStore
from employee_agent.config import Settings, get_settings
from employee_agent.engine.checkpoint import sqlite_checkpointer
from employee_agent.engine.graph import build_graph
from employee_agent.engine.state import new_state
from employee_agent.mcp_tools.client import MCPToolClient
from employee_agent.providers.factory import build_provider
from employee_agent.rag import ingest
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}

router = APIRouter()


def _thread(job_id: str) -> dict:
    return {"configurable": {"thread_id": job_id}}


def _status_from_result(result: dict) -> str:
    if "__interrupt__" in result:
        return "awaiting_human"
    return result.get("status", "done")


def _owned_or_404(request: Request, job_id: str, api_key: str):
    rec = request.app.state.jobs.get(job_id)
    if rec is None or rec.owner != api_key:
        raise HTTPException(status_code=404, detail="job not found")
    return rec


def _graph_for(app, saver):
    return build_graph(
        app.state.provider, app.state.retriever,
        checkpointer=saver, tool_client=app.state.tool_client,
    )


def _rate_key(request: Request) -> str:
    return request.headers.get("x-api-key") or (
        request.client.host if request.client else "anon"
    )


# slowapi invokes the limit provider with no arguments, so the active limit is
# read from this module global, which create_app sets from the app's settings.
_RATE_LIMIT = {"value": "100/minute"}


def _limit_value() -> str:
    return _RATE_LIMIT["value"]


limiter = Limiter(key_func=_rate_key)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/whoami")
async def whoami(api_key: str = Depends(require_api_key)):
    return {"api_key": api_key}


@router.post("/jobs", response_model=CreateJobResponse)
@limiter.limit(_limit_value)
async def create_job(
    request: Request,
    job_description: str = Form(...),
    role: str = Form("hr_analyst"),
    resume: UploadFile = File(...),
    api_key: str = Depends(require_api_key),
):
    suffix = Path(resume.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"unsupported file type: {suffix!r}")
    data = await resume.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")
    try:
        get_role(role)
    except KeyError:
        raise HTTPException(status_code=422, detail=f"unknown role: {role!r}")

    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        resume_text = ingest.load_document(tmp)
    finally:
        os.unlink(tmp)

    job_id = uuid.uuid4().hex
    request.app.state.jobs.create(job_id, owner=api_key, role=role)
    state = new_state(job_id, get_role(role), job_description, resume_text)
    async with sqlite_checkpointer(request.app.state.settings.sqlite_path) as saver:
        graph = _graph_for(request.app, saver)
        result = await graph.ainvoke(state, _thread(job_id))
    status = _status_from_result(result)
    request.app.state.jobs.set_status(job_id, status)
    return CreateJobResponse(job_id=job_id, status=status)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, request: Request, api_key: str = Depends(require_api_key)):
    rec = _owned_or_404(request, job_id, api_key)
    async with sqlite_checkpointer(request.app.state.settings.sqlite_path) as saver:
        graph = _graph_for(request.app, saver)
        snap = await graph.aget_state(_thread(job_id))
    assessment = snap.values.get("assessment")
    return JobResponse(
        job_id=job_id, status=rec.status,
        assessment=assessment.model_dump() if assessment else None,
    )


@router.post("/jobs/{job_id}/approve", response_model=JobResponse)
async def approve_job(
    job_id: str, body: ApproveRequest, request: Request,
    api_key: str = Depends(require_api_key),
):
    _owned_or_404(request, job_id, api_key)
    async with sqlite_checkpointer(request.app.state.settings.sqlite_path) as saver:
        graph = _graph_for(request.app, saver)
        result = await graph.ainvoke(Command(resume=body.model_dump()), _thread(job_id))
    status = _status_from_result(result)
    request.app.state.jobs.set_status(job_id, status)
    assessment = result.get("assessment")
    return JobResponse(
        job_id=job_id, status=status,
        assessment=assessment.model_dump() if assessment else None,
    )


@router.get("/jobs/{job_id}/trace", response_model=TraceResponse)
async def job_trace(job_id: str, request: Request, api_key: str = Depends(require_api_key)):
    _owned_or_404(request, job_id, api_key)
    async with sqlite_checkpointer(request.app.state.settings.sqlite_path) as saver:
        graph = _graph_for(request.app, saver)
        history = [snap async for snap in graph.aget_state_history(_thread(job_id))]
    steps: list[TraceStep] = []
    for snap in reversed(history):
        if snap.next:
            node = snap.next[0]
            if node != "__start__":
                steps.append(TraceStep(step=(snap.metadata or {}).get("step", 0), node=node))
    return TraceResponse(job_id=job_id, steps=steps)


def create_app(settings: Settings | None = None, provider=None) -> FastAPI:
    s = settings or get_settings()
    _RATE_LIMIT["value"] = s.rate_limit
    app = FastAPI(title="Employee Agent API")
    prov = provider or build_provider(s)
    app.state.settings = s
    app.state.provider = prov
    app.state.retriever = Retriever(prov, VectorStore(path=s.chroma_path))
    app.state.tool_client = MCPToolClient(allowlist={"verify_certification"})
    app.state.jobs = JobStore()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(router)
    return app
